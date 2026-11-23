# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""The raw layer against the runner's fixture charms."""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import pathlib
import signal
import zipfile
from typing import Any

import pytest
import scriptlet_testing
from scriptlet_testing import (
    Context,
    Intent,
    NotMeasuredError,
    RunnerError,
    StepLimitError,
    Versions,
)
from scriptlet_testing._process import NativeProcess, find_runner

from .conftest import BUNDLED, TESTDATA, skip_if_bundled, write_charm


def event_file(name: str) -> tuple[str, dict[str, Any]]:
    data = json.loads((TESTDATA / 'events' / f'{name}.json').read_text())
    return data['name'], data.get('attrs') or {}


@pytest.fixture
def set_status():
    with Context.from_charm(TESTDATA / 'set_status') as ctx:
        yield ctx


def pid(ctx: Context) -> int:
    backend = ctx._backend
    assert isinstance(backend, NativeProcess)
    return backend.pid


# --- intents --------------------------------------------------------------


def test_set_status_intent(set_status: Context):
    out = set_status.run_raw(*event_file('config_changed_port'))
    assert out.ok
    assert out.observed
    assert out.intents == [
        Intent('set-status', {'status': 'active', 'message': 'serving on 8080'}),
    ]
    assert out.intents_of_type('set-status') == out.intents
    assert out.intents_of_type('other') == []


def test_float_stays_float(set_status: Context):
    out = set_status.run_raw(*event_file('config_changed_port_float'))
    assert out.intents == [
        Intent('set-status', {'status': 'blocked', 'message': 'port must be an int, got float'}),
    ]


def test_default_message(set_status: Context):
    out = set_status.run_raw('update_status')
    assert out.intents == [Intent('set-status', {'status': 'active', 'message': ''})]


def test_unobserved_event(set_status: Context):
    out = set_status.run_raw('relation_broken')
    assert not out.observed
    assert out.intents == []
    assert out.logs == []


def test_packed_charm(tmp_path: pathlib.Path):
    src = TESTDATA / 'set_status'
    archive = tmp_path / 'set-status.charm'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('metadata.yaml', 'name: set-status\nassumes: [unitless]\n')
        for p in sorted((src / 'scriptlets').rglob('*.star')):
            z.write(p, p.relative_to(src).as_posix())
    with Context.from_charm(archive) as ctx:
        out = ctx.run_raw('config_changed', {'port': 80, 'ratio': 0.5})
    assert out.intents == [Intent('set-status', {'status': 'active', 'message': 'serving on 80'})]
    assert ctx.load.sources == ['charm.star', 'lib/checks.star']


# --- logs -----------------------------------------------------------------


def test_logs_split_load_and_event(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path / 'charm',
        {
            'scriptlets/hook.star': 'print("top level")\n\n'
            'def handler(event):\n'
            '    print("handling", event.n)\n'
            '    debug("detail")\n\n'
            'def init():\n'
            '    print("in init")\n'
            '    juju.observe("config_changed", handler)\n'
        },
    )
    with Context.from_charm(root) as ctx:
        assert ctx.load.logs == ['top level', 'in init']
        out = ctx.run_raw('config_changed', {'n': 3})
    assert out.logs == ['handling 3', 'detail']
    assert [e.level for e in out.log_entries] == ['print', 'debug']
    assert [e.juju_level for e in out.log_entries] == ['INFO', 'DEBUG']
    first = out.log_entries[0]
    assert (first.path, first.line, first.event) == ('hook.star', 4, 'config_changed')
    assert first.juju == 'scriptlet hook.star:4: handling 3'
    assert out.load_logs == ['top level', 'in init']
    assert all(e.event == '<load>' for e in ctx.load.log_entries)


# --- measuring ------------------------------------------------------------


def test_not_measured_by_default(set_status: Context):
    out = set_status.run_raw(*event_file('config_changed_port'))
    assert not out.measured
    assert out.step_budget is not None and out.step_budget.used is None
    with pytest.raises(NotMeasuredError, match='measure=True'):
        _ = out.steps
    with pytest.raises(NotMeasuredError):
        _ = out.allocs
    with pytest.raises(NotMeasuredError):
        _ = out.load_steps


def test_measure_per_call(set_status: Context):
    out = set_status.run_raw(*event_file('config_changed_port'), measure=True)
    assert out.measured
    # Measured once and stable: runs are deterministic.
    assert out.steps == 116
    assert out.allocs == 3416
    assert out.load_steps > 0
    # The next call is back to the context's default.
    assert not set_status.run_raw(*event_file('config_changed_port')).measured


def test_measure_per_context():
    with Context.from_charm(TESTDATA / 'step_limit', measure=True) as ctx:
        assert ctx.load.step_budget is not None and ctx.load.step_budget.measured
        out = ctx.run_raw(*event_file('relation_changed_100'))
        assert out.steps == 5545
        assert out.intents == [
            Intent('set-status', {'status': 'active', 'message': '0 bad of 100'})
        ]
        assert not ctx.run_raw(*event_file('relation_changed_100'), measure=False).measured


def test_unobserved_measures_zero(set_status: Context):
    out = set_status.run_raw('relation_broken', measure=True)
    assert out.steps == 0


def test_limits(tmp_path: pathlib.Path):
    with Context.from_charm(TESTDATA / 'set_status', max_steps=500, max_allocs=50_000) as ctx:
        assert (ctx.max_steps, ctx.max_allocs) == (500, 50_000)
        out = ctx.run_raw('update_status', measure=True)
        assert out.step_budget is not None and out.step_budget.limit == 500
        with pytest.raises(ValueError, match='positive'):
            ctx.run_raw('update_status', max_steps=0)
    with Context.from_charm(TESTDATA / 'set_status') as ctx:
        assert (ctx.max_steps, ctx.max_allocs) == (100_000, 10 << 20)


# --- the process ----------------------------------------------------------


def test_process_reused(set_status: Context):
    first = pid(set_status)
    a = set_status.run_raw('update_status')
    b = set_status.run_raw(*event_file('config_changed_port'))
    assert pid(set_status) == first
    # Loaded once by from_charm, then reused.
    assert a.load is not None and a.load.cached
    assert b.load is not None and b.load.cached


def test_contexts_have_their_own_process():
    with (
        Context.from_charm(TESTDATA / 'set_status') as a,
        Context.from_charm(TESTDATA / 'set_status') as b,
    ):
        assert pid(a) != pid(b)


def test_clean_shutdown():
    ctx = Context.from_charm(TESTDATA / 'set_status')
    backend = ctx._backend
    assert isinstance(backend, NativeProcess)
    ctx.close()
    assert ctx.closed
    assert backend.returncode == 0
    ctx.close()  # idempotent
    with pytest.raises(RunnerError, match='closed'):
        ctx.run_raw('update_status')


def test_context_manager_closes():
    with Context.from_charm(TESTDATA / 'set_status') as ctx:
        pass
    assert ctx.closed


def test_garbage_collected_context_stops_runner():
    ctx = Context.from_charm(TESTDATA / 'set_status')
    backend = ctx._backend
    assert isinstance(backend, NativeProcess)
    del ctx
    gc.collect()
    assert backend.closed
    assert backend.returncode == 0


def test_failed_start_leaves_no_process(tmp_path: pathlib.Path):
    before = set(os.listdir('/proc'))
    with pytest.raises(Exception, match='while loops') as exc:
        Context.from_charm(TESTDATA / 'syntax_error')
    # No gc.collect(), and the traceback (which holds the half-built
    # Context) is still alive: the constructor itself must stop the runner.
    assert exc.traceback
    # No runner child left behind (zombies included).
    children = [
        p for p in set(os.listdir('/proc')) - before if p.isdigit() and _ppid(p) == os.getpid()
    ]
    assert children == []


def _ppid(pid: str) -> int:
    try:
        stat = pathlib.Path(f'/proc/{pid}/stat').read_text()
    except OSError:
        return -1
    return int(stat.rsplit(')', 1)[1].split()[1])


def test_runner_dies(set_status: Context):
    os.kill(pid(set_status), signal.SIGKILL)
    with pytest.raises(RunnerError, match=r'exited|cannot write'):
        set_status.run_raw('update_status')


@pytest.mark.skipif(not BUNDLED, reason='only when testing an installed wheel')
def test_bundled_runner():
    path = find_runner()
    assert path == pathlib.Path(scriptlet_testing.__file__).parent / '_bin' / 'runner'
    assert os.access(path, os.X_OK)


@skip_if_bundled
def test_no_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('SCRIPTLET_TESTING_RUNNER')
    with pytest.raises(RunnerError, match='SCRIPTLET_TESTING_RUNNER'):
        Context.from_charm(TESTDATA / 'set_status')
    with pytest.raises(RunnerError, match='does not exist'):
        Context.from_charm(TESTDATA / 'set_status', runner='/nonexistent/runner')


def _fake_runner(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    script = tmp_path / 'fake-runner'
    script.write_text(
        '#!/usr/bin/env python3\nimport json, sys\n'
        'for line in sys.stdin:\n'
        '    req = json.loads(line)\n'
        f'{body}\n'
        '    sys.stdout.write(json.dumps(resp) + "\\n")\n'
        '    sys.stdout.flush()\n'
    )
    script.chmod(0o755)
    return script


def test_cap_hit_leaves_used_unknown():
    data = json.loads((TESTDATA / 'events' / 'relation_changed_2000.json').read_text())
    with (
        Context.from_charm(TESTDATA / 'step_limit', cap_factor=1) as ctx,
        pytest.raises(StepLimitError) as exc,
    ):
        ctx.run_raw(data['name'], data['attrs'], measure=True)
    result = exc.value.result
    assert result is not None and result.step_budget is not None
    assert result.step_budget.cap == 100_000
    with pytest.raises(NotMeasuredError, match='search cap'):
        _ = result.steps


def test_non_object_response(tmp_path: pathlib.Path):
    fake = _fake_runner(tmp_path, '    resp = [req.get("id")]')
    with pytest.raises(RunnerError, match='non-object'):
        Context.from_charm(TESTDATA / 'set_status', runner=fake)


def test_close_sends_shutdown(tmp_path: pathlib.Path):
    # A fake that only exits on the shutdown op (as a runner whose stdin is
    # also held by a forked child would), recording that it got one.
    marker = tmp_path / 'shutdown'
    fake = _fake_runner(
        tmp_path,
        f'    if req.get("op") == "shutdown":\n'
        f'        open({str(marker)!r}, "w").close()\n'
        f'        sys.exit(0)\n'
        '    resp = {"id": req.get("id"), "ok": True, "protocol": 1,'
        ' "load": {"steps": {"limit": 1}}}',
    )
    ctx = Context.from_charm(TESTDATA / 'set_status', runner=fake)
    ctx.close()
    assert marker.exists()


def test_protocol_mismatch(tmp_path: pathlib.Path):
    fake = _fake_runner(tmp_path, '    resp = {"id": req.get("id"), "ok": True, "protocol": 2}')
    with pytest.raises(RunnerError, match='protocol 2'):
        Context.from_charm(TESTDATA / 'set_status', runner=fake)


def test_response_for_another_request(tmp_path: pathlib.Path):
    fake = _fake_runner(tmp_path, '    resp = {"id": 999, "ok": True, "protocol": 1}')
    with pytest.raises(RunnerError, match='answered request 999'):
        Context.from_charm(TESTDATA / 'set_status', runner=fake)


def test_garbage_response(tmp_path: pathlib.Path):
    script = tmp_path / 'fake-runner'
    script.write_text('#!/bin/sh\nread line\necho not json\n')
    script.chmod(0o755)
    with pytest.raises(RunnerError, match='invalid JSON'):
        Context.from_charm(TESTDATA / 'set_status', runner=script)


def test_ok_false_without_error(tmp_path: pathlib.Path):
    fake = _fake_runner(
        tmp_path,
        '    resp = {"id": req.get("id"), "ok": req.get("op") != "run", "protocol": 1}\n'
        '    if req.get("op") == "load":\n'
        '        resp["load"] = {"steps": {"limit": 1}, "allocs": {"limit": 1}}',
    )
    with (
        Context.from_charm(TESTDATA / 'set_status', runner=fake) as ctx,
        pytest.raises(RunnerError, match='no error'),
    ):
        ctx.run_raw('update_status')


def test_from_charm_uses_load_op(tmp_path: pathlib.Path):
    # from_charm() asks for a load, with the context's options, and runs no
    # event; the load result becomes ctx.load.
    log = tmp_path / 'requests'
    fake = _fake_runner(
        tmp_path,
        f'    open({str(log)!r}, "a").write(json.dumps(req) + "\\n")\n'
        '    resp = {"id": req.get("id"), "ok": True, "protocol": 1,'
        ' "load": {"sources": ["a.star"], "steps": {"limit": 7}, "allocs": {"limit": 9}}}',
    )
    with Context.from_charm(TESTDATA / 'set_status', runner=fake, max_steps=7) as ctx:
        assert ctx.load.sources == ['a.star']
        assert (ctx.max_steps, ctx.max_allocs) == (7, 9)
    reqs = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r['op'] for r in reqs] == ['version', 'load', 'shutdown']
    load = reqs[1]
    assert load['charm'] == str((TESTDATA / 'set_status').resolve())
    assert load['options'] == {'measure': False, 'max_steps': 7}
    assert 'event' not in load


def test_load_failed_without_error(tmp_path: pathlib.Path):
    fake = _fake_runner(
        tmp_path, '    resp = {"id": req.get("id"), "ok": req.get("op") != "load", "protocol": 1}'
    )
    with pytest.raises(RunnerError, match='failed load with no error'):
        Context.from_charm(TESTDATA / 'set_status', runner=fake)


def test_load_without_load_result(tmp_path: pathlib.Path):
    fake = _fake_runner(tmp_path, '    resp = {"id": req.get("id"), "ok": True, "protocol": 1}')
    with pytest.raises(RunnerError, match='no load result'):
        Context.from_charm(TESTDATA / 'set_status', runner=fake)


def test_load_then_run_reuses_load():
    with Context.from_charm(TESTDATA / 'set_status') as ctx:
        assert not ctx.load.cached
        assert ctx.load.sources == ['charm.star', 'lib/checks.star']
        out = ctx.run_raw('update_status')
        assert out.load is not None and out.load.cached


def test_stderr_in_error(tmp_path: pathlib.Path):
    script = tmp_path / 'fake-runner'
    script.write_text('#!/bin/sh\necho "something broke" >&2\nexit 3\n')
    script.chmod(0o755)
    with pytest.raises(RunnerError, match='something broke'):
        Context.from_charm(TESTDATA / 'set_status', runner=script)


def test_versions_report_juju():
    with Context.from_charm(TESTDATA / 'set_status') as ctx:
        v = ctx.versions
    # The pinned commit isn't in a Juju release yet (step 4 log).
    assert v.juju_tag is None
    assert v.juju_version == '4.2-beta1'
    assert v.juju == f'4.2-beta1 ({v.juju_commit[:12]}, untagged)'
    tagged = dataclasses.replace(v, juju_tag='v4.2.0')
    assert tagged.juju == f'v4.2.0 ({v.juju_commit[:12]})'
    # A tagged runner's version output.
    from_json = Versions._from_json({'juju_tag': 'v4.2.0', 'juju_commit': 'abc', 'protocol': 1})
    assert (from_json.juju_tag, from_json.juju) == ('v4.2.0', 'v4.2.0 (abc)')
