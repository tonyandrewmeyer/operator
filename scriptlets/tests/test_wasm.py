# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""The private wasm backend gives the same answers as the native runner."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import zipfile
from typing import Any

import pytest
from scriptlet_testing import (
    AllocLimitError,
    BuiltinArgumentError,
    Context,
    Intent,
    InterpreterPanicError,
    InvalidEventError,
    RunnerError,
    ScriptletRuntimeError,
    ScriptletSyntaxError,
    StepLimitError,
)
from scriptlet_testing._process import WASM_ENV
from scriptlet_testing._wasm import WasmProcess

from .conftest import BUNDLED_ENV, TESTDATA, write_charm

pytest.importorskip('wasmtime')
if os.environ.get(BUNDLED_ENV) and not os.environ.get(WASM_ENV):
    pytest.skip('testing the bundled runner, and no wasm runner named', allow_module_level=True)


def wasm(path: str | pathlib.Path, **kwargs: Any) -> Context:
    return Context(path, _backend='wasm', **kwargs)


def event(name: str) -> tuple[str, dict[str, Any]]:
    data = json.loads((TESTDATA / 'events' / f'{name}.json').read_text())
    return data['name'], data.get('attrs') or {}


@pytest.mark.parametrize(
    'charm,ev', [('set_status', 'config_changed_port'), ('step_limit', 'relation_changed_100')]
)
def test_same_as_native(charm: str, ev: str):
    with wasm(TESTDATA / charm) as w, Context.from_charm(TESTDATA / charm) as n:
        assert w.versions == n.versions
        a = w.run_raw(*event(ev), measure=True)
        b = n.run_raw(*event(ev), measure=True)
    assert a.intents == b.intents
    assert a.log_entries == b.log_entries
    assert (a.steps, a.load_steps) == (b.steps, b.load_steps)
    # Allocation estimates are 8 bytes higher under wasm for the relation
    # event (any number of units), and equal for config_changed. See the
    # step 3 log.
    assert a.allocs - b.allocs == (8 if charm == 'step_limit' else 0)


def test_packed_charm(tmp_path: pathlib.Path):
    src = TESTDATA / 'set_status'
    archive = tmp_path / 'set-status.charm'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('metadata.yaml', 'name: set-status\nassumes: [unitless]\n')
        for p in sorted((src / 'scriptlets').rglob('*.star')):
            z.write(p, p.relative_to(src).as_posix())
    with wasm(archive) as ctx:
        assert ctx.run_raw('update_status').intents == [
            Intent('set-status', {'status': 'active', 'message': ''})
        ]


def test_errors():
    with pytest.raises(ScriptletSyntaxError):
        wasm(TESTDATA / 'syntax_error')
    with wasm(TESTDATA / 'runtime_error') as ctx, pytest.raises(ScriptletRuntimeError):
        ctx.run_raw(*event('config_changed_missing_port'))
    with wasm(TESTDATA / 'builtin_arg') as ctx, pytest.raises(BuiltinArgumentError):
        ctx.run_raw('config_changed')
    with wasm(TESTDATA / 'set_status') as ctx, pytest.raises(InvalidEventError):
        ctx.run_raw('cc')
    with wasm(TESTDATA / 'step_limit') as ctx, pytest.raises(StepLimitError) as exc:
        ctx.run_raw(*event('relation_changed_2000'), measure=True)
    assert exc.value.result is not None and exc.value.result.steps == 110046


def test_interpreter_panic(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path,
        {
            'scriptlets/hook.star': 'def r(d):\n    return 0 if d == 0 else r(d - 1)\n\n'
            'def handler(event):\n    x = "a" * 1000\n    r(40)\n\n'
            'def init():\n    juju.observe("config_changed", handler)\n'
        },
    )
    with wasm(root) as ctx:
        for limit in range(2000, 6000, 7):
            try:
                ctx.run_raw('config_changed', max_allocs=limit)
            except InterpreterPanicError:
                break
            except AllocLimitError:
                pass
        else:
            pytest.fail('no limit panicked under wasm')
        # The guest survived the recovered panic.
        assert ctx.run_raw('config_changed').ok


def test_charm_outside_preopen_is_invisible(tmp_path: pathlib.Path):
    # Only the charm's own directory is preopened, so the guest can't read
    # a sibling path.
    with wasm(TESTDATA / 'set_status') as ctx:
        resp = ctx._backend.request({
            'op': 'run',
            'charm': str(TESTDATA / 'step_limit'),
            'event': {'name': 'start_up'},
        })
    assert resp['error']['kind'] == 'invalid_input'


def test_clean_shutdown():
    ctx = wasm(TESTDATA / 'set_status')
    backend = ctx._backend
    ctx.close()
    assert backend.closed
    assert getattr(backend, 'exit_code', None) == 0
    with pytest.raises(RunnerError, match='closed'):
        ctx.run_raw('update_status')


def test_large_request_to_exited_guest():
    # The FIFO holds 64 KiB; a bigger request to a guest that has stopped
    # reading must fail, not hang.
    with wasm(TESTDATA / 'set_status') as ctx:
        backend = ctx._backend
        assert isinstance(backend, WasmProcess)
        backend._write(b'{"op":"shutdown"}\n')
        backend._thread.join(10)
        errors: list[BaseException] = []

        def big() -> None:
            try:
                ctx.run_raw('config_changed', {'pad': 'x' * 300_000})
            except BaseException as e:
                errors.append(e)

        t = threading.Thread(target=big, daemon=True)
        t.start()
        t.join(10)
        assert not t.is_alive(), 'request to an exited guest hung'
    assert len(errors) == 1 and isinstance(errors[0], RunnerError)


def test_interpreter_exits_cleanly(tmp_path: pathlib.Path):
    # wasmtime panics at exit if a store outlives its guest thread.
    script = (
        'from scriptlet_testing import Context\n'
        f'ctx = Context({str(TESTDATA / "set_status")!r}, _backend="wasm")\n'
        'ctx.run_raw("update_status")\n'
        'ctx.close()\n'
        'keep = ctx\n'
    )
    r = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert 'panicked' not in r.stderr


def test_no_wasm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('SCRIPTLET_TESTING_RUNNER_WASM')
    with pytest.raises(RunnerError, match='SCRIPTLET_TESTING_RUNNER_WASM'):
        wasm(TESTDATA / 'set_status')
