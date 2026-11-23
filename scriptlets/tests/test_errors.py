"""Every runner error kind becomes its own exception class."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from scriptlet_testing import (
    KIND_TO_EXCEPTION,
    AllocLimitError,
    BuiltinArgumentError,
    Context,
    InterpreterPanicError,
    InvalidCharmError,
    InvalidEventError,
    InvalidInputError,
    LoadPathError,
    Position,
    RunnerError,
    RunnerInternalError,
    ScriptletError,
    ScriptletRuntimeError,
    ScriptletSyntaxError,
    ScriptletTestingError,
    StepLimitError,
)
from scriptlet_testing.errors import error_from_json

from .conftest import RUNNER_SRC, TESTDATA, handler, write_charm


def runner_kinds() -> set[str]:
    """The kinds listed in the runner's own Error.Kind comment."""
    src = (RUNNER_SRC / 'internal' / 'run' / 'run.go').read_text()
    m = re.search(r'Kind is one of:(.*?)\n\s*Kind ', src, re.S)
    assert m, 'Error.Kind comment not found in run.go'
    return set(re.findall(r'[a-z_]+', m.group(1).replace('//', ' ')))


def test_table_covers_runner_kinds():
    assert set(KIND_TO_EXCEPTION) == runner_kinds()


def test_one_class_per_kind():
    classes = list(KIND_TO_EXCEPTION.values())
    assert len(set(classes)) == len(classes)
    for cls in classes:
        assert issubclass(cls, ScriptletError)
        assert issubclass(cls, ScriptletTestingError)
        assert cls is not ScriptletError


def test_unknown_kind_is_base():
    e = error_from_json({'kind': 'something_new', 'phase': 'event', 'message': 'hm'})
    assert type(e) is ScriptletError
    assert e.kind == 'something_new'


def test_internal_is_also_a_runner_error():
    e = error_from_json({'kind': 'internal', 'phase': 'event', 'message': 'oops'})
    assert isinstance(e, RunnerError) and isinstance(e, ScriptletError)
    with pytest.raises(RunnerError):
        raise e


def test_internal():
    # Not reachable from outside the runner, so only the mapping is tested.
    e = error_from_json({'kind': 'internal', 'phase': 'event', 'message': 'oops'})
    assert type(e) is RunnerInternalError
    assert str(e) == 'internal (event phase): oops'


def test_syntax_error():
    with pytest.raises(ScriptletSyntaxError) as exc:
        Context.from_charm(TESTDATA / 'syntax_error')
    e = exc.value
    assert (e.kind, e.phase) == ('syntax_error', 'load')
    assert e.position == Position('hook.star', 3, 5)
    assert 'while loops' in e.message
    assert 'cached-' in e.juju_log
    assert e.result is not None and not e.result.ok


def test_runtime_error():
    with (
        Context.from_charm(TESTDATA / 'runtime_error') as ctx,
        pytest.raises(ScriptletRuntimeError) as exc,
    ):
        ctx.run_raw('config_changed', {'config': {'host': 'db.internal'}})
    e = exc.value
    assert e.message == 'key "port" not in dict'
    assert e.position == Position('hook.star', 2, 52)
    assert 'in endpoint' in e.backtrace
    assert e.juju_log == 'handling event "config_changed": key "port" not in dict'
    # The result is still there: logs, and no intents (Juju drops them on error).
    assert e.result is not None
    assert e.result.logs == ['config changed']
    assert e.result.intents == []
    assert 'hook.star:2:52' in str(e)


def test_builtin_argument_error():
    with (
        Context.from_charm(TESTDATA / 'builtin_arg') as ctx,
        pytest.raises(BuiltinArgumentError) as exc,
    ):
        ctx.run_raw('config_changed')
    assert exc.value.builtin == 'set_status'
    assert exc.value.message == 'set_status: got 3 arguments, want at most 2'


def test_step_limit():
    data = json.loads((TESTDATA / 'events' / 'relation_changed_2000.json').read_text())
    name, attrs = data['name'], data['attrs']
    with (
        Context.from_charm(TESTDATA / 'step_limit') as ctx,
        pytest.raises(StepLimitError) as exc,
    ):
        ctx.run_raw(name, attrs, measure=True)
    e = exc.value
    assert e.phase == 'event'
    assert e.position == Position('hook.star', 3, 5)
    assert e.result is not None
    assert e.result.step_budget is not None and e.result.step_budget.limit_hit
    assert e.result.steps == 110046


def test_step_limit_at_load(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path, {'scriptlets/hook.star': 'def init():\n    x = [i for i in range(200)]\n'}
    )
    with pytest.raises(StepLimitError) as exc:
        Context.from_charm(root, max_steps=100)
    assert exc.value.phase == 'load'


def test_alloc_limit(tmp_path: pathlib.Path):
    root = write_charm(tmp_path, handler('    x = [None] * 1000'))
    with Context.from_charm(root) as ctx, pytest.raises(AllocLimitError) as exc:
        ctx.run_raw('config_changed', max_allocs=4096)
    assert exc.value.phase == 'event'


def test_load_path_error(tmp_path: pathlib.Path):
    root = write_charm(tmp_path, {'scriptlets/my-hook.star': 'x = 1\n'})
    with pytest.raises(LoadPathError, match='use "_" instead'):
        Context.from_charm(root)


def test_invalid_event(tmp_path: pathlib.Path):
    with (
        Context.from_charm(TESTDATA / 'set_status') as ctx,
        pytest.raises(InvalidEventError, match='name too short') as exc,
    ):
        ctx.run_raw('cc')
    assert exc.value.phase == 'event'


def test_charm_invalid(tmp_path: pathlib.Path):
    root = write_charm(tmp_path, handler('    pass'), assumes=False)
    with pytest.raises(InvalidCharmError) as exc:
        Context.from_charm(root)
    assert exc.value.phase == 'charm'
    # The check can be skipped, as the runner's flag allows.
    with Context.from_charm(root, skip_charm_checks=True) as ctx:
        assert ctx.run_raw('config_changed').ok


def test_charm_invalid_empty_source(tmp_path: pathlib.Path):
    root = write_charm(tmp_path, {'scriptlets/hook.star': 'x = 1\n', 'scriptlets/empty.star': ''})
    with pytest.raises(InvalidCharmError, match='empty scriptlet source') as exc:
        Context.from_charm(root)
    assert exc.value.phase == 'load'


def test_invalid_input_from_runner():
    with (
        Context.from_charm(TESTDATA / 'set_status') as ctx,
        pytest.raises(InvalidInputError) as exc,
    ):
        ctx.run_raw('config_changed', {'port': 2**63})
    assert exc.value.phase == 'input'


def test_invalid_input_missing_charm(tmp_path: pathlib.Path):
    with pytest.raises(InvalidInputError) as exc:
        Context.from_charm(tmp_path / 'nope')
    assert exc.value.phase == 'charm'


@pytest.mark.parametrize('value', [float('nan'), object(), {1, 2}])
def test_invalid_input_not_json(value: object):
    with Context.from_charm(TESTDATA / 'set_status') as ctx:
        with pytest.raises(InvalidInputError, match='not JSON') as exc:
            ctx.run_raw('config_changed', {'port': value})
        assert exc.value.phase == 'input'
        # The runner is still usable.
        assert ctx.run_raw('update_status').ok


def test_interpreter_panic(tmp_path: pathlib.Path):
    # A starlark allocation failure while growing the call stack panics inside the
    # interpreter; the runner recovers it.
    root = write_charm(
        tmp_path,
        {
            'scriptlets/hook.star': 'def r(d):\n    return 0 if d == 0 else r(d - 1)\n\n'
            'def handler(event):\n    x = "a" * 1000\n    r(40)\n\n'
            'def init():\n    juju.observe("config_changed", handler)\n'
        },
    )
    panics: list[InterpreterPanicError] = []
    with Context.from_charm(root) as ctx:
        for limit in range(2000, 6000, 7):
            try:
                ctx.run_raw('config_changed', max_allocs=limit)
            except InterpreterPanicError as e:
                panics.append(e)
                break
            except AllocLimitError:
                pass
    assert panics, 'no limit panicked; has starlark fixed the stack-growth bug?'
    assert 'nil pointer dereference' in panics[0].message
