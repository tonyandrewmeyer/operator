# ops-testing-scriptlets

Unit tests for Juju 4.2 scriptlet (unitless) charms. A Python layer over
the Go runner in `runner/`, which runs a charm's Starlark scriptlets with
Juju's own executor code. Early days: nothing here is released yet.

```python
from scriptlet_testing import Context, Intent


def test_config_changed_sets_active():
    with Context.from_charm('.') as ctx:
        out = ctx.run_raw('config_changed', {'port': 8080, 'ratio': 1.0})
    assert out.intents == [
        Intent('set-status', {'status': 'active', 'message': 'serving on 8080'}),
    ]
    assert out.logs == ['serving on 8080']
```

The runner binary comes from `$SCRIPTLET_TESTING_RUNNER` if that's set, and
otherwise from the copy bundled in the wheel:

```
cd runner && go build -trimpath -o /tmp/runner . && export SCRIPTLET_TESTING_RUNNER=/tmp/runner
```

Steps and allocations are only measured when asked (`measure=True` on the
context or the call), because measuring costs about 40 times the event.

## Errors

A load error (syntax, a bad `load()`, an `init()` that fails) raises from
`Context.from_charm()`; an event error raises from `run_raw()`, with the
full result on `exc.result`. One class per runner error kind, all
subclasses of `ScriptletError`:

| Runner kind | Exception |
|---|---|
| `syntax_error` | `ScriptletSyntaxError` |
| `runtime_error` | `ScriptletRuntimeError` |
| `builtin_argument_error` | `BuiltinArgumentError` |
| `step_limit` | `StepLimitError` |
| `alloc_limit` | `AllocLimitError` |
| `load_path_error` | `LoadPathError` |
| `invalid_event` | `InvalidEventError` |
| `charm_invalid` | `InvalidCharmError` |
| `invalid_input` | `InvalidInputError` |
| `interpreter_panic` | `InterpreterPanicError` |
| `internal` | `RunnerInternalError` (also a `RunnerError`) |

`RunnerError` is the runner process failing (didn't start, exited, broke
the protocol), and `NotMeasuredError` is reading steps or allocations from
a run that wasn't measured. Everything subclasses `ScriptletTestingError`.

```python
with Context.from_charm('.') as ctx, pytest.raises(StepLimitError) as exc:
    ctx.run_raw('relation_changed', {'units': big})
assert exc.value.phase == 'event'
```

Steps are exact; allocations are starlark's estimate and can differ by a
few bytes between build platforms, so assert them with headroom.

## pytest plugin

Installed with the package (entry point `pytest11`). It adds:

- `scriptlet_ctx`: a `Context` for the charm, one runner per test. The charm
  is `scriptlet_charm` in the ini file (relative to the ini file), or
  `@pytest.mark.scriptlet_charm('path')` on the test, or the rootdir.
- The runner, Juju (tag, or version and commit while the runner's Juju
  commit isn't in a release), starform and starlark versions in the report
  header.
- `scriptlet_budget_fraction` (ini) or `--scriptlet-budget-fraction`: off by
  default. When set, `scriptlet_ctx` measures, every measured run is listed
  in a "scriptlet step budget" summary, and a test fails if a run that
  succeeded used more than that fraction of the step limit (load or event).
  `scriptlet_measure = true` measures and reports without failing anything.

## Development

```
uv sync --package ops-testing-scriptlets
uv run --package ops-testing-scriptlets pytest scriptlets/tests   # builds the runner with go build unless $SCRIPTLET_TESTING_RUNNER is set
```
