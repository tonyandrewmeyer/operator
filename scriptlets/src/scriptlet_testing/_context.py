"""The raw layer: run one event through the runner and return what it did."""

from __future__ import annotations

import json
import os
import pathlib
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from ._process import PROTOCOL, Backend, NativeProcess, find_runner
from ._result import LoadInfo, RawResult, Versions
from .errors import InvalidInputError, RunnerError, error_from_json

# Called with every result a Context produces, errors included. The pytest
# plugin uses this for the step-budget report; anything else should not.
_run_observers: list[Callable[[Context, RawResult], None]] = []


class Context:
    """A charm's scriptlets, loaded once by a runner process and run event by event.

    Measuring steps and allocations is off unless ``measure=True`` is passed
    here or to ``run_raw()``: the runner finds them by bisection, which costs
    about 40 times the event itself.
    """

    def __init__(
        self,
        charm: str | os.PathLike[str],
        *,
        max_steps: int | None = None,
        max_allocs: int | None = None,
        measure: bool = False,
        cap_factor: int | None = None,
        skip_charm_checks: bool = False,
        runner: str | os.PathLike[str] | None = None,
        _backend: Literal['native', 'wasm'] = 'native',
    ):
        self.charm_path = pathlib.Path(charm).resolve()
        self.measure = measure
        self._max_steps = max_steps
        self._max_allocs = max_allocs
        self._cap_factor = cap_factor
        self._skip_charm_checks = skip_charm_checks
        self._backend_name = _backend
        self._backend: Backend
        if _backend == 'native':
            self._backend = NativeProcess(find_runner(runner))
        elif _backend == 'wasm':
            from ._wasm import WasmProcess, find_wasm

            self._backend = WasmProcess(find_wasm(runner), self.charm_path)
        else:
            raise ValueError(f'unknown backend {_backend!r}')
        self._finalizer = weakref.finalize(self, self._backend.close)
        try:
            resp = self._backend.request({'op': 'version'})
            self._check_protocol(resp)
            self.versions = Versions._from_json(resp)
            self.load = self._load()
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_charm(
        cls,
        path: str | os.PathLike[str],
        *,
        max_steps: int | None = None,
        max_allocs: int | None = None,
        measure: bool = False,
        cap_factor: int | None = None,
        skip_charm_checks: bool = False,
        runner: str | os.PathLike[str] | None = None,
    ) -> Context:
        """Start a runner for the charm at ``path`` (a directory or a ``.charm`` file).

        The charm is loaded straight away, so a load error (a syntax error,
        a bad ``load()``, an ``init()`` that fails) is raised here, as Juju
        would fail to start the application's runner.
        """
        return cls(
            path,
            max_steps=max_steps,
            max_allocs=max_allocs,
            measure=measure,
            cap_factor=cap_factor,
            skip_charm_checks=skip_charm_checks,
            runner=runner,
        )

    @property
    def max_steps(self) -> int:
        """The step limit events run under (Juju's 100,000 unless set)."""
        assert self.load.step_budget is not None
        return self.load.step_budget.limit

    @property
    def max_allocs(self) -> int:
        """The allocation limit in bytes (Juju's 10 MiB unless set)."""
        assert self.load.alloc_budget is not None
        return self.load.alloc_budget.limit

    @property
    def closed(self) -> bool:
        return self._backend.closed

    def close(self) -> None:
        """Stop the runner process. Safe to call more than once."""
        self._finalizer()

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = 'closed' if self.closed else 'open'
        return f'<Context {str(self.charm_path)!r} ({self._backend_name}, {state})>'

    def _check_protocol(self, resp: Mapping[str, Any]) -> None:
        if resp.get('protocol') != PROTOCOL:
            raise RunnerError(
                f'runner speaks protocol {resp.get("protocol")!r}, this package {PROTOCOL}'
            )

    def _options(
        self, measure: bool | None, max_steps: int | None, max_allocs: int | None
    ) -> dict[str, Any]:
        # 'measure' is always sent: the runner measures when it's missing.
        opts: dict[str, Any] = {'measure': self.measure if measure is None else measure}
        steps = self._max_steps if max_steps is None else max_steps
        allocs = self._max_allocs if max_allocs is None else max_allocs
        for name, value in (('max_steps', steps), ('max_allocs', allocs)):
            if value is not None:
                if value <= 0:
                    # The runner reads 0 as "Juju's default", not as a limit.
                    raise ValueError(f'{name} must be positive, not {value}')
                opts[name] = value
        if self._cap_factor is not None:
            opts['cap_factor'] = self._cap_factor
        if self._skip_charm_checks:
            opts['skip_charm_checks'] = True
        return opts

    def _run(self, name: str, attrs: Mapping[str, Any] | None, options: dict[str, Any]):
        event: dict[str, Any] = {'name': name}
        if attrs is not None:
            try:
                # Round-trip now, so unserialisable values fail here with a
                # clear message rather than as a runner protocol error.
                event['attrs'] = json.loads(json.dumps(attrs, allow_nan=False))
            except (TypeError, ValueError) as e:
                raise InvalidInputError(
                    f'event attributes are not JSON: {e}', kind='invalid_input', phase='input'
                ) from e
        resp = self._backend.request(
            {'op': 'run', 'charm': str(self.charm_path), 'event': event, 'options': options}
        )
        self._check_protocol(resp)
        return resp

    def _load(self) -> LoadInfo:
        # The runner's load op: read the charm, load the sources and run
        # every init(), with no event. The runner caches the loaded script
        # set, so the first run_raw() with the same options reuses it.
        resp = self._backend.request(
            {
                'op': 'load',
                'charm': str(self.charm_path),
                'options': self._options(None, None, None),
            }
        )
        self._check_protocol(resp)
        err = cast('dict[str, Any] | None', resp.get('error'))
        if err is not None:
            raise error_from_json(err, RawResult._from_json(resp))
        if not resp.get('ok'):
            raise RunnerError(f'runner reported a failed load with no error: {resp!r}')
        load = LoadInfo._from_json(resp.get('load'))
        if load is None:
            raise RunnerError('runner reported no load result')
        return load

    def run_raw(
        self,
        event_name: str,
        attrs: Mapping[str, Any] | None = None,
        *,
        measure: bool | None = None,
        max_steps: int | None = None,
        max_allocs: int | None = None,
    ) -> RawResult:
        """Run one event with these attributes and return what it did.

        Raises a ScriptletError subclass if the load or the handler failed;
        the exception's ``result`` still has the logs and budgets. ``measure``
        overrides the context's setting for this run.
        """
        resp = self._run(event_name, attrs, self._options(measure, max_steps, max_allocs))
        result = RawResult._from_json(resp)
        for observer in list(_run_observers):
            observer(self, result)
        err = resp.get('error')
        if err is not None:
            raise error_from_json(err, result)
        if not result.ok:
            raise RunnerError(f'runner reported failure with no error: {resp!r}')
        return result
