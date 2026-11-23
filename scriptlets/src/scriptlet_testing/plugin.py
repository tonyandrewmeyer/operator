"""pytest plugin: a ``scriptlet_ctx`` fixture, versions in the header, and a step-budget check."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
from collections.abc import Generator, Iterator
from typing import Any

import pytest

from . import _context
from ._context import Context
from ._process import find_runner
from ._result import RawResult, _juju_label
from .errors import RunnerError


@dataclasses.dataclass(frozen=True)
class _Record:
    """One measured run: which phase came closest to its step limit."""

    event: str
    phase: str  # 'event' or 'load'
    used: int | None
    limit: int
    ok: bool

    @property
    def fraction(self) -> float | None:
        return None if self.used is None else self.used / self.limit


def _record(result: RawResult) -> _Record | None:
    candidates: list[_Record] = []
    if result.step_budget is not None and result.step_budget.measured:
        b = result.step_budget
        candidates.append(_Record(result.event, 'event', b.used, b.limit, result.ok))
    if result.load is not None and result.load.step_budget is not None:
        b = result.load.step_budget
        if b.measured:
            candidates.append(_Record(result.event, 'load', b.used, b.limit, result.ok))
    if not candidates:
        return None
    # A phase that hit the cap (used is None) ranks above everything.
    return max(candidates, key=lambda r: float('inf') if r.fraction is None else r.fraction)


@dataclasses.dataclass
class _State:
    fraction: float | None
    measure: bool
    records: dict[str, list[_Record]] = dataclasses.field(default_factory=dict[str, list[_Record]])


_STATE = pytest.StashKey[_State]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        'scriptlet_charm',
        'charm directory or .charm file for scriptlet_ctx, relative to the ini file',
        default='',
    )
    parser.addini(
        'scriptlet_measure',
        'measure steps and allocations in scriptlet_ctx and report them (slow)',
        type='bool',
        default=False,
    )
    parser.addini(
        'scriptlet_budget_fraction',
        'fail a test whose scriptlet run used more than this fraction of the step limit',
        default='',
    )
    group = parser.getgroup('scriptlet', 'scriptlet charm testing')
    group.addoption(
        '--scriptlet-budget-fraction',
        type=float,
        default=None,
        metavar='F',
        help='fail a test whose scriptlet run used more than F (0 < F <= 1) of the step limit; '
        'turns measuring on for scriptlet_ctx',
    )


def _fraction(config: pytest.Config) -> float | None:
    value: Any = config.getoption('--scriptlet-budget-fraction')
    if value is None:
        ini = str(config.getini('scriptlet_budget_fraction')).strip()
        if not ini:
            return None
        try:
            value = float(ini)
        except ValueError:
            raise pytest.UsageError(f'scriptlet_budget_fraction: not a number: {ini!r}') from None
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise pytest.UsageError(f'scriptlet budget fraction must be in (0, 1], not {fraction}')
    return fraction


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        'markers',
        'scriptlet_charm(path): the charm scriptlet_ctx loads, relative to the test file',
    )
    fraction = _fraction(config)
    measure = bool(config.getini('scriptlet_measure')) or fraction is not None
    config.stash[_STATE] = _State(fraction=fraction, measure=measure)


def pytest_report_header(config: pytest.Config) -> list[str]:
    try:
        runner = find_runner()
    except RunnerError as e:
        return [f'scriptlet runner: none ({e})']
    try:
        out = subprocess.run(
            [str(runner), 'version'], capture_output=True, check=True, timeout=30
        ).stdout
        v: dict[str, Any] = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return [f'scriptlet runner: {runner} (version failed: {e})']
    juju = _juju_label(
        None if v.get('juju_tag') is None else str(v['juju_tag']),
        str(v.get('juju_version', '')),
        str(v.get('juju_commit', '')),
    )
    return [
        f'scriptlet runner: {runner}',
        f'scriptlet runner versions: juju {juju}, '
        f'starform {v.get("starform", "?")}, starlark {v.get("starlark", "?")}, '
        f'{v.get("go", "?")}, protocol {v.get("protocol", "?")}',
    ]


def _charm_path(request: pytest.FixtureRequest) -> pathlib.Path:
    marker = request.node.get_closest_marker('scriptlet_charm')
    if marker is not None:
        if len(marker.args) != 1:
            raise pytest.UsageError('scriptlet_charm takes one argument, the charm path')
        return request.path.parent / str(marker.args[0])
    config = request.config
    ini = str(config.getini('scriptlet_charm')).strip()
    if ini:
        base = config.inipath.parent if config.inipath else config.rootpath
        return base / ini
    return config.rootpath


@pytest.fixture
def scriptlet_ctx(request: pytest.FixtureRequest) -> Iterator[Context]:
    """A Context for the charm, with its own runner process, closed after the test."""
    state = request.config.stash[_STATE]
    with Context.from_charm(_charm_path(request), measure=state.measure) as ctx:
        yield ctx


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    state = item.config.stash[_STATE]
    records: list[_Record] = []

    def observe(_ctx: Context, result: RawResult) -> None:
        record = _record(result)
        if record is not None:
            records.append(record)

    _context._run_observers.append(observe)
    try:
        outcome = yield
    finally:
        _context._run_observers.remove(observe)
        if records:
            state.records[item.nodeid] = records
            worst = _record_worst(records)
            item.user_properties.append(('scriptlet_max_step_fraction', worst.fraction))
    if state.fraction is not None:
        over = [
            r for r in records if r.ok and r.fraction is not None and r.fraction > state.fraction
        ]
        if over:
            lines = [
                f'{r.event} ({r.phase}): {r.used:,} of {r.limit:,} steps ({r.fraction:.1%})'
                for r in over
                if r.used is not None and r.fraction is not None
            ]
            pytest.fail(
                f'scriptlet step budget: over {state.fraction:.0%} of the limit\n  '
                + '\n  '.join(lines),
                pytrace=False,
            )
    return outcome


def _record_worst(records: list[_Record]) -> _Record:
    return max(records, key=lambda r: float('inf') if r.fraction is None else r.fraction)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    state = terminalreporter.config.stash.get(_STATE, None)
    if state is None or not state.records:
        return
    terminalreporter.section('scriptlet step budget')
    rows = sorted(
        ((nodeid, _record_worst(rs)) for nodeid, rs in state.records.items()),
        key=lambda row: -(float('inf') if row[1].fraction is None else row[1].fraction),
    )
    for nodeid, r in rows:
        if r.used is None:
            usage = f'over {r.limit:,} steps (search cap hit)'
        else:
            usage = f'{r.used:,} of {r.limit:,} steps ({r.used / r.limit:.1%})'
        note = '' if r.ok else ', failed'
        terminalreporter.write_line(f'{nodeid}: {usage}, {r.event} {r.phase}{note}')
