# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""pytest plugin: a ``scriptlet_ctx`` fixture, versions in the header, and a budget check."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
from collections.abc import Generator, Iterator
from typing import Any, cast

import pytest

from . import _context
from ._context import Context
from ._process import find_runner
from ._result import RawResult, _juju_label
from .errors import RunnerError


@dataclasses.dataclass(frozen=True)
class _Record:
    """One limit of one measured run, in the phase that came closest to it."""

    event: str
    unit: str  # 'steps' or 'bytes'
    phase: str  # 'event' or 'load'
    used: int | None
    limit: int
    ok: bool

    @property
    def fraction(self) -> float | None:
        return None if self.used is None else self.used / self.limit

    @property
    def rank(self) -> float:
        # A phase that hit the cap (used is None) ranks above everything.
        return float('inf') if self.fraction is None else self.fraction

    @property
    def usage(self) -> str:
        if self.used is None:
            return f'over {self.limit:,} {self.unit} (search cap hit)'
        return f'{self.used:,} of {self.limit:,} {self.unit} ({self.used / self.limit:.1%})'

    @property
    def source(self) -> str:
        return f'{self.event} {self.phase}' + ('' if self.ok else ', failed')


def _record(result: RawResult) -> list[_Record]:
    """The steps and allocations of a measured run, or nothing if it wasn't measured."""
    load = result.load
    phases = {
        'steps': {
            'event': result.step_budget,
            'load': load.step_budget if load else None,
        },
        'bytes': {
            'event': result.alloc_budget,
            'load': load.alloc_budget if load else None,
        },
    }
    records: list[_Record] = []
    for unit, budgets in phases.items():
        candidates = [
            _Record(result.event, unit, phase, b.used, b.limit, result.ok)
            for phase, b in budgets.items()
            if b is not None and b.measured
        ]
        if candidates:
            records.append(_worst(candidates))
    return records


def _worst(records: list[_Record]) -> _Record:
    return max(records, key=lambda r: r.rank)


def _worst_by_unit(records: list[_Record]) -> list[_Record]:
    """The worst record for steps, then for bytes."""
    by_unit = {unit: [r for r in records if r.unit == unit] for unit in ('steps', 'bytes')}
    return [_worst(rs) for rs in by_unit.values() if rs]


@dataclasses.dataclass
class _State:
    fraction: float | None
    measure: bool
    records: dict[str, list[_Record]] = dataclasses.field(default_factory=dict[str, list[_Record]])


_STATE = pytest.StashKey[_State]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the plugin's ini settings and command-line options."""
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
        'fail a test whose scriptlet run used more than this fraction of the step or '
        'allocation limit',
        default='',
    )
    group = parser.getgroup('scriptlet', 'scriptlet charm testing')
    group.addoption(
        '--scriptlet-budget-fraction',
        type=float,
        default=None,
        metavar='F',
        help='fail a test whose scriptlet run used more than F (0 < F <= 1) of the step or '
        'allocation limit; turns measuring on for scriptlet_ctx',
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
    """Register the marker and set up the budget report."""
    config.addinivalue_line(
        'markers',
        'scriptlet_charm(path): the charm scriptlet_ctx loads, relative to the test file',
    )
    fraction = _fraction(config)
    measure = bool(config.getini('scriptlet_measure')) or fraction is not None
    config.stash[_STATE] = _State(fraction=fraction, measure=measure)


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Report the runner's Juju, starform and starlark versions."""
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
    # pytest doesn't annotate FixtureRequest.node.
    node = cast('pytest.Item', request.node)  # pyright: ignore[reportUnknownMemberType]
    marker = node.get_closest_marker('scriptlet_charm')
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
    """Record what each run used, and fail the test if one is over the budget fraction."""
    state = item.config.stash[_STATE]
    records: list[_Record] = []

    def observe(_ctx: Context, result: RawResult) -> None:
        records.extend(_record(result))

    _context._run_observers.append(observe)
    try:
        outcome = yield
    finally:
        _context._run_observers.remove(observe)
        if records:
            state.records[item.nodeid] = records
            for worst in _worst_by_unit(records):
                name = 'step' if worst.unit == 'steps' else 'alloc'
                item.user_properties.append((f'scriptlet_max_{name}_fraction', worst.fraction))
    if state.fraction is not None:
        over = [
            r for r in records if r.ok and r.fraction is not None and r.fraction > state.fraction
        ]
        if over:
            lines = [f'{r.event} ({r.phase}): {r.usage}' for r in over]
            pytest.fail(
                f'scriptlet budget: over {state.fraction:.0%} of the limit\n  '
                + '\n  '.join(lines),
                pytrace=False,
            )
    return outcome


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """List every measured run in a budget summary."""
    state = terminalreporter.config.stash.get(_STATE, None)
    if state is None or not state.records:
        return
    terminalreporter.section('scriptlet budget')
    rows = sorted(
        ((nodeid, _worst_by_unit(rs)) for nodeid, rs in state.records.items()),
        key=lambda row: -_worst(row[1]).rank,
    )
    for nodeid, worst in rows:
        if len({r.source for r in worst}) == 1:
            text = ', '.join([*(r.usage for r in worst), worst[0].source])
        else:
            text = '; '.join(f'{r.usage}, {r.source}' for r in worst)
        terminalreporter.write_line(f'{nodeid}: {text}')
