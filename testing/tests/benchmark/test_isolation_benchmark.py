# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Benchmark: isolation transports on the Saddle 4-charm / 20-event workload.

Measures the same workload three ways and compares them:

(a) **in-process Scenario baseline** — ``testing.Context(...).run(...)`` in the
    test process (no isolation);
(b) **spawn-per-event isolated** — a fresh worker subprocess per event
    (``IsolatedContext(spawn_per_event=True)``);
(c) **persistent-worker isolated** — one long-lived worker per charm
    (``IsolatedContext`` default).

The yardstick workload is a 4-charm bundle with 20 events per ``settle()``.
There is no ``Model`` / ``settle`` yet (that is step 4), so the workload is
realised as 4 isolated charm environments, each dispatched 20 sequential events.

Acceptance bar (OP089): two decomposed **absolute** budgets, not a single ratio
against the baseline.  Isolation costs two structurally different things, and
they amortise differently, so they are gated separately:

* **worker spawn** — a fresh interpreter plus ``import ops`` per isolated
  environment, paid once and amortised over however many events that
  environment sees (:data:`_SPAWN_BUDGET`);
* **steady state** — the JSON + IPC cost of moving one event and one ``State``
  across the process boundary, paid on every dispatch
  (:data:`_STEADY_STATE_BUDGET`).

A single ratio conflates the two: its value depends on the events-per-worker
count and on how fast the *baseline* runs on the current hardware, neither of
which the isolation code controls.  The blended ratio is still reported, as
context, but it does not gate.

This module records the measured numbers; it is run via ``tox -e benchmark``
(it is excluded from the unit suite) and is re-run by later steps to catch
regressions.

Run with ``-s`` to see the comparison table on stdout.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import time

import pytest
import yaml

import ops
from ops import testing

HERE = pathlib.Path(__file__).parent
CHARMS = HERE / 'charms' / 'iso_bench'
CHARM_DIRS = sorted(CHARMS.glob('bench_*'))

WORKLOAD_CHARMS = 4
EVENTS_PER_CHARM = 20

# Reps are overridable so the (slow) benchmark can be tuned locally.
_BASELINE_REPS = int(os.environ.get('ISO_BENCH_REPS', '5'))
_PERSISTENT_REPS = int(os.environ.get('ISO_BENCH_REPS', '5'))
# Spawn-per-event is ~80 process spawns per rep; keep its rep count low.
_SPAWN_REPS = int(os.environ.get('ISO_BENCH_SPAWN_REPS', '2'))

# Acceptance budgets.  Both are absolute, and both are ceilings on the *extra*
# cost isolation adds over the equivalent in-process dispatch.
#
# Spawn: measured 236-330ms per environment across four machines, by two
# methodologies.  400ms leaves ~20-70% headroom while still catching a real
# regression such as a new eager import on the ``import ops`` path.
_SPAWN_BUDGET = 0.400
# Steady state: measured 0.15-0.85ms per dispatch at this fixture's small
# ``State``.  2ms leaves 2-13x headroom.  Serde cost grows roughly linearly
# with ``State`` size, so this ceiling is only meaningful against the fixed
# payload below; a size-aware version would need a size-parameterised budget.
_STEADY_STATE_BUDGET = 0.002

assert len(CHARM_DIRS) == WORKLOAD_CHARMS, f'expected {WORKLOAD_CHARMS} charms, got {CHARM_DIRS}'


# Helpers


def _event_kinds(on: testing.CharmEvents):
    """A deterministic 20-event sequence cycling through lifecycle events."""
    cycle = [on.install, on.start, on.config_changed, on.update_status]
    return [cycle[i % len(cycle)]() for i in range(EVENTS_PER_CHARM)]


def _state() -> testing.State:
    return testing.State(config={'log-level': 'debug'})


def _load_inprocess(charm_dir: pathlib.Path, idx: int):
    """Import a charm class under a unique module name for the in-process baseline.

    Each charm's ``src/charm.py`` declares a module named ``charm``; loading them
    under unique names avoids the ``sys.modules['charm']`` collision so all four
    classes coexist in the test process.
    """
    src = charm_dir / 'src' / 'charm.py'
    name = f'_isobench_charm_{idx}'
    spec = importlib.util.spec_from_file_location(name, src)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cls = next(
        c
        for c in vars(module).values()
        if isinstance(c, type) and issubclass(c, ops.CharmBase) and c is not ops.CharmBase
    )
    meta = yaml.safe_load((charm_dir / 'metadata.yaml').read_text())
    config_path = charm_dir / 'config.yaml'
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else None
    return cls, meta, config


# The three workloads


def _run_baseline(record: list[list[float]] | None = None) -> None:
    specs = [_load_inprocess(d, i) for i, d in enumerate(CHARM_DIRS)]
    for cls, meta, config in specs:
        # Build the event sequence once per charm, matching what the isolated
        # runs do, so it is not timed as part of any single dispatch.
        events = _event_kinds(testing.Context(cls, meta=meta, config=config).on)
        per_charm: list[float] = []
        for i in range(EVENTS_PER_CHARM):
            state = _state()
            # The worker builds its own Context inside the dispatch it is timed
            # on, so the baseline times Context construction too -- otherwise
            # the difference between the two counts construction as isolation
            # overhead when both paths in fact pay it.
            start = time.perf_counter()
            ctx = testing.Context(cls, meta=meta, config=config)
            ctx.run(events[i], state)
            per_charm.append(time.perf_counter() - start)
        if record is not None:
            record.append(per_charm)


def _run_persistent(record: list[list[float]] | None = None) -> None:
    ctxs = [testing.IsolatedContext(charm_source=d) for d in CHARM_DIRS]
    try:
        for ctx in ctxs:
            events = _event_kinds(ctx.on)
            per_charm: list[float] = []
            for i in range(EVENTS_PER_CHARM):
                state = _state()
                start = time.perf_counter()
                # The worker spawns lazily, so the first dispatch of each charm
                # -- and only the first -- pays the interpreter + import cost.
                ctx.run(events[i], state)
                per_charm.append(time.perf_counter() - start)
            if record is not None:
                record.append(per_charm)
    finally:
        for ctx in ctxs:
            ctx.close()


def _run_spawn() -> None:
    for d in CHARM_DIRS:
        ctx = testing.IsolatedContext(charm_source=d, spawn_per_event=True)
        try:
            events = _event_kinds(ctx.on)
            for i in range(EVENTS_PER_CHARM):
                ctx.run(events[i], _state())
        finally:
            ctx.close()


def _median(fn, reps: int) -> float:
    times: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return _med(times)


def _med(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _cold_warm(record: list[list[float]]) -> tuple[float, float]:
    """Split per-dispatch timings into (median cold, median warm).

    ``record`` holds one list of per-dispatch times per charm per rep.  The
    first entry of each list is that environment's cold dispatch; the rest are
    warm.  Taking the first-vs-rest split on *both* the baseline and the
    isolated run controls for any first-call cache effects on the baseline side
    too, so the difference between the two is attributable to isolation.
    """
    cold = [c[0] for c in record if c]
    warm = [t for c in record for t in c[1:]]
    return _med(cold), _med(warm)


@pytest.fixture(scope='module')
def results() -> dict[str, float]:
    # Warm import / disk caches so the first timed rep is not penalised.
    _run_baseline()
    _run_persistent()

    # The per-dispatch record is gathered from the timed reps themselves, so
    # the decomposition costs no extra runs.
    baseline_record: list[list[float]] = []
    persistent_record: list[list[float]] = []

    baseline = _median(lambda: _run_baseline(baseline_record), _BASELINE_REPS)
    persistent = _median(lambda: _run_persistent(persistent_record), _PERSISTENT_REPS)
    spawn = _median(_run_spawn, _SPAWN_REPS)

    baseline_cold, baseline_warm = _cold_warm(baseline_record)
    persistent_cold, persistent_warm = _cold_warm(persistent_record)

    data = {
        'baseline': baseline,
        'persistent': persistent,
        'spawn': spawn,
        # The two gated numbers: isolation's extra cost over the equivalent
        # in-process dispatch, split by how it amortises.
        'spawn_extra_per_charm': persistent_cold - baseline_cold,
        'steady_state_gap_per_event': persistent_warm - baseline_warm,
    }

    total_dispatches = WORKLOAD_CHARMS * EVENTS_PER_CHARM
    print('\n')
    print('=' * 64)
    print('Saddle step 3 — isolation benchmark')
    print(
        f'workload: {WORKLOAD_CHARMS} charms x {EVENTS_PER_CHARM} events '
        f'= {total_dispatches} dispatches'
    )
    print('-' * 64)
    print(f'{"mode":<28}{"median (s)":>12}{"per-event":>12}{"vs base":>10}')
    for mode in ('baseline', 'persistent', 'spawn'):
        secs = data[mode]
        print(
            f'{mode:<28}{secs:>12.3f}{secs / total_dispatches * 1000:>10.2f}ms'
            f'{secs / baseline:>9.2f}x'
        )
    print('-' * 64)
    print('isolation overhead, decomposed (persistent - baseline):')
    print(
        f'{"  cold spawn / environment":<40}'
        f'{data["spawn_extra_per_charm"] * 1000:>10.1f}ms'
        f'  (budget {_SPAWN_BUDGET * 1000:.0f}ms)'
    )
    print(
        f'{"  steady state / event":<40}'
        f'{data["steady_state_gap_per_event"] * 1000:>10.3f}ms'
        f'  (budget {_STEADY_STATE_BUDGET * 1000:.0f}ms)'
    )
    print('-' * 64)
    print(f'persistent vs baseline : {persistent / baseline:.2f}x (informational, not gated)')
    print(f'persistent vs spawn    : {spawn / persistent:.2f}x faster')
    print('=' * 64)
    return data


def test_persistent_beats_spawn_per_event(results: dict[str, float]):
    """The whole point of the persistent worker: it must beat spawn-per-event.

    This is the robust regression guard — later steps re-run it to catch a
    persistent-worker performance regression.
    """
    assert results['persistent'] < results['spawn'], (
        f'persistent ({results["persistent"]:.3f}s) should be faster than '
        f'spawn-per-event ({results["spawn"]:.3f}s)'
    )


def test_worker_spawn_cost_under_budget(results: dict[str, float]):
    """Acceptance bar from OP089: cold worker spawn stays under budget.

    This is the interpreter start plus ``import ops`` plus ``import scenario``
    that each isolated environment pays once, on its first dispatch.  It is the
    structural price of process-boundary isolation rather than something the
    harness can optimise away, so the budget guards against it *growing* -- a
    new eager import on the ``import ops`` path, say.
    """
    extra = results['spawn_extra_per_charm']
    assert extra <= _SPAWN_BUDGET, (
        f'cold worker spawn overhead {extra * 1000:.1f}ms exceeds the '
        f'{_SPAWN_BUDGET * 1000:.0f}ms/environment budget'
    )


def test_steady_state_per_event_under_budget(results: dict[str, float]):
    """Acceptance bar from OP089: warm per-event overhead stays under budget.

    Once the worker is up, the only cost isolation adds per dispatch is JSON
    (de)serialisation of the event and ``State`` plus the IPC round trip.  This
    is the number that does *not* amortise as a test dispatches more events, so
    it is the one that matters most as ``settle()`` loops grow.
    """
    gap = results['steady_state_gap_per_event']
    assert gap <= _STEADY_STATE_BUDGET, (
        f'steady-state per-event overhead {gap * 1000:.3f}ms exceeds the '
        f'{_STEADY_STATE_BUDGET * 1000:.0f}ms budget'
    )
