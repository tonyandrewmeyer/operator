# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Settle soak: the same multi-application scenario, settled many times, settles the same way.

Not part of the fast unit run. Run it with::

    tox -e soak

Every run must produce the same trace (event, unit and resulting ``State``
for every dispatch) and the same final ``State`` for every unit, compared as
the bytes of the typed JSON codec.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from . import scenario_run

RUNS = 100


def _soak(isolated: bool) -> None:
    started = time.monotonic()
    first = scenario_run.run(isolated=isolated)
    assert len(first) > 100  # Guard against a vacuous comparison.
    for i in range(1, RUNS):
        lines = scenario_run.run(isolated=isolated)
        if lines != first:
            pairs = zip(lines, first, strict=False)
            at = next((n for n, (a, b) in enumerate(pairs) if a != b), min(len(lines), len(first)))
            pytest.fail(f'Run {i} differs from run 0 from line {at} on.')
    elapsed = time.monotonic() - started
    print(f'\n{RUNS} runs, {len(first)} lines each, identical, {elapsed:.1f}s')


def test_in_process_settles_identically():
    _soak(isolated=False)


def test_with_a_worker_settles_identically():
    _soak(isolated=True)


@pytest.mark.parametrize('isolated', [False, True])
def test_settles_identically_across_hash_seeds(isolated: bool):
    """Different processes, with different string hashing, give the same bytes."""
    expected = scenario_run.digest(scenario_run.run(isolated=isolated))
    args = [sys.executable, '-m', 'tests.soak.scenario_run']
    if isolated:
        args.append('--isolated')
    cwd = os.path.join(os.path.dirname(__file__), '..', '..')
    for seed in range(10):
        env = {**os.environ, 'PYTHONHASHSEED': str(seed * 7919 + 1)}
        result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == expected, f'PYTHONHASHSEED={env["PYTHONHASHSEED"]}'
