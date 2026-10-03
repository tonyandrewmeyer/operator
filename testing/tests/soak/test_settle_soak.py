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
import shutil
import subprocess
import sys
import time
from collections.abc import Generator

import pytest

from .. import local_index
from . import scenario_run

RUNS = 100


@pytest.fixture(scope='module')
def offline_uv(tmp_path_factory: pytest.TempPathFactory) -> Generator[None]:
    """Build isolated environments from a local index, into a temporary cache."""
    if shutil.which('uv') is None:
        pytest.skip('needs uv on PATH')
    index = tmp_path_factory.mktemp('index')
    local_index.ops_dependency_wheels(index)
    environ = local_index.uv_environ(index, tmp_path_factory.mktemp('cache'))
    saved = {name: os.environ.get(name) for name in environ}
    os.environ.update(environ)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _soak(isolated: bool, built: bool = False) -> None:
    started = time.monotonic()
    first = scenario_run.run(isolated=isolated, built=built)
    assert len(first) > 100  # Guard against a vacuous comparison.
    for i in range(1, RUNS):
        lines = scenario_run.run(isolated=isolated, built=built)
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


@pytest.mark.usefixtures('offline_uv')
def test_with_a_built_environment_settles_identically():
    _soak(isolated=False, built=True)
    # And the same as with the worker in the test's own interpreter.
    built = scenario_run.digest(scenario_run.run(built=True))
    assert built == scenario_run.digest(scenario_run.run(isolated=True))


@pytest.mark.parametrize('mode', ['in-process', 'isolated', 'built'])
def test_settles_identically_across_hash_seeds(mode: str, request: pytest.FixtureRequest):
    """Different processes, with different string hashing, give the same bytes."""
    if mode == 'built':
        request.getfixturevalue('offline_uv')
    expected = scenario_run.digest(
        scenario_run.run(isolated=mode == 'isolated', built=mode == 'built')
    )
    args = [sys.executable, '-m', 'tests.soak.scenario_run']
    if mode != 'in-process':
        args.append(f'--{mode}')
    cwd = os.path.join(os.path.dirname(__file__), '..', '..')
    for seed in range(10):
        env = {**os.environ, 'PYTHONHASHSEED': str(seed * 7919 + 1)}
        result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == expected, f'PYTHONHASHSEED={env["PYTHONHASHSEED"]}'
