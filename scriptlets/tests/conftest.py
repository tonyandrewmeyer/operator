# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Build the runner (native and wasip1) once per session unless the environment names one."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest
from scriptlet_testing._process import RUNNER_ENV, WASM_ENV

HERE = pathlib.Path(__file__).parent
RUNNER_SRC = HERE.parent / 'runner'
TESTDATA = RUNNER_SRC / 'testdata'

_build_dir: str | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _build_dir
    # Inner pytester runs inherit the environment, so only the outer session builds.
    if os.environ.get(RUNNER_ENV):
        return
    _build_dir = tempfile.mkdtemp(prefix='scriptlet-runner-')
    out = pathlib.Path(_build_dir) / 'runner'
    subprocess.run(['go', 'build', '-trimpath', '-o', str(out), '.'], cwd=RUNNER_SRC, check=True)
    os.environ[RUNNER_ENV] = str(out)
    if not os.environ.get(WASM_ENV):
        wasm = pathlib.Path(_build_dir) / 'runner.wasm'
        subprocess.run(
            ['go', 'build', '-trimpath', '-o', str(wasm), '.'],
            cwd=RUNNER_SRC,
            check=True,
            env={**os.environ, 'GOOS': 'wasip1', 'GOARCH': 'wasm'},
        )
        os.environ[WASM_ENV] = str(wasm)


def pytest_unconfigure(config: pytest.Config) -> None:
    global _build_dir
    if _build_dir is not None:
        shutil.rmtree(_build_dir, ignore_errors=True)
        os.environ.pop(RUNNER_ENV, None)
        os.environ.pop(WASM_ENV, None)
        _build_dir = None


def write_charm(root: pathlib.Path, files: dict[str, str], assumes: bool = True) -> pathlib.Path:
    """Write a charm directory, as the runner's own Go tests do."""
    root.mkdir(parents=True, exist_ok=True)
    if assumes:
        files = {'charmcraft.yaml': 'assumes: [unitless]\n', **files}
    for name, content in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


def handler(body: str) -> dict[str, str]:
    """A charm with one config_changed handler."""
    return {
        'scriptlets/hook.star': f'def handler(event):\n{body}\n\n'
        'def init():\n    juju.observe("config_changed", handler)\n'
    }
