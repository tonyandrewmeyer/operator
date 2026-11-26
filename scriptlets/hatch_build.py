# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Build the Go runner into the wheel, tagged for the platform it was built for.

The runner is built with ``CGO_ENABLED=0``, so it's a static binary with no
libc dependency, and Go cross-compiles it for any target. The target is Go's
own ``GOOS``/``GOARCH`` if they're set in the environment, and the machine
building the wheel otherwise. The Python code is pure, so the tag is
``py3-none-<platform>``.

Editable installs (``uv sync`` in the workspace) don't build the runner: the
tests build their own, and the package finds one through
``$SCRIPTLET_TESTING_RUNNER``.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# A static binary runs on both glibc and musl, so Linux wheels carry both tags.
# Go 1.26 needs macOS 12 or later.
PLATFORM_TAGS = {
    ('linux', 'amd64'): 'manylinux_2_17_x86_64.manylinux2014_x86_64.musllinux_1_1_x86_64',
    ('linux', 'arm64'): 'manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64',
    ('darwin', 'arm64'): 'macosx_12_0_arm64',
}

RUNNER_PATH = 'scriptlet_testing/_bin/runner'


class RunnerBuildHook(BuildHookInterface):  # type: ignore[type-arg]
    """Compile the runner and add it to the wheel at ``scriptlet_testing/_bin/runner``."""

    PLUGIN_NAME = 'custom'

    _tmp: str | None = None

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Build the runner for the target platform and include it in the wheel."""
        if self.target_name != 'wheel' or version == 'editable':
            return
        go = shutil.which('go')
        if go is None:
            raise RuntimeError(
                'building an ops-testing-scriptlets wheel needs Go (https://go.dev/dl/) '
                'to compile the runner'
            )
        goos, goarch = _target(go)
        try:
            platform = PLATFORM_TAGS[goos, goarch]
        except KeyError:
            supported = ', '.join(f'{o}/{a}' for o, a in PLATFORM_TAGS)
            raise RuntimeError(
                f'no wheel platform tag for {goos}/{goarch}; supported: {supported}'
            ) from None
        self._tmp = tempfile.mkdtemp(prefix='scriptlet-runner-')
        out = pathlib.Path(self._tmp) / 'runner'
        env = {**os.environ, 'CGO_ENABLED': '0', 'GOOS': goos, 'GOARCH': goarch}
        self.app.display_info(f'Building the scriptlet runner for {goos}/{goarch}')
        subprocess.run(
            [go, 'build', '-trimpath', '-buildvcs=false', '-ldflags=-s -w', '-o', str(out), '.'],
            cwd=pathlib.Path(self.root) / 'runner',
            env=env,
            check=True,
        )
        build_data['force_include'][str(out)] = RUNNER_PATH
        build_data['pure_python'] = False
        build_data['tag'] = f'py3-none-{platform}'

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        """Remove the temporary build directory."""
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


def _target(go: str) -> tuple[str, str]:
    """The GOOS and GOARCH Go will build for, honouring the environment."""
    out = subprocess.run(
        [go, 'env', 'GOOS', 'GOARCH'], capture_output=True, text=True, check=True
    ).stdout.split()
    return out[0], out[1]
