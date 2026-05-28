# Copyright 2025 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for ops.testing.IsolatedContext and ops.testing.IsolatedEnv.

These tests cover the §7 step-1 primitive from the Saddle incremental plan:
running *one* charm in an isolated subprocess from a Scenario-style test,
without the full multi-charm Model / convergence loop.

Test structure
--------------
``test_in_process_collision_is_real``
    Sanity check: the two ``confdep`` versions really cannot coexist in one
    interpreter.  This is the problem statement, demonstrated concretely.

``test_isolated_context_runs_single_charm_*``
    Single-charm isolation: alpha charm with confdep v1, beta charm with confdep
    v2.  Each runs in the test process's interpreter but with an extra_sys_path
    pointing at its own dep directory.

``test_independent_runs_see_different_dep_versions``
    The headline test: alpha and beta, each with conflicting confdep versions,
    both produce correct output in the same pytest session.  Without isolation
    this would be impossible.

``test_venv_isolation_*``
    Full real-venv tests (opt-in via ``OPS_VENV_ISOLATION_TEST=1``).  Build a
    genuine venv per charm, pip-install a different confdep version, point
    IsolatedContext at it.

``test_isolation_error_*``
    Error-propagation tests: charm exceptions reach the parent as IsolationError.

``test_isolated_env_*``
    Unit tests for IsolatedEnv itself.

``test_isolated_context_metadata_*``
    Tests for the metadata-reading path (charm_source with no
    python_executable / extra_sys_path).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from scenario import IsolatedContext, IsolatedEnv, IsolationError, State
from scenario.isolation import _read_charm_metadata

HERE = Path(__file__).parent
CHARMS = HERE / 'charms'
DEPS = HERE / 'deps'

# ---------------------------------------------------------------------------
# Problem statement (sanity check)
# ---------------------------------------------------------------------------


def test_in_process_collision_is_real():
    """Demonstrate that the two confdep versions cannot coexist in one process.

    This is the motivation for IsolatedContext: without per-process isolation,
    alpha (needs v1) and beta (needs v2) cannot both be loaded in the same test.
    """
    # Load v1 first.
    sys.path.insert(0, str(DEPS / 'confdep_v1'))
    try:
        sys.modules.pop('confdep', None)
        v1 = importlib.import_module('confdep')
        assert v1.VERSION == '1.0'
        assert hasattr(v1, 'LEGACY_NAME')  # v1-only attribute
    finally:
        sys.path.remove(str(DEPS / 'confdep_v1'))

    # Try to bring in v2 alongside it — Python returns the cached v1 module.
    sys.path.insert(0, str(DEPS / 'confdep_v2'))
    try:
        again = importlib.import_module('confdep')
        # Same cached object: v2's API is NOT reachable from this process.
        assert again is v1
        assert again.VERSION == '1.0'
        assert not hasattr(again, 'NEW_NAME')
    finally:
        sys.path.remove(str(DEPS / 'confdep_v2'))
        sys.modules.pop('confdep', None)


# ---------------------------------------------------------------------------
# Single-charm isolation via extra_sys_path
# ---------------------------------------------------------------------------


class TestIsolatedContextExtraSysPath:
    """Single-charm tests using extra_sys_path (fast, offline, no venv needed)."""

    def test_alpha_charm_sees_confdep_v1(self):
        """Alpha charm runs with confdep v1 injected via extra_sys_path."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            extra_sys_path=(str(DEPS / 'confdep_v1'),),
        )
        state_out = ctx.run(ctx.on.install(), State())

        assert state_out.unit_status.name == 'active'
        msg = state_out.unit_status.message
        assert 'confdep=1.0' in msg
        assert 'legacy=alpha-only-name' in msg
        assert 'compute=1' in msg

    def test_beta_charm_sees_confdep_v2(self):
        """Beta charm runs with confdep v2 injected via extra_sys_path."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'beta',
            extra_sys_path=(str(DEPS / 'confdep_v2'),),
        )
        state_out = ctx.run(ctx.on.install(), State())

        assert state_out.unit_status.name == 'active'
        msg = state_out.unit_status.message
        assert 'confdep=2.0' in msg
        assert 'new=beta-only-name' in msg
        assert 'compute=2' in msg

    def test_independent_runs_see_different_dep_versions(self):
        """Both charms run in the same pytest session and each sees its own dep version.

        This is the headline isolation test: without per-process isolation, the
        second import of ``confdep`` would return the cached module from the
        first run and the assertion would fail.
        """
        alpha_ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            extra_sys_path=(str(DEPS / 'confdep_v1'),),
        )
        beta_ctx = IsolatedContext(
            charm_source=CHARMS / 'beta',
            extra_sys_path=(str(DEPS / 'confdep_v2'),),
        )

        alpha_out = alpha_ctx.run(alpha_ctx.on.install(), State())
        beta_out = beta_ctx.run(beta_ctx.on.install(), State())

        assert 'confdep=1.0' in alpha_out.unit_status.message
        assert 'confdep=2.0' in beta_out.unit_status.message

    def test_multiple_events_on_same_context(self):
        """IsolatedContext can be used to run multiple events in sequence."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            extra_sys_path=(str(DEPS / 'confdep_v1'),),
        )
        state = State()
        for event in (ctx.on.install(), ctx.on.start(), ctx.on.config_changed()):
            state = ctx.run(event, state)
            assert state.unit_status.name == 'active'
            assert 'confdep=1.0' in state.unit_status.message

    def test_state_is_threaded_through(self):
        """State changes from one event are visible as input to the next."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            extra_sys_path=(str(DEPS / 'confdep_v1'),),
        )
        # First run: starts with default State (leader=False).
        state_out1 = ctx.run(ctx.on.install(), State())
        assert state_out1.unit_status.name == 'active'

        # Second run: pass the output of the first as the new input.
        state_out2 = ctx.run(ctx.on.start(), state_out1)
        assert state_out2.unit_status.name == 'active'
        # Status message should be the same since the charm does the same thing.
        assert state_out2.unit_status.message == state_out1.unit_status.message


# ---------------------------------------------------------------------------
# Full venv isolation (opt-in)
# ---------------------------------------------------------------------------


_VENV_TEST_MARKER = pytest.mark.skipif(
    os.environ.get('OPS_VENV_ISOLATION_TEST') != '1',
    reason=(
        'set OPS_VENV_ISOLATION_TEST=1 to run the (slow) real-venv isolation tests. '
        'These build per-charm venvs and pip-install conflicting packages.'
    ),
)


def _make_venv(tmp_path: Path, name: str, confdep_src: Path) -> str:
    """Create a venv with ops installed (inheriting from the parent) and pip-install confdep."""
    venv_dir = tmp_path / name
    # Use --system-site-packages so the venv inherits the parent's ops/scenario
    # install (required for pickle compatibility).
    venv.create(venv_dir, with_pip=True, system_site_packages=True)
    py = venv_dir / 'bin' / 'python'
    subprocess.check_call(
        [str(py), '-m', 'pip', 'install', '-q', str(confdep_src)],
    )
    return str(py)


class TestVenvIsolation:
    """End-to-end venv isolation tests.

    These tests build real venvs and pip-install conflicting confdep versions.
    They are opt-in (``OPS_VENV_ISOLATION_TEST=1``) because they are slow and
    require a working build toolchain.
    """

    @_VENV_TEST_MARKER
    def test_alpha_in_own_venv(self, tmp_path):
        """Alpha charm runs in its own venv with confdep v1 pip-installed."""
        alpha_py = _make_venv(tmp_path, 'alpha-venv', DEPS / 'confdep_v1')
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            python_executable=alpha_py,
        )
        state_out = ctx.run(ctx.on.install(), State())
        assert 'confdep=1.0' in state_out.unit_status.message

    @_VENV_TEST_MARKER
    def test_conflicting_charms_in_separate_venvs(self, tmp_path):
        """Both charms run in the same pytest session, each in its own venv.

        This is the full real-venv demonstration: alpha and beta each have a
        separate venv with a different confdep version pip-installed.
        """
        alpha_py = _make_venv(tmp_path, 'alpha-venv', DEPS / 'confdep_v1')
        beta_py = _make_venv(tmp_path, 'beta-venv', DEPS / 'confdep_v2')

        alpha_ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            python_executable=alpha_py,
        )
        beta_ctx = IsolatedContext(
            charm_source=CHARMS / 'beta',
            python_executable=beta_py,
        )

        alpha_out = alpha_ctx.run(alpha_ctx.on.install(), State())
        beta_out = beta_ctx.run(beta_ctx.on.install(), State())

        assert 'confdep=1.0' in alpha_out.unit_status.message
        assert 'confdep=2.0' in beta_out.unit_status.message


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestIsolationError:
    """IsolationError is raised when the charm or worker fails."""

    def test_missing_dep_raises_isolation_error(self):
        """If the charm cannot import its dependency, IsolationError is raised.

        Running alpha without injecting confdep causes an ImportError inside the
        worker, which the parent surfaces as IsolationError.
        """
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            # Deliberately NOT injecting confdep — the charm will fail to import.
        )
        with pytest.raises(IsolationError, match='confdep'):
            ctx.run(ctx.on.install(), State())

    def test_isolation_error_includes_worker_traceback(self):
        """IsolationError's message includes the worker-side traceback."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
        )
        with pytest.raises(IsolationError) as exc_info:
            ctx.run(ctx.on.install(), State())

        # The traceback should mention the charm file.
        assert 'charm.py' in str(exc_info.value) or 'ModuleNotFoundError' in str(exc_info.value)

    def test_nonexistent_charm_source_raises_value_error(self):
        """IsolatedContext raises ValueError for a non-existent charm_source."""
        with pytest.raises(ValueError, match='does not exist'):
            IsolatedContext(charm_source='/totally/does/not/exist')


# ---------------------------------------------------------------------------
# IsolatedEnv unit tests
# ---------------------------------------------------------------------------


class TestIsolatedEnv:
    """Unit tests for the IsolatedEnv dataclass."""

    def test_defaults(self):
        """IsolatedEnv defaults to the current interpreter and empty sys.path."""
        env = IsolatedEnv(charm_source=CHARMS / 'alpha')
        assert env.python_executable == sys.executable
        assert env.extra_sys_path == ()

    def test_frozen(self):
        """IsolatedEnv is frozen (immutable)."""
        env = IsolatedEnv(charm_source=CHARMS / 'alpha')
        with pytest.raises((AttributeError, TypeError)):
            env.python_executable = '/other/python'  # type: ignore[misc]

    def test_custom_python_executable(self):
        """IsolatedEnv stores a custom python_executable."""
        env = IsolatedEnv(
            charm_source=CHARMS / 'alpha',
            python_executable='/usr/bin/python3',
        )
        assert env.python_executable == '/usr/bin/python3'

    def test_extra_sys_path(self):
        """IsolatedEnv stores extra_sys_path as a tuple."""
        env = IsolatedEnv(
            charm_source=CHARMS / 'alpha',
            extra_sys_path=('/foo', '/bar'),
        )
        assert env.extra_sys_path == ('/foo', '/bar')


# ---------------------------------------------------------------------------
# Metadata resolution
# ---------------------------------------------------------------------------


class TestMetadataResolution:
    """Tests for reading charm metadata without importing the charm."""

    def test_reads_metadata_yaml(self):
        """_read_charm_metadata reads name from metadata.yaml."""
        meta = _read_charm_metadata(CHARMS / 'alpha')
        assert meta['name'] == 'alpha'

    def test_isolated_context_app_name_from_metadata(self):
        """IsolatedContext defaults app_name to the charm name from metadata."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            extra_sys_path=(str(DEPS / 'confdep_v1'),),
        )
        assert ctx._app_name == 'alpha'

    def test_isolated_context_explicit_app_name(self):
        """IsolatedContext accepts an explicit app_name override."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            app_name='my-alpha',
            extra_sys_path=(str(DEPS / 'confdep_v1'),),
        )
        assert ctx._app_name == 'my-alpha'

    def test_isolated_context_explicit_meta(self):
        """IsolatedContext accepts an explicit meta dict."""
        ctx = IsolatedContext(
            charm_source=CHARMS / 'alpha',
            meta={'name': 'override'},
            extra_sys_path=(str(DEPS / 'confdep_v1'),),
        )
        assert ctx._meta['name'] == 'override'

    def test_nonexistent_metadata_raises_runtime_error(self):
        """_read_charm_metadata raises RuntimeError if no metadata is found."""
        # Create a charm directory with no metadata files.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / 'src').mkdir()
            with pytest.raises(RuntimeError, match='metadata'):
                _read_charm_metadata(tmp_path)
