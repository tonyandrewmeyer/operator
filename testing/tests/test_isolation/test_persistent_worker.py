# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the persistent-worker isolation transport (Saddle step 3).

These cover the behaviours the spec calls out for the persistent worker:

- lazy spawn on first dispatch;
- worker reuse across dispatches (one long-lived process);
- explicit teardown via ``close()`` / context-manager, and re-spawn afterwards;
- idle-timeout teardown;
- the spawn-per-event debug mode;
- worker crash surfacing as ``IsolationError`` with no silent re-spawn.
"""

from __future__ import annotations

import pathlib
import time

import pytest
from scenario import IsolatedContext, IsolationError, State

HERE = pathlib.Path(__file__).parent
CHARMS = HERE / 'charms'
DEPS = HERE / 'deps'

ALPHA = CHARMS / 'alpha'
HARDEXIT = CHARMS / 'hardexit'
V1 = (str(DEPS / 'confdep_v1'),)


# Lazy spawn and reuse


class TestPersistentWorkerLifecycle:
    """The persistent worker spawns lazily and is reused across events."""

    def test_persistent_is_the_default(self):
        """A plain IsolatedContext uses the persistent worker, not spawn-per-event."""
        ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1)
        assert ctx._spawn_per_event is False
        ctx.close()

    def test_worker_not_spawned_until_first_run(self):
        """No worker process exists before the first dispatch (lazy spawn)."""
        ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1)
        assert ctx._worker is None  # nothing spawned yet
        out = ctx.run(ctx.on.install(), State())
        assert out.unit_status.name == 'active'
        # Now there is a live worker process.
        assert ctx._worker is not None
        assert ctx._worker._proc is not None
        assert ctx._worker._proc.poll() is None  # still running
        ctx.close()

    def test_worker_is_reused_across_events(self):
        """The same long-lived process handles successive events (same PID)."""
        with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1) as ctx:
            ctx.run(ctx.on.install(), State())
            pid1 = ctx._worker._proc.pid
            ctx.run(ctx.on.start(), State())
            ctx.run(ctx.on.config_changed(), State())
            pid2 = ctx._worker._proc.pid
            assert pid1 == pid2

    def test_close_tears_down_and_is_idempotent(self):
        """close() stops the worker and can be called repeatedly."""
        ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1)
        ctx.run(ctx.on.install(), State())
        proc = ctx._worker._proc
        ctx.close()
        assert ctx._worker is None
        # Process has exited.
        proc.wait(timeout=5)
        assert proc.poll() is not None
        ctx.close()  # idempotent, no error

    def test_run_after_close_respawns(self):
        """A dispatch after close() lazily spawns a fresh worker."""
        ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1)
        ctx.run(ctx.on.install(), State())
        ctx.close()
        out = ctx.run(ctx.on.start(), State())  # re-spawn
        assert out.unit_status.name == 'active'
        assert ctx._worker is not None and ctx._worker._proc.poll() is None
        ctx.close()

    def test_context_manager_closes_worker(self):
        """Using IsolatedContext as a context manager tears the worker down on exit."""
        with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1) as ctx:
            ctx.run(ctx.on.install(), State())
            proc = ctx._worker._proc
            assert proc.poll() is None
        assert ctx._worker is None
        proc.wait(timeout=5)
        assert proc.poll() is not None


# Idle timeout


class TestIdleTimeout:
    """An idle persistent worker is torn down after the idle timeout."""

    def test_idle_timeout_tears_worker_down(self):
        ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1, idle_timeout=0.5)
        ctx.run(ctx.on.install(), State())
        proc = ctx._worker._proc
        assert proc.poll() is None
        # Wait comfortably past the idle timeout.
        time.sleep(2.0)
        assert ctx._worker._proc is None  # idle timer closed it
        proc.wait(timeout=5)
        assert proc.poll() is not None
        ctx.close()

    def test_dispatch_after_idle_timeout_respawns(self):
        ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1, idle_timeout=0.5)
        ctx.run(ctx.on.install(), State())
        first_pid = ctx._worker._proc.pid
        time.sleep(2.0)
        assert ctx._worker._proc is None
        out = ctx.run(ctx.on.start(), State())  # re-spawn a fresh worker
        assert out.unit_status.name == 'active'
        assert ctx._worker._proc.pid != first_pid
        ctx.close()

    def test_idle_timer_does_not_kill_a_busy_worker(self):
        """Activity within the idle window keeps the worker alive."""
        ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1, idle_timeout=1.0)
        ctx.run(ctx.on.install(), State())
        pid = ctx._worker._proc.pid
        # Each dispatch resets the idle timer, so stepping faster than the
        # timeout must not tear the worker down.
        for _ in range(3):
            time.sleep(0.4)
            ctx.run(ctx.on.start(), State())
        assert ctx._worker._proc is not None
        assert ctx._worker._proc.pid == pid  # same process throughout
        ctx.close()


# Spawn-per-event debug mode


class TestSpawnPerEventDebugMode:
    """spawn_per_event=True uses a fresh process per event and no persistent worker."""

    def test_spawn_per_event_runs(self):
        with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1, spawn_per_event=True) as ctx:
            out = ctx.run(ctx.on.install(), State())
            assert 'confdep=1.0' in out.unit_status.message
            # No persistent worker is ever created in this mode.
            assert ctx._worker is None

    def test_spawn_per_event_handles_multiple_events(self):
        with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1, spawn_per_event=True) as ctx:
            state = State()
            for event in (ctx.on.install(), ctx.on.start(), ctx.on.config_changed()):
                state = ctx.run(event, state)
                assert state.unit_status.name == 'active'
            assert ctx._worker is None


# Crash handling


class TestWorkerCrash:
    """A worker that dies mid-dispatch surfaces as IsolationError, with no respawn."""

    def test_crash_raises_isolation_error(self):
        ctx = IsolatedContext(charm_source=HARDEXIT)
        with pytest.raises(IsolationError, match='crashed'):
            ctx.run(ctx.on.start(), State())
        ctx.close()

    def test_crashed_worker_is_not_silently_respawned(self):
        """After a crash, further dispatches fail rather than spawning a new worker.

        The second event (``install``) would succeed on a fresh worker, so if it
        raises IsolationError we know the harness refused to silently re-spawn.
        """
        ctx = IsolatedContext(charm_source=HARDEXIT)
        with pytest.raises(IsolationError):
            ctx.run(ctx.on.start(), State())  # hard-crashes the worker
        with pytest.raises(IsolationError, match='not re-spawned'):
            ctx.run(ctx.on.install(), State())  # would succeed if respawned
        ctx.close()

    def test_fresh_context_recovers_after_crash(self):
        """A crash poisons only its own context; a new context works normally."""
        crashed = IsolatedContext(charm_source=HARDEXIT)
        with pytest.raises(IsolationError):
            crashed.run(crashed.on.start(), State())
        crashed.close()

        with IsolatedContext(charm_source=HARDEXIT) as fresh:
            out = fresh.run(fresh.on.install(), State())
            assert out.unit_status.name == 'active'
            assert out.unit_status.message == 'installed ok'

    def test_charm_exception_keeps_worker_reusable(self):
        """A *caught* charm error (not a crash) leaves the worker alive and reusable."""
        # alpha without confdep raises ImportError inside the worker, which the
        # worker catches and reports — the process survives.
        ctx = IsolatedContext(charm_source=ALPHA)  # no confdep injected
        with pytest.raises(IsolationError, match='confdep'):
            ctx.run(ctx.on.install(), State())
        # The worker is still alive and not marked crashed, so a subsequent
        # (this time satisfiable) dispatch reuses it.
        assert ctx._worker is not None and not ctx._worker._crashed
        ctx.close()
