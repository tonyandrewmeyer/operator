# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the persistent-worker isolation transport.

These cover the behaviours required of the persistent worker:

- lazy spawn on first dispatch;
- worker reuse across dispatches (one long-lived process);
- explicit teardown via ``close()`` / context-manager, and re-spawn afterwards;
- the spawn-per-event debug mode;
- worker crash surfacing as ``IsolationError`` with no silent re-spawn.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from scenario import ActiveStatus, State
from scenario._isolation import IsolatedContext
from scenario.errors import IsolationError

HERE = pathlib.Path(__file__).parent
CHARMS = HERE / 'charms'
DEPS = HERE / 'deps'

ALPHA = CHARMS / 'alpha'
HARDEXIT = CHARMS / 'hardexit'
V1 = (str(DEPS / 'confdep_v1'),)


# Lazy spawn and reuse


def test_persistent_is_the_default():
    """A plain IsolatedContext uses the persistent worker, not spawn-per-event."""
    ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1)
    assert ctx._spawn_per_event is False
    ctx.close()


def test_worker_not_spawned_until_first_run():
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


def test_worker_is_reused_across_events():
    """The same long-lived process handles successive events (same PID)."""
    with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1) as ctx:
        ctx.run(ctx.on.install(), State())
        pid1 = ctx._worker._proc.pid
        ctx.run(ctx.on.start(), State())
        ctx.run(ctx.on.config_changed(), State())
        pid2 = ctx._worker._proc.pid
        assert pid1 == pid2


def test_close_tears_down_and_is_idempotent():
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


def test_run_after_close_respawns():
    """A dispatch after close() lazily spawns a fresh worker."""
    ctx = IsolatedContext(charm_source=ALPHA, extra_sys_path=V1)
    ctx.run(ctx.on.install(), State())
    ctx.close()
    out = ctx.run(ctx.on.start(), State())  # re-spawn
    assert out.unit_status.name == 'active'
    assert ctx._worker is not None and ctx._worker._proc.poll() is None
    ctx.close()


def test_context_manager_closes_worker():
    """Using IsolatedContext as a context manager tears the worker down on exit."""
    with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1) as ctx:
        ctx.run(ctx.on.install(), State())
        proc = ctx._worker._proc
        assert proc.poll() is None
    assert ctx._worker is None
    proc.wait(timeout=5)
    assert proc.poll() is not None


# Spawn-per-event debug mode


def test_spawn_per_event_runs():
    with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1, spawn_per_event=True) as ctx:
        out = ctx.run(ctx.on.install(), State())
        assert 'confdep=1.0' in out.unit_status.message
        # No persistent worker is ever created in this mode.
        assert ctx._worker is None


def test_spawn_per_event_handles_multiple_events():
    with IsolatedContext(charm_source=ALPHA, extra_sys_path=V1, spawn_per_event=True) as ctx:
        state = State()
        for event in (ctx.on.install(), ctx.on.start(), ctx.on.config_changed()):
            state = ctx.run(event, state)
            assert state.unit_status.name == 'active'
        assert ctx._worker is None


# Crash handling


def test_crash_raises_isolation_error():
    ctx = IsolatedContext(charm_source=HARDEXIT)
    with pytest.raises(IsolationError, match='crashed'):
        ctx.run(ctx.on.start(), State())
    ctx.close()


def test_crashed_worker_is_not_silently_respawned():
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


def test_fresh_context_recovers_after_crash():
    """A crash poisons only its own context; a new context works normally."""
    crashed = IsolatedContext(charm_source=HARDEXIT)
    with pytest.raises(IsolationError):
        crashed.run(crashed.on.start(), State())
    crashed.close()

    with IsolatedContext(charm_source=HARDEXIT) as fresh:
        out = fresh.run(fresh.on.install(), State())
        assert out.unit_status.name == 'active'
        assert out.unit_status.message == 'installed ok'


def test_charm_exception_keeps_worker_reusable():
    """A *caught* charm error (not a crash) leaves the worker alive and reusable."""
    # alpha without confdep raises ImportError inside the worker, which the
    # worker catches and reports, and the process survives.
    ctx = IsolatedContext(charm_source=ALPHA)  # no confdep injected
    with pytest.raises(IsolationError, match='confdep'):
        ctx.run(ctx.on.install(), State())
    # A crashed worker would report that it isn't re-spawned; this one is
    # still there to try the import again.
    with pytest.raises(IsolationError, match='confdep') as caught:
        ctx.run(ctx.on.install(), State())
    assert 'not re-spawned' not in str(caught.value)
    ctx.close()


# Stream hygiene and timeouts


def test_charm_output_on_fd_1_does_not_corrupt_the_protocol():
    """A charm writing past sys.stdout must not land in the frame stream."""
    with IsolatedContext(charm_source=CHARMS / 'noisy') as ctx:
        out = ctx.run(ctx.on.install(), State())
        assert out.unit_status == ActiveStatus('survived')
        # The worker is still usable, so the stream is still in sync.
        assert ctx.run(ctx.on.install(), State()).unit_status == ActiveStatus('survived')


def test_dispatch_timeout_kills_the_persistent_worker():
    """A charm that never returns is killed once dispatch_timeout expires."""
    ctx = IsolatedContext(charm_source=CHARMS / 'slow', dispatch_timeout=0.5)
    with pytest.raises(IsolationError, match=re.escape('exceeded 0.5 seconds')):
        ctx.run(ctx.on.install(), State())
    ctx.close()


def test_worker_is_not_respawned_after_a_dispatch_timeout():
    """A timed-out worker is treated as crashed, like any other worker death."""
    ctx = IsolatedContext(charm_source=CHARMS / 'slow', dispatch_timeout=0.5)
    with pytest.raises(IsolationError):
        ctx.run(ctx.on.install(), State())
    with pytest.raises(IsolationError, match='not re-spawned'):
        ctx.run(ctx.on.install(), State())
    ctx.close()
