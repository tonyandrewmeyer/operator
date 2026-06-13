# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Per-charm dependency isolation for ops.testing.

This module provides :class:`IsolatedEnv` and :class:`IsolatedContext` —
primitives that let a Scenario test drive *one* charm's event handler in an
isolated subprocess with its own ``sys.path`` / venv.  No convergence loop, no
multi-charm model: just the ability to run a single on-disk charm when its
dependency set conflicts with the test process's packages.

The isolation mechanism is a subprocess + per-charm interpreter:

* **Each charm runs in a separate process** whose Python interpreter is selected
  per charm (e.g. a per-charm venv's ``bin/python``).
* **The parent test process never imports the charm.**  It reads only the
  charm's metadata (``metadata.yaml`` / ``charmcraft.yaml``) and serialises the
  :class:`~ops.testing.State` and event across the process boundary.
* **Conflicting binary dependencies coexist trivially** because each worker
  process has a genuinely independent ``sys.path`` / site-packages.

Subinterpreters are explicitly *not* used — they do not solve C-extension binary
conflicts and cost the same serialisation overhead.  See §4 of the spec for the
full rationale.

Persistent vs spawn-per-event
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default the worker is **persistent**: one long-lived process per
:class:`IsolatedEnv`, spawned lazily on the first dispatch and reused for every
subsequent event.  The charm module is imported once, so only the first event
pays interpreter startup and ``import ops`` cost.  The worker is torn down at
:meth:`IsolatedContext.close` (or when the context is used as a context manager)
and, optionally, after an idle timeout.

Spawn-per-event remains available as an explicit **debug mode**
(``spawn_per_event=True``): a fresh process per event, so there is no shared
interpreter state between events and it is trivial to attach a debugger.  It is
much slower and is intended only for debugging.

A worker *crash* (the process dies without producing a response) surfaces as an
:class:`IsolationError`; the harness never silently re-spawns a crashed worker
mid-test — create a new :class:`IsolatedContext` to continue.

Serialisation
~~~~~~~~~~~~~
The event and state cross the process boundary as **JSON**.  The wire format is
the typed, schema-versioned codec in :mod:`scenario._isolated_serde` (which
delegates State to the canonical :mod:`scenario._state_serde`); it round-trips
frozen dataclasses, ``set``/``frozenset``/``tuple``, ``datetime``,
``pathlib.Path``, the ``_EntityStatus`` family, pebble enums, ``bytes``, and
``pebble.Layer``.

The parent and the per-charm worker must have **the same** ``ops`` /
``ops.testing`` version installed (the worker reconstructs dataclasses by name,
so the class registry must match).  Only the charm's own runtime dependencies
(``cryptography``, ``pydantic``, charm libs, ...) may differ between the worker
venv and the parent.

Typical usage
~~~~~~~~~~~~~
::

    import pathlib
    from ops import testing

    # Point at an existing venv built for this charm.
    ctx = testing.IsolatedContext(
        charm_source=pathlib.Path('./charms/myapp'),
        python_executable='/path/to/myapp-venv/bin/python',
    )
    state_out = ctx.run(ctx.on.install(), testing.State())
    assert state_out.unit_status == testing.ActiveStatus('ready')
    ctx.close()

For fast offline tests, ``extra_sys_path`` lets you inject a pre-built
dependency directory without a full venv::

    with testing.IsolatedContext(
        charm_source=pathlib.Path('./charms/myapp'),
        extra_sys_path=('./deps/mylib_v2',),
    ) as ctx:
        state_out = ctx.run(ctx.on.start(), testing.State())
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from typing import Any

import yaml

from . import _isolated_serde, _worker_protocol
from .context import _DEFAULT_JUJU_VERSION, CharmEvents
from .state import State, _Event

__all__ = [
    'IsolatedContext',
    'IsolatedEnv',
    'IsolationError',
]


# Public data types


@dataclasses.dataclass(frozen=True)
class IsolatedEnv:
    """Describes the isolated runtime environment for a single charm.

    An :class:`IsolatedEnv` pairs a charm source directory with the Python
    interpreter that should run it.  Use this when the charm has dependencies
    that conflict with the packages installed in the test process.

    Args:
        charm_source: Path to the charm repository root.  Must contain a
            ``src/`` directory (and optionally ``lib/``) whose ``charm.py``
            defines the charm class.  The metadata files
            (``metadata.yaml`` / ``charmcraft.yaml``) are read here in the
            parent; the charm code itself is only imported inside the worker
            subprocess.
        python_executable: The Python interpreter to use for the worker
            subprocess.  Point this at a per-charm venv's ``bin/python`` to
            isolate the charm's dependencies.  Defaults to the current
            interpreter (useful when there are no dependency conflicts).
        extra_sys_path: Directories prepended to ``sys.path`` in the worker
            before the charm is imported.  A lightweight stand-in for (or
            supplement to) a full venv — handy for fast, offline tests where
            the dependency directory is already available on disk.

    Invariant:
        The per-charm venv selected via ``python_executable`` must have **the
        same** ``ops`` version installed as the parent test process.  Only the
        charm's own runtime dependencies may differ between the two environments.
        A mismatch typically surfaces as an :class:`IsolationError` whose
        traceback names an unknown dataclass on the wire.

    Examples::

        # Point at a pre-built venv:
        env = IsolatedEnv(
            charm_source=pathlib.Path('./charms/myapp'),
            python_executable='/path/to/myapp-venv/bin/python',
        )

        # Inject a dependency directory (no venv needed):
        env = IsolatedEnv(
            charm_source=pathlib.Path('./charms/myapp'),
            extra_sys_path=('./deps/mylib_v2',),
        )
    """

    charm_source: pathlib.Path
    python_executable: str = dataclasses.field(default_factory=lambda: sys.executable)
    extra_sys_path: tuple[str, ...] = ()


class IsolationError(RuntimeError):
    """Raised when a charm event fails inside the isolated worker subprocess.

    This wraps any exception raised by the worker — either an uncaught charm
    exception or a worker-infrastructure error (for example, the charm module
    could not be imported, or the worker process crashed without producing a
    response).

    The original traceback from the worker is included in the message.
    """


# Metadata helpers (reads charm metadata without importing the charm)


def _read_yaml(path: pathlib.Path) -> dict[str, Any] | None:
    """Read a YAML file and return its contents, or None if the file does not exist."""
    if not path.exists():
        return None
    with path.open() as fh:
        return yaml.safe_load(fh)


def _read_charm_metadata(charm_root: pathlib.Path) -> dict[str, Any]:
    """Read charm metadata from disk without importing the charm.

    Prefers ``metadata.yaml`` but falls back to the metadata embedded in a
    ``charmcraft.yaml``.

    Raises:
        RuntimeError: if neither file is found or neither contains a ``name``
            key.
    """
    meta = _read_yaml(charm_root / 'metadata.yaml')
    if meta and 'name' in meta:
        return meta
    charmcraft = _read_yaml(charm_root / 'charmcraft.yaml') or {}
    if 'name' in charmcraft:
        return charmcraft
    raise RuntimeError(
        f'Could not find charm metadata in {charm_root} '
        '(looked for metadata.yaml and charmcraft.yaml with a "name" key).'
    )


def _child_environ() -> dict[str, str]:
    """Build the environment for the worker subprocess.

    Prepends the parent's ``scenario`` package source directory to
    ``PYTHONPATH`` so the worker uses the same ``ops.testing`` (scenario)
    classes as the parent, ensuring wire-format compatibility.
    """
    # The scenario package lives at testing/src/scenario/; the importable root
    # is two levels up: testing/src/.
    scenario_src = str(pathlib.Path(__file__).resolve().parent.parent)

    child = dict(os.environ)
    existing = child.get('PYTHONPATH', '')
    parts = [scenario_src] + ([existing] if existing else [])
    child['PYTHONPATH'] = os.pathsep.join(parts)
    return child


# Spawn-per-event dispatch (debug mode)


def _dispatch_spawn(env: IsolatedEnv, child_env: dict[str, str], request: dict[str, Any]) -> State:
    """Run a single charm event in a fresh subprocess and return the output State.

    This is the spawn-per-event debug transport: the request and response cross
    via JSON files in a short-lived temporary directory and the process is torn
    down after the single event.

    Raises:
        IsolationError: if the worker exits without producing a response, or if
            the charm raised an uncaught exception inside the worker.
    """
    with tempfile.TemporaryDirectory(prefix='ops-iso-') as tmp:
        req_file = pathlib.Path(tmp) / 'request.json'
        resp_file = pathlib.Path(tmp) / 'response.json'

        req_file.write_text(json.dumps(request))

        cmd = [
            env.python_executable,
            '-m',
            'scenario._isolated_worker',
            str(req_file),
            str(resp_file),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, env=child_env)

        if not resp_file.exists():
            raise IsolationError(
                'Isolated worker produced no response.\n'
                f'Command: {cmd}\n'
                f'Return code: {proc.returncode}\n'
                f'stdout:\n{proc.stdout}\n'
                f'stderr:\n{proc.stderr}'
            )

        response = json.loads(resp_file.read_text())

    if 'error' in response:
        raise IsolationError(f'Isolated charm run failed:\n{response["error"]}')

    return _isolated_serde.decode_state(response['state_out'])


# Persistent worker (default transport)


class _PersistentWorker:
    """A long-lived worker subprocess for one :class:`IsolatedEnv`.

    The process is spawned lazily on the first :meth:`dispatch` and reused for
    every subsequent event.  Communication is a length-prefixed framed JSON
    protocol over the worker's stdin/stdout (see
    :mod:`scenario._worker_protocol`).

    Thread-safety: a reentrant lock serialises dispatches and the idle-timeout
    teardown, so the idle timer (which fires on a background thread) can never
    race a dispatch.
    """

    def __init__(
        self,
        env: IsolatedEnv,
        child_env: dict[str, str],
        idle_timeout: float | None = None,
    ):
        self._env = env
        self._child_env = child_env
        self._idle_timeout = idle_timeout
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._timer_generation = 0
        self._crashed = False
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=200)
        self._stderr_thread: threading.Thread | None = None

    # Spawn / drain

    def _spawn(self) -> None:
        cmd = [self._env.python_executable, '-m', 'scenario._isolated_worker', '--serve']
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._child_env,
        )
        # Drain stderr on a background thread so the worker can never deadlock on
        # a full stderr pipe, and so we have a tail to report if it crashes.
        self._stderr_tail.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._proc.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, stream: Any) -> None:
        try:
            for line in iter(stream.readline, b''):
                self._stderr_tail.append(line.decode('utf8', 'replace'))
        except (ValueError, OSError):
            pass  # Stream closed underneath us; nothing more to drain.

    # Idle timer

    def _arm_timer(self) -> None:
        if self._idle_timeout is None:
            return
        self._timer_generation += 1
        generation = self._timer_generation
        self._timer = threading.Timer(self._idle_timeout, self._on_idle, args=(generation,))
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        # Bump the generation so any already-fired-but-waiting timer is ignored.
        self._timer_generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_idle(self, generation: int) -> None:
        with self._lock:
            if generation != self._timer_generation:
                return  # A newer dispatch superseded this timer; do nothing.
            # Idle teardown is a *clean* shutdown: the next dispatch may re-spawn.
            self.close()

    # Crash handling

    @staticmethod
    def _close_pipes(proc: subprocess.Popen[bytes]) -> None:
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _mark_crashed(self) -> None:
        self._crashed = True
        self._cancel_timer()
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None
        if proc is not None:
            self._close_pipes(proc)

    def _crash_message(self) -> str:
        returncode = self._proc.poll() if self._proc is not None else None
        tail = ''.join(self._stderr_tail).strip()
        return (
            'The isolated worker process crashed without producing a response.\n'
            f'Return code: {returncode}\n'
            f'Worker stderr (tail):\n{tail}'
        )

    # Dispatch

    def dispatch(self, request: dict[str, Any]) -> State:
        """Send one event request to the worker and return the output State.

        Raises:
            IsolationError: if the worker has crashed (it is not re-spawned), if
                it crashes during this dispatch, or if the charm raised an
                uncaught exception (the worker survives the latter and stays
                reusable).
        """
        with self._lock:
            if self._crashed:
                raise IsolationError(
                    'The isolated worker for this charm crashed earlier in the '
                    'test and is not re-spawned mid-test. Create a new '
                    'IsolatedContext to run further events.'
                )

            self._cancel_timer()

            if self._proc is not None and self._proc.poll() is not None:
                # Started earlier but has since exited unexpectedly: a crash.
                self._mark_crashed()
                raise IsolationError(self._crash_message())

            if self._proc is None:
                self._spawn()  # Lazy spawn on first dispatch (or after idle teardown).

            assert self._proc is not None
            assert self._proc.stdin is not None and self._proc.stdout is not None
            payload = json.dumps({'cmd': 'run', **request}).encode('utf8')
            try:
                _worker_protocol.write_frame(self._proc.stdin, payload)
                raw = _worker_protocol.read_frame(self._proc.stdout)
            except (BrokenPipeError, OSError) as exc:
                self._mark_crashed()
                raise IsolationError(self._crash_message()) from exc

            if raw is None:
                self._mark_crashed()
                raise IsolationError(self._crash_message())

            response = json.loads(raw.decode('utf8'))

            if 'error' in response:
                # A clean charm error: the worker caught it and is still alive,
                # so it stays reusable. Re-arm the idle timer and raise.
                self._arm_timer()
                raise IsolationError(f'Isolated charm run failed:\n{response["error"]}')

            self._arm_timer()
            return _isolated_serde.decode_state(response['state_out'])

    def close(self) -> None:
        """Shut the worker down cleanly (idempotent)."""
        with self._lock:
            self._cancel_timer()
            proc = self._proc
            self._proc = None
            if proc is None:
                return
            if proc.poll() is None and proc.stdin is not None:
                try:
                    _worker_protocol.write_frame(
                        proc.stdin, json.dumps({'cmd': 'shutdown'}).encode('utf8')
                    )
                    proc.stdin.close()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)
                self._stderr_thread = None
            self._close_pipes(proc)


# IsolatedContext — the public Context-like entry point


class IsolatedContext:
    """Run a single on-disk charm's events in an isolated subprocess.

    :class:`IsolatedContext` is the isolated counterpart of
    :class:`~ops.testing.Context`.  Instead of importing the charm into the
    test process, it:

    1. Reads the charm's metadata from disk (without importing the charm).
    2. Serialises the event and input :class:`~ops.testing.State`.
    3. Sends them to a worker subprocess running the charm's own
       interpreter / venv.
    4. Returns the output :class:`~ops.testing.State`.

    The charm class is **never imported into the test process**, making it
    safe to test charms whose dependencies would otherwise conflict with the
    test runner's installed packages.

    By default a single persistent worker is reused across events; call
    :meth:`close` (or use the context as a context manager) to tear it down.

    Args:
        charm_source: Path to the charm repository root (must contain
            ``src/charm.py`` and metadata files).
        python_executable: Interpreter to run the charm with.  Point this at a
            per-charm venv's ``bin/python`` to isolate the charm's
            dependencies.  Defaults to the current interpreter.
        extra_sys_path: Directories prepended to the worker's ``sys.path``
            before the charm is imported.  A lightweight alternative to a full
            venv for offline tests.
        meta: Charm metadata dict (``metadata.yaml`` format).  If omitted,
            read from ``charm_source/metadata.yaml`` (or
            ``charm_source/charmcraft.yaml``).
        config: Charm config dict (``config.yaml`` format).  If omitted, read
            from ``charm_source/config.yaml``.
        actions: Charm actions dict (``actions.yaml`` format).  If omitted, read
            from ``charm_source/actions.yaml``.
        app_name: Application name as seen by the charm.  Defaults to the
            charm name from the metadata.
        unit_id: Unit ID.  Defaults to ``0``.
        juju_version: Juju agent version to simulate.
        spawn_per_event: If ``True``, use the spawn-per-event **debug** transport
            — a fresh process per event, with no shared interpreter state and an
            easy place to attach a debugger.  Much slower; defaults to ``False``
            (the persistent worker).
        idle_timeout: If set, tear the persistent worker down after this many
            seconds of inactivity.  A later event lazily re-spawns a fresh
            worker.  Ignored in ``spawn_per_event`` mode.

    Invariant:
        The per-charm venv must have the **same** ``ops`` version installed as
        the parent test process.  Mismatches surface as
        :class:`IsolationError`.

    Example — point at a pre-built venv::

        import pathlib
        from ops import testing

        ctx = testing.IsolatedContext(
            charm_source=pathlib.Path('./charms/myapp'),
            python_executable='/path/to/myapp-venv/bin/python',
        )
        state_out = ctx.run(ctx.on.install(), testing.State())
        assert state_out.unit_status == testing.ActiveStatus('ready')
        ctx.close()

    Example — ``extra_sys_path`` for fast, offline tests, as a context manager::

        with testing.IsolatedContext(
            charm_source=pathlib.Path('./charms/alpha'),
            extra_sys_path=('./deps/mylib_v1',),
        ) as ctx:
            state_out = ctx.run(ctx.on.start(), testing.State())
    """

    #: Use ``ctx.on.<event>(...)`` to construct events for :meth:`run`.
    on: CharmEvents = CharmEvents()

    def __init__(
        self,
        charm_source: str | pathlib.Path,
        python_executable: str | None = None,
        extra_sys_path: tuple[str, ...] = (),
        *,
        meta: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        actions: dict[str, Any] | None = None,
        app_name: str | None = None,
        unit_id: int = 0,
        juju_version: str = _DEFAULT_JUJU_VERSION,
        spawn_per_event: bool = False,
        idle_timeout: float | None = None,
    ):
        charm_root = pathlib.Path(charm_source)
        if not charm_root.exists():
            raise ValueError(f'charm_source {charm_root!r} does not exist.')

        self._env = IsolatedEnv(
            charm_source=charm_root,
            python_executable=python_executable or sys.executable,
            extra_sys_path=extra_sys_path,
        )

        # Read metadata in the parent — the charm itself is never imported here.
        self._meta = meta if meta is not None else _read_charm_metadata(charm_root)
        self._config = config if config is not None else _read_yaml(charm_root / 'config.yaml')
        self._actions = actions if actions is not None else _read_yaml(charm_root / 'actions.yaml')
        self._app_name = app_name or self._meta.get('name', '')
        self._unit_id = unit_id
        self._juju_version = juju_version

        self._spawn_per_event = spawn_per_event
        self._idle_timeout = idle_timeout
        self._child_env = _child_environ()
        self._worker: _PersistentWorker | None = None

    @property
    def env(self) -> IsolatedEnv:
        """The :class:`IsolatedEnv` that describes this context's execution environment."""
        return self._env

    def _build_request(self, event: _Event, state: State) -> dict[str, Any]:
        return {
            'charm_source': str(self._env.charm_source),
            'extra_sys_path': list(self._env.extra_sys_path),
            'meta': self._meta,
            'config': self._config,
            'actions': self._actions,
            'app_name': self._app_name,
            'unit_id': self._unit_id,
            'event': _isolated_serde.encode_event(event),
            'state_in': _isolated_serde.encode_state(state),
        }

    def run(self, event: _Event, state: State) -> State:
        """Trigger a charm execution with an event and a State.

        Serialises ``event`` and ``state``, dispatches them to a worker
        subprocess running in :attr:`env`'s interpreter, and returns the output
        :class:`~ops.testing.State`.  In the default persistent mode the worker
        is spawned on the first call and reused thereafter.

        .. note::
            Unlike :class:`~ops.testing.Context`, :class:`IsolatedContext` does
            **not** capture ``juju_log``, ``app_status_history``, or other
            side-effect attributes.  Those are internal to the worker process.
            Assertions on side effects must be made via the output ``State``
            (e.g. ``state_out.unit_status``).

        Args:
            event: The event to dispatch.  Use :attr:`on` to construct it,
                e.g. ``ctx.on.install()`` or ``ctx.on.config_changed()``.
            state: The input :class:`~ops.testing.State` for this dispatch.

        Returns:
            The output :class:`~ops.testing.State` produced by the charm.

        Raises:
            IsolationError: if the worker subprocess crashes or the charm raises
                an uncaught exception.

        Example::

            state_out = ctx.run(ctx.on.install(), State())
            assert state_out.unit_status == ActiveStatus('ready')
        """
        request = self._build_request(event, state)

        if self._spawn_per_event:
            return _dispatch_spawn(self._env, self._child_env, request)

        if self._worker is None:
            self._worker = _PersistentWorker(self._env, self._child_env, self._idle_timeout)
        return self._worker.dispatch(request)

    def close(self) -> None:
        """Tear down the persistent worker, if one is running.

        Safe to call more than once and a no-op in ``spawn_per_event`` mode (no
        persistent worker is held there).
        """
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self) -> IsolatedContext:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if the caller forgot to close(); never raise during
        # interpreter shutdown.
        with contextlib.suppress(Exception):
            self.close()
