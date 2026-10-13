# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Run one charm's events in a worker process with its own interpreter.

The parent never imports the charm: it reads the metadata from disk, and the
event and ``State`` cross the process boundary as JSON (see ``_state_serde``).
A worker is spawned on the first dispatch and kept for later ones. A worker
that dies raises :class:`IsolationError` and isn't started again.
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
from collections.abc import Mapping, Sequence
from typing import Any

from . import _state_serde, _worker_protocol
from .context import _DEFAULT_JUJU_VERSION, CharmEvents
from .errors import IsolationError, MetadataNotFoundError
from .state import State, _CharmSpec, _Event


class _HookFailedError(IsolationError):
    """The charm itself raised, as opposed to the worker failing to run it.

    :class:`~ops.testing.Juju` treats this as a failed hook.
    """

    def __init__(self, traceback: str, cause: BaseException | None = None):
        super().__init__(f'Isolated charm run failed:\n{traceback}')
        self.traceback = traceback
        # The exception itself, when the charm ran in this process.
        self.cause = cause


def _isolated_error(response: Mapping[str, Any]) -> IsolationError:
    if response.get('hook_failed'):
        return _HookFailedError(response['error'])
    return IsolationError(f'Isolated charm run failed:\n{response["error"]}')


#: Seconds a single isolated event may run before the worker is killed.
_DEFAULT_DISPATCH_TIMEOUT = 60.0


# Data types


@dataclasses.dataclass(frozen=True)
class _IsolatedEnv:
    """Internal bundle of the isolated runtime settings for a single charm.

    Held by :class:`IsolatedContext`, which takes these as constructor
    arguments and documents them for users. Not public API: nothing accepts
    or returns one.

    Args:
        charm_source: Path to the charm repository root.
        python_executable: The Python interpreter that runs the worker
            subprocess. Defaults to the current interpreter.
        extra_sys_path: Directories prepended to ``sys.path`` in the worker
            before the charm is imported.
    """

    charm_source: pathlib.Path
    python_executable: str = dataclasses.field(default_factory=lambda: sys.executable)
    extra_sys_path: tuple[str, ...] = ()


# Metadata helpers (reads charm metadata without importing the charm)


def _load_charm_spec(
    charm_root: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Read a charm's metadata, config and actions from disk without importing it.

    Uses the same loaders as :class:`~ops.testing.Context`: a unified
    ``charmcraft.yaml`` first (with its ``config`` and ``actions`` split out),
    then the legacy ``metadata.yaml``, ``config.yaml`` and ``actions.yaml``.

    Returns:
        A ``(meta, config, actions)`` tuple. ``config`` and ``actions`` are
        ``None`` when the charm declares none.

    Raises:
        MetadataNotFoundError: if neither form holds any metadata.
    """
    meta, config, actions = _CharmSpec._load_metadata(charm_root)
    if not meta:
        meta, config, actions = _CharmSpec._load_metadata_legacy(charm_root)
    if not meta:
        raise MetadataNotFoundError(
            f'Could not find charm metadata in {charm_root} '
            '(looked for charmcraft.yaml and metadata.yaml).'
        )
    return meta, config, actions


def _child_environ(python_path: Sequence[str] = ()) -> dict[str, str]:
    """Build the environment for the worker subprocess.

    ``python_path`` goes at the front of the worker's ``PYTHONPATH``.

    In a source checkout the ``scenario`` package is importable from
    ``testing/src``, which is not on the worker's ``sys.path``, so that
    directory is prepended to ``PYTHONPATH``. When ``ops-scenario`` is
    installed, the same directory is the parent's ``site-packages``: putting
    that ahead of the worker's own would defeat the isolation, so it is left
    alone and the worker is required to have ``ops-scenario`` installed
    itself.
    """
    child = dict(os.environ)
    scenario_src = pathlib.Path(__file__).resolve().parent.parent
    parts = list(python_path)
    if not _is_installed_layout(scenario_src):
        parts.append(str(scenario_src))
    if not parts:
        return child

    existing = child.get('PYTHONPATH', '')
    parts += [existing] if existing else []
    child['PYTHONPATH'] = os.pathsep.join(parts)
    return child


def _is_installed_layout(path: pathlib.Path) -> bool:
    """Report whether ``path`` is an installed package root rather than a checkout."""
    return path.name in {'site-packages', 'dist-packages'}


# Spawn-per-event dispatch (debug mode)


def _dispatch_spawn(
    env: _IsolatedEnv,
    child_env: dict[str, str],
    request: dict[str, Any],
    timeout: float | None,
) -> State:
    """Run a single charm event in a fresh subprocess and return the output State.

    This is the spawn-per-event debug transport: the request and response cross
    via JSON files in a short-lived temporary directory and the process is torn
    down after the single event.

    Raises:
        IsolationError: if the worker exits without producing a response, if it
            outlives ``timeout``, or if the charm raised an uncaught exception
            inside the worker.
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

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=child_env, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise IsolationError(
                f'Isolated charm run exceeded {timeout} seconds and was killed.\n'
                f'Command: {cmd}\n'
                f'stdout:\n{exc.stdout}\n'
                f'stderr:\n{exc.stderr}'
            ) from exc

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
        raise _isolated_error(response)

    return _state_serde.decode_state(response['state_out'])


# Persistent worker (default transport)


class _PersistentWorker:
    """A long-lived worker subprocess for one :class:`_IsolatedEnv`.

    The process is spawned lazily on the first :meth:`dispatch` and reused for
    every subsequent event. Communication is a length-prefixed framed JSON
    protocol over the worker's stdin/stdout (see
    :mod:`scenario._worker_protocol`).

    Thread-safety: a reentrant lock serialises dispatches and the idle-timeout
    teardown, so the idle timer (which fires on a background thread) can never
    race a dispatch.
    """

    def __init__(
        self,
        env: _IsolatedEnv,
        child_env: dict[str, str],
        idle_timeout: float | None = None,
        dispatch_timeout: float | None = None,
    ):
        self._env = env
        self._child_env = child_env
        self._idle_timeout = idle_timeout
        self._dispatch_timeout = dispatch_timeout
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

    def _read_frame_with_timeout(self, stream: Any) -> bytes | None:
        """Read one frame, giving up after ``dispatch_timeout`` seconds.

        ``read_frame`` blocks, and it is called while holding the dispatch
        lock, so a charm that never returns would otherwise hang the whole test
        session with no diagnostic. The read happens on a helper thread that is
        abandoned once the deadline passes; ``_mark_crashed`` then kills the
        worker, which unblocks it.

        Raises:
            TimeoutError: if no complete frame arrives in time.
        """
        if self._dispatch_timeout is None:
            return _worker_protocol.read_frame(stream)

        result: list[bytes | None] = []
        error: list[BaseException] = []

        def read() -> None:
            try:
                result.append(_worker_protocol.read_frame(stream))
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(self._dispatch_timeout)
        if thread.is_alive():
            raise TimeoutError
        if error:
            raise error[0]
        return result[0]

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
                raw = self._read_frame_with_timeout(self._proc.stdout)
            except TimeoutError as exc:
                self._mark_crashed()
                raise IsolationError(
                    f'Isolated charm run exceeded {self._dispatch_timeout} seconds; '
                    f'the worker was killed.\n{self._crash_message()}'
                ) from exc
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
                raise _isolated_error(response)

            self._arm_timer()
            return _state_serde.decode_state(response['state_out'])

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


# IsolatedContext


class IsolatedContext:
    """Run a charm on disk in a worker process, much as :class:`~ops.testing.Context` runs one.

    The worker is kept between events; call :meth:`close`, or use the context
    as a context manager, to stop it.

    Args:
        charm_source: The charm's source directory, with ``src/charm.py``.
        python_executable: The interpreter to run the charm with.
        extra_sys_path: Directories put at the front of the worker's
            ``sys.path`` before the charm is imported.
        python_path: Directories put at the front of the worker's
            ``PYTHONPATH``.
        meta: The charm's metadata. Read from the charm source if omitted.
        config: The charm's config options. Read from the charm source if
            omitted.
        actions: The charm's actions. Read from the charm source if omitted.
        app_name: The application name. Defaults to the charm's name.
        unit_id: The unit ID.
        juju_version: The Juju agent version to simulate.
        app_trusted: Whether the application has Juju trust.
        charm_root: The charm directory the charm runs in.
        mocking: Keyword arguments for the charm's own mocking. ``None`` runs
            the charm with no mocking at all.
        dispatch_timeout: Seconds to let one event run before the worker is
            killed and :class:`IsolationError` raised. ``None`` waits
            indefinitely.
        spawn_per_event: Start a new process for every event.
        idle_timeout: Stop the persistent worker after this many idle seconds.
            The next event starts a new one.
    """

    #: Use ``ctx.on.<event>(...)`` to construct events for :meth:`run`.
    on: CharmEvents

    def __init__(
        self,
        charm_source: str | pathlib.Path,
        python_executable: str | None = None,
        extra_sys_path: Sequence[str] = (),
        *,
        python_path: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        actions: Mapping[str, Any] | None = None,
        app_name: str | None = None,
        unit_id: int = 0,
        juju_version: str = _DEFAULT_JUJU_VERSION,
        app_trusted: bool = False,
        charm_root: str | pathlib.Path | None = None,
        mocking: Mapping[str, Any] | None = None,
        spawn_per_event: bool = False,
        idle_timeout: float | None = None,
        dispatch_timeout: float | None = _DEFAULT_DISPATCH_TIMEOUT,
    ):
        self.on = CharmEvents()

        source = pathlib.Path(charm_source)
        if not source.exists():
            raise ValueError(f'charm_source {source!r} does not exist.')

        self._env = _IsolatedEnv(
            charm_source=source,
            python_executable=python_executable or sys.executable,
            extra_sys_path=tuple(extra_sys_path),
        )

        # The parent reads the metadata; it never imports the charm.
        if meta is None or config is None or actions is None:
            disk_meta, disk_config, disk_actions = _load_charm_spec(source)
            meta = disk_meta if meta is None else meta
            config = disk_config if config is None else config
            actions = disk_actions if actions is None else actions
        self._meta = dict(meta)
        self._config = dict(config) if config is not None else None
        self._actions = dict(actions) if actions is not None else None
        self.app_name = app_name or self._meta.get('name', '')
        self.unit_id = unit_id
        self.juju_version = juju_version
        self.app_trusted = app_trusted
        self.charm_root = charm_root
        # The unit's own filesystem root, which the filesystem default
        # translates the charm's file access into. Set per dispatch by Juju.
        self.filesystem_root: str | pathlib.Path | None = None
        # The directory that keeps the unit's container and storage
        # filesystems between dispatches. Set per dispatch by Juju.
        self.workload_root: str | pathlib.Path | None = None
        # On a machine model, the machine each unit in the model is on, by
        # unit name. Set per dispatch by Juju.
        self.machines: dict[str, int] = {}
        self._mocking = dict(mocking) if mocking is not None else None

        self.dispatch_timeout = dispatch_timeout
        self._spawn_per_event = spawn_per_event
        self._idle_timeout = idle_timeout
        self._child_env = _child_environ(python_path)
        self._worker: _PersistentWorker | None = None

    def _build_request(
        self,
        event: _Event,
        state: State,
        unit_id: int | None = None,
        secret_seed: str | None = None,
    ) -> dict[str, Any]:
        return {
            'charm_source': str(self._env.charm_source),
            'extra_sys_path': list(self._env.extra_sys_path),
            'meta': self._meta,
            'config': self._config,
            'actions': self._actions,
            'app_name': self.app_name,
            'unit_id': self.unit_id if unit_id is None else unit_id,
            'juju_version': self.juju_version,
            'app_trusted': self.app_trusted,
            'charm_root': None if self.charm_root is None else str(self.charm_root),
            'machines': dict(self.machines),
            'mocking': self._mocking,
            'filesystem_root': None if self.filesystem_root is None else str(self.filesystem_root),
            'workload_root': None if self.workload_root is None else str(self.workload_root),
            'secret_seed': secret_seed,
            'event': _state_serde.encode_event(event),
            'state_in': _state_serde.encode_state(state),
        }

    def run(self, event: _Event, state: State) -> State:
        """Trigger a charm execution with an event and a State.

        Serialises ``event`` and ``state``, dispatches them to a worker
        subprocess running in this context's interpreter, and returns the output
        :class:`~ops.testing.State`. In the default persistent mode the worker
        is spawned on the first call and reused thereafter.

        .. note::
            Unlike :class:`~ops.testing.Context`, :class:`IsolatedContext` does
            **not** capture ``juju_log``, ``app_status_history``, or other
            side-effect attributes. Those are internal to the worker process.
            Assertions on side effects must be made via the output ``State``
            (for example, ``state_out.unit_status``).

        Args:
            event: The event to dispatch. Use :attr:`on` to construct it,
                for example, ``ctx.on.install()`` or ``ctx.on.config_changed()``.
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
        return self._run_as(self.unit_id, event, state)

    def _run_as(
        self, unit_id: int, event: _Event, state: State, *, secret_seed: str | None = None
    ) -> State:
        """Dispatch ``event`` as unit ``unit_id`` rather than this context's own unit.

        The unit ID travels in the request rather than being baked into the
        worker, so a single persistent worker serves every unit of an
        application: one process per application rather than one per unit.
        ``secret_seed`` fixes the IDs of any secrets the charm creates.
        """
        request = self._build_request(event, state, unit_id=unit_id, secret_seed=secret_seed)
        if self._spawn_per_event:
            return _dispatch_spawn(self._env, self._child_env, request, self.dispatch_timeout)
        if self._worker is None:
            self._worker = _PersistentWorker(
                self._env, self._child_env, self._idle_timeout, self.dispatch_timeout
            )
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
