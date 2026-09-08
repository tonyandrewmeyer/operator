# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Per-charm dependency isolation for ops.testing.

:class:`IsolatedContext` lets a test drive *one* charm's event handler in an
isolated subprocess with its own ``sys.path`` / venv. No convergence loop, no
multi-charm model: just the ability to run a single on-disk charm when its
dependency set conflicts with the test process's packages.

The isolation mechanism is a subprocess + per-charm interpreter:

* **Each charm event is dispatched to a separate process** whose Python
  interpreter is selected per charm (for example, a per-charm venv's ``bin/python``).
* **The parent test process never imports the charm.**  It reads only the
  charm's metadata (``metadata.yaml`` / ``charmcraft.yaml``) and serialises the
  :class:`~ops.testing.State` and event across the process boundary.
* **Conflicting binary dependencies coexist** because each worker
  process has a genuinely independent ``sys.path`` / site-packages.

Subinterpreters are explicitly *not* used — they do not solve C-extension binary
conflicts and cost the same serialisation overhead.

Serialisation
~~~~~~~~~~~~~
The event and state cross the process boundary as **JSON** files in a temporary
directory. The wire format is a typed envelope produced by
:mod:`scenario._isolated_serde` that round-trips frozen dataclasses,
``set``/``frozenset``/``tuple``, ``datetime``, ``pathlib.Path``, the
``_EntityStatus`` family, and ``pebble.Layer``.

The parent and the per-charm worker must have **the same** ``ops`` /
``ops.testing`` version installed (the worker reconstructs dataclasses by name,
so the class registry must match). Only the charm's own runtime dependencies
(``cryptography``, ``pydantic``, charm libs, ...) may differ between the worker
venv and the parent.

Not public API: nothing in ``ops.testing`` exports :class:`IsolatedContext`.
It is the building block for running several charms under one test when each
needs its own dependencies.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from . import _isolated_serde
from .context import _DEFAULT_JUJU_VERSION, CharmEvents
from .errors import IsolationError, MetadataNotFoundError
from .state import State, _CharmSpec, _Event

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


# Worker dispatch (spawn-per-event)


def _dispatch(
    env: _IsolatedEnv,
    *,
    meta: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    actions: Mapping[str, Any] | None,
    app_name: str,
    unit_id: int,
    juju_version: str,
    app_trusted: bool,
    charm_root: str | pathlib.Path | None,
    event: _Event,
    state_in: State,
    timeout: float | None,
) -> State:
    """Serialise a charm event request, spawn the worker, and return the output State.

    The event and state cross the process boundary via JSON files in a
    short-lived temporary directory. Both the parent and worker must therefore
    use the same ``ops`` version (the wire format reconstructs dataclasses by
    name).

    Raises:
        IsolationError: if the worker exits without producing a response, if it
            outlives ``timeout``, or if the charm raised an uncaught exception
            inside the worker.
    """
    request = {
        'charm_source': str(env.charm_source),
        'extra_sys_path': list(env.extra_sys_path),
        'meta': meta,
        'config': config,
        'actions': actions,
        'app_name': app_name,
        'unit_id': unit_id,
        'juju_version': juju_version,
        'app_trusted': app_trusted,
        'charm_root': None if charm_root is None else str(charm_root),
        'event': _isolated_serde.encode_event(event),
        'state_in': _isolated_serde.encode_state(state_in),
    }

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

        child_env = _child_environ()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=child_env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise IsolationError(
                f'Isolated charm run for {app_name}/{unit_id} exceeded '
                f'{timeout} seconds and was killed.\n'
                f'Command: {cmd}\n'
                f'stdout:\n{e.stdout}\n'
                f'stderr:\n{e.stderr}'
            ) from e

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
        raise IsolationError(
            f'Isolated charm run failed for {app_name}/{unit_id}:\n{response["error"]}'
        )

    return _isolated_serde.decode_state(response['state_out'])


def _child_environ() -> dict[str, str]:
    """Build the environment for the worker subprocess.

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
    if _is_installed_layout(scenario_src):
        return child

    existing = child.get('PYTHONPATH', '')
    parts = [str(scenario_src)] + ([existing] if existing else [])
    child['PYTHONPATH'] = os.pathsep.join(parts)
    return child


def _is_installed_layout(path: pathlib.Path) -> bool:
    """Report whether ``path`` is an installed package root rather than a checkout."""
    return path.name in {'site-packages', 'dist-packages'}


# IsolatedContext — the public Context-like entry point


class IsolatedContext:
    """Run a single on-disk charm's events in an isolated subprocess.

    :class:`IsolatedContext` is the isolated counterpart of
    :class:`~ops.testing.Context`. Instead of importing the charm into the
    test process, it:

    1. Reads the charm's metadata from disk (without importing the charm).
    2. Serialises the event and input :class:`~ops.testing.State`.
    3. Spawns (or sends to) a worker subprocess running the charm's own
       interpreter / venv.
    4. Returns the output :class:`~ops.testing.State`.

    The charm class is **never imported into the test process**, making it
    safe to test charms whose dependencies would otherwise conflict with the
    test runner's installed packages.

    Args:
        charm_source: Path to the charm repository root (must contain
            ``src/charm.py`` and metadata files).
        python_executable: Interpreter to run the charm with. Point this at a
            per-charm venv's ``bin/python`` to isolate the charm's
            dependencies. Defaults to the current interpreter.
        extra_sys_path: Directories prepended to the worker's ``sys.path``
            before the charm is imported. A lightweight alternative to a full
            venv for offline tests.
        meta: Charm metadata dict (``metadata.yaml`` format). If omitted,
            read from the charm's ``charmcraft.yaml`` (or, for older charms,
            ``metadata.yaml``).
        config: Charm config dict (``config.yaml`` format). If omitted, read
            from the charm source in the same way as ``meta``.
        actions: Charm actions dict (``actions.yaml`` format). If omitted,
            read from the charm source in the same way as ``meta``.
        app_name: Application name as seen by the charm. Defaults to the
            charm name from the metadata.
        unit_id: Unit ID. Defaults to ``0``.
        juju_version: Juju agent version to simulate.
        app_trusted: Whether the application has Juju trust, as for
            :class:`~ops.testing.Context`.
        charm_root: The charm directory the charm runs with, as for
            :class:`~ops.testing.Context`.
        dispatch_timeout: Seconds to let a single event run in the worker
            before the worker is killed and :class:`IsolationError` raised.
            Pass ``None`` to wait indefinitely, which is what a charm being
            stepped through in a debugger needs.

    Invariant:
        The per-charm venv must have the **same** ``ops`` version installed as
        the parent test process. Mismatches surface as
        :class:`IsolationError`.

    For example, to run a charm in an existing virtual environment built for
    it::

        ctx = IsolatedContext(
            charm_source=pathlib.Path('./charms/myapp'),
            python_executable='/path/to/myapp-venv/bin/python',
        )
        state_out = ctx.run(ctx.on.install(), State())
        assert state_out.unit_status == ActiveStatus('ready')
    """

    #: Use ``ctx.on.<event>(...)`` to construct events for :meth:`run`.
    on: CharmEvents

    def __init__(
        self,
        charm_source: str | pathlib.Path,
        python_executable: str | None = None,
        extra_sys_path: Sequence[str] = (),
        *,
        meta: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        actions: Mapping[str, Any] | None = None,
        app_name: str | None = None,
        unit_id: int = 0,
        juju_version: str = _DEFAULT_JUJU_VERSION,
        app_trusted: bool = False,
        charm_root: str | pathlib.Path | None = None,
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

        # Read metadata in the parent — the charm itself is never imported here.
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
        self.dispatch_timeout = dispatch_timeout

    def run(self, event: _Event, state: State) -> State:
        """Trigger a charm execution with an event and a State.

        Serialises ``event`` and ``state``, dispatches them to a worker
        subprocess running in this context's interpreter, and returns the output
        :class:`~ops.testing.State`.

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
        return _dispatch(
            self._env,
            meta=self._meta,
            config=self._config,
            actions=self._actions,
            app_name=self.app_name,
            unit_id=self.unit_id,
            juju_version=self.juju_version,
            app_trusted=self.app_trusted,
            charm_root=self.charm_root,
            event=event,
            state_in=state,
            timeout=self.dispatch_timeout,
        )
