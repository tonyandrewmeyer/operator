# Copyright 2025 Canonical Ltd.
# See LICENSE file for licensing details.

"""Per-charm dependency isolation for ops.testing.

This module provides :class:`IsolatedEnv` and :class:`IsolatedContext` —
primitives that let a Scenario test drive *one* charm's event handler in an
isolated subprocess with its own ``sys.path`` / venv.  No convergence loop, no
multi-charm model: just the ability to run a single on-disk charm when its
dependency set conflicts with the test process's packages.

The isolation mechanism is a subprocess + per-charm interpreter, as recommended
by the spike (see ``saddle-spec.md`` §4 and ``dependency-isolation-findings.md``):

* **Each charm event is dispatched to a separate process** whose Python
  interpreter is selected per charm (e.g. a per-charm venv's ``bin/python``).
* **The parent test process never imports the charm.**  It reads only the
  charm's metadata (``metadata.yaml`` / ``charmcraft.yaml``) and serialises the
  :class:`~ops.testing.State` and event across the process boundary.
* **Conflicting binary dependencies coexist trivially** because each worker
  process has a genuinely independent ``sys.path`` / site-packages.

Subinterpreters are explicitly *not* used — they do not solve C-extension binary
conflicts and cost the same serialisation overhead.  See §4 of the spec for the
full rationale.

Serialisation (step 1)
~~~~~~~~~~~~~~~~~~~~~~
The event and state cross the process boundary via ``pickle`` files in a
temporary directory.  Both the parent test process and the per-charm worker must
therefore have **the same** ``ops`` / ``ops.testing`` version installed — only
the charm's own runtime dependencies (``cryptography``, ``pydantic``, charm
libs, ...) may differ between the worker venv and the parent.

A typed JSON encoder/decoder (step 2 of the incremental plan) will replace this
pickle wire format once it is merged; the interface of :class:`IsolatedEnv` and
:class:`IsolatedContext` will not change.

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

For fast offline tests, ``extra_sys_path`` lets you inject a pre-built
dependency directory without a full venv::

    ctx = testing.IsolatedContext(
        charm_source=pathlib.Path('./charms/myapp'),
        extra_sys_path=('./deps/mylib_v2',),
    )
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from .context import _DEFAULT_JUJU_VERSION, CharmEvents
from .state import State, _Event

__all__ = [
    'IsolatedContext',
    'IsolatedEnv',
    'IsolationError',
]


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


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
        A mismatch will typically surface as a pickle deserialization error
        inside :class:`IsolationError`.

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
    python_executable: str = dataclasses.field(
        default_factory=lambda: sys.executable
    )
    extra_sys_path: tuple[str, ...] = ()


class IsolationError(RuntimeError):
    """Raised when a charm event fails inside the isolated worker subprocess.

    This wraps any exception raised by the worker — either an uncaught charm
    exception or a worker-infrastructure error (e.g. the charm module could not
    be imported, or the worker process crashed without producing a response).

    The original traceback from the worker is included in the message.
    """


# ---------------------------------------------------------------------------
# Metadata helpers (reads charm metadata without importing the charm)
# ---------------------------------------------------------------------------


def _read_yaml(path: pathlib.Path) -> dict[str, Any] | None:
    """Read a YAML file and return its contents, or None if the file does not exist."""
    if not path.exists():
        return None
    with path.open() as fh:
        return yaml.safe_load(fh)


def _read_charm_metadata(charm_root: pathlib.Path) -> dict[str, Any]:
    """Read charm metadata from disk without importing the charm.

    Prefers ``metadata.yaml`` but falls back to the metadata embedded in a
    ``charmcraft.yaml`` (Juju's newer single-file format).

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


# ---------------------------------------------------------------------------
# Worker dispatch (spawn-per-event, step-1 mode)
# ---------------------------------------------------------------------------


def _dispatch(
    env: IsolatedEnv,
    *,
    meta: dict[str, Any],
    config: dict[str, Any] | None,
    actions: dict[str, Any] | None,
    app_name: str,
    unit_id: int,
    event: _Event,
    state_in: State,
) -> State:
    """Serialise a charm event request, spawn the worker, and return the output State.

    This is the *spawn-per-event* path (step 1 of the incremental plan).  A
    persistent-worker mode (step 3) will be added later behind the same
    :class:`IsolatedEnv` interface.

    The event and state cross the process boundary via ``pickle`` files in a
    short-lived temporary directory.  Both the parent and worker must therefore
    use the same ``ops`` version.

    Raises:
        IsolationError: if the worker exits without producing a response, or if
            the charm raised an uncaught exception inside the worker.
    """
    request = {
        'charm_source': str(env.charm_source),
        'extra_sys_path': list(env.extra_sys_path),
        'meta': meta,
        'config': config,
        'actions': actions,
        'app_name': app_name,
        'unit_id': unit_id,
        # event and state are pickled to avoid importing ops.testing types on
        # the parent side with a possibly mismatched worker-side version.
        'event': pickle.dumps(event),
        'state_in': pickle.dumps(state_in),
    }

    with tempfile.TemporaryDirectory(prefix='ops-iso-') as tmp:
        req_file = pathlib.Path(tmp) / 'request.pkl'
        resp_file = pathlib.Path(tmp) / 'response.pkl'

        with req_file.open('wb') as fh:
            pickle.dump(request, fh)

        cmd = [
            env.python_executable,
            '-m',
            'scenario._isolated_worker',
            str(req_file),
            str(resp_file),
        ]

        # Make the ops.testing package importable inside the worker regardless
        # of the worker interpreter's site-packages (the worker venv may not
        # have ops installed in editable/dev mode, but it will have the
        # release package; the PYTHONPATH override ensures the *parent's*
        # testing src is available so the worker can import the same
        # scenario.* classes as the parent for pickle compatibility).
        child_env = _child_environ()

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=child_env,
        )

        if not resp_file.exists():
            raise IsolationError(
                'Isolated worker produced no response.\n'
                f'Command: {cmd}\n'
                f'Return code: {proc.returncode}\n'
                f'stdout:\n{proc.stdout}\n'
                f'stderr:\n{proc.stderr}'
            )

        with resp_file.open('rb') as fh:
            response = pickle.load(fh)

    if 'error' in response:
        raise IsolationError(
            f'Isolated charm run failed for {app_name}/{unit_id}:\n'
            f'{response["error"]}'
        )

    return pickle.loads(response['state_out'])


def _child_environ() -> dict[str, str]:
    """Build the environment for the worker subprocess.

    Prepends the parent's ``scenario`` package source directory to
    ``PYTHONPATH`` so the worker uses the same ``ops.testing`` (scenario)
    classes as the parent, ensuring pickle compatibility.
    """
    # The scenario package lives at testing/src/scenario/; the importable root
    # is two levels up: testing/src/.
    scenario_src = str(pathlib.Path(__file__).resolve().parent.parent)

    child = dict(os.environ)
    existing = child.get('PYTHONPATH', '')
    parts = [scenario_src] + ([existing] if existing else [])
    child['PYTHONPATH'] = os.pathsep.join(parts)
    return child


# ---------------------------------------------------------------------------
# IsolatedContext — the public Context-like entry point
# ---------------------------------------------------------------------------


class IsolatedContext:
    """Run a single on-disk charm's events in an isolated subprocess.

    :class:`IsolatedContext` is the isolated counterpart of
    :class:`~ops.testing.Context`.  Instead of importing the charm into the
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
        actions: Charm actions dict (``actions.yaml`` format).  If omitted,
            read from ``charm_source/actions.yaml``.
        app_name: Application name as seen by the charm.  Defaults to the
            charm name from the metadata.
        unit_id: Unit ID.  Defaults to ``0``.
        juju_version: Juju agent version to simulate.

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

    Example — ``extra_sys_path`` for fast, offline tests::

        ctx = testing.IsolatedContext(
            charm_source=pathlib.Path('./charms/alpha'),
            extra_sys_path=('./deps/mylib_v1',),
        )
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
    ):
        charm_root = pathlib.Path(charm_source)
        if not charm_root.exists():
            raise ValueError(
                f'charm_source {charm_root!r} does not exist.'
            )

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

    @property
    def env(self) -> IsolatedEnv:
        """The :class:`IsolatedEnv` that describes this context's execution environment."""
        return self._env

    def run(self, event: _Event, state: State) -> State:
        """Trigger a charm execution with an event and a State.

        Serialises ``event`` and ``state``, dispatches them to a worker
        subprocess running in :attr:`env`'s interpreter, and returns the output
        :class:`~ops.testing.State`.

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
        return _dispatch(
            self._env,
            meta=self._meta,
            config=self._config,
            actions=self._actions,
            app_name=self._app_name,
            unit_id=self._unit_id,
            event=event,
            state_in=state,
        )
