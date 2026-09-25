# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Subprocess worker that runs charm events in an isolated environment.

The worker runs under the (potentially per-charm) Python interpreter selected in
:class:`scenario._isolation.IsolatedContext`. The whole point is that *this* process may
have a completely different ``sys.path`` / set of installed packages than the
parent test process, so two charms with conflicting dependencies can each run in
their own world.

Two transports
~~~~~~~~~~~~~~
**Persistent (default).** Invoked as::

    python -m scenario._isolated_worker --serve

The worker enters a serve loop, reading framed JSON requests from ``stdin`` and
writing framed JSON responses to ``stdout`` (see :mod:`scenario._worker_protocol`
for the framing). The charm module is imported once and cached, so subsequent
events on the same charm avoid both interpreter startup and ``import`` cost.
This is the mode that makes a convergence run affordable.

**Spawn-per-event (debug).** Invoked as::

    python -m scenario._isolated_worker <request_file> <response_file>

The worker reads one request from ``request_file``, writes one response to
``response_file``, and exits. A fresh process per event means no shared
interpreter state between events, and a debugger can attach to the single
process; it is much slower and is offered only as an explicit debug mode.

Request / response shape
~~~~~~~~~~~~~~~~~~~~~~~~~~
A request dict has the keys:

- ``cmd`` (``str``, persistent only): ``"run"`` or ``"shutdown"``.
- ``charm_source`` (``str``): path to the charm repo root (``src/``, ``lib/``).
- ``extra_sys_path`` (``list[str]``): prepended to ``sys.path`` before import.
- ``meta`` / ``config`` / ``actions`` (``dict | None``): charm spec.
- ``app_name`` (``str``), ``unit_id`` (``int``), ``juju_version`` (``str``).
- ``app_trusted`` (``bool``), ``charm_root`` (``str | None``): as for ``Context``.
- ``mocking`` (``dict | None``): the keyword arguments for the charm's own
  mocking; ``None`` runs the charm with no mocking at all.
- ``event`` (``str``): the JSON wire form of the input ``_Event``.
- ``state_in`` (``str``): the JSON wire form of the input ``State``.

A response dict is either ``{"state_out": <str>}`` (the JSON wire form of the
output ``State``) or ``{"error": <str>}`` (a formatted traceback when the charm
raises). A worker *crash* (process death) is detected by the parent as a
missing response, not via this dict.

Serialisation compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``State`` and ``_Event`` are round-tripped through
:mod:`scenario._state_serde`, which the event codec also uses. The parent and
worker must therefore have the **same** ``ops`` version; only the charm's *own*
runtime dependencies may differ.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import pathlib
import sys
import traceback
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover
    from ops import CharmBase


def _load_charm_type(charm_source: pathlib.Path, module_name: str = 'charm') -> type[CharmBase]:
    """Import the charm module and return its CharmBase subclass.

    Adds ``charm_source/src`` and ``charm_source/lib`` to ``sys.path`` so that
    the charm's own source files and bundled charm libraries are importable.

    Args:
        charm_source: Path to the charm repository root.
        module_name: The name to import ``src/charm.py`` under. The worker
            runs one charm, so it uses ``charm``. Several charms loaded into
            one process each need their own name, and are loaded from the
            file rather than found on ``sys.path``.

    Returns:
        The charm class (a :class:`ops.CharmBase` subclass).

    Raises:
        RuntimeError: if zero or more than one charm class is found.
    """
    from ops import CharmBase

    sources = [str(charm_source / 'src'), str(charm_source / 'lib')]
    for entry in sources:
        if pathlib.Path(entry).exists() and entry not in sys.path:
            sys.path.insert(0, entry)

    if module_name == 'charm':
        module = importlib.import_module('charm')
    elif module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(
            module_name, charm_source / 'src' / 'charm.py'
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Cannot load {charm_source}/src/charm.py.')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    charm_types = [
        t
        for t in module.__dict__.values()
        if isinstance(t, type) and issubclass(t, CharmBase) and t is not CharmBase
    ]
    if not charm_types:
        raise RuntimeError(f'No CharmBase subclass found in {charm_source}/src/charm.py.')
    if len(charm_types) > 1:
        raise RuntimeError(
            f'Multiple CharmBase subclasses found in {charm_source}/src/charm.py: '
            f'{[t.__name__ for t in charm_types]}. '
            'Exactly one is required.'
        )
    return charm_types[0]


def _run(request: dict[str, Any], charm_cache: dict[str, Any] | None = None) -> dict[str, str]:
    """Execute a single charm event and return the serialised output state.

    Args:
        request: The request dict from the parent process.
        charm_cache: Optional ``{charm_source: charm_type}`` cache. In the
            persistent serve loop the charm module is imported once and reused;
            in spawn-per-event mode this is ``None`` (a fresh process each time).

    Returns:
        ``{"state_out": <json-str>}`` on success.

    Note:
        This propagates any exception the charm or ops.testing raises; the
        caller wraps it in ``{"error": traceback_str}``.
    """
    # Make the per-charm dependency set importable BEFORE anything else.
    # extra_sys_path entries are prepended so they take priority over any
    # site-packages already on sys.path (that is, the worker venv's packages).
    for entry in reversed(cast('list[str]', request.get('extra_sys_path', []))):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from scenario import Context, State, _charm_mocking, _isolated_serde

    charm_source = request['charm_source']
    mocking_key = f'mocking:{charm_source}'
    mocking: _charm_mocking.CharmMocking | None = None
    if charm_cache is not None and mocking_key in charm_cache:
        mocking = cast('_charm_mocking.CharmMocking', charm_cache[mocking_key])
    elif request.get('mocking') is not None:
        mocking = _charm_mocking.CharmMocking(
            pathlib.Path(charm_source),
            app_name=request['app_name'],
            mocking=request['mocking'],
            module_name='_ops_testing_mocking',
        )
        if charm_cache is not None:
            charm_cache[mocking_key] = mocking
    if mocking is not None:
        # The charm's mocking module loads before the charm itself, so its
        # import-time patches are in place when the charm is imported.
        mocking.load()

    if charm_cache is not None and charm_source in charm_cache:
        charm_type = charm_cache[charm_source]
    else:
        with mocking.importing() if mocking is not None else contextlib.nullcontext():
            charm_type = _load_charm_type(pathlib.Path(charm_source))
        if charm_cache is not None:
            charm_cache[charm_source] = charm_type

    event = _isolated_serde.decode_event(request['event'])
    state_in = State._from_json(request['state_in'])

    ctx = Context(
        charm_type,
        meta=request['meta'],
        config=request['config'],
        actions=request['actions'],
        app_name=request['app_name'],
        unit_id=request['unit_id'],
        juju_version=request['juju_version'],
        app_trusted=request['app_trusted'],
        charm_root=request['charm_root'],
    )
    if mocking is None:
        state_out = ctx.run(event, state_in)
    else:
        unit_name = f'{request["app_name"]}/{request["unit_id"]}'
        with mocking.dispatching(unit_name, state_in.model.name):
            state_out = ctx.run(event, state_in)
    return {'state_out': state_out._to_json()}


def serve() -> int:
    """Run the persistent serve loop, reading framed requests until EOF/shutdown.

    The charm module is imported once and cached for the lifetime of the
    process. ``stdout`` is reserved for the framed protocol, so the charm's own
    ``stdout`` is redirected to ``stderr`` to keep it from corrupting the stream.

    Returns:
        ``0`` on a clean shutdown (``{"cmd": "shutdown"}`` or stdin closed).
    """
    from . import _worker_protocol
    from .errors import JujuError

    real_stdin = sys.stdin.buffer
    # Anything written to stdout would corrupt the framed protocol. Reassigning
    # sys.stdout only covers Python-level writes: a charm that shells out
    # without capturing, or a C extension calling write(1, ...), goes straight
    # past it. So the protocol moves to a private descriptor and fd 1 itself is
    # pointed at stderr, which the parent drains separately.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    real_stdout = os.fdopen(protocol_fd, 'wb')
    sys.stdout = sys.stderr

    charm_cache: dict[str, Any] = {}

    while True:
        raw = _worker_protocol.read_frame(real_stdin)
        if raw is None:
            return 0  # Parent closed stdin.
        request = json.loads(raw.decode('utf8'))
        if request.get('cmd') == 'shutdown':
            return 0
        try:
            response = _run(request, charm_cache)
        except JujuError as e:
            # A problem with how the charm is set up to run, such as its
            # mocking, rather than a failure inside a hook.
            response = {'error': str(e)}
        except Exception:
            response = {'error': traceback.format_exc()}
        _worker_protocol.write_frame(real_stdout, json.dumps(response).encode('utf8'))


def main(argv: list[str]) -> int:
    """Entry point for the worker subprocess.

    With ``argv[1] == '--serve'`` the worker runs the persistent serve loop.
    Otherwise ``argv[1]`` / ``argv[2]`` are the request / response file paths for
    a single spawn-per-event dispatch.

    Returns:
        ``0`` always (charm errors are communicated via the response, never the
        exit code).
    """
    if argv[1] == '--serve':
        return serve()

    from .errors import JujuError

    request_file, response_file = argv[1], argv[2]

    with open(request_file, encoding='utf8') as fh:
        request = cast('dict[str, Any]', json.load(fh))

    response: dict[str, str]
    try:
        response = _run(request)
    except JujuError as e:
        response = {'error': str(e)}
    except Exception:
        response = {'error': traceback.format_exc()}

    with open(response_file, 'w', encoding='utf8') as fh:
        json.dump(response, fh)

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
