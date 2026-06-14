# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Subprocess worker that runs charm events in an isolated environment.

The worker runs under the (potentially per-charm) Python interpreter selected in
:class:`~ops.testing.IsolatedEnv`.  The whole point is that *this* process may
have a completely different ``sys.path`` / set of installed packages than the
parent test process, so two charms with conflicting dependencies can each run in
their own world.

Two transports
~~~~~~~~~~~~~~
**Persistent (default).** Invoked as::

    python -m scenario._isolated_worker --serve

The worker enters a serve loop, reading framed JSON requests from ``stdin`` and
writing framed JSON responses to ``stdout`` (see :mod:`scenario._worker_protocol`
for the framing).  The charm module is imported once and cached, so subsequent
events on the same charm avoid both interpreter startup and ``import`` cost.
This is the mode that makes a convergence run affordable.

**Spawn-per-event (debug).** Invoked as::

    python -m scenario._isolated_worker <request_file> <response_file>

The worker reads one request from ``request_file``, writes one response to
``response_file``, and exits.  A fresh process per event means no shared
interpreter state between events and an easy place to attach a debugger; it is
much slower and is offered only as an explicit debug mode.

Request / response shape
~~~~~~~~~~~~~~~~~~~~~~~~~~
A request dict has the keys:

- ``cmd`` (``str``, persistent only): ``"run"`` or ``"shutdown"``.
- ``charm_source`` (``str``): path to the charm repo root (``src/``, ``lib/``).
- ``extra_sys_path`` (``list[str]``): prepended to ``sys.path`` before import.
- ``meta`` / ``config`` / ``actions`` (``dict | None``): charm spec.
- ``app_name`` (``str``), ``unit_id`` (``int``).
- ``event`` (``str``): the JSON wire form of the input ``_Event``.
- ``state_in`` (``str``): the JSON wire form of the input ``State``.

A response dict is either ``{"state_out": <str>}`` (the JSON wire form of the
output ``State``) or ``{"error": <str>}`` (a formatted traceback when the charm
raises).  A worker *crash* (process death) is detected by the parent as a
missing response, not via this dict.

Serialisation compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``State`` and ``_Event`` are round-tripped through
:mod:`scenario._isolated_serde` (which delegates State to the canonical
:mod:`scenario._state_serde`).  The parent and worker must therefore have the
**same** ``ops`` version; only the charm's *own* runtime dependencies may differ.
"""

import json
import pathlib
import sys
import traceback
from typing import Any


def _load_charm_type(charm_source: pathlib.Path):
    """Import the charm module and return its CharmBase subclass.

    Adds ``charm_source/src`` and ``charm_source/lib`` to ``sys.path`` so that
    the charm's own source files and bundled charm libraries are importable.

    Args:
        charm_source: Path to the charm repository root.

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

    import importlib

    module = importlib.import_module('charm')

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
            'The isolated worker requires exactly one.'
        )
    return charm_types[0]


def _run(request: dict, charm_cache: dict[str, Any] | None = None) -> dict:
    """Execute a single charm event and return the serialised output state.

    Args:
        request: The request dict from the parent process.
        charm_cache: Optional ``{charm_source: charm_type}`` cache.  In the
            persistent serve loop the charm module is imported once and reused;
            in spawn-per-event mode this is ``None`` (a fresh process each time).

    Returns:
        ``{"state_out": <json-str>}`` on success.

    Raises:
        Any exception raised by the charm or by ops.testing is propagated to
        the caller, which wraps it in ``{"error": traceback_str}``.
    """
    # Make the per-charm dependency set importable BEFORE anything else.
    # extra_sys_path entries are prepended so they take priority over any
    # site-packages already on sys.path (i.e. the worker venv's packages).
    for entry in reversed(request.get('extra_sys_path', [])):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from scenario import Context, _isolated_serde

    charm_source = request['charm_source']
    if charm_cache is not None and charm_source in charm_cache:
        charm_type = charm_cache[charm_source]
    else:
        charm_type = _load_charm_type(pathlib.Path(charm_source))
        if charm_cache is not None:
            charm_cache[charm_source] = charm_type

    event = _isolated_serde.decode_event(request['event'])
    state_in = _isolated_serde.decode_state(request['state_in'])

    ctx = Context(
        charm_type,
        meta=request['meta'],
        config=request['config'],
        actions=request['actions'],
        app_name=request['app_name'],
        unit_id=request['unit_id'],
    )
    state_out = ctx.run(event, state_in)
    return {'state_out': _isolated_serde.encode_state(state_out)}


def serve() -> int:
    """Run the persistent serve loop, reading framed requests until EOF/shutdown.

    The charm module is imported once and cached for the lifetime of the
    process.  ``stdout`` is reserved for the framed protocol, so the charm's own
    ``stdout`` is redirected to ``stderr`` to keep it from corrupting the stream.

    Returns:
        ``0`` on a clean shutdown (``{"cmd": "shutdown"}`` or stdin closed).
    """
    from . import _worker_protocol

    real_stdin = sys.stdin.buffer
    real_stdout = sys.stdout.buffer
    # Anything the charm prints to stdout would corrupt the framed protocol;
    # redirect it to stderr (which the parent drains separately).
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

    request_file, response_file = argv[1], argv[2]

    with open(request_file, encoding='utf8') as fh:
        request = json.load(fh)

    try:
        response = _run(request)
    except Exception:
        response = {'error': traceback.format_exc()}

    with open(response_file, 'w', encoding='utf8') as fh:
        json.dump(response, fh)

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
