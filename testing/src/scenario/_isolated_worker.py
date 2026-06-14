# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Subprocess worker that runs a single charm event in an isolated environment.

This module is executed as::

    python -m scenario._isolated_worker <request_file> <response_file>

by the (potentially per-charm) Python interpreter selected in
:class:`~ops.testing.IsolatedEnv`.  The whole point is that *this* process may
have a completely different ``sys.path`` / set of installed packages than the
parent test process, so two charms with conflicting dependencies can each run in
their own world.

Protocol (spawn-per-event)
~~~~~~~~~~~~~~~~~~~~~~~~~~
``argv[1]`` — path to a JSON file with the request dict.  Keys:

- ``charm_source`` (``str``): path to the charm repo root (``src/``, ``lib/``).
- ``extra_sys_path`` (``list[str]``): prepended to ``sys.path`` before the charm
  is imported.
- ``meta`` / ``config`` / ``actions`` (``dict | None``): charm spec.
- ``app_name`` (``str``), ``unit_id`` (``int``).
- ``event`` (``str``): the JSON wire form of the input ``_Event``.
- ``state_in`` (``str``): the JSON wire form of the input ``State``.

``argv[2]`` — path to write the response JSON file.  Keys:

- ``{"state_out": <str>}`` — the JSON wire form of the output ``State``.
- ``{"error": <str>}`` — formatted traceback on charm failure.

Serialisation compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``State`` and ``_Event`` are round-tripped through
:mod:`scenario._isolated_serde`, which reconstructs scenario dataclasses by
name.  The parent and worker must therefore have the **same** ``ops`` version.
The parent sets ``PYTHONPATH`` to include the ``testing/src`` directory so the
worker imports the matching ``scenario.*`` classes.

Only the charm's *own* runtime dependencies (``cryptography``, ``pydantic``,
charm libs, ...) may differ between the worker venv and the parent process.

Auto-detection
~~~~~~~~~~~~~~
The worker detects the charm class by scanning the imported ``charm`` module for
subclasses of :class:`ops.CharmBase`.  The charm source must therefore expose
a single ``CharmBase`` subclass in ``src/charm.py``.  If multiple or zero charm
classes are found, the worker exits with an error.
"""

import json
import pathlib
import sys
import traceback


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


def _run(request: dict) -> dict:
    """Execute a single charm event and return the serialised output state.

    Args:
        request: The request dict from the parent process.

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

    charm_source = pathlib.Path(request['charm_source'])
    charm_type = _load_charm_type(charm_source)

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


def main(argv: list[str]) -> int:
    """Entry point for the worker subprocess.

    Reads the request JSON file, runs :func:`_run`, and writes the response
    JSON file.  Exceptions from the charm are caught and returned as an
    ``{"error": traceback_str}`` response so the parent can raise
    :class:`~ops.testing.IsolationError` with the full traceback.

    Args:
        argv: ``sys.argv``-style argument list; ``argv[1]`` is the request file
            path and ``argv[2]`` is the response file path.

    Returns:
        ``0`` always (errors are communicated via the response file).
    """
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
