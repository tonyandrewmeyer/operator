# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""The worker process that :mod:`scenario._isolation` starts.

``python -m scenario._isolated_worker --serve`` reads framed JSON requests
from stdin and writes responses to stdout, importing the charm once.
``python -m scenario._isolated_worker <request> <response>`` handles one
request, from a file, and exits.

A response is ``{"state_out": ...}`` or ``{"error": <traceback>}``, with
``"hook_failed": true`` when the error is the charm's own.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
import traceback
from typing import Any, cast


def _run(request: dict[str, Any], charm_cache: dict[str, Any] | None = None) -> dict[str, Any]:
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
    # Ahead of the environment's site-packages, so these win for the charm's
    # own imports.
    for entry in reversed(cast('list[str]', request['extra_sys_path'])):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from scenario import Context, _charm_dispatch, _charm_mocking, _state_serde, _unit_filesystem
    from scenario.errors import UncaughtCharmError

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
            charm_type = _charm_dispatch.load_charm_type(pathlib.Path(charm_source))
        if charm_cache is not None:
            charm_cache[charm_source] = charm_type

    event = _state_serde.decode_event(request['event'])
    state_in = _state_serde.decode_state(request['state_in'])

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
        machine_id=_charm_dispatch.machine_id(
            request['machines'], request['app_name'], request['unit_id']
        ),
    )
    ctx._wrap_charm_errors = True
    if request.get('workload_root') is not None:
        ctx._workload_root = pathlib.Path(request['workload_root'])
    try:
        with (
            # Juju runs every hook in the unit's charm directory.
            _unit_filesystem.working_directory(request['charm_root']),
            _charm_dispatch.secret_ids(request['secret_seed']),
        ):
            if mocking is None:
                state_out = ctx.run(event, state_in)
            else:
                unit_name = f'{request["app_name"]}/{request["unit_id"]}'
                with mocking.dispatching(
                    unit_name,
                    state_in.model.name,
                    model_type=state_in.model.type,
                    model_uuid=state_in.model.uuid,
                    machines=request['machines'],
                    trusted=request['app_trusted'],
                    filesystem_root=request.get('filesystem_root'),
                    allow=_unit_filesystem.framework_paths(ctx, state_in),
                ):
                    state_out = ctx.run(event, state_in)
    except UncaughtCharmError as e:
        return {'error': _charm_dispatch.charm_traceback(e), 'hook_failed': True}
    return {'state_out': _state_serde.encode_state(state_out)}


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

    response: dict[str, Any]
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
