# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Wire codec for the isolated-worker protocol.

``State`` payloads use the typed codec in :mod:`scenario._state_serde` — the
canonical ``ops.testing`` State serialiser (step 2 of the Saddle delivery
plan).  This module re-exports :func:`encode_state` / :func:`decode_state`
from there so the isolation layer has a single import point, and adds the
matching :class:`_Event` (de)serialisation built on the *same* primitives.

Events are encoded with the same typed encoder as ``State`` rather than a
parallel one: an ``_Event`` carries the same leaf types as ``State`` (pebble
enums, ``datetime``, ``pathlib.Path``, ``_EntityStatus`` subclasses, nested
dataclasses such as ``Relation`` / ``Container`` / ``Secret``), so it needs the
same coverage.  The event payload embeds the producing ``ops.testing``
version, mirroring the State payload, and is required to match this
process's for the same reason: every per-charm venv carries the same
``ops.testing`` version as the parent process, so there is no cross-version
negotiation to do.
"""

from __future__ import annotations

import json

import ops.version

from . import _state_serde
from . import state as _state

__all__ = [
    'decode_event',
    'decode_state',
    'encode_event',
    'encode_state',
]

# State codec — re-exported from the canonical serialiser so callers import the
# State and event codecs from one place.
encode_state = _state_serde.encode_state
decode_state = _state_serde.decode_state


def encode_event(event: _state._Event) -> str:
    """Serialise an :class:`_Event` to a JSON string using the typed State codec."""
    payload = {
        'ops_testing_version': ops.version.version,
        'event': _state_serde._encode(event, 'event'),
    }
    return json.dumps(payload)


def decode_event(payload: str) -> _state._Event:
    """Round-trip a JSON string produced by :func:`encode_event` back to an _Event.

    Raises:
        StateVersionMismatchError: if the payload's producing ``ops.testing``
            version does not match this process's.
        TypeError: if the decoded payload is not an ``_Event``.
    """
    data = json.loads(payload)
    producing_version = data.get('ops_testing_version')
    if producing_version != ops.version.version:
        raise _state_serde.StateVersionMismatchError(
            f'Event was encoded by ops.testing {producing_version!r}, but this '
            f'process is running ops.testing {ops.version.version!r}. '
            "Install a matching ops.testing (the 'ops[testing]' extra) in the charm's venv."
        )
    result = _state_serde._decode(data['event'])
    if not isinstance(result, _state._Event):
        raise TypeError(f'Decoded payload is not an _Event: {type(result).__name__}.')
    return result
