# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Wire codec for the isolated-worker protocol.

``State`` payloads use the typed, schema-versioned codec in
:mod:`scenario._state_serde` — the canonical ``ops.testing`` State serialiser
(step 2 of the Saddle delivery plan).  This module re-exports
:func:`encode_state` / :func:`decode_state` from there so the isolation layer
has a single import point, and adds the matching :class:`_Event`
(de)serialisation built on the *same* primitives.

Events are encoded with the same typed encoder as ``State`` rather than a
parallel one: an ``_Event`` carries the same leaf types as ``State`` (pebble
enums, ``datetime``, ``pathlib.Path``, ``_EntityStatus`` subclasses, nested
dataclasses such as ``Relation`` / ``Container`` / ``Secret``), so it needs the
same coverage.  The event payload is wrapped in an ``event_schema_version``
envelope that mirrors the State one and shares the version dispatch table.
"""

from __future__ import annotations

import json

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

_EVENT_SCHEMA_VERSION = _state_serde.STATE_SCHEMA_VERSION


def encode_event(event: _state._Event) -> str:
    """Serialise an :class:`_Event` to a JSON string using the typed State codec."""
    payload = {
        'event_schema_version': _EVENT_SCHEMA_VERSION,
        'event': _state_serde._encode(event, 'event'),
    }
    return json.dumps(payload)


def decode_event(payload: str) -> _state._Event:
    """Round-trip a JSON string produced by :func:`encode_event` back to an _Event.

    Raises:
        StateSchemaVersionError: if the payload's ``event_schema_version`` is not
            supported by this version of ops.
        TypeError: if the decoded payload is not an ``_Event``.
    """
    data = json.loads(payload)
    version = data.get('event_schema_version')
    decode_fn = _state_serde._VERSION_DECODERS.get(version)
    if decode_fn is None:
        raise _state_serde.StateSchemaVersionError(
            f'Unsupported event_schema_version={version!r}. '
            f'Supported versions: {sorted(_state_serde._VERSION_DECODERS)}. '
            'Upgrade ops to a version that supports this payload.'
        )
    result = decode_fn(data['event'])
    if not isinstance(result, _state._Event):
        raise TypeError(f'Decoded payload is not an _Event: {type(result).__name__}.')
    return result
