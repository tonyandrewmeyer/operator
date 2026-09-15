# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Wire codec for the isolated-worker protocol.

``State`` payloads go through :meth:`~ops.testing.State._to_json` and
:meth:`~ops.testing.State._from_json`, which wrap the canonical codec in
:mod:`scenario._state_serde`. This module adds the matching :class:`_Event`
(de)serialisation, built on the same primitives.

Events use the same typed encoder as ``State`` rather than a parallel one: an
``_Event`` carries the same leaf types (pebble enums, ``datetime``,
``pathlib.Path``, ``_EntityStatus`` subclasses, and nested dataclasses such as
``Relation``, ``Container`` and ``Secret``), so it needs the same coverage. The
event payload embeds the producing ``ops.testing`` version, mirroring the
``State`` payload, and the receiving side requires it to match: every per-charm
venv carries the same ``ops.testing`` version as the parent process, so there
is no cross-version negotiation to do.
"""

from __future__ import annotations

from . import _state_serde
from . import state as _state

__all__ = [
    'decode_event',
    'encode_event',
]


def encode_event(event: _state._Event) -> str:
    """Serialise an :class:`_Event` to a JSON string using the typed State codec."""
    return _state_serde._encode_envelope(event, 'event')


def decode_event(payload: str) -> _state._Event:
    """Decode a JSON string produced by :func:`encode_event` back to an ``_Event``.

    Raises:
        StateVersionMismatchError: if the payload's producing ``ops.testing``
            version does not match this process's.
        TypeError: if the decoded payload is not an ``_Event``.
    """
    result = _state_serde._decode_envelope(payload, 'event')
    if not isinstance(result, _state._Event):
        raise TypeError(f'Decoded payload is not an _Event: {type(result).__name__!r}.')
    return result
