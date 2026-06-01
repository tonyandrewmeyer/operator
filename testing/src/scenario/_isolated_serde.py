# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Typed JSON wire format for the isolated worker.

This module encodes :class:`~ops.testing.State` and :class:`_Event` (and the
many dataclasses they hold) into a JSON-compatible structure, and decodes the
same back to the original objects.  It replaces the earlier ``pickle`` wire
format used by the spawn-per-event isolated worker.

Why a hand-rolled encoder rather than ``dataclasses.asdict`` + ``json``:

* ``dataclasses.asdict`` does not recurse into ``set`` / ``frozenset`` fields
  (it passes them through unchanged), so a ``State`` carrying any of the
  default ``frozenset`` collections will not survive ``json.dumps``.
* ``json`` does not handle ``datetime``, ``timedelta``, ``pathlib.Path``,
  ``set`` / ``frozenset``, or ``tuple``-vs-``list`` distinctions on its own.
* The status types (``ActiveStatus`` etc.) are dataclasses with **custom**
  ``__init__`` signatures — round-tripping them through ``cls(**fields)`` does
  not work; they need ``_EntityStatus.from_status_name``.

The wire format wraps each non-primitive value in a small envelope:

    {"__t__": "<kind>", ...}

where ``<kind>`` is one of: ``dc`` (any dataclass), ``status`` (subclasses of
``_EntityStatus`` — special-cased ``__init__``), ``set``, ``frozenset``,
``tuple``, ``datetime``, ``timedelta``, ``path``, ``layer`` (a ``pebble.Layer``).
Plain ``list`` and ``dict`` carry no envelope.
"""

from __future__ import annotations

import dataclasses
import datetime
import inspect
import json
import pathlib
from typing import Any

from ops import pebble

from . import state as _state

_TYPE_KEY = '__t__'

_STATUS_TYPES: dict[str, type[_state._EntityStatus]] = {
    'active': _state.ActiveStatus,
    'blocked': _state.BlockedStatus,
    'waiting': _state.WaitingStatus,
    'maintenance': _state.MaintenanceStatus,
    'error': _state.ErrorStatus,
    'unknown': _state.UnknownStatus,
}

_DC_TYPES: dict[str, type] = {}


def _build_dc_registry() -> None:
    """Populate ``_DC_TYPES`` with every dataclass in ``scenario.state``.

    Called lazily on first decode so import-time cost is paid only by tests
    that actually round-trip a State.
    """
    if _DC_TYPES:
        return
    for name in dir(_state):
        value = getattr(_state, name)
        if (
            inspect.isclass(value)
            and dataclasses.is_dataclass(value)
            and value.__module__ == _state.__name__
        ):
            _DC_TYPES[value.__name__] = value


def _encode(obj: Any) -> Any:
    """Recursively encode ``obj`` into a JSON-compatible structure."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, _state._EntityStatus):
        return {_TYPE_KEY: 'status', 'name': obj.name, 'message': obj.message}
    if isinstance(obj, pebble.Layer):
        return {_TYPE_KEY: 'layer', 'value': obj.to_dict()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            _TYPE_KEY: 'dc',
            'cls': type(obj).__name__,
            'fields': {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, datetime.datetime):
        return {_TYPE_KEY: 'datetime', 'value': obj.isoformat()}
    if isinstance(obj, datetime.timedelta):
        return {_TYPE_KEY: 'timedelta', 'value': obj.total_seconds()}
    if isinstance(obj, pathlib.PurePath):
        return {_TYPE_KEY: 'path', 'value': str(obj)}
    if isinstance(obj, frozenset):
        return {_TYPE_KEY: 'frozenset', 'value': [_encode(x) for x in obj]}
    if isinstance(obj, set):
        return {_TYPE_KEY: 'set', 'value': [_encode(x) for x in obj]}
    if isinstance(obj, tuple):
        return {_TYPE_KEY: 'tuple', 'value': [_encode(x) for x in obj]}
    if isinstance(obj, list):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    raise TypeError(
        f'No JSON encoding for type {type(obj).__name__}: {obj!r}. '
        'Extend testing/src/scenario/_isolated_serde.py to handle it.'
    )


def _decode(obj: Any) -> Any:
    """Recursively decode a value produced by :func:`_encode`."""
    if isinstance(obj, list):
        return [_decode(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    kind = obj.get(_TYPE_KEY)
    if kind is None:
        return {k: _decode(v) for k, v in obj.items()}
    if kind == 'status':
        name = obj['name']
        cls = _STATUS_TYPES[name]
        if name == 'unknown':
            return cls()
        return cls(obj['message'])
    if kind == 'layer':
        return pebble.Layer(obj['value'])
    if kind == 'dc':
        _build_dc_registry()
        cls = _DC_TYPES[obj['cls']]
        fields = {k: _decode(v) for k, v in obj['fields'].items()}
        return cls(**fields)
    if kind == 'datetime':
        return datetime.datetime.fromisoformat(obj['value'])
    if kind == 'timedelta':
        return datetime.timedelta(seconds=obj['value'])
    if kind == 'path':
        return pathlib.Path(obj['value'])
    if kind == 'frozenset':
        return frozenset(_decode(x) for x in obj['value'])
    if kind == 'set':
        return {_decode(x) for x in obj['value']}
    if kind == 'tuple':
        return tuple(_decode(x) for x in obj['value'])
    raise TypeError(f'Unknown wire type {kind!r}.')


def encode_state(state: _state.State) -> str:
    """Serialise a :class:`~ops.testing.State` to a JSON string."""
    return json.dumps(_encode(state))


def decode_state(payload: str) -> _state.State:
    """Round-trip a JSON string produced by :func:`encode_state` back to a State."""
    result = _decode(json.loads(payload))
    if not isinstance(result, _state.State):
        raise TypeError(f'Decoded payload is not a State: {type(result).__name__}.')
    return result


def encode_event(event: _state._Event) -> str:
    """Serialise a :class:`_Event` to a JSON string."""
    return json.dumps(_encode(event))


def decode_event(payload: str) -> _state._Event:
    """Round-trip a JSON string produced by :func:`encode_event` back to an _Event."""
    result = _decode(json.loads(payload))
    if not isinstance(result, _state._Event):
        raise TypeError(f'Decoded payload is not an _Event: {type(result).__name__}.')
    return result
