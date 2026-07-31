# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Typed JSON encoder/decoder for ops.testing.State.

Public surface (re-exported via ops.testing):

    encode_state(state: State) -> str
    decode_state(payload: str) -> State
    StateSchemaVersionError

Wire format
~~~~~~~~~~~
Every payload is a JSON object with a top-level ``state_schema_version``
integer and the encoded state tree::

    {"state_schema_version": 1, "state": <encoded>}

``decode_state`` reads the version and dispatches to a version-specific
decoder.  Unsupported versions raise :class:`StateSchemaVersionError`.

Type envelopes
~~~~~~~~~~~~~~
Non-primitive values are wrapped in a small envelope dict keyed on ``"__t__"``:

    ``"dc"``          - any dataclass from ``scenario.state``
    ``"status"``      - ``_EntityStatus`` subclasses (special ``__init__``)
    ``"frozenset"``   - ``frozenset``
    ``"set"``         - ``set``
    ``"datetime"``    - ISO-8601 string
    ``"timedelta"``   - ``total_seconds()`` float
    ``"Path"``        - ``pathlib.Path`` string
    ``"PurePosixPath"`` - ``pathlib.PurePosixPath`` string
    ``"layer"``       - ``pebble.Layer`` via ``to_dict()``/``Layer(dict)``
    ``"enum"``        - ``enum.Enum`` subclass (class name + member name)
    ``"bytes"``       - base64-encoded bytes
    ``"idict"``       - ``dict`` with ``int`` keys

Tuples use a list-based envelope: ``["__tuple__", elem, ...]``.

Plain ``list`` and ``dict`` with string keys carry no envelope.

StoredState escape hatch
~~~~~~~~~~~~~~~~~~~~~~~~
``StoredState.content`` (and ``Container._base_plan``) may contain types
that are a superset of JSON: ``bytes``, ``tuple``-vs-``list`` distinction,
and bare ``set``s.  The typed escape hatch is **always-on**:

* ``bytes``  → base64 via the ``"bytes"`` envelope.
* ``tuple``  → ``["__tuple__", ...]`` list-based envelope.
* ``set``    → ``"set"`` envelope.

Any value that cannot be encoded raises ``TypeError`` from the encoder,
including the dotted path through ``State`` at which the unrecognised type
was found.  The decoder mirrors: an unknown ``"__t__"`` tag raises
``TypeError``.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import enum
import inspect
import json
import pathlib
from collections.abc import Callable
from typing import Any, TypeAlias, Union, cast

from ops import SecretRotate, pebble

from . import state as _state

#: Any value that survives a ``json.dumps``/``json.loads`` round trip.  The
#: encoder produces one of these and the decoder consumes one, which keeps the
#: untyped ``Any`` from ``json.loads`` confined to a single narrowing point.
_JSON: TypeAlias = Union[bool, int, float, str, 'list[_JSON]', 'dict[str, _JSON]', None]

__all__ = [
    'STATE_SCHEMA_VERSION',
    'StateSchemaVersionError',
    'decode_state',
    'encode_state',
]

STATE_SCHEMA_VERSION = 1

_T = '__t__'
_TUPLE_SENTINEL = '__tuple__'

# Type registries

_PEBBLE_ENUM_TYPES: dict[str, type[enum.Enum]] = {
    'CheckLevel': pebble.CheckLevel,
    'CheckStartup': pebble.CheckStartup,
    'CheckStatus': pebble.CheckStatus,
    'NoticeType': pebble.NoticeType,
    'SecretRotate': SecretRotate,
    'ServiceStartup': pebble.ServiceStartup,
    'ServiceStatus': pebble.ServiceStatus,
}

# ``UnknownStatus`` is deliberately absent: alone among the statuses its
# ``__init__`` takes no message, so it is special-cased in the decoder and every
# entry here shares the one-message-argument signature.
_STATUS_TYPES: dict[str, Callable[[str], _state._EntityStatus]] = {
    'active': _state.ActiveStatus,
    'blocked': _state.BlockedStatus,
    'error': _state.ErrorStatus,
    'maintenance': _state.MaintenanceStatus,
    'waiting': _state.WaitingStatus,
}

_DC_TYPES: dict[str, type[Any]] = {}


def _build_dc_registry() -> None:
    """Populate ``_DC_TYPES`` with every dataclass from ``scenario.state``."""
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


# Errors


class StateSchemaVersionError(Exception):
    """Raised when a payload's ``state_schema_version`` is not supported by this ops version."""


# Encoder


def _encode(obj: Any, path: str = 'state') -> _JSON:
    """Recursively encode *obj* into a JSON-compatible structure.

    Raises:
        TypeError: if *obj* (or any nested value) has no registered encoding,
            including the dotted *path* through ``State`` in the message.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, bytes):
        return {_T: 'bytes', 'v': base64.b64encode(obj).decode('ascii')}

    # _EntityStatus check must precede the generic dataclass check because
    # status subclasses have custom __init__ signatures that cls(**fields) won't satisfy.
    if isinstance(obj, _state._EntityStatus):
        return {_T: 'status', 'name': obj.name, 'msg': obj.message}

    if isinstance(obj, pebble.Layer):
        return {_T: 'layer', 'v': cast('dict[str, _JSON]', obj.to_dict())}

    # Enum check before dataclass: pebble enums are not dataclasses.
    if isinstance(obj, enum.Enum):
        cls_name = type(obj).__name__
        if cls_name not in _PEBBLE_ENUM_TYPES:
            raise TypeError(f'Unrecognised enum type {type(obj).__qualname__!r} at path {path!r}.')
        return {_T: 'enum', 'cls': cls_name, 'name': obj.name}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        encoded_fields: dict[str, _JSON] = {}
        for f in dataclasses.fields(obj):
            # init=False fields (e.g. _Event._juju_name) are derived in
            # __post_init__ from other fields and can't be passed to
            # cls(**fields) on the decode side; skip them.
            if not f.init:
                continue
            val = getattr(obj, f.name)
            encoded_fields[f.name] = _encode(val, f'{path}.{f.name}')
        return {_T: 'dc', 'cls': type(obj).__name__, 'f': encoded_fields}

    if isinstance(obj, datetime.datetime):
        return {_T: 'datetime', 'v': obj.isoformat()}

    if isinstance(obj, datetime.timedelta):
        return {_T: 'timedelta', 'v': obj.total_seconds()}

    # pathlib.Path is a concrete subclass of PurePosixPath on Linux; check it first
    # so that concrete paths always get the 'Path' tag, not 'PurePosixPath'.
    if isinstance(obj, pathlib.Path):
        return {_T: 'Path', 'v': str(obj)}

    if isinstance(obj, pathlib.PurePosixPath):
        return {_T: 'PurePosixPath', 'v': str(obj)}

    if isinstance(obj, pathlib.PurePath):
        # Catch-all for PureWindowsPath and any other PurePath subclasses.
        return {_T: 'PurePosixPath', 'v': str(obj)}

    # The isinstance narrowing below leaves the element types unknown, since
    # *obj* is Any; the casts pin them to Any so that the encoded results are
    # fully typed.
    if isinstance(obj, frozenset):
        frozen = cast('frozenset[Any]', obj)
        return {_T: 'frozenset', 'v': [_encode(x, f'{path}[]') for x in frozen]}

    if isinstance(obj, set):
        members = cast('set[Any]', obj)
        return {_T: 'set', 'v': [_encode(x, f'{path}[]') for x in members]}

    if isinstance(obj, tuple):
        elements = cast('tuple[Any, ...]', obj)
        return [_TUPLE_SENTINEL, *(_encode(x, f'{path}[]') for x in elements)]

    if isinstance(obj, list):
        items = cast('list[Any]', obj)
        return [_encode(x, f'{path}[{i}]') for i, x in enumerate(items)]

    if isinstance(obj, dict):
        mapping = cast('dict[Any, Any]', obj)
        if mapping and all(isinstance(k, int) for k in mapping):
            # JSON requires string keys; preserve int-keyed dicts with a tag.
            return {
                _T: 'idict',
                'v': {str(k): _encode(v, f'{path}[{k}]') for k, v in mapping.items()},
            }
        return {str(k): _encode(v, f'{path}.{k}') for k, v in mapping.items()}

    raise TypeError(f'No JSON encoding for type {type(obj).__qualname__!r} at path {path!r}.')


def encode_state(state: _state.State) -> str:
    """Serialise a :class:`~ops.testing.State` to a JSON string.

    The payload includes a ``state_schema_version`` integer field for forward
    compatibility.  Use :func:`decode_state` to round-trip the result back to
    a ``State``.

    Raises:
        TypeError: if any field value in *state* has no registered encoding.
    """
    payload = {
        'state_schema_version': STATE_SCHEMA_VERSION,
        'state': _encode(state),
    }
    return json.dumps(payload)


# Decoder


def _as_str(value: _JSON, field: str) -> str:
    """Narrow a decoded payload field to ``str``, or say what was wrong with it."""
    if not isinstance(value, str):
        raise TypeError(f'Expected a string for {field} in wire payload, got {value!r}.')
    return value


def _as_number(value: _JSON, field: str) -> float:
    """Narrow a decoded payload field to a number, or say what was wrong with it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'Expected a number for {field} in wire payload, got {value!r}.')
    return value


def _as_list(value: _JSON, field: str) -> list[_JSON]:
    """Narrow a decoded payload field to ``list``, or say what was wrong with it."""
    if not isinstance(value, list):
        raise TypeError(f'Expected a list for {field} in wire payload, got {value!r}.')
    return value


def _as_dict(value: _JSON, field: str) -> dict[str, _JSON]:
    """Narrow a decoded payload field to ``dict``, or say what was wrong with it."""
    if not isinstance(value, dict):
        raise TypeError(f'Expected an object for {field} in wire payload, got {value!r}.')
    return value


def _decode_v1(obj: _JSON) -> Any:
    """Decode a value produced by :func:`_encode` (schema version 1)."""
    if isinstance(obj, list):
        if obj and obj[0] == _TUPLE_SENTINEL:
            return tuple(_decode_v1(x) for x in obj[1:])
        return [_decode_v1(x) for x in obj]

    if not isinstance(obj, dict):
        return obj

    kind = obj.get(_T)

    if kind is None:
        return {k: _decode_v1(v) for k, v in obj.items()}

    if kind == 'status':
        name = _as_str(obj['name'], 'status name')
        if name == 'unknown':
            # Alone among the statuses, UnknownStatus carries no message.
            return _state.UnknownStatus()
        cls = _STATUS_TYPES.get(name)
        if cls is None:
            raise TypeError(f'Unknown status name {name!r} in wire payload.')
        return cls(_as_str(obj['msg'], 'status message'))

    if kind == 'layer':
        return pebble.Layer(cast('pebble.LayerDict', _as_dict(obj['v'], 'layer')))

    if kind == 'enum':
        cls_name = _as_str(obj['cls'], 'enum class')
        enum_cls = _PEBBLE_ENUM_TYPES.get(cls_name)
        if enum_cls is None:
            raise TypeError(f'Unknown enum class {cls_name!r} in wire payload.')
        return enum_cls[_as_str(obj['name'], 'enum member')]

    if kind == 'dc':
        _build_dc_registry()
        cls_name = _as_str(obj['cls'], 'dataclass name')
        dc_cls = _DC_TYPES.get(cls_name)
        if dc_cls is None:
            raise TypeError(f'Unknown dataclass {cls_name!r} in wire payload.')
        fields = {k: _decode_v1(v) for k, v in _as_dict(obj['f'], 'dataclass fields').items()}
        return dc_cls(**fields)

    if kind == 'datetime':
        return datetime.datetime.fromisoformat(_as_str(obj['v'], 'datetime'))

    if kind == 'timedelta':
        return datetime.timedelta(seconds=_as_number(obj['v'], 'timedelta'))

    if kind == 'Path':
        return pathlib.Path(_as_str(obj['v'], 'Path'))

    if kind == 'PurePosixPath':
        return pathlib.PurePosixPath(_as_str(obj['v'], 'PurePosixPath'))

    if kind == 'frozenset':
        return frozenset(_decode_v1(x) for x in _as_list(obj['v'], 'frozenset'))

    if kind == 'set':
        return {_decode_v1(x) for x in _as_list(obj['v'], 'set')}

    if kind == 'bytes':
        return base64.b64decode(_as_str(obj['v'], 'bytes'))

    if kind == 'idict':
        return {int(k): _decode_v1(v) for k, v in _as_dict(obj['v'], 'idict').items()}

    raise TypeError(f'Unknown wire type tag {kind!r} in payload.')


# Dispatch table keyed by schema version; new versions add an entry here.
_VERSION_DECODERS: dict[int, Callable[[_JSON], Any]] = {
    1: _decode_v1,
}


def decode_state(payload: str) -> _state.State:
    """Decode a JSON string produced by :func:`encode_state` back to a :class:`~ops.testing.State`.

    Raises:
        StateSchemaVersionError: if the payload's ``state_schema_version`` is
            not supported by this version of ops.
        TypeError: if the payload contains an unknown type tag.
    """
    data = _as_dict(json.loads(payload), 'payload')
    version = data.get('state_schema_version')
    # A non-integer version is treated the same as an unsupported one: the
    # payload was produced by something this ops version does not understand.
    decode_fn = _VERSION_DECODERS.get(version) if isinstance(version, int) else None
    if decode_fn is None:
        raise StateSchemaVersionError(
            f'Unsupported state_schema_version={version!r}. '
            f'Supported versions: {sorted(_VERSION_DECODERS)}. '
            'Upgrade ops to a version that supports this payload.'
        )
    result = decode_fn(data['state'])
    if not isinstance(result, _state.State):
        raise TypeError(f'Decoded payload root is not a State: got {type(result).__name__!r}.')
    return result
