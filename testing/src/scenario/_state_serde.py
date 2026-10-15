# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""JSON serialisation for :class:`scenario.state.State`.

This module is private because the JSON representation is an implementation
detail and may change between Ops releases.
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

__all__ = ['_decode_state', '_encode_state']

_TYPE_KEY = '__type__'
_JSON: TypeAlias = Union[bool, int, float, str, 'list[_JSON]', 'dict[str, _JSON]', None]


_ENUM_TYPES: dict[str, type[enum.Enum]] = {
    'CheckLevel': pebble.CheckLevel,
    'CheckStartup': pebble.CheckStartup,
    'CheckStatus': pebble.CheckStatus,
    'NoticeType': pebble.NoticeType,
    'SecretRotate': SecretRotate,
    'ServiceStartup': pebble.ServiceStartup,
    'ServiceStatus': pebble.ServiceStatus,
}

_STATUS_TYPES: dict[str, Callable[[str], _state._EntityStatus]] = {
    'active': _state.ActiveStatus,
    'blocked': _state.BlockedStatus,
    'error': _state.ErrorStatus,
    'maintenance': _state.MaintenanceStatus,
    'waiting': _state.WaitingStatus,
}

_DATACLASS_TYPES: dict[str, type[Any]] = {}

# These attributes carry state, but are intentionally not dataclass fields.
_EXTRA_ATTRIBUTES: dict[type[Any], tuple[str, ...]] = {
    _state.Secret: ('_tracked_revision', '_latest_revision'),
}


def _dataclass_types() -> dict[str, type[Any]]:
    """Return the dataclasses defined by :mod:`scenario.state`, keyed by name."""
    if not _DATACLASS_TYPES:
        for name in dir(_state):
            value = getattr(_state, name)
            if (
                inspect.isclass(value)
                and dataclasses.is_dataclass(value)
                and value.__module__ == _state.__name__
            ):
                _DATACLASS_TYPES[name] = value
    return _DATACLASS_TYPES


def _encode(value: Any, path: str) -> _JSON:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return {_TYPE_KEY: 'bytes', 'value': base64.b64encode(value).decode('ascii')}

    # Status classes are dataclasses, but their constructors do not accept all
    # of their fields as keyword arguments.
    if isinstance(value, _state._EntityStatus):
        return {_TYPE_KEY: 'status', 'name': value.name, 'message': value.message}

    if isinstance(value, pebble.Layer):
        return {_TYPE_KEY: 'layer', 'value': cast('dict[str, _JSON]', value.to_dict())}

    if isinstance(value, enum.Enum):
        class_name = type(value).__name__
        if class_name not in _ENUM_TYPES:
            raise TypeError(f'No JSON encoding for enum {class_name!r} at {path!r}.')
        return {_TYPE_KEY: 'enum', 'class': class_name, 'name': value.name}

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: _encode(getattr(value, field.name), f'{path}.{field.name}')
            for field in dataclasses.fields(value)
            if field.init
        }
        encoded: dict[str, _JSON] = {
            _TYPE_KEY: 'dataclass',
            'class': type(value).__name__,
            'fields': fields,
        }
        extra = {
            name: _encode(getattr(value, name), f'{path}.{name}')
            for name in _EXTRA_ATTRIBUTES.get(type(value), ())
        }
        if extra:
            encoded['extra'] = extra
        return encoded

    if isinstance(value, datetime.datetime):
        return {_TYPE_KEY: 'datetime', 'value': value.isoformat()}

    if isinstance(value, datetime.timedelta):
        return {_TYPE_KEY: 'timedelta', 'value': value.total_seconds()}

    # On POSIX, Path is also a PurePosixPath, so test for the concrete type first.
    if isinstance(value, pathlib.Path):
        return {_TYPE_KEY: 'path', 'value': str(value)}
    if isinstance(value, pathlib.PurePath):
        return {_TYPE_KEY: 'pure-path', 'value': str(value)}

    if isinstance(value, frozenset):
        items = cast('frozenset[Any]', value)
        return {
            _TYPE_KEY: 'frozenset',
            'items': [_encode(item, f'{path}[]') for item in items],
        }
    if isinstance(value, set):
        items = cast('set[Any]', value)
        return {_TYPE_KEY: 'set', 'items': [_encode(item, f'{path}[]') for item in items]}
    if isinstance(value, tuple):
        items = cast('tuple[Any, ...]', value)
        return {_TYPE_KEY: 'tuple', 'items': [_encode(item, f'{path}[]') for item in items]}
    if isinstance(value, list):
        items = cast('list[Any]', value)
        return [_encode(item, f'{path}[{index}]') for index, item in enumerate(items)]

    if isinstance(value, dict):
        mapping = cast('dict[Any, Any]', value)
        if mapping and all(type(key) is int for key in mapping):
            return {
                _TYPE_KEY: 'int-keyed-dict',
                'items': {
                    str(key): _encode(item, f'{path}[{key}]') for key, item in mapping.items()
                },
            }
        invalid_keys = [key for key in mapping if not isinstance(key, str)]
        if invalid_keys:
            key = invalid_keys[0]
            raise TypeError(
                f'No JSON encoding for dictionary key {key!r} of type '
                f'{type(key).__name__!r} at {path!r}; keys must be all strings or all integers.'
            )
        string_mapping = cast('dict[str, Any]', mapping)
        encoded = {key: _encode(item, f'{path}.{key}') for key, item in string_mapping.items()}
        if _TYPE_KEY in encoded:
            return {_TYPE_KEY: 'dict', 'items': encoded}
        return encoded

    raise TypeError(f'No JSON encoding for type {type(value).__name__!r} at {path!r}.')


def _as_str(value: _JSON, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f'Expected a string for {field}, got {value!r}.')
    return value


def _as_number(value: _JSON, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'Expected a number for {field}, got {value!r}.')
    return value


def _as_list(value: _JSON, field: str) -> list[_JSON]:
    if not isinstance(value, list):
        raise TypeError(f'Expected a list for {field}, got {value!r}.')
    return value


def _as_dict(value: _JSON, field: str) -> dict[str, _JSON]:
    if not isinstance(value, dict):
        raise TypeError(f'Expected an object for {field}, got {value!r}.')
    return value


def _decode(value: _JSON) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value

    value_type = value.get(_TYPE_KEY)
    if value_type is None:
        return {key: _decode(item) for key, item in value.items()}
    if value_type == 'dict':
        items = _as_dict(value['items'], 'dictionary items')
        return {key: _decode(item) for key, item in items.items()}
    if value_type == 'bytes':
        return base64.b64decode(_as_str(value['value'], 'bytes'))
    if value_type == 'status':
        name = _as_str(value['name'], 'status name')
        if name == 'unknown':
            return _state.UnknownStatus()
        status_type = _STATUS_TYPES.get(name)
        if status_type is None:
            raise TypeError(f'Unknown status name {name!r} in JSON payload.')
        return status_type(_as_str(value['message'], 'status message'))
    if value_type == 'layer':
        layer = cast('pebble.LayerDict', _as_dict(value['value'], 'layer'))
        return pebble.Layer(layer)
    if value_type == 'enum':
        class_name = _as_str(value['class'], 'enum class')
        enum_type = _ENUM_TYPES.get(class_name)
        if enum_type is None:
            raise TypeError(f'Unknown enum class {class_name!r} in JSON payload.')
        return enum_type[_as_str(value['name'], 'enum member')]
    if value_type == 'dataclass':
        class_name = _as_str(value['class'], 'dataclass name')
        dataclass_type = _dataclass_types().get(class_name)
        if dataclass_type is None:
            raise TypeError(f'Unknown dataclass {class_name!r} in JSON payload.')
        encoded_fields = _as_dict(value['fields'], 'dataclass fields')
        fields = {name: _decode(item) for name, item in encoded_fields.items()}
        instance = dataclass_type(**fields)
        extra = _as_dict(value.get('extra', {}), 'dataclass extra attributes')
        for name, item in extra.items():
            object.__setattr__(instance, name, _decode(item))
        return instance
    if value_type == 'datetime':
        return datetime.datetime.fromisoformat(_as_str(value['value'], 'datetime'))
    if value_type == 'timedelta':
        return datetime.timedelta(seconds=_as_number(value['value'], 'timedelta'))
    if value_type == 'path':
        return pathlib.Path(_as_str(value['value'], 'path'))
    if value_type == 'pure-path':
        return pathlib.PurePosixPath(_as_str(value['value'], 'pure path'))
    if value_type == 'frozenset':
        return frozenset(_decode(item) for item in _as_list(value['items'], 'frozenset'))
    if value_type == 'set':
        return {_decode(item) for item in _as_list(value['items'], 'set')}
    if value_type == 'tuple':
        return tuple(_decode(item) for item in _as_list(value['items'], 'tuple'))
    if value_type == 'int-keyed-dict':
        items = _as_dict(value['items'], 'integer-keyed dictionary')
        return {int(key): _decode(item) for key, item in items.items()}

    raise TypeError(f'Unknown type tag {value_type!r} in JSON payload.')


def _encode_state(state: _state.State) -> str:
    """Serialise a :class:`scenario.state.State` to JSON.

    Raises:
        TypeError: If the state contains a value that cannot be represented.
    """
    if not isinstance(state, _state.State):
        raise TypeError(f'Expected State, got {type(state).__name__}.')
    return json.dumps(_encode(state, 'state'))


def _decode_state(payload: str) -> _state.State:
    """Deserialise JSON produced by :func:`_encode_state`."""
    state = _decode(cast('_JSON', json.loads(payload)))
    if not isinstance(state, _state.State):
        raise TypeError(f'Decoded JSON is not a State: got {type(state).__name__}.')
    return state
