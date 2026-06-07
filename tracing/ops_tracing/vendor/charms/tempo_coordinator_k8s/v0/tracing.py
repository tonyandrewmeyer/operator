# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.
"""## Overview.

This document explains how to integrate with the Tempo charm for the purpose of pushing traces to a
tracing endpoint provided by Tempo.

This is a depydantic'd, fork-local copy used by ``ops_tracing`` only: the surface is intentionally
narrower than the upstream charm lib (no ``TracingEndpointProvider``, no charm-tracing helper,
no upstream pydantic data model) and the ``pydantic`` runtime dependency has been removed.

## Requirer Library Usage

Charms seeking to push traces to Tempo, must do so using the `TracingEndpointRequirer`
object from this charm library. For the simplest use cases, using the `TracingEndpointRequirer`
object only requires instantiating it, typically in the constructor of your charm. The
`TracingEndpointRequirer` constructor requires the name of the relation over which a tracing
endpoint is exposed by the Tempo charm, and a list of protocols it intends to send traces with.
This relation must use the `tracing` interface.
"""

import dataclasses
import enum
import json
import logging
import typing
from typing import Any, List, Literal, MutableMapping, Optional, Sequence

from ops.charm import CharmBase, CharmEvents, RelationBrokenEvent, RelationEvent
from ops.framework import EventSource, Object
from ops.model import ModelError, Relation

logger = logging.getLogger(__name__)

DEFAULT_RELATION_NAME = 'tracing'
RELATION_INTERFACE_NAME = 'tracing'

# Supported list rationale https://github.com/canonical/tempo-coordinator-k8s-operator/issues/8
ReceiverProtocol = Literal[
    'zipkin',
    'otlp_grpc',
    'otlp_http',
    'jaeger_grpc',
    'jaeger_thrift_http',
]


class TransportProtocolType(enum.Enum):
    """Receiver Type."""

    grpc = 'grpc'
    http = 'http'


class TracingError(Exception):
    """Base class for custom errors raised by this library."""


class ProtocolNotRequestedError(TracingError):
    """Raised if the user attempts to obtain an endpoint for a protocol it did not request."""


class DataValidationError(TracingError):
    """Raised when data validation fails on IPU relation data."""


class AmbiguousRelationUsageError(TracingError):
    """Raised when one wrongly assumes that there can only be one relation on an endpoint."""


def _coerce(value: Any, tp: Any) -> Any:
    """Coerce a JSON-decoded value into the target dataclass-field type."""
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise TypeError(f'expected dict for {tp.__name__}, got {type(value).__name__}')
        hints = typing.get_type_hints(tp)
        kwargs = {f.name: _coerce(value[f.name], hints[f.name]) for f in dataclasses.fields(tp) if f.name in value}
        return tp(**kwargs)  # type: ignore[operator]
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(value)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is list:
        (elem_tp,) = args
        return [_coerce(item, elem_tp) for item in value]
    if origin is set:
        (elem_tp,) = args
        return {_coerce(item, elem_tp) for item in value}
    return value


def _to_json(value: Any) -> Any:
    """Recursively convert a dataclass / enum / set value into JSON-friendly primitives."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_json(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, set):
        return sorted(_to_json(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    return value


def _databag_load(cls: type, databag: MutableMapping[str, str]) -> Any:
    """Load a frozen dataclass instance from a Juju databag (one JSON value per field key)."""
    try:
        hints = typing.get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):  # type: ignore[arg-type]
            if f.name in databag:
                kwargs[f.name] = _coerce(json.loads(databag[f.name]), hints[f.name])
        return cls(**kwargs)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        msg = f'failed to validate databag: {databag}'
        logger.debug(msg, exc_info=True)
        raise DataValidationError(msg) from e


def _databag_dump(
    self: Any,
    databag: Optional[MutableMapping[str, str]] = None,
    clear: bool = False,
) -> MutableMapping[str, str]:
    """Serialise a frozen dataclass instance into a Juju databag dict."""
    if clear and databag is not None:
        databag.clear()
    if databag is None:
        databag = {}
    for f in dataclasses.fields(self):
        databag[f.name] = json.dumps(_to_json(getattr(self, f.name)))
    return databag


@dataclasses.dataclass(frozen=True)
class ProtocolType:
    """Protocol Type."""

    name: str
    type: TransportProtocolType


@dataclasses.dataclass(frozen=True)
class Receiver:
    """Specification of an active receiver."""

    url: str
    protocol: ProtocolType


@dataclasses.dataclass(frozen=True)
class TracingProviderAppData:
    """Application databag model for the tracing provider."""

    receivers: List[Receiver]

    @classmethod
    def load(cls, databag: MutableMapping[str, str]) -> 'TracingProviderAppData':
        """Load this model from a Juju databag."""
        return _databag_load(cls, databag)

    def dump(
        self,
        databag: Optional[MutableMapping[str, str]] = None,
        clear: bool = False,
    ) -> MutableMapping[str, str]:
        """Write the contents of this model to a Juju databag."""
        return _databag_dump(self, databag, clear)


@dataclasses.dataclass(frozen=True)
class TracingRequirerAppData:
    """Application databag model for the tracing requirer."""

    receivers: List[str]

    @classmethod
    def load(cls, databag: MutableMapping[str, str]) -> 'TracingRequirerAppData':
        """Load this model from a Juju databag."""
        return _databag_load(cls, databag)

    def dump(
        self,
        databag: Optional[MutableMapping[str, str]] = None,
        clear: bool = False,
    ) -> MutableMapping[str, str]:
        """Write the contents of this model to a Juju databag."""
        return _databag_dump(self, databag, clear)


class EndpointRemovedEvent(RelationBrokenEvent):
    """Event representing a change in one of the receiver endpoints."""


class EndpointChangedEvent(RelationEvent):
    """Event representing a change in one of the receiver endpoints."""


class TracingEndpointRequirerEvents(CharmEvents):
    """TracingEndpointRequirer events."""

    endpoint_changed = EventSource(EndpointChangedEvent)
    endpoint_removed = EventSource(EndpointRemovedEvent)


class TracingEndpointRequirer(Object):
    """A tracing endpoint for Tempo."""

    on = TracingEndpointRequirerEvents()  # type: ignore

    def __init__(
        self,
        charm: CharmBase,
        relation_name: str = DEFAULT_RELATION_NAME,
        protocols: Optional[List[ReceiverProtocol]] = None,
    ):
        """Construct a tracing requirer for a Tempo charm.

        Args:
            charm: a `CharmBase` object that manages this
                `TracingEndpointRequirer` object. Typically, this is `self` in the instantiating
                class.
            relation_name: an optional string name of the relation between `charm`
                and the Tempo charmed service. The default is "tracing".
            protocols: optional list of protocols that the charm intends to send traces with.
        """
        super().__init__(charm, f'internal: {relation_name}')

        self._is_single_endpoint = charm.meta.relations[relation_name].limit == 1

        self._charm = charm
        self._relation_name = relation_name

        events = self._charm.on[self._relation_name]
        self.framework.observe(events.relation_changed, self._on_tracing_relation_changed)
        self.framework.observe(events.relation_broken, self._on_tracing_relation_broken)

        if protocols:
            self.request_protocols(protocols)

    def request_protocols(
        self, protocols: Sequence[ReceiverProtocol], relation: Optional[Relation] = None
    ):
        """Publish the list of protocols which the provider should activate."""
        relations = [relation] if relation else self.relations

        if not protocols:
            raise ValueError(
                'You need to pass a nonempty sequence of protocols to `request_protocols`.'
            )

        try:
            if self._charm.unit.is_leader():
                for rel in relations:
                    TracingRequirerAppData(receivers=list(protocols)).dump(
                        rel.data[self._charm.app], clear=True
                    )

        except ModelError as e:
            msg = e.args[0]
            if isinstance(msg, bytes):
                if msg.startswith(
                    b'ERROR cannot read relation application settings: permission denied'
                ):
                    logger.error(
                        'encountered error %s while attempting to request_protocols. '
                        'The relation must be gone.',
                        e,
                    )
                    return
            raise

    @property
    def relations(self) -> List[Relation]:
        """The tracing relations associated with this endpoint."""
        return self._charm.model.relations[self._relation_name]

    @property
    def _relation(self) -> Optional[Relation]:
        """If this wraps a single endpoint, the relation bound to it, if any."""
        if not self._is_single_endpoint:
            objname = type(self).__name__
            raise AmbiguousRelationUsageError(
                f'This {objname} wraps a {self._relation_name} endpoint that has '
                "limit != 1. We can't determine what relation, of the possibly many, you are "
                f'talking about. Please pass a relation instance while calling {objname}, '
                'or set limit=1 in the charm metadata.'
            )
        relations = self.relations
        return relations[0] if relations else None

    def is_ready(self, relation: Optional[Relation] = None) -> bool:
        """Is this endpoint ready?"""
        relation = relation or self._relation
        if not relation:
            logger.debug('no relation on %r: tracing not ready', self._relation_name)
            return False
        if relation.data is None:
            logger.error('relation data is None for %s', relation)
            return False
        if not relation.app:
            logger.error('%s event received but there is no relation.app', relation)
            return False
        try:
            TracingProviderAppData.load(relation.data[relation.app])
        except DataValidationError:
            logger.info('failed validating relation data for %s', relation)
            return False
        return True

    def _on_tracing_relation_changed(self, event: RelationEvent):
        """Notify the providers that there is new endpoint information available."""
        relation = event.relation
        if not self.is_ready(relation):
            self.on.endpoint_removed.emit(relation)  # type: ignore
            return
        self.on.endpoint_changed.emit(relation)  # type: ignore

    def _on_tracing_relation_broken(self, event: RelationBrokenEvent):
        """Notify the providers that the endpoint is broken."""
        relation = event.relation
        self.on.endpoint_removed.emit(relation)  # type: ignore

    def get_all_endpoints(
        self, relation: Optional[Relation] = None
    ) -> Optional[TracingProviderAppData]:
        """Unmarshalled relation data."""
        relation = relation or self._relation
        if not self.is_ready(relation):
            return None
        assert relation is not None and relation.app is not None
        return TracingProviderAppData.load(relation.data[relation.app])

    def _get_endpoint(
        self, relation: Optional[Relation], protocol: ReceiverProtocol
    ) -> Optional[str]:
        app_data = self.get_all_endpoints(relation)
        if not app_data:
            return None
        receivers: List[Receiver] = [
            r for r in app_data.receivers if r.protocol.name == protocol
        ]
        if not receivers:
            logger.warning('no receiver found with protocol=%r.', protocol)
            return None
        if len(receivers) > 1:
            logger.warning(
                'too many receivers with protocol=%r; using first one. Found: %s',
                protocol,
                receivers,
            )

        return receivers[0].url

    def get_endpoint(
        self, protocol: ReceiverProtocol, relation: Optional[Relation] = None
    ) -> Optional[str]:
        """Receiver endpoint for the given protocol.

        Raises:
            ProtocolNotRequestedError: If the charm unit is the leader unit and attempts to
                obtain an endpoint for a protocol it did not request.
        """
        endpoint = self._get_endpoint(relation or self._relation, protocol=protocol)
        if not endpoint:
            requested_protocols: set[str] = set()
            relations = [relation] if relation else self.relations
            for rel in relations:
                try:
                    databag = TracingRequirerAppData.load(rel.data[self._charm.app])
                except DataValidationError:
                    continue

                requested_protocols.update(databag.receivers)

            if protocol not in requested_protocols:
                raise ProtocolNotRequestedError(protocol, relation)

            return None
        return endpoint
