# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.
"""## Overview.

This document explains how to integrate with the Tempo charm for the purpose of
pushing traces to a tracing endpoint provided by Tempo.

## Requirer Library Usage

Charms seeking to push traces to Tempo, must do so using the
`TracingEndpointRequirer` object from this charm library. For the simplest use
cases, using the `TracingEndpointRequirer` object only requires instantiating
it, typically in the constructor of your charm. The `TracingEndpointRequirer`
constructor requires the name of the relation over which a tracing endpoint
is exposed by the Tempo charm, and a list of protocols it intends to send
traces with. This relation must use the `tracing` interface.

Units of requirer charms obtain the tempo endpoint to which they will push
their traces by calling `TracingEndpointRequirer.get_endpoint(protocol: str)`,
where `protocol` is, for example:
- `otlp_grpc`
- `otlp_http`
- `zipkin`
- `tempo`

If the `protocol` is not in the list of protocols that the charm requested at
endpoint set-up time, the library will raise an error.
"""

import dataclasses
import enum
import json
import logging
import typing
from typing import Any, List, Literal, MutableMapping, Optional, Sequence

from ops.charm import (
    CharmBase,
    CharmEvents,
    RelationBrokenEvent,
    RelationEvent,
)
from ops.framework import EventSource, Object
from ops.model import ModelError, Relation

logger = logging.getLogger(__name__)

DEFAULT_RELATION_NAME = "tracing"

# Supported list rationale https://github.com/canonical/tempo-coordinator-k8s-operator/issues/8
ReceiverProtocol = Literal[
    "zipkin",
    "otlp_grpc",
    "otlp_http",
    "jaeger_grpc",
    "jaeger_thrift_http",
]


class TransportProtocolType(enum.Enum):
    """Receiver Type."""

    grpc = "grpc"
    http = "http"


class TracingError(Exception):
    """Base class for custom errors raised by this library."""


class ProtocolNotRequestedError(TracingError):
    """Raised if the user attempts to obtain an endpoint for a protocol it did not request."""


class DataValidationError(TracingError):
    """Raised when data validation fails on IPU relation data."""


class AmbiguousRelationUsageError(TracingError):
    """Raised when one wrongly assumes that there can only be one relation on an endpoint."""


def _coerce(value: Any, type_: Any) -> Any:
    """Coerce a JSON-decoded value into ``type_``."""
    origin = typing.get_origin(type_)
    args = typing.get_args(type_)
    if origin is list:
        return [_coerce(v, args[0]) for v in value]
    if origin is set:
        return {_coerce(v, args[0]) for v in value}
    if dataclasses.is_dataclass(type_):
        return _load_from_dict(type_, value)
    if isinstance(type_, type) and issubclass(type_, enum.Enum):
        return type_(value)
    return value


def _load_from_dict(cls: Any, data: Any) -> Any:
    """Build a dataclass instance from a plain dict (used for nested fields)."""
    hints = typing.get_type_hints(cls)
    kwargs: dict = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name])
    return cls(**kwargs)


def _databag_load(cls: Any, databag: MutableMapping) -> Any:
    """Load this dataclass from a Juju databag.

    Unknown keys are ignored. On any decoding or coercion failure,
    raises :class:`DataValidationError`.
    """
    try:
        hints = typing.get_type_hints(cls)
        kwargs: dict = {}
        for f in dataclasses.fields(cls):
            if f.name in databag:
                kwargs[f.name] = _coerce(json.loads(databag[f.name]), hints[f.name])
        return cls(**kwargs)
    except (TypeError, KeyError, ValueError, json.JSONDecodeError) as e:
        msg = f"failed to load databag into {cls.__name__}: {dict(databag)!r}"
        logger.debug(msg, exc_info=True)
        raise DataValidationError(msg) from e


def _to_serialisable(value: Any) -> Any:
    """Convert a dataclass / enum / collection value into JSON-friendly data.

    Sets are emitted as sorted lists for deterministic wire output.
    """
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_serialisable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, set):
        return sorted(_to_serialisable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    return value


def _databag_dump(
    self: Any,
    databag: Optional[MutableMapping] = None,
    clear: bool = False,
) -> MutableMapping:
    """Write this dataclass to a Juju databag (or a fresh dict)."""
    if clear and databag is not None:
        databag.clear()
    if databag is None:
        databag = {}
    for f in dataclasses.fields(type(self)):
        databag[f.name] = json.dumps(_to_serialisable(getattr(self, f.name)))
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
    def load(cls, databag: MutableMapping) -> "TracingProviderAppData":
        """Load the model from a databag."""
        return _databag_load(cls, databag)

    def dump(
        self,
        databag: Optional[MutableMapping] = None,
        clear: bool = False,
    ) -> MutableMapping:
        """Dump the model into a databag."""
        return _databag_dump(self, databag, clear)


@dataclasses.dataclass(frozen=True)
class TracingRequirerAppData:
    """Application databag model for the tracing requirer."""

    receivers: List[str]

    @classmethod
    def load(cls, databag: MutableMapping) -> "TracingRequirerAppData":
        """Load the model from a databag."""
        return _databag_load(cls, databag)

    def dump(
        self,
        databag: Optional[MutableMapping] = None,
        clear: bool = False,
    ) -> MutableMapping:
        """Dump the model into a databag."""
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
                `TracingEndpointRequirer` object.
            relation_name: an optional string name of the relation between `charm`
                and the Tempo charmed service. The default is "tracing".
            protocols: optional list of protocols that the charm intends to send
                traces with.
        """
        super().__init__(charm, f"internal: {relation_name}")

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
                "You need to pass a nonempty sequence of protocols to `request_protocols`."
            )

        try:
            if self._charm.unit.is_leader():
                for relation in relations:
                    TracingRequirerAppData(
                        receivers=list(protocols),
                    ).dump(relation.data[self._charm.app])

        except ModelError as e:
            # args are bytes
            msg = e.args[0]
            if isinstance(msg, bytes):
                if msg.startswith(
                    b"ERROR cannot read relation application settings: permission denied"
                ):
                    logger.error(
                        f"encountered error {e} while attempting to request_protocols."
                        f"The relation must be gone."
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
                f"This {objname} wraps a {self._relation_name} endpoint that has "
                "limit != 1. We can't determine what relation, of the possibly many, you are "
                f"talking about. Please pass a relation instance while calling {objname}, "
                "or set limit=1 in the charm metadata."
            )
        relations = self.relations
        return relations[0] if relations else None

    def is_ready(self, relation: Optional[Relation] = None):
        """Is this endpoint ready?"""
        relation = relation or self._relation
        if not relation:
            logger.debug(f"no relation on {self._relation_name !r}: tracing not ready")
            return False
        if relation.data is None:
            logger.error(f"relation data is None for {relation}")
            return False
        if not relation.app:
            logger.error(f"{relation} event received but there is no relation.app")
            return False
        try:
            databag = dict(relation.data[relation.app])
            TracingProviderAppData.load(databag)
        except DataValidationError:
            logger.info(f"failed validating relation data for {relation}")
            return False
        return True

    def _on_tracing_relation_changed(self, event):
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
        return TracingProviderAppData.load(relation.data[relation.app])  # type: ignore

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
            logger.warning(f"no receiver found with protocol={protocol!r}.")
            return None
        if len(receivers) > 1:
            logger.warning(
                f"too many receivers with protocol={protocol!r}; using first one. "
                f"Found: {receivers}"
            )

        return receivers[0].url

    def get_endpoint(
        self, protocol: ReceiverProtocol, relation: Optional[Relation] = None
    ) -> Optional[str]:
        """Receiver endpoint for the given protocol.

        Raises:
            ProtocolNotRequestedError: If the charm unit is the leader unit and
                attempts to obtain an endpoint for a protocol it did not request.
        """
        endpoint = self._get_endpoint(relation or self._relation, protocol=protocol)
        if not endpoint:
            requested_protocols: set = set()
            relations = [relation] if relation else self.relations
            for relation in relations:
                try:
                    databag = TracingRequirerAppData.load(relation.data[self._charm.app])
                except DataValidationError:
                    continue

                requested_protocols.update(databag.receivers)

            if protocol not in requested_protocols:
                raise ProtocolNotRequestedError(protocol, relation)

            return None
        return endpoint
