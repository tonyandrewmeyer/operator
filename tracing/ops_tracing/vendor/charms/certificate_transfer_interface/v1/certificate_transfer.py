# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

"""Library for the certificate_transfer relation.

This library contains the Requires class for handling the certificate-transfer
interface. This is a depydantic'd, fork-local copy used by ``ops_tracing`` only:
the surface is intentionally narrower than the upstream charm lib and the
``pydantic`` runtime dependency has been removed.
"""

import dataclasses
import enum
import json
import logging
import typing
from typing import Any, List, MutableMapping, Optional, Set

from ops import (
    CharmEvents,
    EventBase,
    EventSource,
    Handle,
    Relation,
    RelationBrokenEvent,
    RelationChangedEvent,
)
from ops.charm import CharmBase
from ops.framework import Object

logger = logging.getLogger(__name__)


class TLSCertificatesError(Exception):
    """Base class for custom errors raised by this library."""


class DataValidationError(TLSCertificatesError):
    """Raised when data validation fails."""


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
class ProviderApplicationData:
    """App databag model."""

    certificates: Set[str] = dataclasses.field(default_factory=set)

    @classmethod
    def load(cls, databag: MutableMapping[str, str]) -> 'ProviderApplicationData':
        """Load this model from a Juju databag."""
        return _databag_load(cls, databag)

    def dump(
        self,
        databag: Optional[MutableMapping[str, str]] = None,
        clear: bool = False,
    ) -> MutableMapping[str, str]:
        """Write the contents of this model to a Juju databag."""
        return _databag_dump(self, databag, clear)


class CertificatesAvailableEvent(EventBase):
    """Charm Event triggered when the set of provided certificates is updated."""

    def __init__(
        self,
        handle: Handle,
        certificates: Set[str],
        relation_id: int,
    ):
        super().__init__(handle)
        self.certificates = certificates
        self.relation_id = relation_id

    def snapshot(self) -> dict[str, Any]:
        """Return snapshot."""
        return {
            'certificates': self.certificates,
            'relation_id': self.relation_id,
        }

    def restore(self, snapshot: dict[str, Any]):
        """Restore snapshot."""
        self.certificates = snapshot['certificates']
        self.relation_id = snapshot['relation_id']


class CertificatesRemovedEvent(EventBase):
    """Charm Event triggered when the set of provided certificates is removed."""

    def __init__(self, handle: Handle, relation_id: int):
        super().__init__(handle)
        self.relation_id = relation_id

    def snapshot(self) -> dict[str, Any]:
        """Return snapshot."""
        return {'relation_id': self.relation_id}

    def restore(self, snapshot: dict[str, Any]):
        """Restore snapshot."""
        self.relation_id = snapshot['relation_id']


class CertificateTransferRequirerCharmEvents(CharmEvents):
    """List of events that the Certificate Transfer requirer charm can leverage."""

    certificate_set_updated = EventSource(CertificatesAvailableEvent)
    certificates_removed = EventSource(CertificatesRemovedEvent)


class CertificateTransferRequires(Object):
    """Certificate transfer requirer class to be instantiated by charms expecting certificates."""

    on = CertificateTransferRequirerCharmEvents()  # type: ignore

    def __init__(
        self,
        charm: CharmBase,
        relationship_name: str,
    ):
        """Observe events related to the relation.

        Args:
            charm: Charm object
            relationship_name: Juju relation name
        """
        super().__init__(charm, f'internal: {relationship_name}_v1')
        self.relationship_name = relationship_name
        self.charm = charm
        self.framework.observe(
            charm.on[relationship_name].relation_changed, self._on_relation_changed
        )
        self.framework.observe(
            charm.on[relationship_name].relation_broken, self._on_relation_broken
        )

    def _on_relation_changed(self, event: RelationChangedEvent) -> None:
        """Emit certificate set updated event."""
        remote_unit_relation_data = self.get_all_certificates(event.relation.id)
        self.on.certificate_set_updated.emit(
            certificates=remote_unit_relation_data,
            relation_id=event.relation.id,
        )

    def _on_relation_broken(self, event: RelationBrokenEvent) -> None:
        """Handle relation broken event."""
        self.on.certificates_removed.emit(relation_id=event.relation.id)

    def get_all_certificates(self, relation_id: Optional[int] = None) -> Set[str]:
        """Get transferred certificates.

        If no relation id is given, certificates from all relations will be
        provided in a concatenated set.

        Args:
            relation_id: The id of the relation to get the certificates from.
        """
        relations = self._get_relevant_relations(relation_id)
        result: Set[str] = set()
        for relation in relations:
            data = self._get_relation_data(relation)
            result = result.union(data)
        return result

    def is_ready(self, relation: Relation) -> bool:
        """Check if the relation is ready by checking that it has valid relation data."""
        if relation.app is None:
            return False
        databag = relation.data[relation.app]
        try:
            ProviderApplicationData.load(databag)
            return True
        except DataValidationError:
            return False

    def _get_relation_data(self, relation: Relation) -> Set[str]:
        """Get the given relation data."""
        if relation.app is None:
            return set()
        databag = relation.data[relation.app]
        try:
            return ProviderApplicationData.load(databag).certificates
        except DataValidationError as e:
            logger.error(
                'Error parsing relation databag: %s. '
                'Make sure not to interact with the databags '
                'except using the public methods in the provider library '
                'and use version V1.',
                e.args,
            )
            return set()

    def _get_relevant_relations(self, relation_id: Optional[int] = None) -> List[Relation]:
        """Get the relevant relation if relation_id is given, all relations otherwise."""
        if relation_id is not None:
            if relation := self.model.get_relation(
                relation_name=self.relationship_name, relation_id=relation_id
            ):
                return [relation]
        return list(self.model.relations[self.relationship_name])
