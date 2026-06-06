# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

"""Library for the certificate_transfer relation.

This library contains the Requires class for handling the
certificate-transfer interface.
"""

import dataclasses
import enum
import json
import logging
import typing
from typing import Any, MutableMapping, Optional, Set

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
    """Build a dataclass instance from a plain dict (used for nesting)."""
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
class ProviderApplicationData:
    """App databag model for the certificate-transfer provider side."""

    certificates: Set[str] = dataclasses.field(default_factory=set)

    @classmethod
    def load(cls, databag: MutableMapping) -> "ProviderApplicationData":
        """Load the model from a databag."""
        return _databag_load(cls, databag)

    def dump(
        self,
        databag: Optional[MutableMapping] = None,
        clear: bool = False,
    ) -> MutableMapping:
        """Dump the model into a databag."""
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

    def snapshot(self) -> dict:
        """Return snapshot."""
        return {
            "certificates": self.certificates,
            "relation_id": self.relation_id,
        }

    def restore(self, snapshot: dict):
        """Restores snapshot."""
        self.certificates = snapshot["certificates"]
        self.relation_id = snapshot["relation_id"]


class CertificatesRemovedEvent(EventBase):
    """Charm Event triggered when the set of provided certificates is removed."""

    def __init__(self, handle: Handle, relation_id: int):
        super().__init__(handle)
        self.relation_id = relation_id

    def snapshot(self) -> dict:
        """Return snapshot."""
        return {"relation_id": self.relation_id}

    def restore(self, snapshot: dict):
        """Restores snapshot."""
        self.relation_id = snapshot["relation_id"]


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
        super().__init__(charm, f"internal: {relationship_name}_v1")
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
        """
        relations = self._get_relevant_relations(relation_id)
        result: Set[str] = set()
        for relation in relations:
            result = result.union(self._get_relation_data(relation))
        return result

    def is_ready(self, relation: Relation) -> bool:
        """Check if the relation is ready by checking that it has valid relation data."""
        databag = relation.data[relation.app]  # type: ignore[index]
        try:
            ProviderApplicationData.load(databag)
            return True
        except DataValidationError:
            return False

    def _get_relation_data(self, relation: Relation) -> Set[str]:
        """Get the given relation data."""
        databag = relation.data[relation.app]  # type: ignore[index]
        try:
            return ProviderApplicationData.load(databag).certificates
        except DataValidationError as e:
            logger.error(
                "Error parsing relation databag: %s. "
                "Make sure not to interact with the databags except using the public methods "
                "in the provider library and use version V1.",
                e.args,
            )
            return set()

    def _get_relevant_relations(self, relation_id: Optional[int] = None) -> list:
        """Get the relevant relation if relation_id is given, all relations otherwise."""
        if relation_id is not None:
            if relation := self.model.get_relation(
                relation_name=self.relationship_name, relation_id=relation_id
            ):
                return [relation]
        return list(self.model.relations[self.relationship_name])
