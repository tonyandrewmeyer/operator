#!/usr/bin/env python3
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Database charm: shares an endpoint and a password secret with each client."""

import ops

_LABEL = 'credentials'


class DbServerCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.database_relation_joined, self._on_joined)
        framework.observe(self.on.database_relation_changed, self._on_changed)
        framework.observe(self.on.config_changed, self._on_config_changed)
        framework.observe(self.on.secret_remove, self._on_secret_remove)

    def _on_joined(self, event: ops.RelationJoinedEvent):
        event.relation.data[self.unit]['host'] = self.unit.name
        if not self.unit.is_leader():
            return
        secret = self._secret()
        secret.grant(event.relation)
        event.relation.data[self.app]['endpoint'] = f'{self.app.name}.example:5432'
        event.relation.data[self.app]['secret-id'] = secret.id or ''

    def _on_changed(self, event: ops.RelationChangedEvent):
        clients = sorted(
            event.relation.data[unit].get('client', '?') for unit in event.relation.units
        )
        self.unit.status = ops.ActiveStatus(f'clients={",".join(clients)}')

    def _on_config_changed(self, _: ops.ConfigChangedEvent):
        if not self.unit.is_leader():
            return
        try:
            secret = self.model.get_secret(label=_LABEL)
        except ops.SecretNotFoundError:
            return
        password = str(self.config['password'])
        if secret.peek_content().get('password') != password:
            secret.set_content({'password': password})

    def _on_secret_remove(self, event: ops.SecretRemoveEvent):
        event.remove_revision()
        self.unit.status = ops.ActiveStatus(f'removed revision {event.revision}')

    def _secret(self) -> ops.Secret:
        try:
            return self.model.get_secret(label=_LABEL)
        except ops.SecretNotFoundError:
            return self.app.add_secret({'password': str(self.config['password'])}, label=_LABEL)


if __name__ == '__main__':
    ops.main(DbServerCharm)
