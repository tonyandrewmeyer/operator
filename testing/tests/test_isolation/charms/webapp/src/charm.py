#!/usr/bin/env python3
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Web application charm: reports the database it was given in its status."""

import ops


class WebAppCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.database_relation_joined, self._on_joined)
        framework.observe(self.on.database_relation_changed, self._on_changed)
        framework.observe(self.on.database_relation_broken, self._on_broken)
        framework.observe(self.on.secret_changed, self._on_secret_changed)

    def _on_joined(self, event: ops.RelationJoinedEvent):
        event.relation.data[self.unit]['client'] = self.unit.name

    def _on_changed(self, event: ops.RelationChangedEvent):
        self._report(event.relation, refresh=False)

    def _on_secret_changed(self, event: ops.SecretChangedEvent):
        event.secret.get_content(refresh=True)
        relation = self.model.get_relation('database')
        if relation is not None:
            self._report(relation, refresh=False)

    def _on_broken(self, _: ops.RelationBrokenEvent):
        self.unit.status = ops.BlockedStatus('no database')

    def _report(self, relation: ops.Relation, *, refresh: bool):
        assert relation.app is not None
        data = relation.data[relation.app]
        secret_id = data.get('secret-id')
        if not secret_id:
            self.unit.status = ops.WaitingStatus('waiting for database')
            return
        password = self.model.get_secret(id=secret_id).get_content(refresh=refresh)['password']
        hosts = sorted(relation.data[unit].get('host', '?') for unit in relation.units)
        self.unit.status = ops.ActiveStatus(
            f'db={data.get("endpoint")} password={password} hosts={",".join(hosts)}'
        )


if __name__ == '__main__':
    ops.main(WebAppCharm)
