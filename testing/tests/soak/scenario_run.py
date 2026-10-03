# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""The multi-application scenario the settle soak runs, again and again.

Run as a script, it runs the scenario once and prints the digest of what it
produced, so the soak can compare runs in processes with different hash seeds.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from typing import Any

from scenario import _isolated_serde
from scenario._deployment import Juju

import ops
from ops import testing

_CHARMS = pathlib.Path(__file__).parent.parent / 'test_isolation' / 'charms'


class Client(ops.CharmBase):
    """A small database client defined in the test, deployed as a CharmSpec."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['database'].relation_joined, self._on_joined)
        framework.observe(self.on['database'].relation_changed, self._on_changed)
        framework.observe(self.on.secret_changed, self._on_secret_changed)

    def _on_joined(self, event: ops.RelationJoinedEvent):
        event.relation.data[self.unit]['client'] = f'spec-{self.unit.name}'

    def _on_changed(self, event: ops.RelationChangedEvent):
        assert event.app is not None
        secret_id = event.relation.data[event.app].get('secret-id')
        if secret_id:
            password = self.model.get_secret(id=secret_id).get_content()['password']
            self.unit.status = ops.ActiveStatus(f'password={password}')

    def _on_secret_changed(self, event: ops.SecretChangedEvent):
        password = event.secret.get_content(refresh=True)['password']
        self.unit.status = ops.ActiveStatus(f'password={password}')


CLIENT = testing.CharmSpec(
    Client,
    meta={'name': 'client', 'requires': {'database': {'interface': 'dbtest'}}},
)


def run(*, isolated: bool = False, built: bool = False) -> list[str]:
    """Run the scenario, returning every trace entry and every final State, encoded.

    With ``isolated``, ``webapp`` runs in a worker with this interpreter. With
    ``built``, it runs in a worker in an environment built for it by
    ``deploy(isolated=True)``.
    """
    with Juju(model_name='soak', uuid='5a5a5a5a-0000-4000-8000-000000000000') as juju:
        kwargs: dict[str, Any] = {'python_executable': sys.executable} if isolated else {}
        db = juju._deploy(_CHARMS / 'dbserver', num_units=3)
        if built:
            web = juju.deploy(_CHARMS / 'webapp', num_units=3, isolated=True)
        else:
            web = juju._deploy(_CHARMS / 'webapp', num_units=3, **kwargs)
        client = juju.deploy(CLIENT, num_units=2)
        juju.integrate(web, db)
        juju.integrate(client, db)
        out: list[str] = []
        out.extend(_encode_trace(juju.settle()))
        juju.config(db, {'password': 'second'})
        out.extend(_encode_trace(juju.settle()))
        juju.add_unit(web)
        juju.remove_unit(db, num_units=1)
        out.extend(_encode_trace(juju.settle()))
        juju.remove_relation(client, db)
        out.extend(_encode_trace(juju.settle()))
        for app in (db, web, client):
            for unit in app.units:
                out.append(f'final {unit.name} {unit.state._to_json()}')
        return out


def _encode_trace(trace: list[testing.Dispatch]) -> list[str]:
    return [
        f'{d.unit.name} {_isolated_serde.encode_event(d.event)} {d.state._to_json()}'
        for d in trace
    ]


def digest(lines: list[str]) -> str:
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


if __name__ == '__main__':
    print(digest(run(isolated='--isolated' in sys.argv, built='--built' in sys.argv)))
