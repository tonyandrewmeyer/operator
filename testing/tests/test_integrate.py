# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for relating applications under a Juju: integrate, remove_relation, and settling."""

from __future__ import annotations

import contextlib
import pathlib
import sys
from typing import Any, cast

import pytest
from scenario._deployment import Juju

import ops
from ops import testing

_CHARMS = pathlib.Path(__file__).parent / 'test_isolation' / 'charms'
DBSERVER = _CHARMS / 'dbserver'
WEBAPP = _CHARMS / 'webapp'


class ProviderCharm(ops.CharmBase):
    """Publishes its unit name and, as leader, a greeting; records what it sees."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['db'].relation_joined, self._on_joined)
        framework.observe(self.on['db'].relation_changed, self._on_changed)

    def _on_joined(self, event: ops.RelationJoinedEvent):
        event.relation.data[self.unit]['name'] = self.unit.name
        if self.unit.is_leader():
            event.relation.data[self.app]['greeting'] = 'hello'

    def _on_changed(self, event: ops.RelationChangedEvent):
        seen = sorted(event.relation.data[u].get('name', '?') for u in event.relation.units)
        self.unit.status = ops.ActiveStatus(f'seen={",".join(seen)}')


class RequirerCharm(ProviderCharm):
    """The same behaviour, on the requires side."""


PROVIDER_META: dict[str, Any] = {
    'name': 'provider',
    'provides': {'db': {'interface': 'test-db'}},
}
REQUIRER_META: dict[str, Any] = {
    'name': 'requirer',
    'requires': {'db': {'interface': 'test-db'}},
}


def provider(meta: dict[str, Any] = PROVIDER_META) -> testing.CharmSpec[Any]:
    return testing.CharmSpec(ProviderCharm, meta=meta)


def requirer(meta: dict[str, Any] = REQUIRER_META) -> testing.CharmSpec[Any]:
    return testing.CharmSpec(RequirerCharm, meta=meta)


def relation(unit: testing.Unit, endpoint: str = 'db') -> testing.Relation:
    relations = [r for r in unit.state.relations if r.endpoint == endpoint]
    assert len(relations) == 1, relations
    assert isinstance(relations[0], testing.Relation)
    return relations[0]


def relation_events(trace: list[testing.Dispatch], unit: testing.Unit) -> list[tuple[str, Any]]:
    return [
        (d.event.name, d.event.relation_remote_unit_id)
        for d in trace
        if d.unit is unit and d.event.relation is not None
    ]


@pytest.fixture
def juju():
    with testing.Juju(model_name='test-model', uuid='00000000-0000-4000-8000-000000000000') as j:
        yield j


# integrate


def test_integrate_adds_the_relation_to_every_unit_on_both_sides(juju: testing.Juju):
    prov = juju.deploy(provider(), num_units=2)
    req = juju.deploy(requirer(), num_units=3)
    juju.integrate(prov, req)
    ids = {relation(u).id for u in (*prov.units, *req.units)}
    assert len(ids) == 1
    for unit in prov.units:
        assert relation(unit).remote_app_name == 'requirer'
        assert relation(unit).interface == 'test-db'
    for unit in req.units:
        assert relation(unit).remote_app_name == 'provider'


def test_integrate_fires_created_then_joined_and_changed_for_each_remote_unit(
    juju: testing.Juju,
):
    prov = juju.deploy(provider(), num_units=2)
    req = juju.deploy(requirer(), num_units=2)
    juju.settle()
    juju.integrate(prov, req)
    trace = juju.settle()
    for unit, remote in ((prov.units[0], req), (req.units[1], prov)):
        events = relation_events(trace, unit)
        assert events[0] == ('db_relation_created', None)
        joined = [e for e in events if e[0] == 'db_relation_joined']
        assert joined == [('db_relation_joined', u.id) for u in remote.units]
        # Each joined is followed by a changed for the same remote unit.
        for remote_unit in remote.units:
            at = events.index(('db_relation_joined', remote_unit.id))
            assert events[at + 1] == ('db_relation_changed', remote_unit.id)


def test_integrated_units_see_each_other_and_their_data(juju: testing.Juju):
    prov = juju.deploy(provider(), num_units=2)
    req = juju.deploy(requirer(), num_units=2)
    juju.integrate(prov, req)
    juju.settle()
    for unit in req.units:
        view = relation(unit)
        assert view.remote_app_data == {'greeting': 'hello'}
        assert {k: v['name'] for k, v in view.remote_units_data.items()} == {
            0: 'provider/0',
            1: 'provider/1',
        }
        assert unit.state.unit_status == testing.ActiveStatus('seen=provider/0,provider/1')
    for unit in prov.units:
        assert unit.state.unit_status == testing.ActiveStatus('seen=requirer/0,requirer/1')
    # The follower's view of its own application databag matches the leader's.
    assert relation(prov.units[1]).local_app_data == {'greeting': 'hello'}


def test_integrate_takes_an_endpoint_where_more_than_one_pair_matches(juju: testing.Juju):
    meta = {
        'name': 'provider',
        'provides': {'db': {'interface': 'test-db'}, 'replica': {'interface': 'test-db'}},
    }
    prov = juju.deploy(provider(meta))
    req = juju.deploy(requirer())
    with pytest.raises(testing.errors.JujuError, match='more than one way') as exc_info:
        juju.integrate(prov, req)
    assert 'provider:db requirer:db' in str(exc_info.value)
    assert 'provider:replica requirer:db' in str(exc_info.value)
    juju.integrate((prov, 'replica'), req)
    assert relation(prov.leader, 'replica').remote_app_name == 'requirer'


@pytest.mark.parametrize(
    ('prov_meta', 'match'),
    [
        ({'name': 'provider', 'provides': {'db': {'interface': 'other'}}}, 'No endpoints'),
        ({'name': 'provider', 'requires': {'db': {'interface': 'test-db'}}}, 'No endpoints'),
        ({'name': 'provider'}, 'no provides or requires endpoints'),
    ],
)
def test_integrate_rejects_apps_with_no_matching_endpoints(
    juju: testing.Juju, prov_meta: dict[str, Any], match: str
):
    prov = juju.deploy(provider(prov_meta))
    req = juju.deploy(requirer())
    with pytest.raises(testing.errors.JujuError, match=match):
        juju.integrate(prov, req)


def test_integrate_rejects_an_unknown_endpoint(juju: testing.Juju):
    prov = juju.deploy(provider())
    req = juju.deploy(requirer())
    with pytest.raises(testing.errors.JujuError, match="no provides or requires endpoint 'nope'"):
        juju.integrate((prov, 'nope'), req)


def test_integrate_rejects_relating_twice(juju: testing.Juju):
    prov = juju.deploy(provider())
    req = juju.deploy(requirer())
    juju.integrate(prov, req)
    with pytest.raises(testing.errors.JujuError, match='already related'):
        juju.integrate(req, prov)


def test_integrate_rejects_relating_an_app_to_itself(juju: testing.Juju):
    meta = {**PROVIDER_META, 'requires': {'up': {'interface': 'test-db'}}}
    prov = juju.deploy(provider(meta))
    with pytest.raises(testing.errors.JujuError, match='itself'):
        juju.integrate(prov, prov)


def test_integrate_rejects_subordinate_endpoints(juju: testing.Juju):
    meta = {'name': 'sub', 'requires': {'db': {'interface': 'test-db', 'scope': 'container'}}}
    prov = juju.deploy(provider())
    sub = juju.deploy(requirer(meta))
    with pytest.raises(testing.errors.JujuError, match='subordinate'):
        juju.integrate(prov, sub)


def test_integrate_rejects_an_app_from_another_juju(juju: testing.Juju):
    prov = juju.deploy(provider())
    with testing.Juju() as other:
        req = other.deploy(requirer())
        with pytest.raises(testing.errors.JujuError, match='not deployed in this Juju'):
            juju.integrate(prov, req)


def test_relation_ids_are_per_juju(juju: testing.Juju):
    # Juju numbers relations per model, so a test's relation IDs don't depend
    # on what ran before it in the same process.
    prov = juju.deploy(provider())
    req = juju.deploy(requirer())
    juju.integrate(prov, req)
    assert relation(prov.leader).id == 0


# remove_relation


def test_remove_relation_departs_each_remote_unit_then_breaks(juju: testing.Juju):
    prov = juju.deploy(provider(), num_units=2)
    req = juju.deploy(requirer(), num_units=2)
    juju.integrate(prov, req)
    juju.settle()
    juju.remove_relation(prov, req)
    trace = juju.settle()
    for unit, remote in ((prov.units[1], req), (req.units[0], prov)):
        assert relation_events(trace, unit) == [
            *[('db_relation_departed', u.id) for u in remote.units],
            ('db_relation_broken', None),
        ]
        assert unit.state.relations == frozenset()


def test_remove_relation_keeps_the_relation_until_broken_is_dispatched(juju: testing.Juju):
    prov = juju.deploy(provider())
    req = juju.deploy(requirer())
    juju.integrate(prov, req)
    juju.settle()
    juju.remove_relation(prov, req)
    assert relation(prov.leader).id == relation(req.leader).id
    trace = juju.settle()
    broken = next(d for d in trace if d.event.name == 'db_relation_broken')
    # During relation-broken, no remote unit is left in the relation.
    assert isinstance(broken.event.relation, testing.Relation)
    assert broken.event.relation.remote_units_data == {}


def test_remove_relation_before_settling_skips_the_joins(juju: testing.Juju):
    prov = juju.deploy(provider())
    req = juju.deploy(requirer())
    juju.integrate(prov, req)
    juju.remove_relation(prov, req)
    trace = juju.settle()
    for unit in (prov.leader, req.leader):
        assert relation_events(trace, unit) == [
            ('db_relation_created', None),
            ('db_relation_broken', None),
        ]


def test_remove_relation_rejects_apps_that_are_not_related(juju: testing.Juju):
    prov = juju.deploy(provider())
    req = juju.deploy(requirer())
    with pytest.raises(testing.errors.JujuError, match='not related'):
        juju.remove_relation(prov, req)


def test_remove_relation_takes_endpoints(juju: testing.Juju):
    meta = {
        'name': 'provider',
        'provides': {'db': {'interface': 'test-db'}, 'replica': {'interface': 'test-db'}},
    }
    prov = juju.deploy(provider(meta))
    req1 = juju.deploy(requirer(), app='req1')
    juju.integrate((prov, 'db'), req1)
    juju.integrate((prov, 'replica'), juju.deploy(requirer(), app='req2'))
    juju.settle()
    with pytest.raises(testing.errors.JujuError, match='provider:replica and req1 are not'):
        juju.remove_relation((prov, 'replica'), req1)
    juju.remove_relation(req1, (prov, 'db'))
    juju.settle()
    assert [r.endpoint for r in prov.leader.state.relations] == ['replica']


def test_remove_relation_rejects_an_ambiguous_pair(juju: testing.Juju):
    meta = {
        'name': 'provider',
        'provides': {'db': {'interface': 'test-db'}, 'replica': {'interface': 'test-db'}},
    }
    prov = juju.deploy(provider(meta))
    req = juju.deploy(requirer())
    juju.integrate((prov, 'db'), req)
    with pytest.raises(testing.errors.JujuError, match='already related'):
        juju.integrate((prov, 'db'), req)
    # A second relation needs a second requires endpoint.
    req_meta = {
        'name': 'requirer',
        'requires': {'db': {'interface': 'test-db'}, 'db2': {'interface': 'test-db'}},
    }
    req2 = juju.deploy(requirer(req_meta), app='req2')
    juju.integrate((prov, 'db'), (req2, 'db'))
    juju.integrate((prov, 'replica'), (req2, 'db2'))
    with pytest.raises(testing.errors.JujuError, match='more than once') as exc_info:
        juju.remove_relation(prov, req2)
    assert 'provider:db req2:db' in str(exc_info.value)


# add_unit and remove_unit on related applications


def test_add_unit_joins_the_new_unit_to_related_applications(juju: testing.Juju):
    prov = juju.deploy(provider())
    req = juju.deploy(requirer(), num_units=2)
    juju.integrate(prov, req)
    juju.settle()
    new = juju.add_unit(prov)
    trace = juju.settle()
    assert [(d.event.name, d.event.relation_remote_unit_id) for d in trace if d.unit is new] == [
        ('install', None),
        ('db_relation_created', None),
        ('leader_settings_changed', None),
        ('config_changed', None),
        ('start', None),
        ('db_relation_joined', 0),
        ('db_relation_changed', 0),
        ('db_relation_joined', 1),
        ('db_relation_changed', 1),
    ]
    for unit in req.units:
        assert relation_events(trace, unit)[:2] == [
            ('db_relation_joined', 1),
            ('db_relation_changed', 1),
        ]
        assert unit.state.unit_status == testing.ActiveStatus('seen=provider/0,provider/1')
    assert relation(new).remote_app_data == {'greeting': 'hello'}
    assert relation(new).local_app_data == {'greeting': 'hello'}


def test_remove_unit_departs_it_from_related_applications(juju: testing.Juju):
    prov = juju.deploy(provider(), num_units=2)
    req = juju.deploy(requirer(), num_units=2)
    juju.integrate(prov, req)
    juju.settle()
    juju.remove_unit(prov, num_units=1)
    trace = juju.settle()
    for unit in req.units:
        assert relation_events(trace, unit) == [('db_relation_departed', 1)]
        assert list(relation(unit).remote_units_data) == [0]
    departing = [
        (d.event.name, d.event.relation_remote_unit_id)
        for d in trace
        if d.unit.name == 'provider/1'
    ]
    assert departing == [
        ('db_relation_departed', 0),
        ('db_relation_departed', 1),
        ('db_relation_broken', None),
        ('stop', None),
        ('remove', None),
    ]


# Propagation


class ChattyProvider(ops.CharmBase):
    """Bumps a counter in its databag whenever the other side changes."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['db'].relation_changed, self._on_changed)

    def _on_changed(self, event: ops.RelationChangedEvent):
        databag = event.relation.data[self.unit]
        databag['n'] = str(int(databag.get('n', '0')) + 1)


class FlipFlop(ops.CharmBase):
    """The provider flips the other side's value and the requirer copies it: a cycle."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['db'].relation_changed, self._on_changed)

    def _on_changed(self, event: ops.RelationChangedEvent):
        assert event.unit is not None
        other = event.relation.data[event.unit].get('v', '0')
        if self.app.name == 'provider':
            other = '1' if other == '0' else '0'
        event.relation.data[self.unit]['v'] = other


def test_settle_reports_a_cross_application_loop(juju: testing.Juju):
    prov = juju.deploy(testing.CharmSpec(FlipFlop, meta=PROVIDER_META))
    req = juju.deploy(testing.CharmSpec(FlipFlop, meta=REQUIRER_META))
    juju.integrate(prov, req)
    with pytest.raises(testing.errors.JujuError, match='same State') as exc_info:
        juju.settle()
    assert 'db_relation_changed on' in str(exc_info.value)


def test_settle_gives_up_on_charms_that_never_converge(juju: testing.Juju):
    prov = juju.deploy(testing.CharmSpec(ChattyProvider, meta=PROVIDER_META))
    req = juju.deploy(testing.CharmSpec(ChattyProvider, meta=REQUIRER_META))
    juju.integrate(prov, req)
    with pytest.raises(testing.errors.JujuError, match='Did not converge after 200 events'):
        juju.settle()


def test_only_the_leader_can_write_application_data(juju: testing.Juju):
    class Writer(ops.CharmBase):
        def __init__(self, framework: ops.Framework):
            super().__init__(framework)
            framework.observe(self.on['db'].relation_joined, self._on_joined)

        def _on_joined(self, event: ops.RelationJoinedEvent):
            event.relation.data[self.app]['by'] = self.unit.name

    prov = juju.deploy(testing.CharmSpec(Writer, meta=PROVIDER_META), num_units=2)
    req = juju.deploy(requirer())
    juju.integrate(prov, req)
    # ops refuses the write on the follower, as Juju would, so the hook fails.
    with pytest.raises(testing.errors.UncaughtCharmError) as exc_info:
        juju.settle()
    assert isinstance(exc_info.value.__cause__, ops.RelationDataAccessError)


def test_a_follower_application_data_write_that_bypasses_ops_is_refused(juju: testing.Juju):
    prov = juju.deploy(provider(), num_units=2)
    req = juju.deploy(requirer())
    juju.integrate(prov, req)
    runner = prov._runner
    original = runner.run

    def run(unit_id: int, event: Any, state: testing.State, secret_seed: str) -> testing.State:
        out = original(unit_id, event, state, secret_seed)
        if unit_id == 1 and event.name == 'db_relation_created':
            view = next(iter(out.relations))
            cast('dict[str, str]', view.local_app_data)['sneaky'] = 'yes'
        return out

    runner.run = run
    with pytest.raises(testing.errors.JujuError, match='provider/1 is not the leader'):
        juju.settle()


def test_application_status_is_shared_with_followers(juju: testing.Juju):
    class Status(ops.CharmBase):
        def __init__(self, framework: ops.Framework):
            super().__init__(framework)
            framework.observe(self.on.start, self._on_start)

        def _on_start(self, _: ops.StartEvent):
            if self.unit.is_leader():
                self.app.status = ops.ActiveStatus('serving')

    app = juju.deploy(testing.CharmSpec(Status, meta={'name': 'status'}), num_units=2)
    juju.settle()
    assert app.units[1].state.app_status == testing.ActiveStatus('serving')


# Ordering


def test_applications_take_turns(juju: testing.Juju):
    juju.deploy(provider())
    juju.deploy(requirer())
    trace = juju.settle()
    assert [d.unit.name for d in trace[:4]] == [
        'provider/0',
        'requirer/0',
        'provider/0',
        'requirer/0',
    ]


def test_a_waiting_changed_event_is_not_queued_twice(juju: testing.Juju):
    # A config change before the first settle is covered by the
    # config-changed in the startup sequence, which sees the new value.
    meta = {**PROVIDER_META, 'config': {'options': {'x': {'type': 'string'}}}}
    app = juju.deploy(provider(meta))
    juju.config(app, {'x': 'y'})
    trace = juju.settle()
    assert [d.event.name for d in trace].count('config_changed') == 1


# Secrets


@pytest.fixture
def db_and_web(juju: testing.Juju) -> tuple[testing.App, testing.App]:
    db = juju.deploy(DBSERVER, num_units=2)
    web = juju.deploy(WEBAPP, num_units=2)
    juju.integrate(web, db)
    juju.settle()
    return db, web


def test_a_granted_secret_is_readable_on_the_other_side(
    juju: testing.Juju, db_and_web: tuple[testing.App, testing.App]
):
    db, web = db_and_web
    owned = [s for s in db.leader.state.secrets if s.owner == 'app']
    assert len(owned) == 1
    for unit in web.units:
        (secret,) = unit.state.secrets
        assert secret.id == owned[0].id
        assert secret.owner is None
        assert secret.tracked_content == {'password': 'first'}
        assert unit.state.unit_status == testing.ActiveStatus(
            'db=dbserver.example:5432 password=first hosts=dbserver/0,dbserver/1'
        )
    # The other units of the owning application see it as the application's.
    assert [s.owner for s in db.units[1].state.secrets] == ['app']


def test_new_secret_content_reaches_the_other_side(
    juju: testing.Juju, db_and_web: tuple[testing.App, testing.App]
):
    db, web = db_and_web
    juju.config(db, {'password': 'second'})
    trace = juju.settle()
    for unit in web.units:
        assert ('secret_changed', unit.name) in [(d.event.name, d.unit.name) for d in trace]
        assert 'password=second' in unit.state.unit_status.message
        (secret,) = unit.state.secrets
        assert secret.tracked_content == secret.latest_content == {'password': 'second'}
    # Once every reader tracks revision 2, the owner is told revision 1 is unused.
    removes = [d for d in trace if d.event.name == 'secret_remove']
    assert [(d.unit.name, d.event.secret_revision) for d in removes] == [('dbserver/0', 1)]
    assert db.leader.state.unit_status == testing.ActiveStatus('removed revision 1')


def test_removing_the_relation_takes_the_secret_away(
    juju: testing.Juju, db_and_web: tuple[testing.App, testing.App]
):
    db, web = db_and_web
    juju.remove_relation(db, web)
    juju.settle()
    for unit in web.units:
        assert unit.state.secrets == frozenset()
        assert unit.state.unit_status == testing.BlockedStatus('no database')
    (secret,) = [s for s in db.leader.state.secrets if s.owner == 'app']
    assert secret.remote_grants == {}


def test_revoking_a_grant_takes_the_secret_away(juju: testing.Juju):
    class Revoker(ops.CharmBase):
        def __init__(self, framework: ops.Framework):
            super().__init__(framework)
            framework.observe(self.on['db'].relation_created, self._on_created)
            framework.observe(self.on.config_changed, self._on_config_changed)

        def _on_created(self, event: ops.RelationCreatedEvent):
            if self.unit.is_leader():
                secret = self.app.add_secret({'key': 'value'}, label='s')
                secret.grant(event.relation)

        def _on_config_changed(self, _: ops.ConfigChangedEvent):
            relation = self.model.get_relation('db')
            if relation is not None and self.config.get('revoke'):
                self.model.get_secret(label='s').revoke(relation)

    options = {'revoke': {'type': 'boolean', 'default': False}}
    meta = {**PROVIDER_META, 'config': {'options': options}}
    prov = juju.deploy(testing.CharmSpec(Revoker, meta=meta))
    req = juju.deploy(requirer(), num_units=2)
    juju.integrate(prov, req)
    juju.settle()
    assert all(len(u.state.secrets) == 1 for u in req.units)
    juju.config(prov, {'revoke': True})
    juju.settle()
    assert all(u.state.secrets == frozenset() for u in req.units)


def test_secret_ids_are_the_same_on_every_run():
    def run() -> list[str]:
        with testing.Juju(model_name='m', uuid='00000000-0000-4000-8000-000000000001') as j:
            db = j.deploy(DBSERVER)
            web = j.deploy(WEBAPP)
            j.integrate(web, db)
            j.settle()
            return sorted(s.id for s in db.leader.state.secrets)

    first = run()
    assert len(first) == 1
    assert first[0].startswith('secret:')
    assert run() == first


# Charms from different sources


class StandInDb(ops.CharmBase):
    """What a charm library's stand-in for the database would do: answer the request."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['database'].relation_joined, self._on_joined)

    def _on_joined(self, event: ops.RelationJoinedEvent):
        event.relation.data[self.unit]['host'] = 'stand-in'
        if self.unit.is_leader():
            secret = self.app.add_secret({'password': 'from-stand-in'})
            secret.grant(event.relation)
            event.relation.data[self.app]['endpoint'] = 'stand-in:5432'
            event.relation.data[self.app]['secret-id'] = secret.id or ''


STAND_IN_META = {'name': 'db-stand-in', 'provides': {'database': {'interface': 'dbtest'}}}


def test_a_charm_spec_stand_in_relates_to_a_charm_on_disk(juju: testing.Juju):
    calls: list[str] = []

    @contextlib.contextmanager
    def mocking():
        calls.append('open')
        yield

    db = juju.deploy(testing.CharmSpec(StandInDb, meta=STAND_IN_META, mocking=mocking))
    web = juju.deploy(WEBAPP)
    juju.integrate(web, db)
    juju.settle()
    assert web.leader.state.unit_status == testing.ActiveStatus(
        'db=stand-in:5432 password=from-stand-in hosts=stand-in'
    )
    assert calls  # The stand-in's own mocking was open around its dispatches.


@pytest.mark.parametrize('isolated', ['webapp', 'dbserver'])
def test_relations_reach_a_charm_in_a_worker(isolated: str):
    with Juju(model_name='m') as juju:

        def kwargs(name: str) -> dict[str, Any]:
            return {'python_executable': sys.executable} if name == isolated else {}

        db = juju._deploy(DBSERVER, num_units=2, **kwargs('dbserver'))
        web = juju._deploy(WEBAPP, num_units=2, **kwargs('webapp'))
        juju.integrate(web, db)
        juju.settle()
        for unit in web.units:
            assert unit.state.unit_status == testing.ActiveStatus(
                'db=dbserver.example:5432 password=first hosts=dbserver/0,dbserver/1'
            )
        juju.config(db, {'password': 'second'})
        juju.settle()
        for unit in web.units:
            assert 'password=second' in unit.state.unit_status.message
        for unit in db.units:
            assert unit.state.unit_status.message.startswith(('clients=', 'removed revision'))
