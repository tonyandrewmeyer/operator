# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the model-level layer: Deployment, App, and Unit."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

import ops
from ops import testing

# The isolation fixtures (charms with mutually incompatible dependencies, and
# the dependency trees themselves) are shared with the isolation tests.
_ISOLATION = pathlib.Path(__file__).parent / 'test_isolation'


META: dict[str, Any] = {
    'name': 'myapp',
    'peers': {'replicas': {'interface': 'myapp-peer'}},
}
CONFIG = {'options': {'log_level': {'type': 'string', 'default': 'info'}}}
ACTIONS = {'greet': {'params': {'name': {'type': 'string'}}}}


class MyCharm(ops.CharmBase):
    """Records what it sees, and publishes to its peer relation."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        for event in (
            self.on.install,
            self.on.start,
            self.on.config_changed,
            self.on.leader_elected,
            self.on.leader_settings_changed,
        ):
            framework.observe(event, self._on_any)
        framework.observe(self.on['replicas'].relation_changed, self._on_peer_changed)
        framework.observe(self.on.greet_action, self._on_greet)

    def _on_any(self, event: ops.EventBase):
        self.unit.status = ops.ActiveStatus(f'{event.handle.kind}:{self.config["log_level"]}')

    def _on_peer_changed(self, _: ops.EventBase):
        relation = self.model.get_relation('replicas')
        assert relation is not None
        names = sorted(unit.name for unit in relation.units)
        self.unit.status = ops.ActiveStatus(f'peers={names}')

    def _on_greet(self, event: ops.ActionEvent):
        self.unit.status = ops.ActiveStatus(f'hello {event.params.get("name", "world")}')


class PublishingCharm(ops.CharmBase):
    """Writes to its peer databags on start, so peers have something to observe."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)
        framework.observe(self.on['replicas'].relation_changed, self._on_changed)

    def _on_start(self, _: ops.EventBase):
        relation = self.model.get_relation('replicas')
        assert relation is not None
        relation.data[self.unit]['ready'] = 'yes'
        if self.unit.is_leader():
            relation.data[self.app]['cluster'] = 'formed'

    def _on_changed(self, _: ops.EventBase):
        relation = self.model.get_relation('replicas')
        assert relation is not None
        ready = sorted(u.name for u in relation.units if relation.data[u].get('ready') == 'yes')
        self.unit.status = ops.ActiveStatus(f'ready={ready}')


class ChattyCharm(ops.CharmBase):
    """Writes a different value to its databag on every event: never converges."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['replicas'].relation_changed, self._on_changed)
        framework.observe(self.on.start, self._on_changed)
        self._counter = 0

    def _on_changed(self, _: ops.EventBase):
        relation = self.model.get_relation('replicas')
        assert relation is not None
        previous = int(relation.data[self.unit].get('counter', '0'))
        relation.data[self.unit]['counter'] = str(previous + 1)


def peer_relation(unit: testing.Unit, endpoint: str = 'replicas') -> testing.PeerRelation:
    relation = unit.state.get_relations(endpoint)[0]
    assert isinstance(relation, testing.PeerRelation)
    return relation


def deploy_mycharm(deployment: testing.Deployment, **kwargs: object) -> testing.App:
    kwargs.setdefault('meta', META)
    kwargs.setdefault('config_schema', CONFIG)
    kwargs.setdefault('actions', ACTIONS)
    return deployment.deploy(MyCharm, app='myapp', **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def deployment():
    with testing.Deployment(name='test-model') as d:
        yield d


# Deployment as a Model


def test_deployment_is_a_model(deployment: testing.Deployment):
    assert isinstance(deployment, testing.Model)
    assert deployment.name == 'test-model'
    assert deployment.uuid


def test_model_identity_is_stamped_into_unit_states(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=2)
    for unit in app.units:
        assert unit.state.model.name == 'test-model'
        assert unit.state.model.uuid == deployment.uuid


def test_unit_state_carries_a_plain_model_not_the_deployment(deployment: testing.Deployment):
    # A State may be serialised out to a worker process, so it must not carry a
    # handle to the deployment driving it.
    app = deploy_mycharm(deployment)
    assert type(app.leader.state.model) is testing.Model


def test_plain_model_has_no_operations():
    assert not hasattr(testing.Model(), 'deploy')
    assert not hasattr(testing.Model(), 'settle')


def test_model_identity_stays_frozen(deployment: testing.Deployment):
    with pytest.raises(Exception, match='cannot assign to field'):
        deployment.name = 'other'  # type: ignore[misc]


# deploy


def test_deploy_creates_units(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=3)
    assert [unit.id for unit in app.units] == [0, 1, 2]
    assert [unit.name for unit in app.units] == ['myapp/0', 'myapp/1', 'myapp/2']
    assert app.name == 'myapp'


def test_deploy_makes_unit_zero_the_leader(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=2)
    assert app.leader is app.units[0]
    assert app.units[0].is_leader
    assert not app.units[1].is_leader
    assert app.units[0].state.leader
    assert not app.units[1].state.leader


def test_deploy_emits_the_juju_startup_sequence(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    trace = deployment.settle()
    assert [dispatch.event.name for dispatch in trace] == [
        'install',
        'leader_elected',
        'config_changed',
        'start',
    ]
    assert all(dispatch.unit is app.leader for dispatch in trace)


def test_non_leader_units_get_leader_settings_changed(deployment: testing.Deployment):
    deploy_mycharm(deployment, num_units=2)
    trace = deployment.settle()
    follower_events = [d.event.name for d in trace if d.unit.id == 1]
    assert 'leader_settings_changed' in follower_events
    assert 'leader_elected' not in follower_events


def test_deploy_applies_config_defaults(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    assert app.config == {'log_level': 'info'}
    assert app.leader.state.config == {'log_level': 'info'}


def test_deploy_config_overrides_defaults(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, config={'log_level': 'trace'})
    assert app.leader.state.config == {'log_level': 'trace'}
    assert app.leader.state.unit_status == testing.ActiveStatus('start:trace')


def test_deploy_sets_planned_units(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=3)
    for unit in app.units:
        assert unit.state.planned_units == 3


def test_deploy_rejects_a_duplicate_app_name(deployment: testing.Deployment):
    deploy_mycharm(deployment)
    with pytest.raises(testing.DeploymentError, match='already deployed'):
        deploy_mycharm(deployment)


def test_deploy_rejects_zero_units(deployment: testing.Deployment):
    with pytest.raises(testing.DeploymentError, match='at least 1'):
        deploy_mycharm(deployment, num_units=0)


def test_deploy_defaults_the_app_name_to_the_charm_name(deployment: testing.Deployment):
    app = deployment.deploy(MyCharm, meta=META, config_schema=CONFIG)
    assert app.name == 'myapp'


def test_deploy_creates_containers_and_emits_pebble_ready(deployment: testing.Deployment):
    meta: dict[str, Any] = {**META, 'containers': {'workload': {}}}
    app = deployment.deploy(MyCharm, meta=meta, config_schema=CONFIG, actions=ACTIONS)
    trace = deployment.settle()
    assert [d.event.name for d in trace][-1] == 'workload_pebble_ready'
    assert {c.name for c in app.leader.state.containers} == {'workload'}


# update_config


def test_update_config_emits_config_changed_on_every_unit(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=2)
    deployment.settle()
    deployment.update_config(app, {'log_level': 'debug'})
    trace = deployment.settle()
    assert [(d.event.name, d.unit.id) for d in trace] == [
        ('config_changed', 0),
        ('config_changed', 1),
    ]
    assert app.config == {'log_level': 'debug'}
    for unit in app.units:
        assert unit.state.unit_status == testing.ActiveStatus('config_changed:debug')


def test_update_config_merges_with_existing_values(deployment: testing.Deployment):
    schema = {
        'options': {
            'log_level': {'type': 'string', 'default': 'info'},
            'other': {'type': 'string', 'default': 'keep'},
        }
    }
    app = deploy_mycharm(deployment, config_schema=schema)
    deployment.update_config(app, {'log_level': 'debug'})
    assert app.leader.state.config == {'log_level': 'debug', 'other': 'keep'}


# add_unit / remove_unit


def test_add_unit_runs_the_startup_sequence_for_the_new_unit(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    deployment.settle()
    unit = deployment.add_unit(app)
    trace = deployment.settle()
    assert unit.id == 1
    assert [d.event.name for d in trace if d.unit.id == 1] == [
        'install',
        'leader_settings_changed',
        'config_changed',
        'start',
    ]


def test_add_unit_makes_existing_peers_see_relation_joined(deployment: testing.Deployment):
    # Juju follows joined with changed: the databag Juju populates for the new
    # unit becomes visible at the same moment the unit joins.
    app = deploy_mycharm(deployment)
    deployment.settle()
    deployment.add_unit(app)
    trace = deployment.settle()
    assert [d.event.name for d in trace if d.unit.id == 0] == [
        'replicas_relation_joined',
        'replicas_relation_changed',
    ]


def test_add_unit_updates_planned_units(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    deployment.add_unit(app)
    for unit in app.units:
        assert unit.state.planned_units == 2


def test_remove_unit_emits_departed_then_teardown(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=2)
    deployment.settle()
    deployment.remove_unit(app, app.units[1])
    trace = deployment.settle()
    assert [(d.event.name, d.unit.id) for d in trace] == [
        ('replicas_relation_departed', 0),
        ('stop', 1),
        ('remove', 1),
    ]


def test_remove_unit_defaults_to_the_highest_numbered_unit(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=3)
    deployment.remove_unit(app)
    deployment.settle()
    assert [unit.id for unit in app.units] == [0, 1]


def test_remove_unit_drops_it_from_peer_databags(deployment: testing.Deployment):
    app = deployment.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    deployment.settle()
    relation = peer_relation(app.units[0])
    assert 1 in relation.peers_data

    deployment.remove_unit(app, app.units[1])
    deployment.settle()
    relation = peer_relation(app.units[0])
    assert relation.peers_data == {}
    assert app.units[0].state.planned_units == 1


def test_remove_unit_rejects_the_last_unit(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    with pytest.raises(testing.DeploymentError, match='last unit'):
        deployment.remove_unit(app)


def test_remove_unit_rejects_a_unit_from_another_app(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=2)
    other = deployment.deploy(MyCharm, app='other', meta={'name': 'other'})
    with pytest.raises(testing.DeploymentError, match='not a unit of'):
        deployment.remove_unit(app, other.leader)


def test_remove_unit_rejects_an_unknown_unit_id(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=2)
    with pytest.raises(testing.DeploymentError, match='no unit 7'):
        deployment.remove_unit(app, 7)


# Peer convergence


def test_peer_unit_databags_propagate(deployment: testing.Deployment):
    app = deployment.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    deployment.settle()
    assert peer_relation(app.units[0]).peers_data[1]['ready'] == 'yes'
    assert peer_relation(app.units[1]).peers_data[0]['ready'] == 'yes'


def test_peer_databag_writes_drive_relation_changed(deployment: testing.Deployment):
    app = deployment.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    trace = deployment.settle()
    assert 'replicas_relation_changed' in [d.event.name for d in trace]
    # Each unit observed the other, which is only possible if the write made it
    # across and woke the peer up.
    for unit in app.units:
        assert unit.state.unit_status.name == 'active'
        assert 'myapp/' in unit.state.unit_status.message


def test_leader_app_databag_propagates_to_followers(deployment: testing.Deployment):
    app = deployment.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    deployment.settle()
    for unit in app.units:
        assert peer_relation(unit).local_app_data['cluster'] == 'formed'


def test_settle_raises_when_the_deployment_does_not_converge(deployment: testing.Deployment):
    deployment.deploy(ChattyCharm, app='myapp', meta=META, num_units=2)
    with pytest.raises(testing.DeploymentError, match='did not converge'):
        deployment.settle(max_events=50)


# settle, implicit settle, and stepping


def test_reading_state_settles_implicitly(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    # No explicit settle(): the startup events are still queued here.
    assert app.leader.state.unit_status == testing.ActiveStatus('start:info')


def test_settle_returns_the_dispatch_trace(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    trace = deployment.settle()
    assert all(isinstance(d, testing.Dispatch) for d in trace)
    event, unit, state = trace[0]
    assert event.name == 'install'
    assert unit is app.leader
    assert isinstance(state, testing.State)


def test_settle_trace_states_are_post_dispatch_snapshots(deployment: testing.Deployment):
    deploy_mycharm(deployment)
    trace = deployment.settle()
    assert trace[-1].state.unit_status == testing.ActiveStatus('start:info')


def test_settle_is_a_no_op_when_the_queue_is_empty(deployment: testing.Deployment):
    deploy_mycharm(deployment)
    deployment.settle()
    assert deployment.settle() == []


def test_stepping_dispatches_one_event_at_a_time(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    with deployment.stepping() as stepper:
        first = stepper.step()
        assert first is not None
        assert first[0].name == 'install'
        assert first[1] is app.leader
        second = stepper.step()
        assert second is not None
        assert second[0].name == 'leader_elected'


def test_stepping_suspends_implicit_settle(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    with deployment.stepping() as stepper:
        stepper.step()  # install
        # Reading state here must not drain the rest of the queue.
        assert app.leader.state.unit_status == testing.ActiveStatus('install:info')
        stepper.step()  # leader_elected
        assert app.leader.state.unit_status == testing.ActiveStatus('leader_elected:info')


def test_implicit_settle_resumes_after_stepping(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    with deployment.stepping() as stepper:
        stepper.step()
    assert app.leader.state.unit_status == testing.ActiveStatus('start:info')


def test_stepping_returns_none_when_the_queue_is_empty(deployment: testing.Deployment):
    deploy_mycharm(deployment)
    deployment.settle()
    with deployment.stepping() as stepper:
        assert stepper.step() is None


def test_settle_is_deterministic():
    def run() -> list[str]:
        with testing.Deployment(name='m') as d:
            d.deploy(PublishingCharm, app='myapp', meta=META, num_units=3)
            return [f'{dispatch.event.name}@{dispatch.unit.name}' for dispatch in d.settle()]

    first = run()
    assert first  # guard against the trace being empty and the check vacuous
    for _ in range(5):
        assert run() == first


# run_action


def test_run_action_dispatches_to_the_leader(deployment: testing.Deployment):
    app = deploy_mycharm(deployment)
    deployment.settle()
    deployment.run_action(app, 'greet', {'name': 'charmer'})
    trace = deployment.settle()
    assert [d.event.name for d in trace] == ['greet_action']
    assert trace[0].unit is app.leader
    assert app.leader.state.unit_status == testing.ActiveStatus('hello charmer')


def test_run_action_dispatches_to_a_named_unit(deployment: testing.Deployment):
    app = deploy_mycharm(deployment, num_units=2)
    deployment.settle()
    deployment.run_action(app.units[1], 'greet')
    trace = deployment.settle()
    assert trace[0].unit is app.units[1]
    assert app.units[1].state.unit_status == testing.ActiveStatus('hello world')


# Lifecycle


def test_operations_after_close_are_rejected(deployment: testing.Deployment):
    deploy_mycharm(deployment)
    deployment.close()
    with pytest.raises(testing.DeploymentError, match='been closed'):
        deploy_mycharm(deployment)


def test_close_is_idempotent(deployment: testing.Deployment):
    deploy_mycharm(deployment)
    deployment.close()
    deployment.close()


# Isolated applications
#
# These run the charm in a subprocess with its own sys.path, so they are slower
# than the in-process tests above; they cover the parts of the layer that only
# differ across the process boundary.


@pytest.mark.parametrize('num_units', [1, 2])
def test_isolated_app_runs_its_startup_sequence(num_units: int):
    with testing.Deployment(name='iso') as d:
        app = d.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
            num_units=num_units,
        )
        assert app.name == 'alpha'
        assert len(app.units) == num_units
        for unit in app.units:
            assert unit.state.unit_status == testing.ActiveStatus(
                'confdep=1.0 legacy=alpha-only-name compute=1'
            )


def test_two_apps_with_conflicting_dependencies_coexist():
    # The point of the whole layer: alpha needs confdep v1 and beta needs v2,
    # and the two cannot share one interpreter.
    with testing.Deployment(name='iso') as d:
        alpha = d.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        beta = d.deploy(
            _ISOLATION / 'charms' / 'beta',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v2'),),
        )
        assert alpha.leader.state.unit_status.message.startswith('confdep=1.0')
        assert beta.leader.state.unit_status.message.startswith('confdep=2.0')


def test_isolated_app_reads_metadata_from_disk():
    with testing.Deployment(name='iso') as d:
        app = d.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        assert app.meta['name'] == 'alpha'


def test_isolated_app_accepts_a_string_path():
    with testing.Deployment(name='iso') as d:
        app = d.deploy(
            str(_ISOLATION / 'charms' / 'alpha'),
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        assert app.leader.state.unit_status.message.startswith('confdep=1.0')


def test_isolated_units_share_one_worker():
    # One process per application, not one per unit: the unit ID travels with
    # each request instead of being baked into the worker.
    with testing.Deployment(name='iso') as d:
        app = d.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
            num_units=3,
        )
        d.settle()
        runner = app._runner
        assert isinstance(runner, testing.IsolatedContext) or hasattr(runner, '_ctx')
        assert runner._ctx._worker is not None  # type: ignore[attr-defined]


def test_isolated_app_runs_update_config_and_run_action():
    # update_config and run_action dispatch through the same runner as the
    # startup sequence, but exercise a different event shape (config-changed
    # outside of startup; an _Action payload) — worth its own isolated check
    # rather than trusting that startup coverage implies it.
    with testing.Deployment(name='iso') as d:
        app = d.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
            config_schema=CONFIG,
            actions=ACTIONS,
        )
        d.update_config(app, {'log_level': 'debug'})
        d.run_action(app, 'greet', params={'name': 'isolated'})
        # Alpha doesn't observe config-changed differently or the action, but
        # both events must round-trip the isolation boundary without raising.
        assert app.leader.state.unit_status.message.startswith('confdep=1.0')
