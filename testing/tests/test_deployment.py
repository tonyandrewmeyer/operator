# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the model-level layer: Juju, App, and Unit."""

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


def deploy_mycharm(juju: testing.Juju, **kwargs: object) -> testing.App:
    kwargs.setdefault('meta', META)
    kwargs.setdefault('config_schema', CONFIG)
    kwargs.setdefault('actions', ACTIONS)
    return juju.deploy(MyCharm, app='myapp', **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def juju():
    with testing.Juju(name='test-model') as j:
        yield j


@pytest.fixture
def machine_juju():
    # 'lxd' is Saddle's stand-in for "machine", matching testing.Model.type.
    with testing.Juju(name='test-model', type='lxd') as j:
        yield j


# Juju's model identity


def test_juju_is_not_a_model(juju: testing.Juju):
    # Juju produces the Model values that go into each unit's State; it is
    # not substitutable for one.
    assert not isinstance(juju, testing.Model)
    assert juju.name == 'test-model'
    assert juju.uuid


def test_model_identity_is_stamped_into_unit_states(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    for unit in app.units:
        assert unit.state.model.name == 'test-model'
        assert unit.state.model.uuid == juju.uuid


def test_unit_state_carries_a_plain_model_not_juju(juju: testing.Juju):
    # A State may be serialised out to a worker process, so it must not carry
    # a handle to the Juju instance driving it.
    app = deploy_mycharm(juju)
    assert type(app.leader.state.model) is testing.Model


def test_plain_model_has_no_operations():
    assert not hasattr(testing.Model(), 'deploy')
    assert not hasattr(testing.Model(), 'settle')


# deploy


def test_deploy_creates_units(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=3)
    assert [unit.id for unit in app.units] == [0, 1, 2]
    assert [unit.name for unit in app.units] == ['myapp/0', 'myapp/1', 'myapp/2']
    assert app.name == 'myapp'


def test_deploy_makes_unit_zero_the_leader(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    assert app.leader is app.units[0]
    assert app.units[0].is_leader
    assert not app.units[1].is_leader
    assert app.units[0].state.leader
    assert not app.units[1].state.leader


def test_deploy_emits_the_juju_startup_sequence(juju: testing.Juju):
    app = deploy_mycharm(juju)
    trace = juju.settle()
    assert [dispatch.event.name for dispatch in trace] == [
        'install',
        'leader_elected',
        'config_changed',
        'start',
    ]
    assert all(dispatch.unit is app.leader for dispatch in trace)


def test_non_leader_units_get_leader_settings_changed(juju: testing.Juju):
    deploy_mycharm(juju, num_units=2)
    trace = juju.settle()
    follower_events = [d.event.name for d in trace if d.unit.id == 1]
    assert 'leader_settings_changed' in follower_events
    assert 'leader_elected' not in follower_events


def test_deploy_applies_config_defaults(juju: testing.Juju):
    app = deploy_mycharm(juju)
    assert app.config == {'log_level': 'info'}
    assert app.leader.state.config == {'log_level': 'info'}


def test_deploy_config_overrides_defaults(juju: testing.Juju):
    app = deploy_mycharm(juju, config={'log_level': 'trace'})
    assert app.leader.state.config == {'log_level': 'trace'}
    assert app.leader.state.unit_status == testing.ActiveStatus('start:trace')


def test_deploy_sets_planned_units(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=3)
    for unit in app.units:
        assert unit.state.planned_units == 3


def test_deploy_rejects_a_duplicate_app_name(juju: testing.Juju):
    deploy_mycharm(juju)
    with pytest.raises(testing.JujuError, match='already deployed'):
        deploy_mycharm(juju)


def test_deploy_rejects_zero_units(juju: testing.Juju):
    with pytest.raises(testing.JujuError, match='at least 1'):
        deploy_mycharm(juju, num_units=0)


def test_deploy_defaults_the_app_name_to_the_charm_name(juju: testing.Juju):
    app = juju.deploy(MyCharm, meta=META, config_schema=CONFIG)
    assert app.name == 'myapp'


def test_deploy_creates_containers_and_emits_pebble_ready(juju: testing.Juju):
    meta: dict[str, Any] = {**META, 'containers': {'workload': {}}}
    app = juju.deploy(MyCharm, meta=meta, config_schema=CONFIG, actions=ACTIONS)
    trace = juju.settle()
    assert [d.event.name for d in trace][-1] == 'workload_pebble_ready'
    assert {c.name for c in app.leader.state.containers} == {'workload'}


# config


def test_config_emits_config_changed_on_every_unit(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    juju.settle()
    juju.config(app, {'log_level': 'debug'})
    trace = juju.settle()
    assert [(d.event.name, d.unit.id) for d in trace] == [
        ('config_changed', 0),
        ('config_changed', 1),
    ]
    assert app.config == {'log_level': 'debug'}
    for unit in app.units:
        assert unit.state.unit_status == testing.ActiveStatus('config_changed:debug')


def test_config_merges_with_existing_values(juju: testing.Juju):
    schema = {
        'options': {
            'log_level': {'type': 'string', 'default': 'info'},
            'other': {'type': 'string', 'default': 'keep'},
        }
    }
    app = deploy_mycharm(juju, config_schema=schema)
    juju.config(app, {'log_level': 'debug'})
    assert app.leader.state.config == {'log_level': 'debug', 'other': 'keep'}


# add_unit


def test_add_unit_runs_the_startup_sequence_for_the_new_unit(juju: testing.Juju):
    app = deploy_mycharm(juju)
    juju.settle()
    unit = juju.add_unit(app)
    trace = juju.settle()
    assert unit.id == 1
    assert [d.event.name for d in trace if d.unit.id == 1] == [
        'install',
        'leader_settings_changed',
        'config_changed',
        'start',
    ]


def test_add_unit_makes_existing_peers_see_relation_joined(juju: testing.Juju):
    # Juju follows joined with changed: the databag Juju populates for the new
    # unit becomes visible at the same moment the unit joins.
    app = deploy_mycharm(juju)
    juju.settle()
    juju.add_unit(app)
    trace = juju.settle()
    assert [d.event.name for d in trace if d.unit.id == 0] == [
        'replicas_relation_joined',
        'replicas_relation_changed',
    ]


def test_add_unit_updates_planned_units(juju: testing.Juju):
    app = deploy_mycharm(juju)
    juju.add_unit(app)
    for unit in app.units:
        assert unit.state.planned_units == 2


# remove_unit
#
# Which form applies -- scale-down-by-count or named-unit -- is decided by the
# application's substrate, so these are split across the default (Kubernetes)
# `juju` fixture and the `machine_juju` ('lxd') fixture.


def test_remove_unit_kubernetes_scales_down_by_count(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    juju.settle()
    juju.remove_unit(app, num_units=1)
    trace = juju.settle()
    assert [(d.event.name, d.unit.id) for d in trace] == [
        ('replicas_relation_departed', 0),
        ('stop', 1),
        ('remove', 1),
    ]


def test_remove_unit_kubernetes_removes_the_highest_numbered_units(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=3)
    juju.remove_unit(app, num_units=1)
    juju.settle()
    assert [unit.id for unit in app.units] == [0, 1]


def test_remove_unit_kubernetes_rejects_named_units(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    with pytest.raises(testing.JujuError, match='count'):
        juju.remove_unit(app.units[1])


def test_remove_unit_kubernetes_rejects_the_last_unit(juju: testing.Juju):
    app = deploy_mycharm(juju)
    with pytest.raises(testing.JujuError, match='only 1'):
        juju.remove_unit(app, num_units=1)


def test_remove_unit_machine_removes_the_named_unit(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    machine_juju.settle()
    machine_juju.remove_unit(app.units[1])
    trace = machine_juju.settle()
    assert [(d.event.name, d.unit.id) for d in trace] == [
        ('replicas_relation_departed', 0),
        ('stop', 1),
        ('remove', 1),
    ]


def test_remove_unit_machine_is_variadic_over_units(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=3)
    machine_juju.settle()
    machine_juju.remove_unit(app.units[1], app.units[2])
    machine_juju.settle()
    assert [unit.id for unit in app.units] == [0]


def test_remove_unit_machine_rejects_the_count_form(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    with pytest.raises(testing.JujuError, match='lxd substrate'):
        machine_juju.remove_unit(app, num_units=1)


def test_remove_unit_rejects_num_units_with_unit_objects(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    with pytest.raises(testing.JujuError, match='num_units'):
        machine_juju.remove_unit(app.units[1], num_units=1)


def test_remove_unit_machine_rejects_the_last_unit(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju)
    with pytest.raises(testing.JujuError, match='last unit'):
        machine_juju.remove_unit(app.units[0])


def test_remove_unit_drops_it_from_peer_databags(machine_juju: testing.Juju):
    app = machine_juju.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    machine_juju.settle()
    relation = peer_relation(app.units[0])
    assert 1 in relation.peers_data

    machine_juju.remove_unit(app.units[1])
    machine_juju.settle()
    relation = peer_relation(app.units[0])
    assert relation.peers_data == {}
    assert app.units[0].state.planned_units == 1


def test_remove_unit_requires_at_least_one_argument(juju: testing.Juju):
    with pytest.raises(testing.JujuError, match='at least one'):
        juju.remove_unit()


def test_remove_unit_rejects_mixing_apps_and_units(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    with pytest.raises(testing.JujuError, match='not a mix'):
        machine_juju.remove_unit(app, app.units[1])


# Peer convergence


def test_peer_unit_databags_propagate(juju: testing.Juju):
    app = juju.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    juju.settle()
    assert peer_relation(app.units[0]).peers_data[1]['ready'] == 'yes'
    assert peer_relation(app.units[1]).peers_data[0]['ready'] == 'yes'


def test_peer_databag_writes_drive_relation_changed(juju: testing.Juju):
    app = juju.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    trace = juju.settle()
    assert 'replicas_relation_changed' in [d.event.name for d in trace]
    # Each unit observed the other, which is only possible if the write made it
    # across and woke the peer up.
    for unit in app.units:
        assert unit.state.unit_status.name == 'active'
        assert 'myapp/' in unit.state.unit_status.message


def test_leader_app_databag_propagates_to_followers(juju: testing.Juju):
    app = juju.deploy(PublishingCharm, app='myapp', meta=META, num_units=2)
    juju.settle()
    for unit in app.units:
        assert peer_relation(unit).local_app_data['cluster'] == 'formed'


def test_settle_raises_when_juju_does_not_converge(juju: testing.Juju):
    juju.deploy(ChattyCharm, app='myapp', meta=META, num_units=2)
    with pytest.raises(testing.JujuError, match='Did not converge'):
        juju.settle(max_events=50)


# settle, implicit settle, and stepping


def test_reading_state_settles_implicitly(juju: testing.Juju):
    app = deploy_mycharm(juju)
    # No explicit settle(): the startup events are still queued here.
    assert app.leader.state.unit_status == testing.ActiveStatus('start:info')


def test_settle_returns_the_dispatch_trace(juju: testing.Juju):
    app = deploy_mycharm(juju)
    trace = juju.settle()
    assert all(isinstance(d, testing.Dispatch) for d in trace)
    event, unit, state = trace[0]
    assert event.name == 'install'
    assert unit is app.leader
    assert isinstance(state, testing.State)


def test_settle_trace_states_are_post_dispatch_snapshots(juju: testing.Juju):
    deploy_mycharm(juju)
    trace = juju.settle()
    assert trace[-1].state.unit_status == testing.ActiveStatus('start:info')


def test_settle_is_a_no_op_when_the_queue_is_empty(juju: testing.Juju):
    deploy_mycharm(juju)
    juju.settle()
    assert juju.settle() == []


def test_stepping_dispatches_one_event_at_a_time(juju: testing.Juju):
    app = deploy_mycharm(juju)
    with juju.stepping() as stepper:
        first = stepper.step()
        assert first is not None
        assert first.event.name == 'install'
        assert first.unit is app.leader
        second = stepper.step()
        assert second is not None
        assert second.event.name == 'leader_elected'


def test_stepping_suspends_implicit_settle(juju: testing.Juju):
    app = deploy_mycharm(juju)
    with juju.stepping() as stepper:
        stepper.step()  # install
        # Reading state here must not drain the rest of the queue.
        assert app.leader.state.unit_status == testing.ActiveStatus('install:info')
        stepper.step()  # leader_elected
        assert app.leader.state.unit_status == testing.ActiveStatus('leader_elected:info')


def test_implicit_settle_resumes_after_stepping(juju: testing.Juju):
    app = deploy_mycharm(juju)
    with juju.stepping() as stepper:
        stepper.step()
    assert app.leader.state.unit_status == testing.ActiveStatus('start:info')


def test_stepping_returns_none_when_the_queue_is_empty(juju: testing.Juju):
    deploy_mycharm(juju)
    juju.settle()
    with juju.stepping() as stepper:
        assert stepper.step() is None


def test_stepping_drains_the_queue_across_a_removed_unit(machine_juju: testing.Juju):
    # Regression test: `_step()` used to return None for a `_RemoveUnit`
    # marker as well as for an empty queue, so `while stepper.step():` stopped
    # early and left events queued after the removal undispatched.
    app = deploy_mycharm(machine_juju, num_units=2)
    machine_juju.settle()
    machine_juju.remove_unit(app.units[1])
    # Queue a further, unrelated event -- targeting only the surviving unit,
    # so removing unit 1 doesn't also remove the thing being asserted on --
    # behind the removal's teardown and marker, so a step loop that stops at
    # the marker leaves it stranded.
    machine_juju.run(app.units[0], 'greet')

    dispatched: list[testing.Dispatch] = []
    with machine_juju.stepping() as stepper:
        while (dispatch := stepper.step()) is not None:
            dispatched.append(dispatch)

    assert [d.event.name for d in dispatched if d.unit.id == 0][-1] == 'greet_action'
    assert app.units[0].state.unit_status == testing.ActiveStatus('hello world')


def test_stepping_drains_the_queue_across_a_vanished_rebind_target(juju: testing.Juju):
    # Regression test: an event whose rebind target (a relation or container)
    # went away between queueing and dispatch also used to return None from
    # `_step()`, indistinguishable from an empty queue. There's no public way
    # to make a rebind target vanish -- containers are meta-declared and peer
    # relations are never dropped from a unit's own state -- so this reaches
    # into the queue directly, the way test_isolated_units_share_one_worker
    # reaches into the runner.
    from scenario.deployment import _Queued, _Rebind
    from scenario.state import _Event

    app = deploy_mycharm(juju)
    juju.settle()
    juju._d.queue.append(
        _Queued(app, 0, _Event('missing_pebble_ready'), _Rebind('container', 'missing'))
    )
    juju.run(app.leader, 'greet')

    dispatched: list[testing.Dispatch] = []
    with juju.stepping() as stepper:
        while (dispatch := stepper.step()) is not None:
            dispatched.append(dispatch)

    assert [d.event.name for d in dispatched][-1] == 'greet_action'


def test_settle_is_deterministic():
    def run() -> list[str]:
        with testing.Juju(name='m') as j:
            j.deploy(PublishingCharm, app='myapp', meta=META, num_units=3)
            return [f'{dispatch.event.name}@{dispatch.unit.name}' for dispatch in j.settle()]

    first = run()
    assert first  # guard against the trace being empty and the check vacuous
    for _ in range(5):
        assert run() == first


# run


def test_run_dispatches_to_the_leader(juju: testing.Juju):
    app = deploy_mycharm(juju)
    juju.settle()
    juju.run(app, 'greet', {'name': 'charmer'})
    trace = juju.settle()
    assert [d.event.name for d in trace] == ['greet_action']
    assert trace[0].unit is app.leader
    assert app.leader.state.unit_status == testing.ActiveStatus('hello charmer')


def test_run_dispatches_to_a_named_unit(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    juju.settle()
    juju.run(app.units[1], 'greet')
    trace = juju.settle()
    assert trace[0].unit is app.units[1]
    assert app.units[1].state.unit_status == testing.ActiveStatus('hello world')


# Lifecycle


def test_operations_after_close_are_rejected(juju: testing.Juju):
    deploy_mycharm(juju)
    juju.close()
    with pytest.raises(testing.JujuError, match='been closed'):
        deploy_mycharm(juju)


def test_close_is_idempotent(juju: testing.Juju):
    deploy_mycharm(juju)
    juju.close()
    juju.close()


# Isolated applications
#
# These run the charm in a subprocess with its own sys.path, so they are slower
# than the in-process tests above; they cover the parts of the layer that only
# differ across the process boundary.


@pytest.mark.parametrize('num_units', [1, 2])
def test_isolated_app_runs_its_startup_sequence(num_units: int):
    with testing.Juju(name='iso') as j:
        app = j.deploy(
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
    with testing.Juju(name='iso') as j:
        alpha = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        beta = j.deploy(
            _ISOLATION / 'charms' / 'beta',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v2'),),
        )
        assert alpha.leader.state.unit_status.message.startswith('confdep=1.0')
        assert beta.leader.state.unit_status.message.startswith('confdep=2.0')


def test_isolated_app_reads_metadata_from_disk():
    with testing.Juju(name='iso') as j:
        app = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        assert app.meta['name'] == 'alpha'


def test_isolated_app_accepts_a_string_path():
    with testing.Juju(name='iso') as j:
        app = j.deploy(
            str(_ISOLATION / 'charms' / 'alpha'),
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        assert app.leader.state.unit_status.message.startswith('confdep=1.0')


def test_isolated_units_share_one_worker():
    # One process per application, not one per unit: the unit ID travels with
    # each request instead of being baked into the worker.
    with testing.Juju(name='iso') as j:
        app = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
            num_units=3,
        )
        j.settle()
        runner = app._runner
        assert isinstance(runner, testing.IsolatedContext) or hasattr(runner, '_ctx')
        assert runner._ctx._worker is not None  # type: ignore[attr-defined]


def test_isolated_app_runs_config_and_run():
    # config and run dispatch through the same runner as the startup
    # sequence, but exercise a different event shape (config-changed outside
    # of startup; an _Action payload) — worth its own isolated check rather
    # than trusting that startup coverage implies it.
    with testing.Juju(name='iso') as j:
        app = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
            config_schema=CONFIG,
            actions=ACTIONS,
        )
        j.config(app, {'log_level': 'debug'})
        j.run(app, 'greet', params={'name': 'isolated'})
        # Alpha doesn't observe config-changed differently or the action, but
        # both events must round-trip the isolation boundary without raising.
        assert app.leader.state.unit_status.message.startswith('confdep=1.0')
