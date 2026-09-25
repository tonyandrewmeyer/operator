# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the model-level layer: Juju, App, and Unit."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import textwrap
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

    def _on_any(self, event: ops.EventBase):
        self.unit.status = ops.ActiveStatus(f'{event.handle.kind}:{self.config["log_level"]}')

    def _on_peer_changed(self, _: ops.EventBase):
        relation = self.model.get_relation('replicas')
        assert relation is not None
        names = sorted(unit.name for unit in relation.units)
        self.unit.status = ops.ActiveStatus(f'peers={names}')


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
    return juju.deploy(MyCharm, app='myapp', **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def juju():
    with testing.Juju(model_name='test-model') as j:
        yield j


@pytest.fixture
def machine_juju():
    # 'lxd' is the stand-in for "machine", matching testing.Model.type.
    with testing.Juju(model_name='test-model', type='lxd') as j:
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
    juju.settle()
    assert app.leader.state.config == {'log_level': 'trace'}
    assert app.leader.state.unit_status == testing.ActiveStatus('start:trace')


def test_deploy_sets_planned_units(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=3)
    for unit in app.units:
        assert unit.state.planned_units == 3


def test_deploy_rejects_a_duplicate_app_name(juju: testing.Juju):
    deploy_mycharm(juju)
    with pytest.raises(testing.errors.JujuError, match='already deployed'):
        deploy_mycharm(juju)


def test_deploy_rejects_zero_units(juju: testing.Juju):
    with pytest.raises(testing.errors.JujuError, match='at least 1'):
        deploy_mycharm(juju, num_units=0)


def test_deploy_defaults_the_app_name_to_the_charm_name(juju: testing.Juju):
    app = juju.deploy(MyCharm, meta=META, config_schema=CONFIG)
    assert app.name == 'myapp'


def test_deploy_creates_containers_and_emits_pebble_ready(juju: testing.Juju):
    meta: dict[str, Any] = {**META, 'containers': {'workload': {}}}
    app = juju.deploy(MyCharm, meta=meta, config_schema=CONFIG)
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
# Which form applies — scale-down-by-count or named-unit — is decided by the
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
    with pytest.raises(testing.errors.JujuError, match='count'):
        juju.remove_unit(app.units[1])


def test_remove_unit_kubernetes_rejects_the_last_unit(juju: testing.Juju):
    app = deploy_mycharm(juju)
    with pytest.raises(testing.errors.JujuError, match='only 1'):
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
    with pytest.raises(testing.errors.JujuError, match='lxd substrate'):
        machine_juju.remove_unit(app, num_units=1)


def test_remove_unit_rejects_num_units_with_unit_objects(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    with pytest.raises(testing.errors.JujuError, match='num_units'):
        machine_juju.remove_unit(app.units[1], num_units=1)


def test_remove_unit_machine_rejects_the_last_unit(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju)
    with pytest.raises(testing.errors.JujuError, match='last unit'):
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
    with pytest.raises(testing.errors.JujuError, match='at least one'):
        juju.remove_unit()


def test_remove_unit_rejects_mixing_apps_and_units(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    with pytest.raises(testing.errors.JujuError, match='not a mix'):
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
    with pytest.raises(testing.errors.JujuError, match='Did not converge') as exc_info:
        juju.settle()
    # The message ends with the tail of the trace.
    assert 'replicas_relation_changed on myapp/' in str(exc_info.value)


# settle


def test_reading_state_does_not_dispatch_anything(juju: testing.Juju):
    """Reading Unit.state never runs charm code; the queue is untouched."""
    app = deploy_mycharm(juju)
    queued = len(juju._state.queue)
    assert app.leader.state.unit_status == testing.UnknownStatus()
    assert len(juju._state.queue) == queued


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


def test_settle_is_deterministic():
    def run() -> list[str]:
        with testing.Juju(model_name='m') as j:
            j.deploy(PublishingCharm, app='myapp', meta=META, num_units=3)
            return [f'{dispatch.event.name}@{dispatch.unit.name}' for dispatch in j.settle()]

    first = run()
    assert first  # guard against the trace being empty and the check vacuous
    for _ in range(5):
        assert run() == first


# Lifecycle


def test_operations_after_close_are_rejected(juju: testing.Juju):
    deploy_mycharm(juju)
    juju.close()
    with pytest.raises(testing.errors.JujuError, match='been closed'):
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
    with testing.Juju(model_name='iso') as j:
        app = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
            num_units=num_units,
        )
        j.settle()
        assert app.name == 'alpha'
        assert len(app.units) == num_units
        for unit in app.units:
            assert unit.state.unit_status == testing.ActiveStatus(
                'confdep=1.0 legacy=alpha-only-name compute=1'
            )


def test_two_apps_with_conflicting_dependencies_coexist():
    # The point of the whole layer: alpha needs confdep v1 and beta needs v2,
    # and the two cannot share one interpreter.
    with testing.Juju(model_name='iso') as j:
        alpha = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        beta = j.deploy(
            _ISOLATION / 'charms' / 'beta',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v2'),),
        )
        j.settle()
        assert alpha.leader.state.unit_status.message.startswith('confdep=1.0')
        assert beta.leader.state.unit_status.message.startswith('confdep=2.0')


def test_isolated_app_reads_metadata_from_disk():
    with testing.Juju(model_name='iso') as j:
        app = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        assert app.meta['name'] == 'alpha'


def test_isolated_app_accepts_a_relative_string_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(_ISOLATION)
    with testing.Juju(model_name='iso') as j:
        app = j.deploy(
            './charms/alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        j.settle()
        assert app.leader.state.unit_status.message.startswith('confdep=1.0')


def test_isolated_units_share_one_worker():
    # One process per application, not one per unit: the unit ID travels with
    # each request instead of being baked into the worker.
    with testing.Juju(model_name='iso') as j:
        app = j.deploy(
            _ISOLATION / 'charms' / 'pid',
            python_executable=sys.executable,
            num_units=3,
        )
        j.settle()
        pids = {unit.state.unit_status.message for unit in app.units}
        assert len(pids) == 1
        assert pids != {f'pid={os.getpid()}'}  # and it is not the test process


def test_isolated_app_runs_config():
    # config dispatches through the same runner as the startup sequence, but
    # config-changed outside startup is worth its own isolated check.
    with testing.Juju(model_name='iso') as j:
        app = j.deploy(
            _ISOLATION / 'charms' / 'alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
            config_schema=CONFIG,
        )
        j.settle()
        j.config(app, {'log_level': 'debug'})
        trace = j.settle()
        assert [d.event.name for d in trace] == ['config_changed']
        assert app.leader.state.config == {'log_level': 'debug'}


# Leadership


def test_removing_the_leader_is_refused(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    machine_juju.settle()
    with pytest.raises(testing.errors.JujuError, match='it is the leader'):
        machine_juju.remove_unit(app.leader)


def test_removing_a_non_leader_is_allowed(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    machine_juju.settle()
    machine_juju.remove_unit(app.units[1])
    machine_juju.settle()
    assert [u.id for u in app.units] == [0]


# Charms deployed from a path, in the test process


def write_charm(
    root: pathlib.Path,
    source: str,
    *,
    metadata: str = 'name: ondisk\n',
    files: dict[str, str] | None = None,
) -> pathlib.Path:
    """Write a charm's source tree under ``root`` and return its directory."""
    (root / 'src').mkdir(parents=True)
    (root / 'src' / 'charm.py').write_text(textwrap.dedent(source))
    if metadata:
        (root / 'metadata.yaml').write_text(metadata)
    for name, content in (files or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


PID_CHARM = """
    import os

    import ops


    class PidCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            self.unit.status = ops.ActiveStatus(f'{type(self).__name__} pid={os.getpid()}')
"""


def test_a_path_without_an_interpreter_runs_in_process(juju: testing.Juju, tmp_path: pathlib.Path):
    app = juju.deploy(write_charm(tmp_path / 'ondisk', PID_CHARM))
    juju.settle()
    assert app.leader.state.unit_status == testing.ActiveStatus(f'PidCharm pid={os.getpid()}')


def test_two_in_process_charms_from_paths_do_not_collide(
    juju: testing.Juju, tmp_path: pathlib.Path
):
    # Both charms are src/charm.py, so importing each as 'charm' would give
    # the second application the first one's class.
    first = juju.deploy(write_charm(tmp_path / 'one', PID_CHARM), app='one')
    second_source = PID_CHARM.replace('PidCharm', 'OtherCharm')
    second = juju.deploy(write_charm(tmp_path / 'two', second_source), app='two')
    juju.settle()
    assert first.leader.state.unit_status.message.startswith('PidCharm ')
    assert second.leader.state.unit_status.message.startswith('OtherCharm ')


def test_a_unified_charmcraft_yaml_supplies_config_defaults(
    juju: testing.Juju, tmp_path: pathlib.Path
):
    charmcraft = textwrap.dedent("""\
        type: charm
        name: unified
        summary: s
        description: d
        config:
          options:
            port: {type: int, default: 8080}
    """)
    root = write_charm(tmp_path / 'unified', PID_CHARM, metadata='')
    (root / 'charmcraft.yaml').write_text(charmcraft)
    app = juju.deploy(root)
    assert app.name == 'unified'
    assert app.config == {'port': 8080}


def test_a_charm_class_reads_its_metadata_from_disk(
    juju: testing.Juju, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    root = write_charm(
        tmp_path / 'classy',
        PID_CHARM,
        metadata='name: classy\ncontainers:\n  workload: {}\n',
    )
    spec = importlib.util.spec_from_file_location('classy_charm', root / 'src' / 'charm.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'classy_charm', module)
    spec.loader.exec_module(module)
    app = juju.deploy(module.PidCharm)
    assert app.name == 'classy'
    assert {c.name for c in app.leader.state.containers} == {'workload'}


def test_a_str_path_must_be_relative(juju: testing.Juju, tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'ondisk', PID_CHARM)
    with pytest.raises(testing.errors.JujuError, match='must be relative'):
        juju.deploy(str(root))


def test_a_missing_charm_directory_is_rejected(juju: testing.Juju, tmp_path: pathlib.Path):
    with pytest.raises(testing.errors.JujuError, match='No charm source'):
        juju.deploy(tmp_path / 'nothing-here')


def test_extra_sys_path_needs_an_interpreter(juju: testing.Juju, tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'ondisk', PID_CHARM)
    with pytest.raises(testing.errors.JujuError, match='python_executable'):
        juju.deploy(root, extra_sys_path=('/somewhere',))


def test_an_interpreter_needs_a_charm_on_disk(juju: testing.Juju):
    with pytest.raises(testing.errors.JujuError, match='charm on disk'):
        juju.deploy(MyCharm, meta=META, python_executable=sys.executable)


# The charm directory


SHIPPED_FILE_CHARM = """
    import ops


    class ShippedFileCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            greeting = (self.charm_dir / 'templates' / 'greeting.txt').read_text()
            self.unit.status = ops.ActiveStatus(greeting.strip())
"""


@pytest.mark.parametrize('isolated', [False, True])
def test_the_charm_finds_files_it_ships(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(
        tmp_path / 'shipped',
        SHIPPED_FILE_CHARM,
        files={'templates/greeting.txt': 'hello from the charm\n'},
    )
    kwargs: dict[str, Any] = {'python_executable': sys.executable} if isolated else {}
    with testing.Juju() as juju:
        app = juju.deploy(root, num_units=2, **kwargs)
        juju.settle()
        for unit in app.units:
            assert unit.state.unit_status == testing.ActiveStatus('hello from the charm')
    # Context writes metadata files into the charm directory while it runs;
    # none of them may land in the source tree.
    assert sorted(p.name for p in root.iterdir()) == ['metadata.yaml', 'src', 'templates']
    assert (root / 'metadata.yaml').read_text() == 'name: ondisk\n'


# trust


class TrustCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_install)

    def _on_install(self, _: ops.EventBase):
        try:
            self.model.get_cloud_spec()
        except ops.ModelError as e:
            trusted = 'not trusted' not in str(e)
        else:
            trusted = True
        self.unit.status = ops.ActiveStatus('trusted' if trusted else 'untrusted')


@pytest.mark.parametrize('trust', [False, True])
def test_trust_reaches_the_charm(juju: testing.Juju, trust: bool):
    app = juju.deploy(TrustCharm, meta={'name': 'trusty'}, trust=trust)
    juju.settle()
    expected = 'trusted' if trust else 'untrusted'
    assert app.leader.state.unit_status == testing.ActiveStatus(expected)


# state_template


class ExecCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['workload'].pebble_ready, self._on_ready)

    def _on_ready(self, event: ops.PebbleReadyEvent):
        stdout, _ = event.workload.exec(['update-ca-certificates']).wait_output()
        self.unit.status = ops.ActiveStatus(stdout.strip())


EXEC_META: dict[str, Any] = {
    'name': 'execy',
    'containers': {'workload': {}},
    'peers': {'replicas': {'interface': 'execy-peer'}},
}
EXEC_TEMPLATE = testing.State(
    containers={
        testing.Container(
            'workload',
            can_connect=True,
            execs={testing.Exec(['update-ca-certificates'], stdout='updated\n')},
        )
    },
)


def test_state_template_reaches_every_unit(juju: testing.Juju):
    app = juju.deploy(ExecCharm, meta=EXEC_META, state_template=EXEC_TEMPLATE, num_units=2)
    juju.settle()
    for unit in app.units:
        assert unit.state.unit_status == testing.ActiveStatus('updated')


def test_state_template_reaches_units_added_later(juju: testing.Juju):
    app = juju.deploy(ExecCharm, meta=EXEC_META, state_template=EXEC_TEMPLATE)
    juju.settle()
    unit = juju.add_unit(app)
    juju.settle()
    assert unit.state.unit_status == testing.ActiveStatus('updated')


def test_state_template_leaves_the_juju_owned_fields_to_juju(juju: testing.Juju):
    template = testing.State(workload_version='1.2.3')
    app = juju.deploy(MyCharm, meta=META, config_schema=CONFIG, state_template=template)
    unit = app.leader
    assert unit.state.workload_version == '1.2.3'
    assert unit.state.leader
    assert unit.state.model.name == 'test-model'
    assert unit.state.config == {'log_level': 'info'}
    assert [r.endpoint for r in unit.state.relations] == ['replicas']


@pytest.mark.parametrize(
    ('field', 'template'),
    [
        ('leader', testing.State(leader=True)),
        ('planned_units', testing.State(planned_units=3)),
        ('model', testing.State(model=testing.Model(type='lxd'))),
        ('config', testing.State(config={'log_level': 'debug'})),
        ('relations', testing.State(relations={testing.PeerRelation('replicas')})),
    ],
)
def test_state_template_may_not_set_juju_owned_fields(
    juju: testing.Juju, field: str, template: testing.State
):
    with pytest.raises(testing.errors.JujuError, match=f'may not set {field}'):
        juju.deploy(MyCharm, meta=META, state_template=template)


# Application-owned secrets


class SecretCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.leader_elected, self._on_leader_elected)

    def _on_leader_elected(self, _: ops.EventBase):
        self.app.add_secret({'password': 'hunter2'}, label='admin')


def test_application_secrets_reach_every_unit(juju: testing.Juju):
    app = juju.deploy(SecretCharm, meta=META, num_units=2)
    juju.settle()
    leader_secrets = app.leader.state.secrets
    assert len(leader_secrets) == 1
    for unit in app.units:
        assert unit.state.secrets == leader_secrets
    added = juju.add_unit(app)
    assert added.state.secrets == leader_secrets


# Loops


class FlipFlopCharm(ops.CharmBase):
    """Each unit flips its published flag whenever a peer changes: never converges."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_changed)
        framework.observe(self.on['replicas'].relation_changed, self._on_changed)

    def _on_changed(self, _: ops.EventBase):
        relation = self.model.get_relation('replicas')
        assert relation is not None
        flag = relation.data[self.unit].get('flag')
        relation.data[self.unit]['flag'] = 'off' if flag == 'on' else 'on'


def test_settle_reports_a_repeated_dispatch_as_a_loop(juju: testing.Juju):
    juju.deploy(FlipFlopCharm, meta=META, num_units=2)
    with pytest.raises(testing.errors.JujuError, match='same State'):
        juju.settle()
