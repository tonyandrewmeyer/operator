# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the model-level layer: Juju, App, and Unit."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import itertools
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Generator
from typing import Any, Literal

import pytest
from scenario import _deployment

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


def spec(
    charm_type: type[ops.CharmBase],
    meta: dict[str, Any] = META,
    config: dict[str, Any] | None = CONFIG,
) -> testing.CharmSpec[Any]:
    """A CharmSpec with the config options in charmcraft.yaml shape."""
    full = {**meta, 'config': config} if config is not None else meta
    return testing.CharmSpec(charm_type, meta=full)


def deploy_mycharm(
    juju: testing.Juju, *, config_schema: dict[str, Any] = CONFIG, **kwargs: Any
) -> testing.App:
    return juju.deploy(spec(MyCharm, config=config_schema), app='myapp', **kwargs)


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


# Juju-model-name validity: lowercase letters, digits and hyphens, with no
# hyphen at either end.
_VALID_MODEL_NAME = re.compile(r'[a-z0-9]([a-z0-9-]*[a-z0-9])?')


def _identity_for(monkeypatch: pytest.MonkeyPatch, node_id: str) -> tuple[str, str]:
    monkeypatch.setenv('PYTEST_CURRENT_TEST', f'{node_id} (call)')
    with testing.Juju() as j:
        return j.name, j.uuid


def test_default_name_and_uuid_come_from_the_test(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'tests/unit/test_charm.py::test_ingress (call)')
    with testing.Juju() as j:
        assert re.fullmatch(r'test-ingress-[0-9a-f]{8}', j.name)
        assert str(uuid.UUID(j.uuid)) == j.uuid
        # A version 4 UUID, as Juju's are, which some charm libraries check.
        assert str(uuid.UUID(j.uuid, version=4)) == j.uuid
        node_id = 'tests/unit/test_charm.py::test_ingress'
        name_based = uuid.uuid5(_deployment._MODEL_UUID_NAMESPACE, node_id)
        assert j.uuid == str(uuid.UUID(bytes=name_based.bytes, version=4))


def test_default_name_and_uuid_are_the_same_for_the_same_test(monkeypatch: pytest.MonkeyPatch):
    first = _identity_for(monkeypatch, 'tests/unit/test_charm.py::test_ingress')
    second = _identity_for(monkeypatch, 'tests/unit/test_charm.py::test_ingress')
    assert first == second


def test_default_name_and_uuid_differ_between_tests(monkeypatch: pytest.MonkeyPatch):
    ingress = _identity_for(monkeypatch, 'tests/unit/test_charm.py::test_ingress')
    other = _identity_for(monkeypatch, 'tests/unit/test_charm.py::test_other')
    elsewhere = _identity_for(monkeypatch, 'tests/unit/test_other.py::test_ingress')
    assert len({ingress[0], other[0], elsewhere[0]}) == 3
    assert len({ingress[1], other[1], elsewhere[1]}) == 3
    # The name keeps the function name; the hash tells the modules apart.
    assert elsewhere[0].startswith('test-ingress-')


def test_default_name_and_uuid_differ_between_parametrised_cases(
    monkeypatch: pytest.MonkeyPatch,
):
    one = _identity_for(monkeypatch, 'tests/unit/test_charm.py::test_scale[1]')
    two = _identity_for(monkeypatch, 'tests/unit/test_charm.py::test_scale[2]')
    assert one[0].startswith('test-scale-')
    assert two[0].startswith('test-scale-')
    assert one[0] != two[0]
    assert one[1] != two[1]


@pytest.mark.parametrize('phase', ['setup', 'call', 'teardown'])
def test_default_name_and_uuid_ignore_the_test_phase(monkeypatch: pytest.MonkeyPatch, phase: str):
    monkeypatch.setenv('PYTEST_CURRENT_TEST', f'tests/test_x.py::test_a ({phase})')
    with testing.Juju() as j:
        in_phase = j.name, j.uuid
    assert in_phase == _identity_for(monkeypatch, 'tests/test_x.py::test_a')


@pytest.fixture
def default_juju():
    with testing.Juju() as j:
        yield j


def test_a_fixture_s_juju_matches_one_made_in_the_test(default_juju: testing.Juju):
    name, uuid_ = default_juju.name, default_juju.uuid
    assert name.startswith('test-a-fixture-s-juju-matches-')
    default_juju.close()
    with testing.Juju() as j:
        assert (j.name, j.uuid) == (name, uuid_)


def test_a_second_juju_in_a_test_gets_a_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'tests/test_x.py::test_two_models (call)')
    with testing.Juju() as first, testing.Juju() as second:
        assert second.name == f'{first.name}-2'
        assert second.uuid != first.uuid
        with testing.Juju() as third:
            assert third.name == f'{first.name}-3'
            assert len({first.uuid, second.uuid, third.uuid}) == 3
    # Closing a Juju frees its name: the next one is the first again.
    with testing.Juju() as again:
        assert (again.name, again.uuid) == (first.name, first.uuid)


def test_closing_the_first_juju_frees_its_name_for_the_next(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'tests/test_x.py::test_slots (call)')
    first = testing.Juju()
    second = testing.Juju()
    first.close()
    first.close()  # Closing twice doesn't free anything twice.
    with testing.Juju() as third, testing.Juju() as fourth:
        assert third.name == first.name
        assert fourth.name == f'{first.name}-3'
    second.close()


def test_explicit_name_and_uuid_are_kept(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'tests/test_x.py::test_explicit (call)')
    with testing.Juju('my-model', uuid='a2c4e6f8-0000-4000-8000-000000000000') as j:
        assert (j.name, j.uuid) == ('my-model', 'a2c4e6f8-0000-4000-8000-000000000000')
        # Neither is derived, so no slot is held.
        with testing.Juju() as other:
            assert not other.name.endswith('-2')
    with testing.Juju('my-model') as named:
        identity = 'tests/test_x.py::test_explicit'
        assert named.uuid == _deployment._model_identity(identity, 'test_explicit')[1]


_NAMED_TESTS = """
import json, os, pathlib

import pytest

from ops import testing


def record(j):
    out = pathlib.Path(os.environ['OUT_DIR'])
    test = os.environ['PYTEST_CURRENT_TEST'].split(' ')[0].replace('/', '_')
    (out / test.replace(':', '_')).write_text(json.dumps([j.name, j.uuid]))


@pytest.fixture
def juju():
    with testing.Juju() as j:
        yield j


def test_a():
    with testing.Juju() as j:
        record(j)


def test_b(juju):
    record(juju)


@pytest.mark.parametrize('n', [1, 2])
def test_p(n):
    with testing.Juju() as j:
        record(j)
"""


def _run_named_tests(tmp_path: pathlib.Path, name: str, *args: str) -> dict[str, list[str]]:
    out = tmp_path / name
    out.mkdir()
    env = {**os.environ, 'OUT_DIR': str(out)}
    env.pop('PYTEST_XDIST_WORKER', None)
    subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *args],
        cwd=tmp_path / 'project',
        env=env,
        check=True,
        capture_output=True,
    )
    return {f.name: json.loads(f.read_text()) for f in out.iterdir()}


def test_default_name_and_uuid_survive_selection_order_and_xdist(tmp_path: pathlib.Path):
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'test_models.py').write_text(_NAMED_TESTS)
    everything = _run_named_tests(tmp_path, 'all')
    assert len(everything) == 4
    assert len({name for name, _ in everything.values()}) == 4
    assert len({uuid_ for _, uuid_ in everything.values()}) == 4
    assert _run_named_tests(tmp_path, 'again') == everything
    reordered = _run_named_tests(
        tmp_path, 'reordered', 'test_models.py::test_p', 'test_models.py::test_b', '-k', 'not 1'
    )
    assert reordered == {k: v for k, v in everything.items() if 'test_a' not in k and '1' not in k}
    if importlib.util.find_spec('xdist') is not None:
        assert _run_named_tests(tmp_path, 'xdist', '-n', '2') == everything


def _make_juju_here() -> tuple[str, str]:
    with testing.Juju() as j:
        return j.name, j.uuid


def test_outside_pytest_the_caller_names_the_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('PYTEST_CURRENT_TEST')
    name, uuid_ = _make_juju_here()
    identity = f'{__name__}::_make_juju_here'
    assert (name, uuid_) == _deployment._model_identity(identity, '_make_juju_here')
    assert re.fullmatch(r'make-juju-here-[0-9a-f]{8}', name)
    assert _make_juju_here() == (name, uuid_)


@pytest.mark.parametrize(
    'function',
    [
        'test_' + 'very_long_name_' * 10,
        'test_ünïcödé_ñame',
        '__test__double__underscores__',
        '1st_test_starting_with_digits',
        'TestCamelCase',
        '_',
        'тест',
    ],
)
def test_default_name_is_a_valid_model_name(monkeypatch: pytest.MonkeyPatch, function: str):
    name, _ = _identity_for(monkeypatch, f'tests/test_x.py::TestClass::{function}[a-b/c]')
    assert _VALID_MODEL_NAME.fullmatch(name), name
    assert len(name) <= _deployment._MODEL_SLUG_LENGTH + 9
    name, _ = _deployment._model_identity('x', function, slot=12)
    assert _VALID_MODEL_NAME.fullmatch(name), name


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
    assert all(dispatch.unit_name == app.leader.name for dispatch in trace)


def test_non_leader_units_get_leader_settings_changed(juju: testing.Juju):
    deploy_mycharm(juju, num_units=2)
    trace = juju.settle()
    follower_events = [d.event.name for d in trace if d.unit_id == 1]
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
    app = juju.deploy(spec(MyCharm))
    assert app.name == 'myapp'


def test_deploy_creates_containers_and_emits_pebble_ready(juju: testing.Juju):
    meta: dict[str, Any] = {**META, 'containers': {'workload': {}}}
    app = juju.deploy(spec(MyCharm, meta))
    trace = juju.settle()
    assert [d.event.name for d in trace][-1] == 'workload_pebble_ready'
    assert {c.name for c in app.leader.state.containers} == {'workload'}


# config


def test_config_emits_config_changed_on_every_unit(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    juju.settle()
    juju.config(app, {'log_level': 'debug'})
    trace = juju.settle()
    assert [(d.event.name, d.unit_id) for d in trace] == [
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
    assert [d.event.name for d in trace if d.unit_id == 1] == [
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
    assert [d.event.name for d in trace if d.unit_id == 0] == [
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
    assert [(d.event.name, d.unit_id) for d in trace] == [
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
    assert [(d.event.name, d.unit_id) for d in trace] == [
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
    app = machine_juju.deploy(spec(PublishingCharm, config=None), app='myapp', num_units=2)
    machine_juju.settle()
    relation = peer_relation(app.units[0])
    assert 1 in relation.peers_data

    machine_juju.remove_unit(app.units[1])
    machine_juju.settle()
    relation = peer_relation(app.units[0])
    assert relation.peers_data == {}
    assert app.units[0].state.planned_units == 1


PLANNED_UNITS_SEEN: list[tuple[str, str, int]] = []


class PlannedUnitsCharm(ops.CharmBase):
    """Records what planned_units() is in each hook a scale-down runs."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        for event in (
            self.on['replicas'].relation_departed,
            self.on.leader_elected,
            self.on.stop,
            self.on.remove,
        ):
            framework.observe(event, self._record)

    def _record(self, event: ops.EventBase):
        PLANNED_UNITS_SEEN.append((self.unit.name, event.handle.kind, self.app.planned_units()))


@pytest.mark.parametrize('substrate', ['kubernetes', 'lxd'])
def test_remove_unit_lowers_planned_units_before_any_hook(
    substrate: Literal['kubernetes', 'lxd'],
):
    PLANNED_UNITS_SEEN.clear()
    with testing.Juju(type=substrate) as juju:
        app = juju.deploy(spec(PlannedUnitsCharm, config=None), app='myapp', num_units=3)
        juju.settle()
        PLANNED_UNITS_SEEN.clear()

        if substrate == 'kubernetes':
            juju.remove_unit(app, num_units=2)
        else:
            juju.remove_unit(app.units[1], app.units[2])
        # Juju marks the units as dying straight away, so nothing waits for a hook.
        assert [unit.state.planned_units for unit in app.units] == [1, 1, 1]
        juju.settle()

    seen = {(unit, kind) for unit, kind, _ in PLANNED_UNITS_SEEN}
    assert ('myapp/0', 'replicas_relation_departed') in seen
    assert ('myapp/2', 'stop') in seen
    assert ('myapp/1', 'remove') in seen
    assert {planned for _, _, planned in PLANNED_UNITS_SEEN} == {1}
    assert app.units[0].state.planned_units == 1


def test_add_unit_while_a_unit_is_leaving_counts_only_the_staying_units(juju: testing.Juju):
    app = juju.deploy(spec(PlannedUnitsCharm, config=None), app='myapp', num_units=2)
    juju.settle()

    juju.remove_unit(app, num_units=1)
    juju.add_unit(app)
    assert [unit.state.planned_units for unit in app.units] == [2, 2, 2]


def test_remove_unit_requires_at_least_one_argument(juju: testing.Juju):
    with pytest.raises(testing.errors.JujuError, match='at least one'):
        juju.remove_unit()


def test_remove_unit_rejects_mixing_apps_and_units(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=2)
    with pytest.raises(testing.errors.JujuError, match='not a mix'):
        machine_juju.remove_unit(app, app.units[1])


# Peer convergence


def test_peer_unit_databags_propagate(juju: testing.Juju):
    app = juju.deploy(spec(PublishingCharm, config=None), app='myapp', num_units=2)
    juju.settle()
    assert peer_relation(app.units[0]).peers_data[1]['ready'] == 'yes'
    assert peer_relation(app.units[1]).peers_data[0]['ready'] == 'yes'


def test_peer_databag_writes_drive_relation_changed(juju: testing.Juju):
    app = juju.deploy(spec(PublishingCharm, config=None), app='myapp', num_units=2)
    trace = juju.settle()
    assert 'replicas_relation_changed' in [d.event.name for d in trace]
    # Each unit observed the other, which is only possible if the write made it
    # across and woke the peer up.
    for unit in app.units:
        assert unit.state.unit_status.name == 'active'
        assert 'myapp/' in unit.state.unit_status.message


def test_leader_app_databag_propagates_to_followers(juju: testing.Juju):
    app = juju.deploy(spec(PublishingCharm, config=None), app='myapp', num_units=2)
    juju.settle()
    for unit in app.units:
        assert peer_relation(unit).local_app_data['cluster'] == 'formed'


def test_settle_raises_when_juju_does_not_converge(juju: testing.Juju):
    juju.deploy(spec(ChattyCharm, config=None), app='myapp', num_units=2)
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
    first = trace[0]
    assert first.event.name == 'install'
    assert (first.app, first.unit_id, first.unit_name) == (app.name, 0, app.leader.name)
    assert isinstance(first.state_in, testing.State)
    assert isinstance(first.state_out, testing.State)


def test_a_dispatch_s_state_in_is_the_previous_state_out_for_one_unit(juju: testing.Juju):
    deploy_mycharm(juju)
    trace = juju.settle()
    for before, after in itertools.pairwise(trace):
        assert after.state_in == before.state_out


def test_dispatches_compare_by_value_and_hold_no_live_handles(juju: testing.Juju):
    app = deploy_mycharm(juju)
    trace = juju.settle()
    install = trace[0]
    assert install == dataclasses.replace(install)
    juju.config(app, {'log_level': 'debug'})
    juju.settle()
    # The record still describes the unit as it was.
    assert install.state_out.config != app.leader.state.config
    assert juju.apps[install.app] is app


def test_dispatch_to_context_runs_the_dispatch_again(juju: testing.Juju):
    deploy_mycharm(juju)
    trace = juju.settle()
    start = trace[-1]
    assert start.event.name == 'start'
    ctx = start.to_context()
    state_out = ctx.run(start.event, start.state_in)
    assert state_out.unit_status == start.state_out.unit_status


def test_settle_trace_states_are_post_dispatch_snapshots(juju: testing.Juju):
    deploy_mycharm(juju)
    trace = juju.settle()
    assert trace[-1].state_out.unit_status == testing.ActiveStatus('start:info')


def test_settle_is_a_no_op_when_the_queue_is_empty(juju: testing.Juju):
    deploy_mycharm(juju)
    juju.settle()
    assert juju.settle() == []


def test_settle_is_deterministic():
    def run() -> list[str]:
        with testing.Juju(model_name='m') as j:
            j.deploy(spec(PublishingCharm, config=None), app='myapp', num_units=3)
            return [f'{dispatch.event.name}@{dispatch.unit_name}' for dispatch in j.settle()]

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
        app = j._deploy(
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
        alpha = j._deploy(
            _ISOLATION / 'charms' / 'alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        beta = j._deploy(
            _ISOLATION / 'charms' / 'beta',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v2'),),
        )
        j.settle()
        assert alpha.leader.state.unit_status.message.startswith('confdep=1.0')
        assert beta.leader.state.unit_status.message.startswith('confdep=2.0')


def test_isolated_app_reads_metadata_from_disk():
    with testing.Juju(model_name='iso') as j:
        app = j._deploy(
            _ISOLATION / 'charms' / 'alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        assert app.meta['name'] == 'alpha'


def test_isolated_app_accepts_a_relative_string_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(_ISOLATION)
    with testing.Juju(model_name='iso') as j:
        app = j._deploy(
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
        app = j._deploy(
            _ISOLATION / 'charms' / 'pid',
            python_executable=sys.executable,
            num_units=3,
        )
        j.settle()
        pids = {unit.state.unit_status.message for unit in app.units}
        assert len(pids) == 1
        assert pids != {f'pid={os.getpid()}'}  # and it is not the test process


def test_isolated_app_runs_config(tmp_path: pathlib.Path):
    # config dispatches through the same runner as the startup sequence, but
    # config-changed outside startup is worth its own isolated check.
    root = tmp_path / 'alpha'
    shutil.copytree(_ISOLATION / 'charms' / 'alpha', root)
    (root / 'config.yaml').write_text('options:\n  log_level: {type: string, default: info}\n')
    with testing.Juju(model_name='iso') as j:
        app = j._deploy(
            root,
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        j.settle()
        j.config(app, {'log_level': 'debug'})
        trace = j.settle()
        assert [d.event.name for d in trace] == ['config_changed']
        assert app.leader.state.config == {'log_level': 'debug'}


# Leadership


def test_removing_the_leader_elects_a_new_one(machine_juju: testing.Juju):
    app = deploy_mycharm(machine_juju, num_units=3)
    machine_juju.settle()
    old_leader = app.leader
    machine_juju.remove_unit(old_leader)
    trace = machine_juju.settle()
    assert [u.id for u in app.units] == [1, 2]
    assert app.leader.id == 1
    assert app.leader.state.leader
    assert not app.units[1].state.leader
    elected = [d.unit_name for d in trace if d.event.name == 'leader_elected']
    assert elected == [f'{app.name}/1']
    # The new leader is elected once the old one has gone.
    names = [(d.event.name, d.unit_name) for d in trace]
    assert names.index(('remove', f'{app.name}/0')) < names.index((
        'leader_elected',
        f'{app.name}/1',
    ))


def test_scaling_down_past_a_moved_leader_elects_another(juju: testing.Juju):
    app = deploy_mycharm(juju, num_units=2)
    juju.settle()
    app._leader_id = 1  # As if leadership had moved.
    juju.remove_unit(app, num_units=1)
    juju.settle()
    assert [u.id for u in app.units] == [0]
    assert app.leader.id == 0


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


class NoMetadataCharm(ops.CharmBase):
    pass


def test_a_charm_class_with_no_metadata_needs_a_charm_spec(juju: testing.Juju):
    with pytest.raises(testing.errors.JujuError, match='Wrap it in a CharmSpec'):
        juju.deploy(NoMetadataCharm)


def test_a_charm_spec_can_be_deployed_more_than_once(juju: testing.Juju):
    charm = spec(MyCharm)
    one = juju.deploy(charm, app='one')
    two = juju.deploy(charm, app='two')
    juju.settle()
    assert one.leader.state.unit_status == testing.ActiveStatus('start:info')
    assert two.leader.state.unit_status == testing.ActiveStatus('start:info')


def test_app_meta_is_charmcraft_shaped(juju: testing.Juju, tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'ondisk', PID_CHARM)
    (root / 'config.yaml').write_text('options:\n  port: {type: int, default: 80}\n')
    (root / 'actions.yaml').write_text('backup: {}\n')
    app = juju.deploy(root)
    assert app.meta == {
        'name': 'ondisk',
        'config': {'options': {'port': {'type': 'int', 'default': 80}}},
        'actions': {'backup': {}},
    }


@pytest.mark.parametrize('kwargs', [{'isolated': True}, {'requirements': 'requirements.txt'}])
def test_isolation_needs_a_charm_on_disk(juju: testing.Juju, kwargs: dict[str, Any]):
    for charm in (spec(MyCharm), MyCharm):
        with pytest.raises(testing.errors.JujuError, match='charm on disk'):
            juju.deploy(charm, **kwargs)


def test_requirements_needs_isolated(juju: testing.Juju, tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'ondisk', PID_CHARM)
    with pytest.raises(testing.errors.JujuError, match='isolated=True'):
        juju.deploy(root, requirements=tmp_path / 'requirements.txt')


def test_building_an_isolated_environment_is_not_implemented(
    juju: testing.Juju, tmp_path: pathlib.Path
):
    root = write_charm(tmp_path / 'ondisk', PID_CHARM)
    with pytest.raises(NotImplementedError, match='isolated=True'):
        juju.deploy(root, isolated=True)
    with pytest.raises(NotImplementedError, match='isolated=True'):
        juju.deploy(root, isolated=True, requirements=tmp_path / 'requirements.txt')


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
        app = juju._deploy(root, num_units=2, **kwargs)
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
    app = juju.deploy(testing.CharmSpec(TrustCharm, meta={'name': 'trusty'}), trust=trust)
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
    app = juju.deploy(
        testing.CharmSpec(ExecCharm, meta=EXEC_META), state_template=EXEC_TEMPLATE, num_units=2
    )
    juju.settle()
    for unit in app.units:
        assert unit.state.unit_status == testing.ActiveStatus('updated')


def test_state_template_reaches_units_added_later(juju: testing.Juju):
    app = juju.deploy(testing.CharmSpec(ExecCharm, meta=EXEC_META), state_template=EXEC_TEMPLATE)
    juju.settle()
    unit = juju.add_unit(app)
    juju.settle()
    assert unit.state.unit_status == testing.ActiveStatus('updated')


def test_state_template_leaves_the_juju_owned_fields_to_juju(juju: testing.Juju):
    template = testing.State(workload_version='1.2.3')
    app = juju.deploy(spec(MyCharm), state_template=template)
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
        juju.deploy(spec(MyCharm), state_template=template)


# Application-owned secrets


class SecretCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.leader_elected, self._on_leader_elected)

    def _on_leader_elected(self, _: ops.EventBase):
        self.app.add_secret({'password': 'hunter2'}, label='admin')


def test_application_secrets_reach_every_unit(juju: testing.Juju):
    app = juju.deploy(spec(SecretCharm, config=None), num_units=2)
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
    juju.deploy(spec(FlipFlopCharm, config=None), num_units=2)
    with pytest.raises(testing.errors.JujuError, match='same State'):
        juju.settle()


# CharmSpec mocking


class ReportsMockingCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_install)

    def _on_install(self, _: ops.EventBase):
        self.unit.status = ops.ActiveStatus(f'mocked={_MOCK_STATE["open"]}')


_MOCK_STATE = {'open': False, 'entered': 0}


@contextlib.contextmanager
def _flag_mocking() -> Generator[None]:
    _MOCK_STATE['open'] = True
    _MOCK_STATE['entered'] += 1
    try:
        yield
    finally:
        _MOCK_STATE['open'] = False


def test_charm_spec_mocking_is_open_around_each_dispatch(juju: testing.Juju):
    _MOCK_STATE['entered'] = 0
    charm = testing.CharmSpec(ReportsMockingCharm, meta={'name': 'mocky'}, mocking=_flag_mocking)
    app = juju.deploy(charm, num_units=2)
    trace = juju.settle()
    assert app.leader.state.unit_status == testing.ActiveStatus('mocked=True')
    assert {'open': False, 'entered': len(trace)} == _MOCK_STATE


# to_context


ACTION_CHARM = """
    import ops


    class ActionCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.backup_action, self._on_backup)

        def _on_backup(self, event):
            event.set_results({'unit': self.unit.name, 'level': self.config['log_level']})
"""


def write_action_charm(root: pathlib.Path) -> pathlib.Path:
    return write_charm(
        root,
        ACTION_CHARM,
        metadata='name: actor\n',
        files={
            'config.yaml': 'options:\n  log_level: {type: string, default: info}\n',
            'actions.yaml': 'backup: {}\n',
        },
    )


def test_to_context_runs_an_action_on_a_unit(juju: testing.Juju, tmp_path: pathlib.Path):
    app = juju.deploy(write_action_charm(tmp_path / 'actor'), num_units=2)
    juju.settle()
    juju.config(app, {'log_level': 'debug'})
    juju.settle()
    unit = app.units[1]
    ctx = unit.to_context()
    ctx.run(ctx.on.action('backup'), unit.state)
    assert ctx.action_results == {'unit': 'actor/1', 'level': 'debug'}


def test_to_context_does_not_feed_back_into_juju(juju: testing.Juju):
    app = deploy_mycharm(juju)
    juju.settle()
    before = app.leader.state
    ctx = app.leader.to_context()
    state_out = ctx.run(ctx.on.config_changed(), dataclasses.replace(before, config={}))
    assert state_out.unit_status != before.unit_status
    assert app.leader.state is before
    assert juju.settle() == []


def test_to_context_is_new_each_time(juju: testing.Juju):
    app = deploy_mycharm(juju)
    juju.settle()
    first = app.leader.to_context()
    first.run(first.on.start(), app.leader.state)
    assert app.leader.to_context() is not first


def test_to_context_carries_the_app_settings(juju: testing.Juju):
    app = juju.deploy(spec(MyCharm), app='renamed', trust=True, num_units=2)
    ctx = app.units[1].to_context()
    assert ctx.app_name == 'renamed'
    assert ctx.unit_id == 1
    assert ctx.app_trusted
    state_out = ctx.run(ctx.on.start(), app.units[1].state)
    assert state_out.unit_status == testing.ActiveStatus('start:info')


def test_to_context_imports_an_isolated_charm(tmp_path: pathlib.Path):
    root = write_action_charm(tmp_path / 'actor')
    with testing.Juju() as j:
        app = j._deploy(root, python_executable=sys.executable)
        j.settle()
        ctx = app.leader.to_context()
        ctx.run(ctx.on.action('backup'), app.leader.state)
        assert ctx.action_results is not None
        assert ctx.action_results['unit'] == 'actor/0'


def test_to_context_raises_isolation_error_if_the_charm_cannot_be_imported():
    # alpha imports confdep, which is only on the worker's sys.path.
    with testing.Juju() as j:
        app = j._deploy(
            _ISOLATION / 'charms' / 'alpha',
            python_executable=sys.executable,
            extra_sys_path=(str(_ISOLATION / 'deps' / 'confdep_v1'),),
        )
        with pytest.raises(testing.errors.IsolationError, match='test process'):
            app.leader.to_context()
