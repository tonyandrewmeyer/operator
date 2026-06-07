# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for Context.service_ops_history and Context.add_layer_history."""

from __future__ import annotations

import pytest
from scenario import AddLayer, Context, ServiceOp
from scenario.errors import UncaughtCharmError
from scenario.state import Container, State

import ops
from ops import pebble

_TWO_SERVICES_LAYER: pebble.LayerDict = {
    'summary': 'two services',
    'description': '',
    'services': {
        'svc-a': {'override': 'replace', 'startup': 'enabled', 'command': '/bin/sleep 1'},
        'svc-b': {'override': 'replace', 'startup': 'disabled', 'command': '/bin/sleep 1'},
    },
}


def _container_with_layer(name: str = 'foo', *, can_connect: bool = True) -> Container:
    return Container(
        name=name,
        can_connect=can_connect,
        layers={'base': pebble.Layer(_TWO_SERVICES_LAYER)},
    )


class _ContainerActionCharm(ops.CharmBase):
    """A charm whose config_changed handler runs a configurable container action."""

    container_name: str = 'foo'
    action: str = ''

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.config_changed, self._on_config_changed)

    def _on_config_changed(self, _: ops.EventBase):
        container = self.unit.get_container(self.container_name)
        action = self.action
        if action == 'restart-a':
            container.restart('svc-a')
        elif action == 'start-a':
            container.start('svc-a')
        elif action == 'stop-a':
            container.stop('svc-a')
        elif action == 'replan':
            container.replan()
        elif action == 'autostart':
            container.autostart()
        elif action == 'sequence':
            container.start('svc-a')
            container.stop('svc-a')
            container.restart('svc-a', 'svc-b')
        elif action == 'add-layer':
            container.add_layer(
                'extra',
                {
                    'summary': 'extra layer',
                    'services': {
                        'svc-c': {
                            'override': 'replace',
                            'startup': 'enabled',
                            'command': '/bin/sleep 1',
                        },
                    },
                },
            )
        elif action == 'add-two-layers':
            container.add_layer(
                'l1',
                pebble.Layer({
                    'services': {
                        'svc-c': {
                            'override': 'replace',
                            'startup': 'enabled',
                            'command': '/bin/sleep 1',
                        },
                    }
                }),
            )
            container.add_layer(
                'l2',
                pebble.Layer({
                    'services': {
                        'svc-d': {
                            'override': 'replace',
                            'startup': 'disabled',
                            'command': '/bin/sleep 1',
                        },
                    }
                }),
            )


def _make_ctx() -> Context[_ContainerActionCharm]:
    return Context(_ContainerActionCharm, meta={'name': 'foo', 'containers': {'foo': {}}})


def test_restart_records_a_single_op(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'restart-a')
    ctx = _make_ctx()
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    assert ctx.service_ops_history == {
        'foo': [ServiceOp('restart', ('svc-a',))],
    }


def test_start_records_only_start_not_restart(monkeypatch: pytest.MonkeyPatch):
    """Container.start should record a 'start' op, not a 'restart'."""
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'start-a')
    ctx = _make_ctx()
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    ops_for_foo = ctx.service_ops_history['foo']
    assert ServiceOp('start', ('svc-a',)) in ops_for_foo
    assert not any(o.op == 'restart' for o in ops_for_foo)


def test_stop_records_a_stop_op(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'stop-a')
    ctx = _make_ctx()
    container = Container(
        name='foo',
        can_connect=True,
        layers={'base': pebble.Layer(_TWO_SERVICES_LAYER)},
        service_statuses={'svc-a': pebble.ServiceStatus.ACTIVE},
    )
    ctx.run(ctx.on.config_changed(), State(containers={container}))
    assert ctx.service_ops_history['foo'] == [ServiceOp('stop', ('svc-a',))]


def test_replan_records_enabled_services_from_plan(monkeypatch: pytest.MonkeyPatch):
    """replan should record only services with startup=enabled, in plan order."""
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'replan')
    ctx = _make_ctx()
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    # The base class implements replan as calling autostart internally, so both
    # ops appear; both should list only the enabled service.
    ops_for_foo = ctx.service_ops_history['foo']
    assert ServiceOp('replan', ('svc-a',)) in ops_for_foo
    assert ServiceOp('autostart', ('svc-a',)) in ops_for_foo
    # 'replan' is recorded before its internal autostart call.
    assert ops_for_foo.index(ServiceOp('replan', ('svc-a',))) < ops_for_foo.index(
        ServiceOp('autostart', ('svc-a',))
    )


def test_autostart_records_enabled_services(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'autostart')
    ctx = _make_ctx()
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    assert ctx.service_ops_history['foo'] == [ServiceOp('autostart', ('svc-a',))]


def test_get_service_ops_filters_by_service(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'sequence')
    ctx = _make_ctx()
    container = Container(
        name='foo',
        can_connect=True,
        layers={'base': pebble.Layer(_TWO_SERVICES_LAYER)},
        service_statuses={'svc-a': pebble.ServiceStatus.ACTIVE},
    )
    ctx.run(ctx.on.config_changed(), State(containers={container}))
    # All ops on svc-a (start, stop, then restart targeting both):
    a_ops = ctx.get_service_ops('foo', 'svc-a')
    assert a_ops == [
        ServiceOp('start', ('svc-a',)),
        ServiceOp('stop', ('svc-a',)),
        ServiceOp('restart', ('svc-a', 'svc-b')),
    ]
    # svc-b is only touched by the restart call:
    b_ops = ctx.get_service_ops('foo', 'svc-b')
    assert b_ops == [ServiceOp('restart', ('svc-a', 'svc-b'))]
    # Unfiltered returns everything for the container.
    assert ctx.get_service_ops('foo') == ctx.service_ops_history['foo']
    # Unknown container returns an empty list, not a KeyError.
    assert ctx.get_service_ops('does-not-exist') == []


def test_can_not_connect_does_not_record(monkeypatch: pytest.MonkeyPatch):
    """A ConnectionError must not produce a history entry."""
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'restart-a')
    ctx = _make_ctx()
    container = Container(
        name='foo',
        can_connect=False,
        layers={'base': pebble.Layer(_TWO_SERVICES_LAYER)},
    )
    with pytest.raises(UncaughtCharmError) as exc_info:
        ctx.run(ctx.on.config_changed(), State(containers={container}))
    assert isinstance(exc_info.value.__cause__, pebble.ConnectionError)
    assert ctx.service_ops_history == {}
    assert ctx.add_layer_history == {}


def test_multiple_ops_recorded_in_order(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'sequence')
    ctx = _make_ctx()
    container = Container(
        name='foo',
        can_connect=True,
        layers={'base': pebble.Layer(_TWO_SERVICES_LAYER)},
        service_statuses={'svc-a': pebble.ServiceStatus.ACTIVE},
    )
    ctx.run(ctx.on.config_changed(), State(containers={container}))
    assert ctx.service_ops_history['foo'] == [
        ServiceOp('start', ('svc-a',)),
        ServiceOp('stop', ('svc-a',)),
        ServiceOp('restart', ('svc-a', 'svc-b')),
    ]


def test_history_is_reset_between_runs(monkeypatch: pytest.MonkeyPatch):
    """A reused Context should not leak history from one run into the next."""
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'restart-a')
    ctx = _make_ctx()
    state = State(containers={_container_with_layer()})
    ctx.run(ctx.on.config_changed(), state)
    assert ctx.service_ops_history['foo'] == [ServiceOp('restart', ('svc-a',))]

    # Second run with a different action: history should reflect only the new run.
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'stop-a')
    container = Container(
        name='foo',
        can_connect=True,
        layers={'base': pebble.Layer(_TWO_SERVICES_LAYER)},
        service_statuses={'svc-a': pebble.ServiceStatus.ACTIVE},
    )
    ctx.run(ctx.on.config_changed(), State(containers={container}))
    assert ctx.service_ops_history['foo'] == [ServiceOp('stop', ('svc-a',))]


def test_add_layer_records_label_and_layer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'add-layer')
    ctx = _make_ctx()
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    history = ctx.add_layer_history['foo']
    assert len(history) == 1
    assert history[0].label == 'extra'
    assert history[0].combine is False
    assert 'svc-c' in history[0].layer.services


def test_add_layer_preserves_order(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'add-two-layers')
    ctx = _make_ctx()
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    history = ctx.add_layer_history['foo']
    assert [r.label for r in history] == ['l1', 'l2']
    assert 'svc-c' in history[0].layer.services
    assert 'svc-d' in history[1].layer.services


def test_add_layer_history_is_reset_between_runs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ContainerActionCharm, 'action', 'add-layer')
    ctx = _make_ctx()
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    assert ctx.add_layer_history['foo'] == [
        AddLayer(
            label='extra',
            layer=ctx.add_layer_history['foo'][0].layer,
            combine=False,
        )
    ]

    monkeypatch.setattr(_ContainerActionCharm, 'action', 'restart-a')
    ctx.run(ctx.on.config_changed(), State(containers={_container_with_layer()}))
    assert ctx.add_layer_history == {}


def test_service_op_normalises_services_to_tuple():
    """ServiceOp should freeze any sequence-typed services arg to a tuple."""
    op = ServiceOp('start', ['a', 'b'])  # type: ignore[arg-type]
    assert op.services == ('a', 'b')
    assert isinstance(op.services, tuple)
