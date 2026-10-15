# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

from __future__ import annotations

import datetime
import json
import pathlib

import pytest
from scenario import state
from scenario._state_serde import _decode_state, _encode_state

from ops import SecretRotate, pebble


def test_default_state_round_trips_through_json():
    original = state.State()

    payload = _encode_state(original)
    restored = _decode_state(payload)

    assert json.loads(payload)
    assert restored == original


def test_state_special_types_round_trip(tmp_path: pathlib.Path):
    now = datetime.datetime(2026, 10, 14, 12, 30, tzinfo=datetime.timezone.utc)
    secret = state.Secret(
        {'password': 'old'},
        latest_content={'password': 'new'},
        id='secret:123',
        owner='app',
        remote_grants={7: {'remote', 'remote/0'}},
        expire=now,
        rotate=SecretRotate.DAILY,
    )
    object.__setattr__(secret, '_tracked_revision', 2)
    object.__setattr__(secret, '_latest_revision', 3)
    notice = state.Notice(
        'example.com/notice',
        id='10',
        type=pebble.NoticeType.CUSTOM,
        first_occurred=now,
        last_occurred=now + datetime.timedelta(seconds=1),
        last_repeated=now + datetime.timedelta(seconds=2),
        repeat_after=datetime.timedelta(minutes=5),
        expire_after=datetime.timedelta(hours=1),
    )
    layer = pebble.Layer({
        'summary': 'test layer',
        'services': {
            'web': {
                'override': 'replace',
                'command': '/bin/web',
                'startup': 'enabled',
            }
        },
    })
    original = state.State(
        config={'count': 3, 'enabled': True},
        relations={
            state.Relation(
                'database',
                'postgresql_client',
                id=7,
                remote_units_data={0: {'host': 'db.example.com'}},
            ),
            state.PeerRelation('peers', 'cluster', id=8, peers_data={1: {'role': 'replica'}}),
        },
        networks={
            state.Network(
                'database',
                [state.BindAddress([state.Address('192.0.2.1')], interface_name='eth0')],
            )
        },
        containers={
            state.Container(
                'workload',
                can_connect=True,
                _base_plan={'__type__': 'user data'},
                layers={'charm': layer},
                service_statuses={'web': pebble.ServiceStatus.ACTIVE},
                mounts={
                    'config': state.Mount(
                        location=pathlib.PurePosixPath('/etc/web/config'),
                        source=tmp_path / 'config',
                    )
                },
                execs={state.Exec(('echo', 'hello'), stdout='hello\n', _change_id=12)},
                notices=[notice],
                check_infos={
                    state.CheckInfo(
                        'ready',
                        level=pebble.CheckLevel.READY,
                        startup=pebble.CheckStartup.ENABLED,
                        status=pebble.CheckStatus.UP,
                        change_id=pebble.ChangeID('13'),
                    )
                },
            )
        },
        storages={state.Storage('data', index=2)},
        opened_ports={state.TCPPort(8080), state.ICMPPort()},
        secrets={secret},
        resources={state.Resource(name='binary', path=tmp_path / 'web')},
        deferred=[
            state.DeferredEvent(
                'MyCharm/on/start[1]',
                'MyCharm',
                '_on_start',
                snapshot_data={'payload': b'\x00\xff'},
            )
        ],
        stored_states={
            state.StoredState(
                content={
                    'bytes': b'\x00\xff',
                    'tuple': ('one', 2),
                    'set': {'one', 'two'},
                    'list': ['one', 2],
                    '__type__': {'nested': 'value'},
                }
            )
        },
        app_status=state.ActiveStatus('ready'),
        unit_status=state.MaintenanceStatus('upgrading'),
    )

    restored = _decode_state(_encode_state(original))

    assert restored.config == original.config
    assert restored.relations == original.relations
    assert restored.networks == original.networks
    assert restored.storages == original.storages
    assert restored.opened_ports == original.opened_ports
    assert restored.resources == original.resources
    assert restored.deferred == original.deferred
    assert restored.app_status == original.app_status
    assert restored.unit_status == original.unit_status

    restored_secret = restored.get_secret(id='secret:123')
    assert restored_secret == secret
    assert restored_secret.remote_grants == {7: frozenset({'remote', 'remote/0'})}
    assert restored_secret._tracked_revision == 2
    assert restored_secret._latest_revision == 3

    restored_stored_state = restored.get_stored_state('_stored')
    assert restored_stored_state.content == original.get_stored_state('_stored').content
    assert isinstance(restored_stored_state.content['bytes'], bytes)
    assert isinstance(restored_stored_state.content['tuple'], tuple)
    assert isinstance(restored_stored_state.content['set'], set)
    assert isinstance(restored_stored_state.content['list'], list)

    restored_container = restored.get_container('workload')
    original_container = original.get_container('workload')
    assert restored_container._base_plan == original_container._base_plan
    assert restored_container.layers['charm'].to_dict() == layer.to_dict()
    assert restored_container.service_statuses == original_container.service_statuses
    assert restored_container.mounts == original_container.mounts
    assert restored_container.execs == original_container.execs
    assert restored_container.notices == original_container.notices
    assert restored_container.check_infos == original_container.check_infos


def test_unsupported_value_reports_state_path():
    original = state.State(stored_states={state.StoredState(content={'bad': object()})})

    with pytest.raises(TypeError, match=r'object.*state\.stored_states\[\]\.content\.bad'):
        _encode_state(original)


def test_decode_rejects_non_state_json():
    with pytest.raises(TypeError, match='not a State'):
        _decode_state('{}')
