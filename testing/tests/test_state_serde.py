# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit tests for the typed State JSON encoder/decoder (Saddle step 2)."""

from __future__ import annotations

import base64
import datetime
import json
import pathlib

import pytest
from ops import SecretRotate, pebble

from scenario._state_serde import (
    STATE_SCHEMA_VERSION,
    StateSchemaVersionError,
    _decode_v1,
    decode_state,
    encode_state,
)
from scenario.state import (
    ActiveStatus,
    BlockedStatus,
    CheckInfo,
    Container,
    DeferredEvent,
    Exec,
    MaintenanceStatus,
    Model,
    Mount,
    Network,
    Notice,
    PeerRelation,
    Relation,
    Resource,
    Secret,
    State,
    Storage,
    StoredState,
    TCPPort,
    UnknownStatus,
    WaitingStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roundtrip(state: State) -> State:
    return decode_state(encode_state(state))


# ---------------------------------------------------------------------------
# Leaf-type round-trips
# ---------------------------------------------------------------------------


class TestFrozensetRoundtrip:
    def test_empty(self):
        state = State()
        out = _roundtrip(state)
        assert out.relations == frozenset()
        assert out.containers == frozenset()

    def test_with_relations(self):
        state = State(
            relations=frozenset([
                Relation(endpoint='db', remote_app_name='pg'),
            ])
        )
        out = _roundtrip(state)
        assert len(out.relations) == 1
        rel = next(iter(out.relations))
        assert rel.endpoint == 'db'
        assert rel.remote_app_name == 'pg'


class TestDatetimeRoundtrip:
    def test_naive_datetime(self):
        dt = datetime.datetime(2030, 6, 15, 12, 0, 0)
        state = State(secrets=frozenset([
            Secret(
                tracked_content={'k': 'v'},
                expire=dt,
            )
        ]))
        out = _roundtrip(state)
        secret = next(iter(out.secrets))
        assert secret.expire == dt

    def test_aware_datetime(self):
        dt = datetime.datetime(2030, 1, 1, tzinfo=datetime.timezone.utc)
        state = State(secrets=frozenset([
            Secret(tracked_content={'k': 'v'}, expire=dt)
        ]))
        out = _roundtrip(state)
        assert next(iter(out.secrets)).expire == dt

    def test_none_datetime(self):
        state = State(secrets=frozenset([Secret(tracked_content={'k': 'v'})]))
        out = _roundtrip(state)
        assert next(iter(out.secrets)).expire is None


class TestTimedeltaRoundtrip:
    def test_repeat_after(self):
        td = datetime.timedelta(hours=2, minutes=30)
        notice = Notice(key='example.com/test', repeat_after=td)
        state = State(containers=frozenset([
            Container(name='c', notices=[notice])
        ]))
        out = _roundtrip(state)
        container = next(iter(out.containers))
        assert container.notices[0].repeat_after == td

    def test_expire_after(self):
        td = datetime.timedelta(days=7)
        notice = Notice(key='example.com/test', expire_after=td)
        state = State(containers=frozenset([
            Container(name='c', notices=[notice])
        ]))
        out = _roundtrip(state)
        assert next(iter(out.containers)).notices[0].expire_after == td

    def test_none_timedelta(self):
        notice = Notice(key='example.com/test')
        state = State(containers=frozenset([Container(name='c', notices=[notice])]))
        out = _roundtrip(state)
        assert next(iter(out.containers)).notices[0].repeat_after is None


class TestPathRoundtrip:
    def test_path_in_resource(self):
        p = pathlib.Path('/tmp/resource.tar.gz')
        state = State(resources=frozenset([Resource(name='oci-image', path=p)]))
        out = _roundtrip(state)
        resource = next(iter(out.resources))
        assert resource.path == p
        assert isinstance(resource.path, pathlib.Path)

    def test_string_path_stays_string(self):
        state = State(resources=frozenset([Resource(name='bin', path='/usr/bin/tool')]))
        out = _roundtrip(state)
        resource = next(iter(out.resources))
        assert resource.path == '/usr/bin/tool'
        assert isinstance(resource.path, str)


class TestPurePosixPathRoundtrip:
    def test_pure_posix_path_in_mount(self):
        loc = pathlib.PurePosixPath('/etc/config')
        state = State(containers=frozenset([
            Container(
                name='myapp',
                mounts={'cfg': Mount(location=loc, source=pathlib.Path('/tmp/cfg'))},
            )
        ]))
        out = _roundtrip(state)
        container = next(iter(out.containers))
        mount = container.mounts['cfg']
        assert mount.location == loc
        assert isinstance(mount.location, pathlib.PurePosixPath)

    def test_string_location_stays_string(self):
        state = State(containers=frozenset([
            Container(
                name='myapp',
                mounts={'cfg': Mount(location='/etc/config', source='/tmp/cfg')},
            )
        ]))
        out = _roundtrip(state)
        container = next(iter(out.containers))
        assert container.mounts['cfg'].location == '/etc/config'
        assert isinstance(container.mounts['cfg'].location, str)


class TestPebbleEnumRoundtrip:
    def test_service_status(self):
        state = State(containers=frozenset([
            Container(
                name='app',
                service_statuses={'svc': pebble.ServiceStatus.ACTIVE},
            )
        ]))
        out = _roundtrip(state)
        container = next(iter(out.containers))
        assert container.service_statuses['svc'] is pebble.ServiceStatus.ACTIVE

    def test_notice_type(self):
        notice = Notice(key='example.com/n', type=pebble.NoticeType.CUSTOM)
        state = State(containers=frozenset([Container(name='c', notices=[notice])]))
        out = _roundtrip(state)
        assert next(iter(out.containers)).notices[0].type is pebble.NoticeType.CUSTOM

    def test_check_status(self):
        ci = CheckInfo(name='http', status=pebble.CheckStatus.DOWN, failures=3)
        state = State(containers=frozenset([
            Container(name='app', check_infos=frozenset([ci]))
        ]))
        out = _roundtrip(state)
        container = next(iter(out.containers))
        info = container.get_check_info('http')
        assert info.status is pebble.CheckStatus.DOWN
        assert info.failures == 3

    def test_check_level(self):
        ci = CheckInfo(name='ready', level=pebble.CheckLevel.READY)
        state = State(containers=frozenset([Container(name='app', check_infos=frozenset([ci]))]))
        out = _roundtrip(state)
        info = next(iter(out.containers)).get_check_info('ready')
        assert info.level is pebble.CheckLevel.READY

    def test_secret_rotate(self):
        state = State(secrets=frozenset([
            Secret(tracked_content={'k': 'v'}, rotate=SecretRotate.HOURLY, owner='app')
        ]))
        out = _roundtrip(state)
        secret = next(iter(out.secrets))
        assert secret.rotate is SecretRotate.HOURLY


class TestPebbleLayerRoundtrip:
    def test_layer_in_container(self):
        layer = pebble.Layer({
            'services': {
                'app': {
                    'command': '/bin/app --port 8080',
                    'startup': 'enabled',
                    'override': 'replace',
                }
            }
        })
        state = State(containers=frozenset([
            Container(name='app', layers={'base': layer})
        ]))
        out = _roundtrip(state)
        container = next(iter(out.containers))
        assert 'app' in container.layers['base'].services

    def test_empty_layer(self):
        layer = pebble.Layer({})
        state = State(containers=frozenset([Container(name='c', layers={'l': layer})]))
        out = _roundtrip(state)
        assert 'l' in next(iter(out.containers)).layers


class TestIntKeyedDictRoundtrip:
    def test_remote_grants(self):
        state = State(secrets=frozenset([
            Secret(
                tracked_content={'pass': 'abc'},
                owner='app',
                remote_grants={0: {'related-app'}, 2: {'other-app/0'}},
            )
        ]))
        out = _roundtrip(state)
        secret = next(iter(out.secrets))
        assert secret.remote_grants[0] == {'related-app'}
        assert secret.remote_grants[2] == {'other-app/0'}
        assert all(isinstance(k, int) for k in secret.remote_grants)

    def test_remote_units_data(self):
        state = State(relations=frozenset([
            Relation(
                endpoint='db',
                remote_units_data={0: {'key': 'val'}, 1: {'key': 'other'}},
            )
        ]))
        out = _roundtrip(state)
        rel = next(iter(out.relations))
        assert rel.remote_units_data[0] == {'key': 'val'}
        assert rel.remote_units_data[1] == {'key': 'other'}
        assert all(isinstance(k, int) for k in rel.remote_units_data)


# ---------------------------------------------------------------------------
# StoredState escape hatch
# ---------------------------------------------------------------------------


class TestBytesEscapeHatch:
    def test_bytes_in_stored_state(self):
        data = b'\x00\x01\x02\xff'
        state = State(stored_states=frozenset([
            StoredState(content={'blob': data})
        ]))
        out = _roundtrip(state)
        ss = next(iter(out.stored_states))
        assert ss.content['blob'] == data
        assert isinstance(ss.content['blob'], bytes)

    def test_empty_bytes(self):
        state = State(stored_states=frozenset([StoredState(content={'b': b''})]))
        out = _roundtrip(state)
        assert next(iter(out.stored_states)).content['b'] == b''


class TestTupleEscapeHatch:
    def test_tuple_in_stored_state(self):
        pair = (1, 'hello', True)
        state = State(stored_states=frozenset([StoredState(content={'pair': pair})]))
        out = _roundtrip(state)
        ss = next(iter(out.stored_states))
        assert ss.content['pair'] == pair
        assert isinstance(ss.content['pair'], tuple)

    def test_nested_tuple(self):
        nested = ((1, 2), (3, 4))
        state = State(stored_states=frozenset([StoredState(content={'n': nested})]))
        out = _roundtrip(state)
        assert next(iter(out.stored_states)).content['n'] == nested

    def test_list_stays_list(self):
        lst = [1, 2, 3]
        state = State(stored_states=frozenset([StoredState(content={'lst': lst})]))
        out = _roundtrip(state)
        ss_content = next(iter(out.stored_states)).content['lst']
        assert ss_content == lst
        assert isinstance(ss_content, list)


class TestSetEscapeHatch:
    def test_set_in_stored_state(self):
        s = {'a', 'b', 'c'}
        state = State(stored_states=frozenset([StoredState(content={'s': s})]))
        out = _roundtrip(state)
        ss = next(iter(out.stored_states))
        assert ss.content['s'] == s
        assert isinstance(ss.content['s'], set)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestEncoderTypeError:
    def test_unrecognised_type_raises(self):
        state = State(stored_states=frozenset([
            StoredState(content={'obj': object()})
        ]))
        with pytest.raises(TypeError, match="No JSON encoding for type 'object'"):
            encode_state(state)

    def test_error_includes_path(self):
        state = State(stored_states=frozenset([
            StoredState(content={'bad': object()})
        ]))
        with pytest.raises(TypeError, match="path"):
            encode_state(state)

    def test_unrecognised_enum_raises(self):
        import enum

        class MyEnum(enum.Enum):
            VAL = 'val'

        state = State(stored_states=frozenset([
            StoredState(content={'e': MyEnum.VAL})
        ]))
        with pytest.raises(TypeError, match="Unrecognised enum type"):
            encode_state(state)


class TestDecoderTypeError:
    def test_unknown_type_tag_raises(self):
        with pytest.raises(TypeError, match="Unknown wire type tag"):
            _decode_v1({'__t__': '__no_such_type__', 'v': None})

    def test_unknown_dataclass_raises(self):
        with pytest.raises(TypeError, match="Unknown dataclass"):
            _decode_v1({'__t__': 'dc', 'cls': 'NoSuchClass', 'f': {}})

    def test_unknown_enum_class_raises(self):
        with pytest.raises(TypeError, match="Unknown enum class"):
            _decode_v1({'__t__': 'enum', 'cls': 'NoSuchEnum', 'name': 'FOO'})


class TestSchemaVersionMismatch:
    def test_unknown_version_raises(self):
        payload = json.dumps({'state_schema_version': 9999, 'state': {}})
        with pytest.raises(StateSchemaVersionError, match="9999"):
            decode_state(payload)

    def test_missing_version_raises(self):
        payload = json.dumps({'state': {}})
        with pytest.raises(StateSchemaVersionError):
            decode_state(payload)

    def test_current_version_accepted(self):
        state = State()
        encoded = encode_state(state)
        data = json.loads(encoded)
        assert data['state_schema_version'] == STATE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Status round-trips
# ---------------------------------------------------------------------------


class TestStatusRoundtrip:
    @pytest.mark.parametrize('status,cls', [
        (ActiveStatus('ready'), ActiveStatus),
        (BlockedStatus('needs config'), BlockedStatus),
        (WaitingStatus('waiting for db'), WaitingStatus),
        (MaintenanceStatus('updating'), MaintenanceStatus),
        (UnknownStatus(), UnknownStatus),
    ])
    def test_status_roundtrip(self, status, cls):
        state = State(unit_status=status, app_status=status)
        out = _roundtrip(state)
        assert isinstance(out.unit_status, cls)
        assert out.unit_status == status


# ---------------------------------------------------------------------------
# Full State round-trip (non-trivial fixture)
# ---------------------------------------------------------------------------


class TestFullStateRoundtrip:
    def test_non_trivial_state(self):
        _first = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        state = State(
            config={'log_level': 'debug', 'port': 8080, 'enabled': True},
            relations=frozenset([
                Relation(
                    endpoint='db',
                    remote_app_name='postgresql',
                    remote_units_data={0: {'host': '10.0.0.1'}, 1: {'host': '10.0.0.2'}},
                    remote_app_data={'version': '14'},
                ),
                PeerRelation(
                    endpoint='peers',
                    peers_data={1: {'leader': 'true'}},
                ),
            ]),
            containers=frozenset([
                Container(
                    name='myapp',
                    can_connect=True,
                    layers={
                        'base': pebble.Layer({
                            'services': {
                                'app': {
                                    'command': '/bin/app',
                                    'startup': 'enabled',
                                    'override': 'replace',
                                }
                            }
                        })
                    },
                    service_statuses={'app': pebble.ServiceStatus.ACTIVE},
                    notices=[
                        Notice(
                            key='example.com/event',
                            type=pebble.NoticeType.CUSTOM,
                            repeat_after=datetime.timedelta(minutes=30),
                            first_occurred=_first,
                            last_occurred=_first,
                            last_repeated=_first,
                        )
                    ],
                    check_infos=frozenset([
                        CheckInfo(
                            name='http',
                            level=pebble.CheckLevel.ALIVE,
                            status=pebble.CheckStatus.UP,
                        )
                    ]),
                )
            ]),
            secrets=frozenset([
                Secret(
                    tracked_content={'password': 'hunter2'},
                    owner='app',
                    expire=datetime.datetime(2030, 12, 31, tzinfo=datetime.timezone.utc),
                    rotate=SecretRotate.DAILY,
                    remote_grants={0: {'related-app'}},
                )
            ]),
            resources=frozenset([
                Resource(name='oci-image', path=pathlib.Path('/tmp/image.tar'))
            ]),
            stored_states=frozenset([
                StoredState(
                    name='_stored',
                    owner_path='MyCharm',
                    content={
                        'count': 42,
                        'raw': b'\xde\xad\xbe\xef',
                        'coords': (1, 2, 3),
                        'tags': {'alpha', 'beta'},
                        'mapping': {'nested': True},
                    },
                )
            ]),
            opened_ports=frozenset([TCPPort(8080)]),
            leader=True,
            unit_status=ActiveStatus('ready'),
            app_status=ActiveStatus('running'),
            workload_version='1.2.3',
            planned_units=3,
            deferred=[
                DeferredEvent(
                    handle_path='MyCharm/on/config_changed[1]',
                    owner='MyCharm',
                    observer='_on_config_changed',
                )
            ],
        )

        out = _roundtrip(state)

        # Config
        assert out.config == state.config

        # Relations
        db_in = next(r for r in state.relations if r.endpoint == 'db')
        db_out = next(r for r in out.relations if r.endpoint == 'db')
        assert db_out.remote_units_data == db_in.remote_units_data
        assert db_out.remote_app_data == db_in.remote_app_data
        assert all(isinstance(k, int) for k in db_out.remote_units_data)

        # Peers
        peer_in = next(r for r in state.relations if r.endpoint == 'peers')
        peer_out = next(r for r in out.relations if r.endpoint == 'peers')
        assert peer_out.peers_data == peer_in.peers_data

        # Container
        c_out = next(iter(out.containers))
        assert c_out.name == 'myapp'
        assert c_out.can_connect is True
        assert 'app' in c_out.layers['base'].services
        assert c_out.service_statuses['app'] is pebble.ServiceStatus.ACTIVE
        notice_out = c_out.notices[0]
        assert notice_out.key == 'example.com/event'
        assert notice_out.type is pebble.NoticeType.CUSTOM
        assert notice_out.repeat_after == datetime.timedelta(minutes=30)
        ci_out = c_out.get_check_info('http')
        assert ci_out.status is pebble.CheckStatus.UP
        assert ci_out.level is pebble.CheckLevel.ALIVE

        # Secret
        secret_out = next(iter(out.secrets))
        assert secret_out.tracked_content == {'password': 'hunter2'}
        assert secret_out.rotate is SecretRotate.DAILY
        assert secret_out.remote_grants[0] == {'related-app'}
        assert isinstance(list(secret_out.remote_grants.keys())[0], int)

        # Resource
        res_out = next(iter(out.resources))
        assert res_out.name == 'oci-image'
        assert isinstance(res_out.path, pathlib.Path)

        # StoredState escape hatch
        ss_out = next(iter(out.stored_states))
        assert ss_out.content['count'] == 42
        assert ss_out.content['raw'] == b'\xde\xad\xbe\xef'
        assert isinstance(ss_out.content['raw'], bytes)
        assert ss_out.content['coords'] == (1, 2, 3)
        assert isinstance(ss_out.content['coords'], tuple)
        assert ss_out.content['tags'] == {'alpha', 'beta'}
        assert isinstance(ss_out.content['tags'], set)

        # Ports
        assert out.opened_ports == frozenset([TCPPort(8080)])

        # Metadata
        assert out.leader is True
        assert out.unit_status == ActiveStatus('ready')
        assert out.app_status == ActiveStatus('running')
        assert out.workload_version == '1.2.3'
        assert out.planned_units == 3

        # Deferred events
        assert len(out.deferred) == 1
        assert out.deferred[0].handle_path == 'MyCharm/on/config_changed[1]'
        assert out.deferred[0].owner == 'MyCharm'
