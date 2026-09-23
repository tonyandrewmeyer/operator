# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.

from __future__ import annotations

import dataclasses
import datetime
import io
import pathlib
from typing import TYPE_CHECKING

import pytest
from scenario import Context
from scenario.state import (
    CheckInfo,
    Container,
    Exec,
    Mount,
    Notice,
    ServiceBehaviour,
    ServiceExitCode,
    ServiceFailureMode,
    ServiceStart,
    State,
)

import ops
from ops import CharmBase, Framework, pebble
from ops.log import _get_juju_log_and_app_id

from ..helpers import state_delta, trigger

if TYPE_CHECKING:
    from ops.pebble import LayerDict, ServiceDict


class Charm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        for evt in self.on.events().values():
            framework.observe(evt, self._on_event)

    def _on_event(self, event: ops.EventBase):
        pass


def test_no_containers():
    def callback(self: ops.CharmBase):
        assert not self.unit.containers

    trigger(
        State(),
        charm_type=Charm,
        meta={'name': 'foo'},
        event='start',
        post_event=callback,
    )


def test_containers_from_meta():
    def callback(self: ops.CharmBase):
        assert self.unit.containers
        assert self.unit.get_container('foo')

    trigger(
        State(),
        charm_type=Charm,
        meta={'name': 'foo', 'containers': {'foo': {}}},
        event='start',
        post_event=callback,
    )


@pytest.mark.parametrize('can_connect', (True, False))
def test_connectivity(can_connect: bool):
    def callback(self: ops.CharmBase):
        assert can_connect == self.unit.get_container('foo').can_connect()

    trigger(
        State(containers={Container(name='foo', can_connect=can_connect)}),
        charm_type=Charm,
        meta={'name': 'foo', 'containers': {'foo': {}}},
        event='start',
        post_event=callback,
    )


def test_fs_push(tmp_path: pathlib.Path):
    text = 'lorem ipsum/n alles amat gloriae foo'

    pth = tmp_path / 'textfile'
    pth.write_text(text)

    def callback(self: ops.CharmBase):
        container = self.unit.get_container('foo')
        with container.pull('/bar/baz.txt') as baz:
            assert baz.read() == text

    trigger(
        State(
            containers={
                Container(
                    name='foo',
                    can_connect=True,
                    mounts={'bar': Mount(location='/bar/baz.txt', source=pth)},
                )
            }
        ),
        charm_type=Charm,
        meta={'name': 'foo', 'containers': {'foo': {}}},
        event='start',
        post_event=callback,
    )


@pytest.mark.parametrize('make_dirs', (True, False))
def test_fs_pull(tmp_path: pathlib.Path, make_dirs: bool):
    text = 'lorem ipsum/n alles amat gloriae foo'

    def callback(self: ops.CharmBase):
        container = self.unit.get_container('foo')
        if make_dirs:
            container.push('/foo/bar/baz.txt', text, make_dirs=make_dirs)
            # check that pulling immediately 'works'
            with container.pull('/foo/bar/baz.txt') as baz:
                assert baz.read() == text
        else:
            with pytest.raises(ops.pebble.PathError):
                container.push('/foo/bar/baz.txt', text, make_dirs=make_dirs)

            # check that nothing was changed
            with pytest.raises((FileNotFoundError, ops.pebble.PathError)):
                container.pull('/foo/bar/baz.txt')

    container = Container(
        name='foo',
        can_connect=True,
        mounts={'foo': Mount(location='/foo', source=tmp_path)},
    )
    state = State(containers={container})

    ctx = Context(
        charm_type=Charm,
        meta={'name': 'foo', 'containers': {'foo': {}}},
    )
    with ctx(ctx.on.start(), state=state) as mgr:
        out = mgr.run()
        callback(mgr.charm)

    if make_dirs:
        # this is one way to retrieve the file
        file = tmp_path / 'bar' / 'baz.txt'

        # another is:
        base = pathlib.Path(out.get_container('foo').mounts['foo'].source)
        assert file == base / 'bar' / 'baz.txt'

        # but that is actually a symlink to the context's root tmp folder:
        base = ctx._tmp_path
        assert (base / 'containers' / 'foo' / 'foo' / 'bar' / 'baz.txt').read_text() == text
        assert file.read_text() == text

        # shortcut for API niceness purposes:
        file = container.get_filesystem(ctx) / 'foo' / 'bar' / 'baz.txt'
        assert file.read_text() == text

    else:
        # nothing has changed
        out_purged = dataclasses.replace(out, stored_states=state.stored_states)
        assert not state_delta(out_purged, state)


LS = """
.rw-rw-r--  228 ubuntu ubuntu 18 jan 12:05 -- charmcraft.yaml
.rw-rw-r--  497 ubuntu ubuntu 18 jan 12:05 -- config.yaml
.rw-rw-r--  900 ubuntu ubuntu 18 jan 12:05 -- CONTRIBUTING.md
drwxrwxr-x    - ubuntu ubuntu 18 jan 12:06 -- lib
.rw-rw-r--  11k ubuntu ubuntu 18 jan 12:05 -- LICENSE
.rw-rw-r-- 1,6k ubuntu ubuntu 18 jan 12:05 -- metadata.yaml
.rw-rw-r--  845 ubuntu ubuntu 18 jan 12:05 -- pyproject.toml
.rw-rw-r--  831 ubuntu ubuntu 18 jan 12:05 -- README.md
.rw-rw-r--   13 ubuntu ubuntu 18 jan 12:05 -- requirements.txt
drwxrwxr-x    - ubuntu ubuntu 18 jan 12:05 -- src
drwxrwxr-x    - ubuntu ubuntu 18 jan 12:05 -- tests
.rw-rw-r-- 1,9k ubuntu ubuntu 18 jan 12:05 -- tox.ini
"""
PS = """
    PID TTY          TIME CMD
 298238 pts/3    00:00:04 zsh
1992454 pts/3    00:00:00 ps
"""


@pytest.mark.parametrize(
    'cmd, out',
    (
        ('ls', LS),
        ('ps', PS),
    ),
)
def test_exec(cmd: str, out: str):
    def callback(self: ops.CharmBase):
        container = self.unit.get_container('foo')
        proc = container.exec([cmd])
        proc.wait()
        assert proc.stdout is not None
        assert proc.stdout.read() == out

    trigger(
        State(
            containers={
                Container(
                    name='foo',
                    can_connect=True,
                    execs={Exec([cmd], stdout=out)},
                )
            }
        ),
        charm_type=Charm,
        meta={'name': 'foo', 'containers': {'foo': {}}},
        event='start',
        post_event=callback,
    )


class ExecCharm(ops.CharmBase):
    stdin: str | io.StringIO | None
    write: str | None

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        self.framework.observe(self.on.foo_pebble_ready, self._on_ready)

    def _on_ready(self, _: ops.EventBase):
        proc = self.unit.get_container('foo').exec(['ls'], stdin=self.stdin)
        if self.write:
            assert proc.stdin is not None
            proc.stdin.write(self.write)
        proc.wait()


@pytest.mark.parametrize(
    'stdin,write',
    (
        [None, 'hello world!'],
        ['hello world!', None],
        [io.StringIO('hello world!'), None],
    ),
)
def test_exec_history_stdin(
    monkeypatch: pytest.MonkeyPatch, stdin: str | io.StringIO | None, write: str | None
):
    monkeypatch.setattr(ExecCharm, 'stdin', stdin, raising=False)
    monkeypatch.setattr(ExecCharm, 'write', write, raising=False)
    ctx = Context(ExecCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    container = Container(name='foo', can_connect=True, execs={Exec([])})
    ctx.run(ctx.on.pebble_ready(container=container), State(containers={container}))
    assert ctx.exec_history[container.name][0].stdin == 'hello world!'


def test_pebble_ready():
    def callback(self: ops.CharmBase):
        foo = self.unit.get_container('foo')
        assert foo.can_connect()

    container = Container(name='foo', can_connect=True)

    trigger(
        State(containers={container}),
        charm_type=Charm,
        meta={'name': 'foo', 'containers': {'foo': {}}},
        event='pebble_ready',
        post_event=callback,
    )


class PlanCharm(ops.CharmBase):
    starting_service_status: ops.pebble.ServiceStatus

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_ready, self._on_ready)

    def _on_ready(self, event: ops.PebbleReadyEvent):
        foo = event.workload

        assert foo.get_plan().to_dict() == {'services': {'fooserv': {'startup': 'enabled'}}}
        fooserv = foo.get_services('fooserv')['fooserv']
        assert fooserv.startup == ops.pebble.ServiceStartup.ENABLED
        assert fooserv.current == ops.pebble.ServiceStatus.ACTIVE

        foo.add_layer(
            'bar',
            {
                'summary': 'bla',
                'description': 'deadbeef',
                'services': {'barserv': {'startup': 'disabled'}},
            },
        )

        foo.replan()
        assert foo.get_plan().to_dict() == {
            'services': {
                'barserv': {'startup': 'disabled'},
                'fooserv': {'startup': 'enabled'},
            }
        }

        assert foo.get_service('barserv').current == self.starting_service_status
        foo.start('barserv')
        # whatever the original state, starting a service sets it to active
        assert foo.get_service('barserv').current == ops.pebble.ServiceStatus.ACTIVE


@pytest.mark.parametrize('starting_service_status', ops.pebble.ServiceStatus)
def test_pebble_plan(
    monkeypatch: pytest.MonkeyPatch, starting_service_status: ops.pebble.ServiceStatus
):
    monkeypatch.setattr(
        PlanCharm, 'starting_service_status', starting_service_status, raising=False
    )

    container = Container(
        name='foo',
        can_connect=True,
        layers={
            'foo': ops.pebble.Layer({
                'summary': 'bla',
                'description': 'deadbeef',
                'services': {'fooserv': {'startup': 'enabled'}},
            })
        },
        service_statuses={
            'fooserv': ops.pebble.ServiceStatus.ACTIVE,
            # todo: should we disallow setting status for services that aren't known YET?
            'barserv': starting_service_status,
        },
    )

    out = trigger(
        State(containers={container}),
        charm_type=PlanCharm,
        meta={'name': 'foo', 'containers': {'foo': {}}},
        event='pebble_ready',
    )

    def serv(name: str, obj: ServiceDict) -> ops.pebble.Service:
        return ops.pebble.Service(name, raw=obj)

    container = out.get_container(container.name)
    assert container.plan.services == {
        'barserv': serv('barserv', {'startup': 'disabled'}),
        'fooserv': serv('fooserv', {'startup': 'enabled'}),
    }
    assert container.services['fooserv'].current == ops.pebble.ServiceStatus.ACTIVE
    assert container.services['fooserv'].startup == ops.pebble.ServiceStartup.ENABLED

    assert container.services['barserv'].current == ops.pebble.ServiceStatus.ACTIVE
    assert container.services['barserv'].startup == ops.pebble.ServiceStartup.DISABLED


def test_exec_wait_error():
    state = State(
        containers={
            Container(
                name='foo',
                can_connect=True,
                execs={Exec(['foo'], stdout='hello pebble', return_code=1)},
            )
        }
    )

    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), state) as mgr:
        container = mgr.charm.unit.get_container('foo')
        proc = container.exec(['foo'])
        with pytest.raises(ops.pebble.ExecError) as exc_info:  # type: ignore
            proc.wait_output()
        assert exc_info.value.stdout == 'hello pebble'  # type: ignore


@pytest.mark.parametrize('command', (['foo'], ['foo', 'bar'], ['foo', 'bar', 'baz']))
def test_exec_wait_output(command: list[str]):
    state = State(
        containers={
            Container(
                name='foo',
                can_connect=True,
                execs={Exec(['foo'], stdout='hello pebble', stderr='oepsie')},
            )
        }
    )

    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), state) as mgr:
        container = mgr.charm.unit.get_container('foo')
        proc = container.exec(command)
        out, err = proc.wait_output()
        assert out == 'hello pebble'
        assert err == 'oepsie'
        assert ctx.exec_history[container.name][0].command == command


def test_exec_wait_output_error():
    state = State(
        containers={
            Container(
                name='foo',
                can_connect=True,
                execs={Exec(['foo'], stdout='hello pebble', return_code=1)},
            )
        }
    )

    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), state) as mgr:
        container = mgr.charm.unit.get_container('foo')
        proc = container.exec(['foo'])
        with pytest.raises(ops.pebble.ExecError):
            proc.wait_output()


def test_pebble_custom_notice():
    notices = [
        Notice(key='example.com/foo'),
        Notice(key='example.com/bar', last_data={'a': 'b'}),
        Notice(key='example.com/baz', occurrences=42),
    ]
    container = Container(
        name='foo',
        can_connect=True,
        notices=notices,
    )

    state = State(containers=[container])
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.pebble_custom_notice(container=container, notice=notices[-1]), state) as mgr:
        container = mgr.charm.unit.get_container('foo')
        assert container.get_notices() == [n._to_ops() for n in notices]


class CustomNoticeCharm(ops.CharmBase):
    key: str
    data: dict[str, str]
    user_id: int
    first_occurred: datetime.datetime
    last_occurred: datetime.datetime
    last_repeated: datetime.datetime
    occurrences: int
    repeat_after: datetime.timedelta
    expire_after: datetime.timedelta

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_custom_notice, self._on_custom_notice)

    def _on_custom_notice(self, event: ops.PebbleCustomNoticeEvent):
        notice = event.notice
        assert notice.type == ops.pebble.NoticeType.CUSTOM
        assert notice.key == self.key
        assert notice.last_data == self.data
        assert notice.user_id == self.user_id
        assert notice.first_occurred == self.first_occurred
        assert notice.last_occurred == self.last_occurred
        assert notice.last_repeated == self.last_repeated
        assert notice.occurrences == self.occurrences
        assert notice.repeat_after == self.repeat_after
        assert notice.expire_after == self.expire_after


def test_pebble_custom_notice_in_charm(monkeypatch: pytest.MonkeyPatch):
    key = 'example.com/test/charm'
    data = {'foo': 'bar'}
    user_id = 100
    first_occurred = datetime.datetime(1979, 1, 25, 11, 0, 0)
    last_occurred = datetime.datetime(2006, 8, 28, 13, 28, 0)
    last_repeated = datetime.datetime(2023, 9, 4, 9, 0, 0)
    occurrences = 42
    repeat_after = datetime.timedelta(days=7)
    expire_after = datetime.timedelta(days=365)

    monkeypatch.setattr(CustomNoticeCharm, 'key', key, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'data', data, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'user_id', user_id, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'first_occurred', first_occurred, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'last_occurred', last_occurred, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'last_repeated', last_repeated, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'occurrences', occurrences, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'repeat_after', repeat_after, raising=False)
    monkeypatch.setattr(CustomNoticeCharm, 'expire_after', expire_after, raising=False)

    notices = [
        Notice('example.com/test/other'),
        Notice('example.org/test/charm', last_data={'foo': 'baz'}),
        Notice(
            key,
            last_data=data,
            user_id=user_id,
            first_occurred=first_occurred,
            last_occurred=last_occurred,
            last_repeated=last_repeated,
            occurrences=occurrences,
            repeat_after=repeat_after,
            expire_after=expire_after,
        ),
    ]
    container = Container(
        name='foo',
        can_connect=True,
        notices=notices,
    )
    state = State(containers=[container])
    ctx = Context(CustomNoticeCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    ctx.run(ctx.on.pebble_custom_notice(container=container, notice=notices[-1]), state)


class CheckFailedCharm(ops.CharmBase):
    infos: list[ops.LazyCheckInfo]

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_check_failed, self._on_check_failed)

    def _on_check_failed(self, event: ops.PebbleCheckFailedEvent):
        self.infos.append(event.info)


@pytest.fixture
def capture_info_failure_charm(monkeypatch: pytest.MonkeyPatch) -> type[CheckFailedCharm]:
    monkeypatch.setattr(CheckFailedCharm, 'infos', [], raising=False)
    return CheckFailedCharm


def test_pebble_check_failed(capture_info_failure_charm: type[CheckFailedCharm]):
    ctx = Context(capture_info_failure_charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    layer = ops.pebble.Layer({
        'checks': {'http-check': {'override': 'replace', 'startup': 'enabled', 'threshold': 3}}
    })
    assert layer.checks['http-check'].threshold is not None
    check = CheckInfo(
        'http-check',
        successes=3,
        failures=7,
        status=ops.pebble.CheckStatus.DOWN,
        level=ops.pebble.CheckLevel(layer.checks['http-check'].level),
        startup=ops.pebble.CheckStartup(layer.checks['http-check'].startup),
        threshold=layer.checks['http-check'].threshold,
    )
    container = Container('foo', check_infos={check}, layers={'layer1': layer})
    state = State(containers={container})
    ctx.run(ctx.on.pebble_check_failed(container, check), state=state)
    infos = capture_info_failure_charm.infos
    assert len(infos) == 1
    assert infos[0].name == 'http-check'
    assert infos[0].status == ops.pebble.CheckStatus.DOWN
    assert infos[0].successes == 3
    assert infos[0].failures == 7


class CheckRecoveredCharm(ops.CharmBase):
    infos: list[ops.LazyCheckInfo]

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_check_recovered, self._on_check_recovered)

    def _on_check_recovered(self, event: ops.PebbleCheckRecoveredEvent):
        self.infos.append(event.info)


@pytest.fixture
def capture_info_recovered_charm(monkeypatch: pytest.MonkeyPatch) -> type[CheckRecoveredCharm]:
    monkeypatch.setattr(CheckRecoveredCharm, 'infos', [], raising=False)
    return CheckRecoveredCharm


def test_pebble_check_recovered(capture_info_recovered_charm: type[CheckRecoveredCharm]):
    ctx = Context(capture_info_recovered_charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    layer = ops.pebble.Layer({
        'checks': {'http-check': {'override': 'replace', 'startup': 'enabled', 'threshold': 3}}
    })
    assert layer.checks['http-check'].threshold is not None
    check = CheckInfo(
        'http-check',
        successes=None,
        status=ops.pebble.CheckStatus.UP,
        level=ops.pebble.CheckLevel(layer.checks['http-check'].level),
        startup=ops.pebble.CheckStartup(layer.checks['http-check'].startup),
        threshold=layer.checks['http-check'].threshold,
    )
    container = Container('foo', check_infos={check}, layers={'layer1': layer})
    state = State(containers={container})
    ctx.run(ctx.on.pebble_check_recovered(container, check), state=state)
    infos = capture_info_recovered_charm.infos
    assert len(infos) == 1
    assert infos[0].name == 'http-check'
    assert infos[0].status == ops.pebble.CheckStatus.UP
    assert infos[0].successes is None
    assert infos[0].failures == 0


class DoubleCharm(ops.CharmBase):
    foo_infos: list[ops.LazyCheckInfo]
    bar_infos: list[ops.LazyCheckInfo]

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_check_failed, self._on_foo_check_failed)
        framework.observe(self.on.bar_pebble_check_failed, self._on_bar_check_failed)

    def _on_foo_check_failed(self, event: ops.PebbleCheckFailedEvent):
        self.foo_infos.append(event.info)

    def _on_bar_check_failed(self, event: ops.PebbleCheckFailedEvent):
        self.bar_infos.append(event.info)


@pytest.fixture
def capture_info_double_charm(monkeypatch: pytest.MonkeyPatch) -> type[DoubleCharm]:
    monkeypatch.setattr(DoubleCharm, 'foo_infos', [], raising=False)
    monkeypatch.setattr(DoubleCharm, 'bar_infos', [], raising=False)
    return DoubleCharm


def test_pebble_check_failed_two_containers(capture_info_double_charm: type[DoubleCharm]):
    ctx = Context(
        capture_info_double_charm, meta={'name': 'foo', 'containers': {'foo': {}, 'bar': {}}}
    )

    layer = ops.pebble.Layer({
        'checks': {'http-check': {'override': 'replace', 'startup': 'enabled', 'threshold': 3}}
    })
    assert layer.checks['http-check'].threshold is not None
    check = CheckInfo(
        'http-check',
        failures=7,
        status=ops.pebble.CheckStatus.DOWN,
        level=ops.pebble.CheckLevel(layer.checks['http-check'].level),
        startup=ops.pebble.CheckStartup(layer.checks['http-check'].startup),
        threshold=layer.checks['http-check'].threshold,
    )
    foo_container = Container('foo', check_infos={check}, layers={'layer1': layer})
    bar_container = Container('bar', check_infos={check}, layers={'layer1': layer})
    state = State(containers={foo_container, bar_container})
    ctx.run(ctx.on.pebble_check_failed(foo_container, check), state=state)
    foo_infos = DoubleCharm.foo_infos
    bar_infos = DoubleCharm.bar_infos
    assert len(foo_infos) == 1
    assert foo_infos[0].name == 'http-check'
    assert foo_infos[0].status == ops.pebble.CheckStatus.DOWN
    assert foo_infos[0].successes == 0
    assert foo_infos[0].failures == 7
    assert len(bar_infos) == 0


class LayerCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_ready, self._on_foo_ready)

    def _on_foo_ready(self, _: ops.EventBase):
        self.unit.get_container('foo').add_layer(
            'foo',
            {'checks': {'chk1': {'override': 'replace'}}},
        )


def test_pebble_add_layer():
    ctx = Context(LayerCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    container = Container('foo', can_connect=True)
    state_out = ctx.run(ctx.on.pebble_ready(container), state=State(containers={container}))
    chk1_info = state_out.get_container('foo').get_check_info('chk1')
    assert chk1_info.status == ops.pebble.CheckStatus.UP


class StartCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_ready, self._on_foo_ready)
        framework.observe(self.on.config_changed, self._on_config_changed)

    def _on_foo_ready(self, _: ops.EventBase):
        container = self.unit.get_container('foo')
        container.add_layer(
            'foo',
            {
                'checks': {
                    'chk1': {
                        'override': 'replace',
                        'startup': 'disabled',
                        'threshold': 3,
                    }
                }
            },
        )

    def _on_config_changed(self, _: ops.EventBase):
        container = self.unit.get_container('foo')
        container.start_checks('chk1')


def test_pebble_start_check():
    ctx = Context(StartCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    container = Container('foo', can_connect=True)

    # Ensure that it starts as inactive.
    state_out = ctx.run(ctx.on.pebble_ready(container), state=State(containers={container}))
    chk1_info = state_out.get_container('foo').get_check_info('chk1')
    assert chk1_info.status == ops.pebble.CheckStatus.INACTIVE

    # Verify that start_checks works.
    state_out = ctx.run(ctx.on.config_changed(), state=state_out)
    chk1_info = state_out.get_container('foo').get_check_info('chk1')
    assert chk1_info.status == ops.pebble.CheckStatus.UP


@pytest.fixture
def reset_security_logging():
    """Ensure that we get a fresh juju-log for the security logging."""
    _get_juju_log_and_app_id.cache_clear()
    yield


class StopCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.config_changed, self._on_config_changed)

    def _on_config_changed(self, _: ops.EventBase):
        container = self.unit.get_container('foo')
        container.stop_checks('chk1')


def test_pebble_stop_check(reset_security_logging: None):
    ctx = Context(StopCharm, meta={'name': 'foo', 'containers': {'foo': {}}})

    layer = ops.pebble.Layer({
        'checks': {'chk1': {'override': 'replace', 'startup': 'enabled', 'threshold': 3}}
    })
    assert layer.checks['chk1'].threshold is not None
    info_in = CheckInfo(
        'chk1',
        status=ops.pebble.CheckStatus.UP,
        level=ops.pebble.CheckLevel(layer.checks['chk1'].level),
        startup=ops.pebble.CheckStartup(layer.checks['chk1'].startup),
        threshold=layer.checks['chk1'].threshold,
    )
    container = Container(
        'foo',
        can_connect=True,
        check_infos=frozenset({info_in}),
        layers={'layer1': layer},
    )
    state_out = ctx.run(ctx.on.config_changed(), state=State(containers={container}))
    info_out = state_out.get_container('foo').get_check_info('chk1')
    assert info_out.status == ops.pebble.CheckStatus.INACTIVE


class ReplanCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.config_changed, self._on_config_changed)

    def _on_config_changed(self, _: ops.EventBase):
        container = self.unit.get_container('foo')
        container.replan()


def test_pebble_replan_checks():
    ctx = Context(ReplanCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    layer = ops.pebble.Layer({
        'checks': {'chk1': {'override': 'replace', 'startup': 'enabled', 'threshold': 3}}
    })
    assert layer.checks['chk1'].threshold is not None
    info_in = CheckInfo(
        'chk1',
        status=ops.pebble.CheckStatus.INACTIVE,
        level=ops.pebble.CheckLevel(layer.checks['chk1'].level),
        startup=ops.pebble.CheckStartup(layer.checks['chk1'].startup),
        threshold=layer.checks['chk1'].threshold,
    )
    container = Container(
        'foo',
        can_connect=True,
        check_infos=frozenset({info_in}),
        layers={'layer1': layer},
    )
    state_out = ctx.run(ctx.on.config_changed(), state=State(containers={container}))
    info_out = state_out.get_container('foo').get_check_info('chk1')
    assert info_out.status == ops.pebble.CheckStatus.UP


class CombineLayerCharm(ops.CharmBase):
    layer_name: str
    layer_dict: LayerDict
    combine: bool

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['my-container'].pebble_ready, self._on_pebble_ready)

    def _on_pebble_ready(self, _: ops.PebbleReadyEvent):
        container = self.unit.get_container('my-container')
        container.add_layer(
            self.layer_name, ops.pebble.Layer(self.layer_dict), combine=self.combine
        )


@pytest.mark.parametrize(
    'combine,new_layer_name',
    [
        (False, 'new-layer'),
        (True, 'base'),
    ],
)
@pytest.mark.parametrize(
    'new_layer_dict',
    [
        {
            'checks': {
                'server-ready': {
                    'override': 'merge',
                    'level': 'ready',
                    'http': {'url': 'http://localhost:5050/version'},
                }
            }
        },
        {
            'checks': {
                'server-ready': {
                    'override': 'merge',
                    'level': 'alive',
                    'threshold': 30,
                    'startup': 'disabled',
                    'http': {'url': 'http://localhost:5050/version'},
                }
            }
        },
    ],
)
def test_add_layer_merge_check(
    monkeypatch: pytest.MonkeyPatch, new_layer_name: str, combine: bool, new_layer_dict: LayerDict
):
    monkeypatch.setattr(CombineLayerCharm, 'layer_name', new_layer_name, raising=False)
    monkeypatch.setattr(CombineLayerCharm, 'layer_dict', new_layer_dict, raising=False)
    monkeypatch.setattr(CombineLayerCharm, 'combine', combine, raising=False)

    ctx = Context(CombineLayerCharm, meta={'name': 'foo', 'containers': {'my-container': {}}})
    layer_in = ops.pebble.Layer({
        'checks': {
            'server-ready': {
                'override': 'replace',
                'level': 'ready',
                'startup': 'enabled',
                'threshold': 10,
                'http': {'url': 'http://localhost:5000/version'},
            }
        }
    })
    assert layer_in.checks['server-ready'].threshold is not None
    check_in = CheckInfo(
        'server-ready',
        level=ops.pebble.CheckLevel(layer_in.checks['server-ready'].level),
        threshold=layer_in.checks['server-ready'].threshold,
        startup=ops.pebble.CheckStartup(layer_in.checks['server-ready'].startup),
    )
    container_in = Container(
        'my-container',
        can_connect=True,
        layers={'base': layer_in},
        check_infos={check_in},
    )
    assert container_in.get_check_info('server-ready').level == ops.pebble.CheckLevel.READY
    state_in = State(containers={container_in})

    state_out = ctx.run(ctx.on.pebble_ready(container_in), state_in)

    check_out = state_out.get_container(container_in.name).get_check_info('server-ready')
    new_layer_check = new_layer_dict.get('checks', {}).get('server-ready', {})
    assert check_out.level == ops.pebble.CheckLevel(new_layer_check.get('level', 'ready'))
    assert check_out.startup == ops.pebble.CheckStartup(new_layer_check.get('startup', 'enabled'))
    assert check_out.threshold == new_layer_check.get('threshold', 10)


@pytest.mark.parametrize('layer1_name,layer2_name', [('a-base', 'b-base'), ('b-base', 'a-base')])
def test_layers_merge_in_plan(layer1_name: str, layer2_name: str):
    layer1_dict: LayerDict = {
        'services': {
            'server': {
                'override': 'replace',
                'command': '/bin/sleep 10',
                'summary': 'sum',
                'description': 'desc',
                'startup': 'enabled',
            },
        },
        'checks': {
            'server-ready': {
                'override': 'replace',
                'level': 'ready',
                'startup': 'enabled',
                'threshold': 10,
                'period': '1s',
                'timeout': '28s',
                'http': {'url': 'http://localhost:5000/version'},
            }
        },
        'log-targets': {
            'loki': {
                'override': 'replace',
                'type': 'loki',
                'location': 'https://loki.example.com',
                'services': ['server'],
                'labels': {'foo': 'bar'},
            }
        },
    }
    layer2_dict: LayerDict = {
        'services': {
            'server': {
                'override': 'merge',
                'command': '/bin/sleep 20',
            }
        },
        'checks': {
            'server-ready': {
                'override': 'merge',
                'level': 'alive',
                'http': {'url': 'http://localhost:5050/version'},
            }
        },
        'log-targets': {
            'loki': {
                'override': 'merge',
                'location': 'https://loki2.example.com',
            },
        },
    }
    layer1 = ops.pebble.Layer(layer1_dict)
    layer2 = ops.pebble.Layer(layer2_dict)

    ctx = Context(ops.CharmBase, meta={'name': 'foo', 'containers': {'my-container': {}}})
    # TODO also a starting layer.
    container = Container('my-container', can_connect=True)

    with ctx(ctx.on.update_status(), State(containers={container})) as mgr:
        mgr.charm.unit.get_container('my-container').add_layer(layer1_name, layer1)
        mgr.charm.unit.get_container('my-container').add_layer(layer2_name, layer2)
        state_out = mgr.run()

    plan = state_out.get_container(container.name).plan

    service = plan.services['server']
    assert service.summary == 'sum'
    assert service.description == 'desc'
    # Service.startup is always a string, even though we have the enum.
    assert service.startup == ops.pebble.ServiceStartup.ENABLED.value
    assert service.override == 'merge'
    assert service.command == '/bin/sleep 20'

    check = plan.checks['server-ready']
    assert check.startup == ops.pebble.CheckStartup.ENABLED
    assert check.threshold == 10
    assert check.period == '1s'
    assert check.timeout == '28s'
    assert check.override == 'merge'
    assert check.level == ops.pebble.CheckLevel.ALIVE
    http = check.http
    assert http is not None
    assert http.get('url') == 'http://localhost:5050/version'

    log_target = plan.log_targets['loki']
    assert log_target.type == 'loki'
    assert log_target.services == ['server']
    assert log_target.labels == {'foo': 'bar'}
    assert log_target.override == 'merge'
    assert log_target.location == 'https://loki2.example.com'


def test_plan_accessed_twice_does_not_accumulate_list_fields():
    """Regression test: accessing .plan multiple times must not mutate original layers.

    When rendering services/checks/log_targets, storing a direct reference to
    a layer's object means that a subsequent merge call mutates the original.
    This causes list fields (after, before, requires, services) to accumulate
    duplicates on each access.
    """
    layer1 = pebble.Layer({
        'services': {
            'svc-a': {
                'override': 'replace',
                'command': '/bin/a',
                'after': ['other'],
                'before': ['another'],
                'requires': ['dep'],
            },
        },
        'checks': {
            'chk-a': {
                'override': 'replace',
                'level': 'ready',
                'http': {'url': 'http://localhost:8080'},
            },
        },
        'log-targets': {
            'lt-a': {
                'override': 'replace',
                'type': 'loki',
                'location': 'https://loki.example.com',
                'services': ['svc-a'],
            },
        },
    })
    layer2 = pebble.Layer({
        'services': {
            'svc-a': {
                'override': 'merge',
                'command': '/bin/a2',
                'after': ['redis'],
                'before': ['cleanup'],
                'requires': ['dep2'],
            },
        },
        'checks': {
            'chk-a': {
                'override': 'merge',
                'level': 'alive',
            },
        },
        'log-targets': {
            'lt-a': {
                'override': 'merge',
                'location': 'https://loki2.example.com',
                'services': ['svc-b'],
            },
        },
    })

    container = Container(
        'my-container',
        can_connect=True,
        layers={'base': layer1, 'override': layer2},
    )

    plan1 = container.plan
    svc1_after = list(plan1.services['svc-a'].after)
    svc1_before = list(plan1.services['svc-a'].before)
    svc1_requires = list(plan1.services['svc-a'].requires)
    chk1_level = plan1.checks['chk-a'].level
    lt1_services = list(plan1.log_targets['lt-a'].services)

    plan2 = container.plan
    svc2_after = list(plan2.services['svc-a'].after)
    svc2_before = list(plan2.services['svc-a'].before)
    svc2_requires = list(plan2.services['svc-a'].requires)
    chk2_level = plan2.checks['chk-a'].level
    lt2_services = list(plan2.log_targets['lt-a'].services)

    # Service list fields must not accumulate duplicates.
    assert svc1_after == svc2_after
    assert svc1_before == svc2_before
    assert svc1_requires == svc2_requires

    # Check fields must be stable across accesses.
    assert chk1_level == chk2_level

    # Log target list fields must not accumulate duplicates.
    assert lt1_services == lt2_services

    # Also verify the original layer objects are not mutated.
    assert layer1.services['svc-a'].after == ['other']
    assert layer1.services['svc-a'].before == ['another']
    assert layer1.services['svc-a'].requires == ['dep']
    assert layer1.log_targets['lt-a'].services == ['svc-a']


def test_warning_on_non_empty_container():
    class MyCharm(CharmBase):
        def __init__(self, framework: Framework):
            super().__init__(framework)
            self.framework.observe(self.on.start, self._on_start)

        def _on_start(self, _: object):
            self.unit.get_container('mycontainer').push('/foo.txt', 'hello')

    ctx = Context(
        MyCharm,
        meta={'name': 'foo', 'containers': {'mycontainer': {}}},
    )
    container = Container(name='mycontainer', can_connect=True)
    state = State(containers={container})

    # First run populates the container root with a file.
    ctx.run(ctx.on.start(), state)

    # Second run should warn that the container root is non-empty.
    ctx.run(ctx.on.start(), state)

    assert any(
        'mycontainer' in line.message and 'non-empty' in line.message for line in ctx.juju_log
    )


def test_no_warning_on_empty_container():
    ctx = Context(
        CharmBase,
        meta={'name': 'foo', 'containers': {'mycontainer': {}}},
    )
    container = Container(name='mycontainer', can_connect=True)
    state = State(containers={container})

    # First run creates the container root.
    ctx.run(ctx.on.start(), state)

    # Second run should not warn since the container root is empty.
    ctx.run(ctx.on.start(), state)

    assert not any(
        'mycontainer' in line.message and 'non-empty' in line.message for line in ctx.juju_log
    )


def _crash_layer(on_failure: str | None = None) -> ops.pebble.Layer:
    service: ServiceDict = {
        'override': 'replace',
        'command': '/bin/false',
        'startup': 'enabled',
    }
    if on_failure is not None:
        service['on-failure'] = on_failure
    return ops.pebble.Layer({'services': {'svc': service}})


def test_service_fails_on_failure_ignore_yields_error_status():
    layer = _crash_layer(on_failure='ignore')
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('svc', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('svc')
        assert 'will ignore' in exc_info.value.err
        assert workload.get_service('svc').current == ops.pebble.ServiceStatus.ERROR


@pytest.mark.parametrize('on_failure', (None, 'restart'))
def test_service_fails_default_on_failure_yields_backoff_string(on_failure: str | None):
    layer = _crash_layer(on_failure=on_failure)
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('svc', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('svc')
        assert 'will restart' in exc_info.value.err
        current = workload.get_service('svc').current
        # 'backoff' has no ServiceStatus member, so real Pebble (and this mock)
        # reports the raw string -- see WORKLOAD-MOCK-DESIGN.md §11.1.
        assert current == 'backoff'
        assert not isinstance(current, ops.pebble.ServiceStatus)


def test_service_fails_exec_error_yields_inactive():
    layer = ops.pebble.Layer({
        'services': {
            'svc': {
                'override': 'replace',
                'command': '/definitely/not/a/binary',
                'startup': 'enabled',
                'on-failure': 'ignore',
            }
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={
            ServiceBehaviour(
                'svc', start=ServiceStart.FAILS, failure_mode=ServiceFailureMode.EXEC_ERROR
            )
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('svc')
        assert (
            'fork/exec /definitely/not/a/binary: no such file or directory' in exc_info.value.err
        )
        assert workload.get_service('svc').current == ops.pebble.ServiceStatus.INACTIVE


def test_service_fails_change_error_shape():
    """The raised ChangeError should match the shape measured against real Pebble."""
    layer = _crash_layer(on_failure='ignore')
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('svc', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('svc')
        err = exc_info.value
        change = err.change

        assert err.err == (
            'cannot perform the following tasks:\n'
            '- Start service "svc" (service start attempt: exited quickly with code 1, '
            'will ignore)'
        )
        assert change.err == err.err
        assert change.kind == 'start'
        assert change.status == 'Error'
        assert change.ready is True
        assert change.spawn_time is not None
        assert change.ready_time is not None
        assert change.ready_time >= change.spawn_time

        assert len(change.tasks) == 1
        task = change.tasks[0]
        assert task.kind == 'start'
        assert task.status == 'Error'
        assert task.summary == 'Start service "svc"'
        assert task.progress.label == ''
        assert task.progress.done == 1
        assert task.progress.total == 1
        assert len(task.log) == 2
        assert task.log[0].endswith('INFO Most recent service output:')
        assert task.log[1].endswith(
            'ERROR service start attempt: exited quickly with code 1, will ignore'
        )


def test_service_fails_exec_error_log_has_one_entry():
    layer = ops.pebble.Layer({
        'services': {
            'svc': {
                'override': 'replace',
                'command': '/definitely/not/a/binary',
                'startup': 'enabled',
            }
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={
            ServiceBehaviour(
                'svc', start=ServiceStart.FAILS, failure_mode=ServiceFailureMode.EXEC_ERROR
            )
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('svc')
        task = exc_info.value.change.tasks[0]
        assert len(task.log) == 1
        assert task.log[0].endswith(
            'ERROR cannot start service: fork/exec /definitely/not/a/binary: '
            'no such file or directory'
        )


@pytest.mark.parametrize('trigger_op', ('restart', 'replan'))
def test_service_fails_via_restart_and_replan(trigger_op: str):
    layer = _crash_layer(on_failure='ignore')
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('svc', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError):
            if trigger_op == 'replan':
                workload.replan()
            else:
                workload.restart('svc')


def test_service_with_no_behaviour_still_runs():
    """A service with no declared ServiceBehaviour is unaffected (the RUNS default)."""
    layer = ops.pebble.Layer({
        'services': {
            'svc': {'override': 'replace', 'command': '/bin/true', 'startup': 'enabled'},
        }
    })
    container = Container('foo', can_connect=True, layers={'base': layer})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.start('svc')
        assert workload.get_service('svc').current == ops.pebble.ServiceStatus.ACTIVE


def test_service_fails_mixed_batch_leaves_other_service_active():
    """One FAILS service in a multi-service start doesn't block the others."""
    layer = ops.pebble.Layer({
        'services': {
            'good': {'override': 'replace', 'command': '/bin/true', 'startup': 'enabled'},
            'bad': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('bad', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError):
            workload.start('good', 'bad')
        assert workload.get_service('good').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('bad').current == ops.pebble.ServiceStatus.ERROR


def test_service_fails_unsupported_on_failure_policy_raises():
    layer = _crash_layer(on_failure='shutdown')
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('svc', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(NotImplementedError):
            workload.start('svc')


def _mixed_layer() -> ops.pebble.Layer:
    return ops.pebble.Layer({
        'services': {
            'ok1': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
            'ok2': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
            'fail': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
        }
    })


def _mixed_container() -> Container:
    return Container(
        'foo',
        can_connect=True,
        layers={'base': _mixed_layer()},
        service_behaviours={ServiceBehaviour('fail', start=ServiceStart.FAILS)},
    )


@pytest.mark.parametrize(
    'op, expected_kind, expected_summary',
    [
        ('start', 'start', 'Start service "fail"'),
        ('restart', 'restart', 'Restart service "fail"'),
    ],
)
def test_service_fails_change_kind_follows_the_entry_point(
    op: str, expected_kind: str, expected_summary: str
):
    """The change kind is the entry point's, not always 'start'."""
    layer = ops.pebble.Layer({
        'services': {
            'fail': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('fail', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        trigger = workload.start if op == 'start' else workload.restart
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            trigger('fail')
        change = exc_info.value.change
        assert change.kind == expected_kind
        assert change.summary == expected_summary


def test_service_fails_restart_emits_a_done_stop_task_first():
    """A restart change carries a Done 'stop' task before 'start'."""
    layer = _crash_layer(on_failure='ignore')
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('svc', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.restart('svc')
        tasks = exc_info.value.change.tasks
        assert [(t.kind, t.status) for t in tasks] == [('stop', 'Done'), ('start', 'Error')]
        assert tasks[0].summary == 'Stop service "svc"'


def test_service_fails_replan_uses_replan_kind():
    """A failing replan raises with kind='replan'."""
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={_mixed_container()})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.replan()
        change = exc_info.value.change
        assert change.kind == 'replan'
        assert change.summary == 'Replan service "fail" and 2 more'


def test_service_fails_multi_service_summary_counts_all_requested():
    """The summary counts every requested service and quotes the name.

    The task list is alphabetical regardless of request order. The summary's
    leading name, for start/restart, is the *first-requested* service, not
    the alphabetically-first one -- see
    test_service_fails_summary_leading_name_by_entry_point for the case that
    tells these two apart. Here "ok1" was requested first and is also
    alphabetically first, so this case alone doesn't disambiguate them.
    """
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={_mixed_container()})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('ok1', 'fail', 'ok2')
        change = exc_info.value.change
        assert change.summary == 'Start service "ok1" and 2 more'
        # A Done task per service that did start, interleaved alphabetically
        # with the failure rather than grouped after it.
        assert [(t.summary, t.status) for t in change.tasks] == [
            ('Start service "fail"', 'Error'),
            ('Start service "ok1"', 'Done'),
            ('Start service "ok2"', 'Done'),
        ]


def test_service_fails_summary_leading_name_by_entry_point():
    """start/restart lead with request order; autostart/replan lead alphabetically.

    Real Pebble's start/restart handler uses the client's request order
    verbatim for the summary's leading name (``payload.Services[0]`` in
    ``api_services.go``); autostart/replan resolve to an
    alphabetically-sorted service list server-side before building the
    summary. Three services, declared and requested in different orders, so
    each ordering gives a different answer if it were the one in force.
    """
    layer = ops.pebble.Layer({
        'services': {
            'foxtrot': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
            'delta': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
            'mike': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('delta', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('mike', 'foxtrot', 'delta')
        assert exc_info.value.change.summary == 'Start service "mike" and 2 more'

    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.autostart()
        assert exc_info.value.change.summary == 'Autostart service "delta" and 2 more'


def test_service_fails_restart_groups_all_stops_before_all_starts():
    """A multi-service restart's tasks are grouped, not interleaved per service."""
    layer = ops.pebble.Layer({
        'services': {
            'foxtrot': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
            'delta': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
            'mike': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('delta', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.restart('mike', 'foxtrot', 'delta')
        tasks = exc_info.value.change.tasks
        assert [(t.kind, t.summary, t.status) for t in tasks] == [
            ('stop', 'Stop service "delta"', 'Done'),
            ('stop', 'Stop service "foxtrot"', 'Done'),
            ('stop', 'Stop service "mike"', 'Done'),
            ('start', 'Start service "delta"', 'Error'),
            ('start', 'Start service "foxtrot"', 'Done'),
            ('start', 'Start service "mike"', 'Done'),
        ]


def test_service_fails_autostart_multiple_failures():
    """autostart with two failing services -- both get a bullet, both get a status."""
    layer = ops.pebble.Layer({
        'services': {
            'alpha': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
            'bravo': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
            'charlie': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
            'delta': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={
            ServiceBehaviour('bravo', start=ServiceStart.FAILS),
            ServiceBehaviour('delta', start=ServiceStart.FAILS),
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.autostart()
        change = exc_info.value.change
        assert change.summary == 'Autostart service "alpha" and 3 more'
        assert [(t.summary, t.status) for t in change.tasks] == [
            ('Start service "alpha"', 'Done'),
            ('Start service "bravo"', 'Error'),
            ('Start service "charlie"', 'Done'),
            ('Start service "delta"', 'Error'),
        ]
        assert exc_info.value.err == (
            'cannot perform the following tasks:\n'
            '- Start service "bravo" '
            '(service start attempt: exited quickly with code 1, will ignore)\n'
            '- Start service "delta" '
            '(service start attempt: exited quickly with code 1, will ignore)'
        )
        assert workload.get_service('alpha').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('bravo').current == ops.pebble.ServiceStatus.ERROR
        assert workload.get_service('charlie').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('delta').current == ops.pebble.ServiceStatus.ERROR


def _exit_layer(on_success: str | None = None, on_failure: str | None = None) -> ops.pebble.Layer:
    service: ServiceDict = {
        'override': 'replace',
        'command': "sh -c 'sleep 1000'",
        'startup': 'enabled',
    }
    if on_success is not None:
        service['on-success'] = on_success
    if on_failure is not None:
        service['on-failure'] = on_failure
    return ops.pebble.Layer({'services': {'svc': service}})


@pytest.mark.parametrize(
    'exit_code, on_success, on_failure, expected',
    [
        (ServiceExitCode.FAILURE, None, 'ignore', ops.pebble.ServiceStatus.ERROR),
        (ServiceExitCode.FAILURE, None, None, 'backoff'),
        (ServiceExitCode.FAILURE, None, 'restart', 'backoff'),
        (ServiceExitCode.SUCCESS, 'ignore', None, ops.pebble.ServiceStatus.INACTIVE),
        (ServiceExitCode.SUCCESS, None, None, 'backoff'),
        (ServiceExitCode.SUCCESS, 'restart', None, 'backoff'),
    ],
)
def test_service_exits_status_by_exit_code_and_policy(
    exit_code: ServiceExitCode,
    on_success: str | None,
    on_failure: str | None,
    expected: ops.pebble.ServiceStatus | str,
):
    """The resulting status is a function of exit_code and the relevant policy."""
    layer = _exit_layer(on_success=on_success, on_failure=on_failure)
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={
            ServiceBehaviour('svc', start=ServiceStart.EXITS, exit_code=exit_code)
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        # Unlike FAILS, the call succeeds -- no ChangeError.
        workload.start('svc')
        assert workload.get_service('svc').current == expected


@pytest.mark.parametrize('op', ['start', 'restart', 'replan', 'autostart'])
def test_service_exits_all_entry_points_succeed(op: str):
    """EXITS never raises ChangeError, for any entry point."""
    layer = _exit_layer(on_failure='ignore')
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={ServiceBehaviour('svc', start=ServiceStart.EXITS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        if op == 'start':
            workload.start('svc')
        elif op == 'restart':
            workload.restart('svc')
        elif op == 'replan':
            workload.replan()
        else:
            workload.autostart()
        assert workload.get_service('svc').current == ops.pebble.ServiceStatus.ERROR


def test_service_exits_unsupported_on_success_policy_raises():
    """on-success values beyond ignore/restart raise NotImplementedError, like FAILS."""
    layer = _exit_layer(on_success='shutdown')
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={
            ServiceBehaviour('svc', start=ServiceStart.EXITS, exit_code=ServiceExitCode.SUCCESS)
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(NotImplementedError):
            workload.start('svc')


def test_service_exits_mixed_with_fails_and_runs():
    """Each service's outcome is its own.

    A single start() call requesting a RUNS, a FAILS, and an EXITS service:
    the FAILS one raises ChangeError for the whole call, but the EXITS one
    still resolves to its own declared status rather than being swept to
    ACTIVE along with the RUNS one.
    """
    layer = ops.pebble.Layer({
        'services': {
            'healthy': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
            'crashy': {
                'override': 'replace',
                'command': '/bin/false',
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
            'exity': {
                'override': 'replace',
                'command': "sh -c 'sleep 1000'",
                'startup': 'enabled',
                'on-failure': 'ignore',
            },
        }
    })
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': layer},
        service_behaviours={
            ServiceBehaviour('crashy', start=ServiceStart.FAILS),
            ServiceBehaviour('exity', start=ServiceStart.EXITS),
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError):
            workload.start('healthy', 'crashy', 'exity')
        assert workload.get_service('healthy').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('crashy').current == ops.pebble.ServiceStatus.ERROR
        assert workload.get_service('exity').current == ops.pebble.ServiceStatus.ERROR


def _dependency_layer() -> ops.pebble.Layer:
    """The exact layer from Real-Pebble probe #3, WORKLOAD-MOCK-DESIGN.md §22.1/§22.2.

    delta must start before bravo; charlie must start after bravo; alpha has
    no declared dependency. All four are startup: disabled, so only an
    explicit start/stop/restart call touches them.
    """
    return ops.pebble.Layer({
        'services': {
            'alpha': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'bravo': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'charlie': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'after': ['bravo'],
            },
            'delta': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'before': ['bravo'],
            },
        }
    })


def test_service_dependency_start_order_matches_probe():
    """StartOrder places dependencies ahead of dependants, independents alphabetically.

    Real-Pebble probe #3 (WORKLOAD-MOCK-DESIGN.md §22.1): requesting
    charlie, delta, alpha, bravo -- deliberately neither alphabetical nor
    dependency order -- real Pebble's tasks came back alpha, delta, bravo,
    charlie: the delta -> bravo -> charlie chain honoured exactly, alpha
    placed first by the alphabetical tie-break among services with no
    outstanding dependency.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _dependency_layer()},
        # A FAILS declaration is what makes Scenario build a per-service
        # task list at all -- with no failing service, start_services()
        # takes the no-task-list fast path. bravo is mid-chain: delta must
        # start before it, charlie must start after it.
        service_behaviours={ServiceBehaviour('bravo', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('charlie', 'delta', 'alpha', 'bravo')
        tasks = exc_info.value.change.tasks
        assert [t.summary for t in tasks] == [
            'Start service "alpha"',
            'Start service "delta"',
            'Start service "bravo"',
            'Start service "charlie"',
        ]
        assert [t.status for t in tasks] == ['Done', 'Done', 'Error', 'Done']
        # alpha sorts ahead of the failing service and is unaffected by it.
        assert workload.get_service('alpha').current == ops.pebble.ServiceStatus.ACTIVE
        # What this does NOT show: whether real Pebble holds a task whose
        # dependency errored, rather than running it -- charlie depends on
        # bravo here, and probe #3 didn't combine FAILS with a dependency
        # (§22.1 used startup: disabled services with no failing behaviour).
        # Scenario follows "declare, don't derive" throughout this design
        # and resolves each service's status from its own ServiceBehaviour
        # only, so charlie -- which has none -- still reaches ACTIVE despite
        # its failed dependency. A test wanting charlie held down needs its
        # own declared FAILS/EXITS on charlie; this mock does not infer it
        # from bravo's failure.
        assert workload.get_service('charlie').current == ops.pebble.ServiceStatus.ACTIVE


def test_service_dependency_restart_uses_independent_stop_and_start_orders():
    """restart's stop pass is StopOrder; its start pass is StartOrder.

    Neither is derived from the other. Confirms the §14.1 "independent
    StopOrder pass, then independent StartOrder pass" mechanism still holds
    once dependencies are involved: the two passes disagree about more than
    direction here (see test_service_dependency_stop_order_matches_probe).
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _dependency_layer()},
        service_behaviours={ServiceBehaviour('bravo', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.restart('charlie', 'delta', 'alpha', 'bravo')
        tasks = exc_info.value.change.tasks
        assert [t.summary for t in tasks if t.kind == 'stop'] == [
            'Stop service "alpha"',
            'Stop service "charlie"',
            'Stop service "bravo"',
            'Stop service "delta"',
        ]
        assert [t.summary for t in tasks if t.kind == 'start'] == [
            'Start service "alpha"',
            'Start service "delta"',
            'Start service "bravo"',
            'Start service "charlie"',
        ]


def test_service_dependency_stop_order_matches_probe():
    """StopOrder is not StartOrder's list reversed -- each dependency edge points the other way.

    Real-Pebble probe #3, WORKLOAD-MOCK-DESIGN.md §22.2: the same layer and
    request as the start case (charlie, delta, alpha, bravo) stopped instead
    of started came back alpha, charlie, bravo, delta. Reversing the start
    order (alpha, delta, bravo, charlie) would give charlie, bravo, delta,
    alpha, which is a different sequence -- stopping recomputes the
    topological sort over the inverted dependency graph, it does not just
    reverse the start list.
    """
    container = Container('foo', can_connect=True, layers={'base': _dependency_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        change_id = workload.pebble.stop_services(['charlie', 'delta', 'alpha', 'bravo'])
        change = workload.pebble.get_change(change_id)
        assert change.kind == 'stop'
        assert [t.kind for t in change.tasks] == ['stop'] * 4
        assert [t.summary for t in change.tasks] == [
            'Stop service "alpha"',
            'Stop service "charlie"',
            'Stop service "bravo"',
            'Stop service "delta"',
        ]
        # Matches start/restart's rule, not autostart/replan's: the summary
        # leads with the caller's first-requested name (§22.3) -- stop has
        # no autostart/replan-style entry point to lead alphabetically
        # instead.
        assert change.summary == 'Stop service "charlie" and 3 more'
        for name in ('alpha', 'bravo', 'charlie', 'delta'):
            assert workload.get_service(name).current == ops.pebble.ServiceStatus.INACTIVE


def _requires_only_layer() -> ops.pebble.Layer:
    """The exact shape from Real-Pebble probe #4, WORKLOAD-MOCK-DESIGN.md §24.3
    (probe6-layers/004-requires-only.yaml): yankee requires zulu, with no
    `after` at all. Both startup: disabled.
    """
    return ops.pebble.Layer({
        'services': {
            'zulu': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'yankee': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['zulu'],
            },
        }
    })


def test_service_requires_pulls_in_undeclared_member_without_ordering():
    """`requires` is membership, `after` is ordering -- and they are separate.

    Real-Pebble probe #4 (WORKLOAD-MOCK-DESIGN.md §24.3): `pebble start
    yankee`, where yankee requires zulu with no `after`, produced a change
    containing both services, with yankee running *first* -- alphabetically,
    despite being the one with the requirement. zulu was never requested.
    Confirms the mock pulls zulu into the change anyway, and that a bare
    `requires` edge is not treated as an ordering edge -- the tie-break
    stays alphabetical, not requirement-first.
    """
    container = Container('foo', can_connect=True, layers={'base': _requires_only_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        change_id = workload.pebble.start_services(['yankee'])
        change = workload.pebble.get_change(change_id)
        assert change.kind == 'start'
        assert [t.summary for t in change.tasks] == [
            'Start service "yankee"',
            'Start service "zulu"',
        ]
        assert [t.status for t in change.tasks] == ['Done', 'Done']
        assert change.summary == 'Start service "yankee" and 1 more'
        assert workload.get_service('yankee').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('zulu').current == ops.pebble.ServiceStatus.ACTIVE


def test_service_requires_only_dependent_not_held_when_it_sorts_first():
    """A bare `requires` dependent is not held if it sorts ahead of the failure.

    Real-Pebble probe #8 (WORKLOAD-MOCK-DESIGN.md §32.5/§32.8, and probe #4's
    own §31.2 recheck with this exact yankee/zulu shape): `Hold` is not
    decided by `requires` closure membership alone -- a service also has to
    be ordered *after* the failing one in the realized
    `_service_dependency_order` output, and with no `after` edge at all that
    position comes from the alphabetical tie-break. yankee requires zulu
    with no `after`; `yankee` < `zulu` alphabetically, so yankee is ordered
    first, runs to completion before zulu's failure is known, and reaches
    `Done`/`ACTIVE` -- it is *not* held, even though zulu is in its
    `requires` closure and fails. This corrects the mock's earlier
    over-holding behaviour on exactly this shape (§31); see
    test_service_requires_only_dependent_held_when_it_sorts_last for the
    paired shape where renaming flips the tie-break and the dependent *is*
    held.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _requires_only_layer()},
        service_behaviours={ServiceBehaviour('zulu', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('yankee')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "yankee"',
            'Start service "zulu"',
        ]
        assert [t.status for t in change.tasks] == ['Done', 'Error']
        assert workload.get_service('yankee').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('zulu').current != ops.pebble.ServiceStatus.ACTIVE


def _requires_only_layer_tiebreak_reversed() -> ops.pebble.Layer:
    """The same declared shape as _requires_only_layer -- one bare `requires`
    edge, no `after` at all -- renamed so the alphabetical tie-break sorts
    the dependent *after* the target instead of before it (Real-Pebble
    probe #8, WORKLOAD-MOCK-DESIGN.md §32.5's paired layers 005/006: same
    edges, only the names change). `zdep` requires `atgt`; `atgt` < `zdep`.
    """
    return ops.pebble.Layer({
        'services': {
            'atgt': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'zdep': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['atgt'],
            },
        }
    })


def test_service_requires_only_dependent_held_when_it_sorts_last():
    """The same bare `requires` shape, renamed so the tie-break flips -- and the outcome with it.

    Real-Pebble probe #8 (WORKLOAD-MOCK-DESIGN.md §32.5/§32.8): identical
    declared edges to
    test_service_requires_only_dependent_not_held_when_it_sorts_first --
    one `requires` edge, no `after` -- with the two names swapped so `zdep`
    (the dependent) sorts *after* `atgt` (the target) instead of before it.
    Real Pebble held the renamed dependent in exactly this shape (§32.5's
    `atop`/`zbot`/`mmid` and `aamid`/`aatop`/`zzbot` layers); only a check
    against realized-order position, not graph reachability, gets both
    members of this pair right at once -- a reachability check would call
    this one `Done` too, since neither shape has an `after` edge to walk.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _requires_only_layer_tiebreak_reversed()},
        service_behaviours={ServiceBehaviour('atgt', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('zdep')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "atgt"',
            'Start service "zdep"',
        ]
        assert [t.status for t in change.tasks] == ['Error', 'Hold']
        assert workload.get_service('zdep').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('atgt').current != ops.pebble.ServiceStatus.ACTIVE


def _transitive_requires_layer() -> ops.pebble.Layer:
    """ctop requires bmid requires afail (Real-Pebble probe #5,
    WORKLOAD-MOCK-DESIGN.md §25.1; probe7-layers/001-transitive-hold.yaml).
    ctop names only bmid -- afail is two `requires` hops away and is never
    named directly by the service that pulls it in.
    """
    return ops.pebble.Layer({
        'services': {
            'afail': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'bmid': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['afail'],
                'after': ['afail'],
            },
            'ctop': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['bmid'],
                'after': ['bmid'],
            },
        }
    })


def test_service_requires_transitive_closure_pulls_in_full_chain():
    """A `requires` chain is held for its full length, not just one hop.

    Real-Pebble probe #5 (WORKLOAD-MOCK-DESIGN.md §25.1): ctop requires
    bmid requires afail, and ctop never names afail -- only bmid. afail
    fails; real Pebble holds bmid's task (the immediate dependent) *and*
    ctop's (two `requires` hops away, with no direct relationship to afail
    at all). An implementation that only checked a service's immediate
    `requires` list against its own status would get bmid right and ctop
    wrong -- this asserts the deep dependent specifically, not just the
    near one.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _transitive_requires_layer()},
        service_behaviours={ServiceBehaviour('afail', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('ctop')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "afail"',
            'Start service "bmid"',
            'Start service "ctop"',
        ]
        assert [t.status for t in change.tasks] == ['Error', 'Hold', 'Hold']
        assert change.summary == 'Start service "ctop" and 2 more'
        # Real Pebble's error message names only afail, the task that
        # actually failed -- not the two it held because of it.
        assert 'afail' in str(exc_info.value)
        assert 'bmid' not in str(exc_info.value)
        assert 'ctop' not in str(exc_info.value)
        assert workload.get_service('bmid').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('ctop').current == ops.pebble.ServiceStatus.INACTIVE


def _two_failures_layer() -> ops.pebble.Layer:
    """duomdep requires duoafail and duozfail independently, both of which fail.

    Real-Pebble probe #9 (WORKLOAD-MOCK-DESIGN.md §34.1;
    probe11-layers/001-two-failures-tiebreak-favourable.yaml). No `after`
    edges anywhere, and the two failures have no `requires` relationship to
    each other -- they are siblings under the one dependent, which is named
    so it sorts alphabetically *between* them.
    """
    return ops.pebble.Layer({
        'services': {
            'duoafail': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'duozfail': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'duomdep': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['duoafail', 'duozfail'],
            },
        }
    })


def test_service_requires_two_failures_hold_the_second_failure_and_the_dependent():
    """A second failure in the same lane is held, and never fails on its own.

    Real-Pebble probe #9 (WORKLOAD-MOCK-DESIGN.md §34.1): duomdep requires
    duoafail and duozfail, both declared FAILS, neither requiring the other.
    Real Pebble errored duoafail (first in the realized order), then held
    *both* duomdep and duozfail. duozfail holding is the discriminating
    result: nothing in its own `requires` closure fails -- it requires
    nothing at all -- and it declares FAILS itself, yet it never gets a turn
    to run, because Pebble chains the tasks of one lane serially and duozfail
    shares duomdep's lane (§34.5). A rule that asked only whether a failing
    service sorts earlier in this service's own closure would let duozfail
    error independently; a rule about lanes holds it. See
    test_service_requires_two_failures_hold_the_second_failure_not_the_leading_dependent
    for the same edges renamed so the tie-break moves the dependent ahead of
    both failures.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _two_failures_layer()},
        service_behaviours={
            ServiceBehaviour('duoafail', start=ServiceStart.FAILS),
            ServiceBehaviour('duozfail', start=ServiceStart.FAILS),
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('duomdep')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "duoafail"',
            'Start service "duomdep"',
            'Start service "duozfail"',
        ]
        assert [t.status for t in change.tasks] == ['Error', 'Hold', 'Hold']
        assert change.summary == 'Start service "duomdep" and 2 more'
        # Only the task that actually errored is named: duozfail's own
        # failure never manifested, so there is nothing to report about it.
        assert 'duoafail' in str(exc_info.value)
        assert 'duozfail' not in str(exc_info.value)
        assert workload.get_service('duoafail').current == 'backoff'
        assert workload.get_service('duomdep').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('duozfail').current == ops.pebble.ServiceStatus.INACTIVE


def _two_failures_layer_tiebreak_reversed() -> ops.pebble.Layer:
    """The same declared shape as _two_failures_layer -- one dependent requiring
    two independently-failing services, no `after` at all -- renamed so the
    dependent sorts *before* both failures instead of between them
    (Real-Pebble probe #9, WORKLOAD-MOCK-DESIGN.md §34.2;
    probe11-layers/002-two-failures-tiebreak-reversed.yaml).
    """
    return ops.pebble.Layer({
        'services': {
            'duqhead': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['duqmfail', 'duqzfail'],
            },
            'duqmfail': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'duqzfail': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
        }
    })


def test_service_requires_two_failures_hold_the_second_failure_not_the_leading_dependent():
    """The same two-failure shape, renamed so the dependent leads -- and runs.

    Real-Pebble probe #9 (WORKLOAD-MOCK-DESIGN.md §34.2): identical declared
    edges to
    test_service_requires_two_failures_hold_the_second_failure_and_the_dependent,
    renamed so duqhead sorts ahead of both failures. Real Pebble ran duqhead
    to completion (Done/active), errored duqmfail, and held duqzfail. The
    pair is what separates the two readings: holding duqzfail is not
    explained by anything in its own `requires` closure in either naming, but
    duqhead's fate does flip with the tie-break, so a lane-position rule is
    the only one that gets all six answers right across the pair.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _two_failures_layer_tiebreak_reversed()},
        service_behaviours={
            ServiceBehaviour('duqmfail', start=ServiceStart.FAILS),
            ServiceBehaviour('duqzfail', start=ServiceStart.FAILS),
        },
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('duqhead')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "duqhead"',
            'Start service "duqmfail"',
            'Start service "duqzfail"',
        ]
        assert [t.status for t in change.tasks] == ['Done', 'Error', 'Hold']
        assert 'duqmfail' in str(exc_info.value)
        assert 'duqzfail' not in str(exc_info.value)
        assert workload.get_service('duqhead').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('duqmfail').current == 'backoff'
        assert workload.get_service('duqzfail').current == ops.pebble.ServiceStatus.INACTIVE


def _branch_requires_layer() -> ops.pebble.Layer:
    """A branching `requires` graph: brntop requires brnmreq and brnnfree,
    brnmreq requires brnfail, and brnnfree requires nothing at all.

    Real-Pebble probe #9 (WORKLOAD-MOCK-DESIGN.md §34.3;
    probe11-layers/003-branch-graph-tiebreak-favourable.yaml). No `after`
    edges; named so the failure sorts first. brnnfree is the branch with no
    causal path to the failure in either direction -- it is joined to it only
    by brntop, which requires both.
    """
    return ops.pebble.Layer({
        'services': {
            'brnfail': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'brnmreq': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['brnfail'],
            },
            'brnnfree': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'brntop': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['brnmreq', 'brnnfree'],
            },
        }
    })


def test_service_requires_branch_sibling_held_through_the_shared_lane():
    """A service with no path to the failure still holds, if a lane-mate merges them.

    Real-Pebble probe #9 (WORKLOAD-MOCK-DESIGN.md §34.3): brnnfree requires
    nothing, is required only by brntop, and has no `requires` relationship
    to brnfail in either direction, direct or transitive -- its own closure
    is just itself. Real Pebble held it anyway, along with brnmreq (the
    direct dependent) and brntop. Pebble's `createLanes` walks `requires`
    edges undirected, so brntop requiring both branches merges them into one
    lane, and every task in a lane after the erroring one holds (§34.5).
    This is the shape §33 named as untested and the mock got wrong: a
    per-service closure check reaches brnnfree by no route at all and lets it
    reach Done/ACTIVE.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _branch_requires_layer()},
        service_behaviours={ServiceBehaviour('brnfail', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('brntop')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "brnfail"',
            'Start service "brnmreq"',
            'Start service "brnnfree"',
            'Start service "brntop"',
        ]
        assert [t.status for t in change.tasks] == ['Error', 'Hold', 'Hold', 'Hold']
        assert change.summary == 'Start service "brntop" and 3 more'
        assert workload.get_service('brnfail').current == 'backoff'
        assert workload.get_service('brnmreq').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('brnnfree').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('brntop').current == ops.pebble.ServiceStatus.INACTIVE


def _branch_requires_layer_tiebreak_reversed() -> ops.pebble.Layer:
    """The same branching shape as _branch_requires_layer, renamed so the
    failure sorts *last* rather than first (Real-Pebble probe #9,
    WORKLOAD-MOCK-DESIGN.md §34.4;
    probe11-layers/004-branch-graph-tiebreak-reversed.yaml). brqhead requires
    brqmreq and brqnfree; brqmreq requires brqzfail; no `after` anywhere.
    """
    return ops.pebble.Layer({
        'services': {
            'brqhead': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['brqmreq', 'brqnfree'],
            },
            'brqmreq': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['brqzfail'],
            },
            'brqnfree': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'brqzfail': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
        }
    })


def test_service_requires_branch_sibling_not_held_when_the_failure_sorts_last():
    """The same branching shape, renamed so nothing holds at all.

    Real-Pebble probe #9 (WORKLOAD-MOCK-DESIGN.md §34.4): identical declared
    edges to test_service_requires_branch_sibling_held_through_the_shared_lane
    -- one lane again -- but with the failure last in the realized order, so
    every other member runs to completion before it errors, the direct
    dependent brqmreq included. Being in the failure's lane is not on its own
    enough to hold a service; being in it *after* the failure is. Without
    this half of the pair, the other half could equally be read as "anything
    sharing a lane with a failure holds".
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _branch_requires_layer_tiebreak_reversed()},
        service_behaviours={ServiceBehaviour('brqzfail', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('brqhead')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "brqhead"',
            'Start service "brqmreq"',
            'Start service "brqnfree"',
            'Start service "brqzfail"',
        ]
        assert [t.status for t in change.tasks] == ['Done', 'Done', 'Done', 'Error']
        assert workload.get_service('brqhead').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('brqmreq').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('brqnfree').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('brqzfail').current == 'backoff'


def _requires_cycle_layer() -> ops.pebble.Layer:
    """A `requires` cycle: loop_a requires loop_b requires loop_a.

    Real Pebble's own behaviour on a `requires` cycle is unmeasured by any
    probe (WORKLOAD-MOCK-DESIGN.md §26's probe #6 question list) -- this
    fixture only exercises the mock's defensive choice of terminating the
    closure rather than recursing forever, not a claimed match to Pebble.
    """
    return ops.pebble.Layer({
        'services': {
            'loop_a': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['loop_b'],
            },
            'loop_b': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['loop_a'],
            },
        }
    })


def test_service_requires_cycle_terminates_instead_of_hanging():
    """A `requires` cycle does not hang the membership closure.

    Not a claim about what real Pebble does with a `requires` cycle -- see
    WORKLOAD-MOCK-DESIGN.md §26's probe #6 question list -- only that this
    mock's closure walks a visited set iteratively, so a cycle terminates
    it rather than looping forever or raising.
    """
    container = Container('foo', can_connect=True, layers={'base': _requires_cycle_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        change_id = workload.pebble.start_services(['loop_a'])
        change = workload.pebble.get_change(change_id)
        assert {t.summary for t in change.tasks} == {
            'Start service "loop_a"',
            'Start service "loop_b"',
        }
        assert change.status == 'Done'
        assert workload.get_service('loop_a').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('loop_b').current == ops.pebble.ServiceStatus.ACTIVE


def _stop_requires_dependant_layer() -> ops.pebble.Layer:
    """sdep requires/after stgt (Real-Pebble probe #6, WORKLOAD-MOCK-DESIGN.md
    §28.2; probe8-layers/002-stop-with-dependants.yaml). Both long-running,
    both startup: disabled.
    """
    return ops.pebble.Layer({
        'services': {
            'stgt': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'sdep': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['stgt'],
                'after': ['stgt'],
            },
        }
    })


def test_service_requires_stop_expands_to_dependant_and_stops_it_first():
    """`stop` expands membership through `requires`, in reverse, dependent first.

    Real-Pebble probe #6 (WORKLOAD-MOCK-DESIGN.md §28.2): sdep requires/after
    stgt; stopping only stgt pulled sdep into the change too, and stopped it
    *first* -- the reverse of start order, which is the only order that
    makes sense if `requires` means what it says. Confirms stop_services,
    which previously expanded nothing (§26/§27 both left it untouched),
    now matches.
    """
    container = Container(
        'foo', can_connect=True, layers={'base': _stop_requires_dependant_layer()}
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['sdep'])
        change_id = workload.pebble.stop_services(['stgt'])
        change = workload.pebble.get_change(change_id)
        assert change.kind == 'stop'
        assert [t.summary for t in change.tasks] == [
            'Stop service "sdep"',
            'Stop service "stgt"',
        ]
        assert [t.status for t in change.tasks] == ['Done', 'Done']
        assert change.summary == 'Stop service "stgt" and 1 more'
        assert workload.get_service('sdep').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('stgt').current == ops.pebble.ServiceStatus.INACTIVE


def test_service_requires_stop_reverse_expansion_is_transitive():
    """Stop's reverse `requires` expansion walks a full chain, not one hop.

    Real-Pebble probe #6 (§28.2) measured only a single hop (sdep/stgt).
    This mock's ``_service_requires_closure(reverse=True)`` is transitive
    by construction, the same visited-set walk as the forward direction
    (§25.1) -- an inference by symmetry with the forward case, not
    something any probe has measured on real Pebble
    (WORKLOAD-MOCK-DESIGN.md §29). Reuses _transitive_requires_layer's
    afail/bmid/ctop chain (bmid requires afail, ctop requires bmid) to show
    a service two `requires` hops away -- ctop, which never names afail
    directly -- is pulled into a stop of afail and stopped ahead of it.
    """
    container = Container('foo', can_connect=True, layers={'base': _transitive_requires_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['ctop'])
        change_id = workload.pebble.stop_services(['afail'])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks] == [
            'Stop service "ctop"',
            'Stop service "bmid"',
            'Stop service "afail"',
        ]
        assert change.summary == 'Stop service "afail" and 2 more'
        for name in ('afail', 'bmid', 'ctop'):
            assert workload.get_service(name).current == ops.pebble.ServiceStatus.INACTIVE


def test_service_requires_stop_cycle_terminates_instead_of_hanging():
    """A `requires` cycle does not hang stop's reverse membership closure either.

    Mirrors test_service_requires_cycle_terminates_instead_of_hanging for
    the reverse direction this stage adds -- not a claim about real Pebble
    (WORKLOAD-MOCK-DESIGN.md §26/§28.3 leave that open), only that the
    reverse walk is exactly as cycle-safe as the forward one.
    """
    container = Container('foo', can_connect=True, layers={'base': _requires_cycle_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        change_id = workload.pebble.stop_services(['loop_a'])
        change = workload.pebble.get_change(change_id)
        assert {t.summary for t in change.tasks} == {
            'Stop service "loop_a"',
            'Stop service "loop_b"',
        }
        assert change.status == 'Done'
        assert workload.get_service('loop_a').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('loop_b').current == ops.pebble.ServiceStatus.INACTIVE


def _restart_near_hop_layer() -> ops.pebble.Layer:
    """rnabot <- rnamid <- rnatop by `requires`, with `after` on the near hop only.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.1/§36.4;
    probe12-layers/001-restart-near-hop-only.yaml). rnamid declares both
    `requires` and `after` on rnabot; rnatop declares only `requires` on
    rnamid, so the `after` graph is partial. Named so the alphabetical
    tie-break ascends with the chain.
    """
    return ops.pebble.Layer({
        'services': {
            'rnabot': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'rnamid': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['rnabot'],
                'after': ['rnabot'],
            },
            'rnatop': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['rnamid'],
            },
        }
    })


def test_service_requires_restart_stop_pass_covers_only_the_requested_names():
    """restart's stop pass does not expand; its start pass does.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.1/§36.4/§36.5):
    `pebble restart rnamid rnatop` gave two stop tasks and three start
    tasks. `api_services.go` runs StopOrder and then discards its
    reverse-`requires` expansion again with
    `intersectOrdered(payload.Services, lanes)`, while StartOrder's forward
    expansion survives -- so a restart is not a stop followed by a start,
    and restarting a service neither takes down the services that require
    it nor stops the ones it requires. The mock used to build both passes
    from the start pass's membership, so it emitted a third stop task for
    the pulled-in rnabot.

    A proper subset is the only request shape that shows this: restarting
    one name gives a one-task stop pass with no order to get wrong, and
    restarting the whole set needs no expansion at all, so it takes the
    no-task-list fast path (§36.4).

    rnamid's own `after: [rnabot]` edge is dropped rather than pulling
    rnabot in, leaving the stop pass to the alphabetical tie-break.
    """
    container = Container('foo', can_connect=True, layers={'base': _restart_near_hop_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['rnatop'])
        change_id = workload.pebble.restart_services(['rnamid', 'rnatop'])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks if t.kind == 'stop'] == [
            'Stop service "rnamid"',
            'Stop service "rnatop"',
        ]
        assert [t.summary for t in change.tasks if t.kind == 'start'] == [
            'Start service "rnabot"',
            'Start service "rnamid"',
            'Start service "rnatop"',
        ]
        # The count is the start pass's membership, not the stop pass's
        # (§36.5: the summary reads `lanes` as it stands after the start
        # pass), so three and not two.
        assert change.summary == 'Restart service "rnamid" and 2 more'
        for name in ('rnabot', 'rnamid', 'rnatop'):
            assert workload.get_service(name).current == ops.pebble.ServiceStatus.ACTIVE


def _restart_near_hop_layer_tiebreak_reversed() -> ops.pebble.Layer:
    """The same declared shape as _restart_near_hop_layer, renamed so the
    alphabetical tie-break runs opposite to the `requires` chain: top sorts
    first, bot last (Real-Pebble probe #10, WORKLOAD-MOCK-DESIGN.md
    §36.3/§36.4; probe12-layers/005-restart-near-hop-tiebreak-reversed.yaml).
    """
    return ops.pebble.Layer({
        'services': {
            'rqnatop': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['rqnmmid'],
            },
            'rqnmmid': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['rqnzbot'],
                'after': ['rqnzbot'],
            },
            'rqnzbot': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
        }
    })


def test_service_requires_restart_stop_pass_is_ordered_not_the_request_order():
    """The restricted stop pass is still ordered, not the caller's list.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.4): the tie-break
    twin of test_service_requires_restart_stop_pass_covers_only_the_requested_names.
    The same request shape (`restart <mid> <top>`) against the same declared
    edges, renamed so the tie-break runs the other way, gives
    `rqnatop, rqnmmid` where the first gave `rnamid, rnatop`. Without the
    pair, the first test reads just as well as "the stop pass is the
    requested names in request order", which is wrong.

    The start pass is the one recorded in §36.3's table for variant E:
    rqnatop starts *ahead* of the chain it requires, because nothing
    constrains it and the tie-break sorts it first.
    """
    container = Container(
        'foo', can_connect=True, layers={'base': _restart_near_hop_layer_tiebreak_reversed()}
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['rqnatop'])
        change_id = workload.pebble.restart_services(['rqnmmid', 'rqnatop'])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks if t.kind == 'stop'] == [
            'Stop service "rqnatop"',
            'Stop service "rqnmmid"',
        ]
        assert [t.summary for t in change.tasks if t.kind == 'start'] == [
            'Start service "rqnatop"',
            'Start service "rqnzbot"',
            'Start service "rqnmmid"',
        ]
        assert change.summary == 'Restart service "rqnmmid" and 2 more'


def _restart_far_hop_layer() -> ops.pebble.Layer:
    """rfabot <- rfamid <- rfatop by `requires`, with `after` on the far hop
    only (rfatop after rfamid) -- Real-Pebble probe #10,
    WORKLOAD-MOCK-DESIGN.md §36.4; probe12-layers/002-restart-far-hop-only.yaml.
    """
    return ops.pebble.Layer({
        'services': {
            'rfabot': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'rfamid': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['rfabot'],
            },
            'rfatop': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['rfamid'],
                'after': ['rfamid'],
            },
        }
    })


def test_service_requires_restart_stop_pass_inverts_edges_inside_the_request():
    """An `after` edge between two requested names still inverts in the stop pass.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.4, variant B):
    `pebble restart rfamid rfatop` stopped rfatop first. Restricting the
    stop pass to the requested names (the fix above) must not also flatten
    it to alphabetical order -- rfatop declares `after: [rfamid]`, both are
    in the request, so the edge survives the restriction and inverts, which
    is the opposite of the name order. The start pass is unchanged and
    still runs bot-first.
    """
    container = Container('foo', can_connect=True, layers={'base': _restart_far_hop_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['rfatop'])
        change_id = workload.pebble.restart_services(['rfamid', 'rfatop'])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks if t.kind == 'stop'] == [
            'Stop service "rfatop"',
            'Stop service "rfamid"',
        ]
        assert [t.summary for t in change.tasks if t.kind == 'start'] == [
            'Start service "rfabot"',
            'Start service "rfamid"',
            'Start service "rfatop"',
        ]


def _two_lanes_interleaved_layer() -> ops.pebble.Layer:
    """Two `requires` lanes whose members interleave in the alphabetical order.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.6;
    probe12-layers/009-stop-two-lanes-interleaved.yaml). slcthree requires
    slaone and sldfour requires slbtwo, so the lanes are
    {slaone, slcthree} and {slbtwo, sldfour} while the realized order is
    slaone, slbtwo, slcthree, sldfour. No `after` edges anywhere, so the
    realized order is pure tie-break and lane grouping is the only thing
    that can move a task.
    """
    return ops.pebble.Layer({
        'services': {
            'slaone': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'slbtwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'slcthree': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slaone'],
            },
            'sldfour': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slbtwo'],
            },
        }
    })


def test_service_requires_stop_emits_tasks_grouped_by_lane():
    """A stop change's tasks come out lane by lane, not in the realized order.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.6/§36.9): stopping
    all four gave slaone, slcthree, slbtwo, sldfour, where the flat realized
    order is slaone, slbtwo, slcthree, sldfour. `StopOrder` lanes its own
    output with the same `createLanes` `StartOrder` uses, and
    `servstate/request.go`'s `Stop` builds the task list lane by lane. The
    mock emitted the flat order until now; every shape measured before this
    probe was single-lane, all-singleton, or contiguous by accident, so the
    two coincided.
    """
    container = Container('foo', can_connect=True, layers={'base': _two_lanes_interleaved_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['slaone', 'slbtwo', 'slcthree', 'sldfour'])
        change_id = workload.pebble.stop_services(['slaone', 'slbtwo', 'slcthree', 'sldfour'])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks] == [
            'Stop service "slaone"',
            'Stop service "slcthree"',
            'Stop service "slbtwo"',
            'Stop service "sldfour"',
        ]
        assert change.summary == 'Stop service "slaone" and 3 more'
        for name in ('slaone', 'slbtwo', 'slcthree', 'sldfour'):
            assert workload.get_service(name).current == ops.pebble.ServiceStatus.INACTIVE


def _two_lanes_pairing_swapped_layer() -> ops.pebble.Layer:
    """The same four alphabetical positions as _two_lanes_interleaved_layer,
    with the `requires` edges pairing first-with-last and second-with-third
    instead (Real-Pebble probe #10, WORKLOAD-MOCK-DESIGN.md §36.6;
    probe12-layers/010-stop-two-lanes-pairing-swapped.yaml).
    """
    return ops.pebble.Layer({
        'services': {
            'slqaone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'slqbtwo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'slqcthree': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slqbtwo'],
            },
            'slqdfour': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slqaone'],
            },
        }
    })


def test_service_requires_stop_lane_grouping_follows_the_edges_not_the_names():
    """The pairing twin: same names, different edges, different task order.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.6): slqaone,
    slqdfour, slqbtwo, slqcthree, against slaone, slcthree, slbtwo,
    sldfour for the layer above under an identical alphabetical order. One
    layer alone could be read as a naming artefact; the pair cannot, and a
    rule that ignored lanes would give a, b, c, d for both.
    """
    container = Container(
        'foo', can_connect=True, layers={'base': _two_lanes_pairing_swapped_layer()}
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['slqaone', 'slqbtwo', 'slqcthree', 'slqdfour'])
        change_id = workload.pebble.stop_services([
            'slqaone',
            'slqbtwo',
            'slqcthree',
            'slqdfour',
        ])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks] == [
            'Stop service "slqaone"',
            'Stop service "slqdfour"',
            'Stop service "slqbtwo"',
            'Stop service "slqcthree"',
        ]


def test_service_requires_stop_reverse_expanded_members_are_lane_grouped_too():
    """Members pulled in by stop's reverse expansion are laned like the rest.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.6): `pebble stop
    slaone slbtwo` pulls in slcthree and sldfour (§28.2's reverse
    expansion) and still emits slaone, slcthree, slbtwo, sldfour. The
    laning happens after membership is settled, so it does not matter
    whether a member was requested or pulled in.
    """
    container = Container('foo', can_connect=True, layers={'base': _two_lanes_interleaved_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['slaone', 'slbtwo', 'slcthree', 'sldfour'])
        change_id = workload.pebble.stop_services(['slaone', 'slbtwo'])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks] == [
            'Stop service "slaone"',
            'Stop service "slcthree"',
            'Stop service "slbtwo"',
            'Stop service "sldfour"',
        ]
        assert change.summary == 'Stop service "slaone" and 3 more'


def _two_lanes_with_after_layer() -> ops.pebble.Layer:
    """_two_lanes_interleaved_layer with an `after` edge inside each lane, so
    stop's `before`/`after` swap has something to act on (Real-Pebble probe
    #10, WORKLOAD-MOCK-DESIGN.md §36.7;
    probe12-layers/011-stop-two-lanes-with-after.yaml).
    """
    return ops.pebble.Layer({
        'services': {
            'slraone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'slrbtwo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'slrcthree': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slraone'],
                'after': ['slraone'],
            },
            'slrdfour': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slrbtwo'],
                'after': ['slrbtwo'],
            },
        }
    })


def test_service_requires_stop_lanes_are_computed_on_stops_own_realized_order():
    """The lanes come out in stop order, with each lane's contents reversed.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.7): slrcthree,
    slraone, slrdfour, slrbtwo. The lanes are the same two as the start
    pass's and come in the same order, but each lane's contents are
    reversed -- so `StopOrder` lanes the result of its own
    `order(..., stop=true)` rather than reusing a partition computed on the
    start order. This shape's task list happens to be lane-grouped already,
    which is why it agreed with the mock before the fix and why the
    interleaving pair above was needed to find the bug.
    """
    container = Container('foo', can_connect=True, layers={'base': _two_lanes_with_after_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['slraone', 'slrbtwo', 'slrcthree', 'slrdfour'])
        change_id = workload.pebble.stop_services([
            'slraone',
            'slrbtwo',
            'slrcthree',
            'slrdfour',
        ])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks] == [
            'Stop service "slrcthree"',
            'Stop service "slraone"',
            'Stop service "slrdfour"',
            'Stop service "slrbtwo"',
        ]


def _two_lanes_one_failing_layer() -> ops.pebble.Layer:
    """Two lanes, one of which contains a failing service (Real-Pebble probe
    #10, WORKLOAD-MOCK-DESIGN.md §36.8/§36.9;
    probe12-layers/012-stop-two-lanes-one-failing.yaml). slfcdep requires
    slfafail, slfdok requires slfbok; no `after` edges.
    """
    return ops.pebble.Layer({
        'services': {
            'slfafail': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'slfbok': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'disabled'},
            'slfcdep': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slfafail'],
            },
            'slfdok': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['slfbok'],
            },
        }
    })


def test_service_requires_start_emits_tasks_grouped_by_lane_with_one_lane_failing():
    """A start change's tasks are lane-grouped too, and a clean lane is untouched.

    Real-Pebble probe #10 (WORKLOAD-MOCK-DESIGN.md §36.8/§36.9): starting
    all four gave slfafail (Error), slfcdep (Hold), slfbok (Done), slfdok
    (Done) -- lane-grouped, where the flat realized order is slfafail,
    slfbok, slfcdep, slfdok. This is also the first measurement of §34/§35's
    lane hold rule on a genuinely multi-lane graph: every shape in §34 was
    one lane, so "held because of a lane-mate" and "held because of anything
    earlier in the change" could not be separated there. Lane B is untouched
    by lane A's failure.
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _two_lanes_one_failing_layer()},
        service_behaviours={ServiceBehaviour('slfafail', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.start('slfafail', 'slfbok', 'slfcdep', 'slfdok')
        change = exc_info.value.change
        assert [t.summary for t in change.tasks] == [
            'Start service "slfafail"',
            'Start service "slfcdep"',
            'Start service "slfbok"',
            'Start service "slfdok"',
        ]
        assert [t.status for t in change.tasks] == ['Error', 'Hold', 'Done', 'Done']
        assert change.summary == 'Start service "slfafail" and 3 more'
        assert workload.get_service('slfafail').current == 'backoff'
        assert workload.get_service('slfcdep').current == ops.pebble.ServiceStatus.INACTIVE
        assert workload.get_service('slfbok').current == ops.pebble.ServiceStatus.ACTIVE
        assert workload.get_service('slfdok').current == ops.pebble.ServiceStatus.ACTIVE
        assert 'slfafail' in str(exc_info.value)
        assert 'slfcdep' not in str(exc_info.value)


def _required_autostart_layer() -> ops.pebble.Layer:
    """rdep requires/after rfail, both startup: enabled (Real-Pebble probe #5,

    WORKLOAD-MOCK-DESIGN.md §25.2/§25.3; probe7-layers/002-required-autostart.yaml).
    rfail is not itself requested by anything outside this layer -- autostart
    and replan pick it up because it's enabled, the same as rdep.
    """
    return ops.pebble.Layer({
        'services': {
            'rdep': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'enabled',
                'requires': ['rfail'],
                'after': ['rfail'],
            },
            'rfail': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
        }
    })


@pytest.mark.parametrize('op', ['autostart', 'replan'])
def test_service_requires_autostart_replan_share_cascade_logic(op: str):
    """autostart and replan hold a required-but-not-requested failure exactly like start.

    Real-Pebble probe #5 §25.2: rdep requires/after rfail; rfail fails.
    Under both autostart and replan the dependency's task is `Error`, the
    dependent's is `Hold`, and the dependent service never starts -- no
    special-casing needed for the cascade mechanism between entry points
    (unlike the summary's leading name, which does diverge -- see §25.3 and
    test_service_requires_autostart_replan_leading_name_diverges).
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _required_autostart_layer()},
        service_behaviours={ServiceBehaviour('rfail', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            getattr(workload, op)()
        change = exc_info.value.change
        assert change.kind == op
        assert [(t.summary, t.status) for t in change.tasks] == [
            ('Start service "rfail"', 'Error'),
            ('Start service "rdep"', 'Hold'),
        ]
        assert workload.get_service('rdep').current == ops.pebble.ServiceStatus.INACTIVE


def test_service_requires_autostart_replan_leading_name_diverges():
    """autostart and replan diverge once `requires` pulls in an out-of-order member.

    Real-Pebble probe #5 §25.3: rdep requires/after rfail -- alphabetically
    `rdep` < `rfail`, but the dependency edge forces `rfail` first in start
    order. autostart's summary leads with the first *task* (`"rfail"`,
    topological); replan's leads with the alphabetically-first member of the
    affected union (`"rdep"`). The task order itself is identical between
    the two changes -- only the summary's leading name differs, which is
    exactly what makes this a change to the summary logic, not to cascade
    (test_service_requires_autostart_replan_share_cascade_logic covers
    cascade itself and deliberately doesn't assert on the summary).
    """
    container = Container(
        'foo',
        can_connect=True,
        layers={'base': _required_autostart_layer()},
        service_behaviours={ServiceBehaviour('rfail', start=ServiceStart.FAILS)},
    )
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.autostart()
        assert exc_info.value.change.summary == 'Autostart service "rfail" and 1 more'

    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(ops.pebble.ChangeError) as exc_info:
            workload.replan()
        change = exc_info.value.change
        assert change.summary == 'Replan service "rdep" and 1 more'
        # Same task order both times -- only the summary's leading name moved.
        assert [t.summary for t in change.tasks] == [
            'Start service "rfail"',
            'Start service "rdep"',
        ]


class NotifyingStartCharm(ops.CharmBase):
    """Drives one Pebble entry point, then records what the container reports."""

    entry_point = 'start'
    services: tuple[str, ...] = ('myapp',)
    seen: list[ops.pebble.Notice]
    errors: list[ops.pebble.ChangeError]

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.config_changed, self._on_config_changed)

    def _on_config_changed(self, _: ops.EventBase):
        container = self.unit.get_container('foo')
        try:
            if self.entry_point == 'start':
                container.start(*self.services)
            elif self.entry_point == 'restart':
                container.restart(*self.services)
            elif self.entry_point == 'replan':
                container.replan()
            else:
                container.autostart()
        except ops.pebble.ChangeError as e:
            NotifyingStartCharm.errors.append(e)
        # Scenario runs a subclass of the charm, so mutate rather than assign.
        NotifyingStartCharm.seen.extend(container.pebble.get_notices())


NOTICE_LAYER = ops.pebble.Layer({
    'services': {
        'myapp': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
        'other': {'override': 'replace', 'command': '/bin/sleep 1000', 'startup': 'enabled'},
    }
})


def _notice_state(
    behaviours: set[ServiceBehaviour],
    entry_point: str = 'start',
    services: tuple[str, ...] = ('myapp',),
) -> tuple[Context[NotifyingStartCharm], State]:
    NotifyingStartCharm.seen = []
    NotifyingStartCharm.errors = []
    NotifyingStartCharm.entry_point = entry_point
    NotifyingStartCharm.services = services
    ctx = Context(NotifyingStartCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    container = Container(
        'foo',
        can_connect=True,
        layers={'layer1': NOTICE_LAYER},
        service_behaviours=frozenset(behaviours),
    )
    return ctx, State(containers={container})


@pytest.mark.parametrize('entry_point', ['start', 'restart', 'replan', 'autostart'])
def test_service_notice_emitted_when_the_service_starts(entry_point: str):
    """Every entry point that starts the service records its declared notices."""
    ctx, state = _notice_state(
        behaviours={ServiceBehaviour('myapp', emits=[Notice('example.com/started')])},
        entry_point=entry_point,
    )
    state_out = ctx.run(ctx.on.config_changed(), state=state)
    assert [notice.key for notice in NotifyingStartCharm.seen] == ['example.com/started']
    # And it is in the output state, which is what lets the next run use it.
    assert [notice.key for notice in state_out.get_container('foo').notices] == [
        'example.com/started'
    ]


def test_service_notice_carries_its_type_and_data():
    ctx, state = _notice_state(
        behaviours={
            ServiceBehaviour(
                'myapp',
                emits=[Notice('example.com/started', last_data={'version': '2'})],
            )
        },
    )
    state_out = ctx.run(ctx.on.config_changed(), state=state)
    (notice,) = state_out.get_container('foo').notices
    assert notice.type == ops.pebble.NoticeType.CUSTOM
    assert notice.last_data == {'version': '2'}
    # Pebble assigns the ID and the timestamps, so they are not the declared ones.
    assert notice.occurrences == 1


def test_service_notice_repeats_rather_than_duplicating():
    """Starting a service twice gives one notice with two occurrences, as Pebble does."""
    ctx, state = _notice_state(
        behaviours={ServiceBehaviour('myapp', emits=[Notice('example.com/started')])},
    )
    state_out = ctx.run(ctx.on.config_changed(), state=state)
    state_out = ctx.run(ctx.on.config_changed(), state=state_out)
    (notice,) = state_out.get_container('foo').notices
    assert notice.occurrences == 2


def test_failing_service_emits_nothing():
    """A service that never starts can't have notified anything."""
    ctx, state = _notice_state(
        behaviours={
            ServiceBehaviour(
                'myapp',
                start=ServiceStart.FAILS,
                emits=[Notice('example.com/started')],
            )
        },
    )
    state_out = ctx.run(ctx.on.config_changed(), state=state)
    assert NotifyingStartCharm.errors
    assert NotifyingStartCharm.seen == []
    assert state_out.get_container('foo').notices == []


def test_exiting_service_still_emits():
    """An EXITS service does start, so its notices are recorded."""
    ctx, state = _notice_state(
        behaviours={
            ServiceBehaviour(
                'myapp',
                start=ServiceStart.EXITS,
                exit_code=ServiceExitCode.FAILURE,
                emits=[Notice('example.com/started')],
            )
        },
    )
    state_out = ctx.run(ctx.on.config_changed(), state=state)
    assert [notice.key for notice in state_out.get_container('foo').notices] == [
        'example.com/started'
    ]


def test_service_notice_emitted_beside_a_failing_service():
    """One service failing doesn't silence a service that did start."""
    ctx, state = _notice_state(
        behaviours={
            ServiceBehaviour('myapp', start=ServiceStart.FAILS),
            ServiceBehaviour('other', emits=[Notice('example.com/other-started')]),
        },
        services=('myapp', 'other'),
    )
    state_out = ctx.run(ctx.on.config_changed(), state=state)
    assert NotifyingStartCharm.errors
    assert [notice.key for notice in state_out.get_container('foo').notices] == [
        'example.com/other-started'
    ]


class NoticeHandlerCharm(ops.CharmBase):
    handled: list[str]

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.foo_pebble_custom_notice, self._on_notice)

    def _on_notice(self, event: ops.PebbleCustomNoticeEvent):
        NoticeHandlerCharm.handled.append(event.notice.key)


def test_service_notice_can_be_handled_by_the_next_run():
    """The point of the notice reaching the output state: the next run handles it."""
    ctx, state = _notice_state(
        behaviours={ServiceBehaviour('myapp', emits=[Notice('example.com/started')])},
    )
    state_out = ctx.run(ctx.on.config_changed(), state=state)
    container_out = state_out.get_container('foo')
    (notice,) = container_out.notices

    NoticeHandlerCharm.handled = []
    handler_ctx = Context(NoticeHandlerCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    handler_ctx.run(
        handler_ctx.on.pebble_custom_notice(container=container_out, notice=notice),
        state=state_out,
    )
    assert NoticeHandlerCharm.handled == ['example.com/started']


class SelfNotifyingCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.config_changed, self._on_config_changed)

    def _on_config_changed(self, _: ops.EventBase):
        container = self.unit.get_container('foo')
        container.pebble.notify(ops.pebble.NoticeType.CUSTOM, 'example.com/charm-said')


def test_charm_notify_reaches_the_output_state():
    """A notice the charm itself records is kept, as Pebble keeps it."""
    ctx = Context(SelfNotifyingCharm, meta={'name': 'foo', 'containers': {'foo': {}}})
    container = Container('foo', can_connect=True)
    state_out = ctx.run(ctx.on.config_changed(), state=State(containers={container}))
    assert [notice.key for notice in state_out.get_container('foo').notices] == [
        'example.com/charm-said'
    ]


def _ordering_cycle_layer() -> ops.pebble.Layer:
    """ant/bee ordered after each other (Real-Pebble probe #12,
    WORKLOAD-MOCK-DESIGN.md §39.1; the shape the daemon rejected with
    `400 Bad Request: services in before/after loop: ant, bee`).
    """
    return ops.pebble.Layer({
        'services': {
            'bee': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'before': ['ant'],
            },
            'ant': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'before': ['bee'],
            },
        },
    })


def _self_edge_layer() -> ops.pebble.Layer:
    """solo declares itself in its own `after` (probe #12 §39.2).

    Real Pebble accepts this and starts the service normally, which is why
    it is a separate fixture from `_ordering_cycle_layer` rather than
    another case of it.
    """
    return ops.pebble.Layer({
        'services': {
            'solo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'after': ['solo'],
            },
        },
    })


def test_add_layer_rejects_an_ordering_cycle():
    """`add_layer` refuses a layer that puts the plan in a before/after loop.

    Status and message are the daemon's own, read off the unix socket in
    Real-Pebble probe #12 (WORKLOAD-MOCK-DESIGN.md §39.1).
    """
    container = Container('foo', can_connect=True)
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(RuntimeError) as excinfo:
            workload.pebble.add_layer('cyc', _ordering_cycle_layer())
        assert '400 Bad Request: services in before/after loop: ant, bee' in str(excinfo.value)


def test_add_layer_rejecting_a_cycle_leaves_the_plan_alone():
    """A refused layer does not land, matching `pebble plan` after a rejection.

    Probe #12 §39.1: the three rejected layers never appeared in the plan.
    """
    container = Container('foo', can_connect=True)
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(RuntimeError):
            workload.pebble.add_layer('cyc', _ordering_cycle_layer())
        assert workload.pebble.get_plan().services == {}
        # And the container is still usable afterwards, rather than wedged.
        workload.pebble.add_layer('ok', _self_edge_layer())
        assert set(workload.pebble.get_plan().services) == {'solo'}


def test_ordering_cycle_in_a_hand_written_plan_is_rejected():
    """A cyclic `Container(layers=...)` is refused too, not just `add_layer`.

    Real Pebble checks at plan load as well as at layer-add time
    (WORKLOAD-MOCK-DESIGN.md §28.3), and a plan written straight into state
    never goes through `add_layer`.
    """
    container = Container('foo', can_connect=True, layers={'base': _ordering_cycle_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(RuntimeError) as excinfo:
            workload.pebble.start_services(['ant'])
        assert 'services in before/after loop: ant, bee' in str(excinfo.value)


def test_self_edge_is_not_an_ordering_cycle():
    """A service in its own `after` is accepted and starts (probe #12 §39.2).

    The one place this mock's cycle check differs from a textbook one, and
    it differs because Pebble does.
    """
    container = Container('foo', can_connect=True, layers={'base': _self_edge_layer()})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        workload.pebble.start_services(['solo'])
        assert workload.get_service('solo').current == ops.pebble.ServiceStatus.ACTIVE


def test_ordering_cycle_names_only_the_cycle_members():
    """Free services either side of a cycle are not named in the error.

    Probe #12 §39.1 added `free1`/`free2` beside a `bravo`/`zulu` loop and
    the daemon reported only `bravo, zulu`.
    """
    layer = ops.pebble.Layer({
        'services': {
            'free1': {'override': 'replace', 'command': '/bin/sleep 1000'},
            'zulu': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['bravo'],
            },
            'bravo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['zulu'],
            },
            'free2': {'override': 'replace', 'command': '/bin/sleep 1000'},
        },
    })
    container = Container('foo', can_connect=True)
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        with pytest.raises(RuntimeError) as excinfo:
            workload.pebble.add_layer('cyc', layer)
        message = str(excinfo.value)
        assert 'services in before/after loop: bravo, zulu' in message
        assert 'free1' not in message
        assert 'free2' not in message


def _add_layer_cycle_message(layer: ops.pebble.Layer) -> str:
    """The rejection message `add_layer` gives for `layer`, or '' if accepted.

    The shapes below are the layers of Real-Pebble probes #13, #14 and #15
    (WORKLOAD-MOCK-DESIGN.md §40.4/§40.5, §42 and §43), and nearly all of
    them ask the same question of `add_layer` -- what, if anything, it
    names -- so the scaffolding is shared and each test carries only its
    shape and its measured answer.

    Each shape is the probe layer with `startup: disabled` dropped, since
    the cycle check runs before anything looks at startup and every one of
    these layers is rejected. Everything the check reads -- the service
    names, their declaration order, and the contents and order of each
    `before`/`after` list -- is the layer's.
    """
    container = Container('foo', can_connect=True)
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        try:
            workload.pebble.add_layer('probe', layer)
        except RuntimeError as exc:
            return str(exc)
        return ''


def test_ordering_cycle_of_four_services_is_named_in_full():
    """A four-service loop names all four, sorted (probe #13 §40.4, `007`).

    Longer than anything probe #12 measured, which did two and three, and
    declared in neither alphabetical nor graph order so the sorted answer
    has something to disagree with: bolt after quad after zap after noel
    after bolt.
    """
    layer = ops.pebble.Layer({
        'services': {
            'quad': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zap']},
            'noel': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bolt']},
            'zap': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['noel']},
            'bolt': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['quad']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: bolt, noel, quad, zap'
    )


def test_self_edge_written_with_before_is_not_an_ordering_cycle():
    """`before: [ouro]` is accepted too, not just `after` (probe #13 §40.5, `008`).

    Probe #12 measured the self-edge only as `after` (§39.2) and measured
    separately that `before` and `after` are one constraint for a
    two-service cycle (§39.1); this is the two together, so the
    skip-self-edges rule is not written against half the syntax.
    """
    layer = ops.pebble.Layer({
        'services': {
            'ouro': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'before': ['ouro'],
            },
        },
    })
    assert _add_layer_cycle_message(layer) == ''


def test_self_edge_on_a_service_with_real_edges_leaves_the_chain_ordered():
    """A self-edge is dropped without disturbing the edges around it.

    Probe #13 §40.5, `009`: `selfmid` is `after` itself *and* requires
    `selfbase`, with `selftop` on top, and a real daemon starts the chain
    `selfbase, selfmid, selftop`. Probe #12's isolated `solo` could not
    have shown this -- an implementation that let the self-edge poison
    `selfmid`'s in-degree loses the whole chain, not just the one service.
    """
    layer = ops.pebble.Layer({
        'services': {
            'selfbase': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
            },
            'selfmid': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['selfbase'],
                'after': ['selfmid', 'selfbase'],
            },
            'selftop': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'startup': 'disabled',
                'requires': ['selfmid'],
                'after': ['selfmid'],
            },
        },
    })
    container = Container('foo', can_connect=True, layers={'base': layer})
    ctx = Context(Charm, meta={'name': 'foo', 'containers': {'foo': {}}})
    with ctx(ctx.on.start(), State(containers={container})) as mgr:
        workload = mgr.charm.unit.get_container('foo')
        change_id = workload.pebble.start_services(['selftop'])
        change = workload.pebble.get_change(change_id)
        assert [t.summary for t in change.tasks] == [
            'Start service "selfbase"',
            'Start service "selfmid"',
            'Start service "selftop"',
        ]
        for name in ('selfbase', 'selfmid', 'selftop'):
            assert workload.get_service(name).current == ops.pebble.ServiceStatus.ACTIVE


def test_self_edge_beside_a_real_cycle_is_not_named_with_it():
    """Another cycle forcing the check down the reporting path does not drag
    a self-edge in with it (probe #13 §40.5, `010`).

    `lonewolf` is `after` itself; `ring1`/`ring2` loop. Real Pebble reports
    `ring1, ring2`, although `lonewolf` sorts between them -- so a report
    that includes it is unmistakable rather than a detail at the end of a
    list.
    """
    layer = ops.pebble.Layer({
        'services': {
            'ring2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['ring1']},
            'lonewolf': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['lonewolf'],
            },
            'ring1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['ring2']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: ring1, ring2'
    )


def test_two_disjoint_cycles_name_the_members_of_only_one():
    """With two loops in one plan, Pebble names one of them (probe #13 §40.4, `006`).

    `apple`/`mango` and `fig`/`pear` loop, `kiwi` is free, and the daemon
    reported `apple, mango` -- saying nothing about `fig`/`pear` at all.
    The names are chosen so that one sorted list of every cycle member
    would interleave the two loops, which is what this mock used to emit.
    """
    layer = ops.pebble.Layer({
        'services': {
            'mango': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['apple']},
            'kiwi': {'override': 'replace', 'command': '/bin/sleep 1000'},
            'apple': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mango']},
            'pear': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['fig']},
            'fig': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['pear']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: apple, mango'
    )


def test_which_of_two_disjoint_cycles_is_named_is_not_declaration_order():
    """Declaration order does not decide which of two loops is named.

    Probe #13 §40.4, `012`: the tie-break twin of `006`, which cannot say
    *which* cycle because its first-declared service (`mango`) and its
    alphabetically-first (`apple`) are in the same one. Here they disagree
    -- first-declared is `zebra`, alphabetically-first is `alpha` -- and
    the daemon reported `alpha, beta`.

    What this shape rules out is declaration order, and that is all it was
    ever entitled to rule out. It was read as measuring "the
    alphabetically-first cyclic service", and §40.7 warned at the time
    that two agreeing shapes did not rule out an artefact of the daemon's
    own discovery order. Probe #14 showed it was one (§42.3): a walk
    visiting services alphabetically starts at `alpha` here, so the two
    rules agree on this shape and part company on eight others.
    """
    layer = ops.pebble.Layer({
        'services': {
            'zebra': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['yak']},
            'yak': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zebra']},
            'alpha': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['beta']},
            'beta': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['alpha']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: alpha, beta'
    )


def test_a_service_ordered_after_a_cycle_member_is_not_named():
    """Downstream of a cycle is not in the cycle (probe #13 §40.4, `011`).

    `dsring1`/`dsring2` loop and `dsafter` is `after dsring1`, so `dsafter`
    can never start either -- but the daemon reported `dsring1, dsring2`
    and left it out. It sorts first, so a report that includes it leads
    with it. This is the shape that rules out reporting whatever a
    topological drain cannot drain.
    """
    layer = ops.pebble.Layer({
        'services': {
            'dsring2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['dsring1']},
            'dsring1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['dsring2']},
            'dsafter': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['dsring1']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: dsring1, dsring2'
    )


def test_the_cycle_named_is_the_one_a_sorted_walk_completes_first():
    """The cycle named is not the one holding the alphabetically-first member.

    Real-Pebble probe #14 (WORKLOAD-MOCK-DESIGN.md §42.2, `013`) is the
    shape §40.9 asked for and it took the previous rule down: `aentry` is
    alphabetically first in the plan, carries no cycle, and is ordered
    after a member of the `mloop`/`nloop` loop, while `bloop`/`cloop` holds
    the alphabetically-first *cyclic* service. The daemon reported
    `mloop, nloop`.

    What fits is a depth-first walk that visits services alphabetically and
    follows each service's `after` list in written order, reporting the
    first cycle it completes. Read with the twin below, which moves the
    structure and leaves every name where it is.
    """
    layer = ops.pebble.Layer({
        'services': {
            'mloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['nloop']},
            'nloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mloop']},
            'aentry': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mloop']},
            'bloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['cloop']},
            'cloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bloop']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: mloop, nloop'
    )


def test_the_named_cycle_moves_with_the_structure_and_not_the_names():
    """The twin of the shape above (probe #14 §42.2, `014`).

    `aentry` is ordered after `bloop` instead of `mloop`; every service
    keeps its name and its alphabetical position, and the only thing that
    changes is which loop the entry service leads into. The daemon
    reported `bloop, cloop`.

    A rule that picked by name could not move here, so the pair is what
    rules out naming rules rather than either layer alone.
    """
    layer = ops.pebble.Layer({
        'services': {
            'mloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['nloop']},
            'nloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mloop']},
            'aentry': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bloop']},
            'bloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['cloop']},
            'cloop': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bloop']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: bloop, cloop'
    )


def test_a_cycle_is_not_chosen_by_being_the_largest():
    """Size does not decide, part one (probe #14 §42.2, `015`).

    Two loops, one of two services and one of three, with the
    alphabetically-first cyclic service in the smaller. The daemon
    reported `bringone, bringtwo`, the smaller one, which is also where
    the walk arrives first.

    Probe #13's shapes could not ask this: `006` and `012` both held two
    loops of two services, so neither size rule had ever been put a
    question it could fail. This shape and its twin below kill both
    between them; each on its own agrees with too much.
    """
    layer = ops.pebble.Layer({
        'services': {
            'wheelone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['wheelthree'],
            },
            'wheeltwo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['wheelone'],
            },
            'wheelthree': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['wheeltwo'],
            },
            'bringone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['bringtwo'],
            },
            'bringtwo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['bringone'],
            },
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: bringone, bringtwo'
    )


def test_a_cycle_is_not_chosen_by_being_the_smallest():
    """Size does not decide, part two (probe #14 §42.2, `016`).

    The twin of the shape above with the alphabetically-first cyclic
    service moved into the three-service loop. The daemon reported
    `apexone, apexthree, apextwo`, the larger one.

    If size governed, one of this pair would have to disagree with the
    other. Neither does. This is one of the shapes that fails to
    discriminate between the walk and the rule it replaced, and it is here
    to stop "the old rule was wrong" being read as "everything was wrong".
    """
    layer = ops.pebble.Layer({
        'services': {
            'zringone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['zringtwo'],
            },
            'zringtwo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['zringone'],
            },
            'apexone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['apexthree'],
            },
            'apextwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['apexone']},
            'apexthree': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['apextwo'],
            },
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: apexone, apexthree, apextwo'
    )


def test_three_cycles_and_only_the_one_the_walk_reaches_is_named():
    """Three loops, one answer, and it is neither naming rule (probe #14 §42.2, `017`).

    The first-declared cyclic service is `monoone`, the
    alphabetically-first is `betaone`, and the daemon reported
    `zetaone, zetatwo` -- the loop reached from `aalead`, which sorts first
    in the plan and is on no cycle. Three candidate rules, three different
    answers, in one layer.

    It also repeats two things measured on two-loop plans: exactly one
    cycle is named, and neither the free service (`freebie`) nor the
    downstream one (`aalead`) is named at all.
    """
    layer = ops.pebble.Layer({
        'services': {
            'monoone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['monotwo']},
            'monotwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['monoone']},
            'zetaone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zetatwo']},
            'zetatwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zetaone']},
            'betaone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['betatwo']},
            'betatwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['betaone']},
            'aalead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zetaone']},
            'freebie': {'override': 'replace', 'command': '/bin/sleep 1000'},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: zetaone, zetatwo'
    )


def test_the_cycle_named_is_the_first_completed_not_the_first_entered():
    """First entered and first completed are not the same (probe #14 §42.2, `018`).

    The two loops here are not disjoint components: `downone` is ordered
    after `upone`, so the walk starts at `downone`, which is itself
    cyclic, and reaches `upone`/`uptwo` through the loop it is standing
    in. `upone`/`uptwo` has no edges leading out of it and so finishes
    first. The daemon reported `upone, uptwo`.

    So it is the first cycle *completed*, not the first entered and not
    the last completed. No shape before probe #14 could separate those,
    because they only come apart when one cycle is reached through
    another.
    """
    layer = ops.pebble.Layer({
        'services': {
            'upone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['uptwo']},
            'uptwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['upone']},
            'downone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['downtwo', 'upone'],
            },
            'downtwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['downone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: upone, uptwo'
    )


def test_the_first_completed_cycle_is_named_when_it_also_sorts_first():
    """The twin of the shape above (probe #14 §42.2, `019`).

    Same chain, names permuted so that the sink loop is also the one
    holding the alphabetically-first cyclic service. The daemon reported
    `alfaone, alfatwo`.

    This is the control for the pair: the answer stays on the sink when
    the names move onto it, so the previous shape's result is not an
    artefact of the sink happening to sort second.
    """
    layer = ops.pebble.Layer({
        'services': {
            'zuluone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['zulutwo', 'alfaone'],
            },
            'zulutwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zuluone']},
            'alfaone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['alfatwo']},
            'alfatwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['alfaone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: alfaone, alfatwo'
    )


def test_a_lead_chain_carries_the_walk_past_two_unrelated_cycles():
    """A three-hop acyclic chain decides the answer (probe #14 §42.3, `020`).

    Pre-registered: written after the shapes above had been measured and
    before it was run, to test the rule they fitted rather than to fit it
    again. `aaa1` sorts first in the plan and leads through `aaa2` and
    `aaa3` into the `mid` loop; the alphabetically-first cyclic service is
    `bx1`, in a loop the chain never touches; and `zz1`/`zz2` is declared
    first and reachable from nothing. The daemon reported
    `midone, midtwo`.

    Three rules, three different answers, and the committed prediction is
    the one that came out.
    """
    layer = ops.pebble.Layer({
        'services': {
            'zz1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zz2']},
            'zz2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zz1']},
            'bx1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bx2']},
            'bx2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bx1']},
            'midone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['midtwo']},
            'midtwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['midone']},
            'aaa1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['aaa2']},
            'aaa2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['aaa3']},
            'aaa3': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['midone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: midone, midtwo'
    )


def test_the_written_order_of_an_after_list_decides_which_cycle_is_named():
    """Two names in one list, and their order is the whole answer (probe #14 §42.4, `021`).

    Pre-registered. `alead` carries two branches in one `after` list,
    written `[zcyc1, mcyc1]`, each leading into a different loop, with a
    third loop holding the alphabetically-first cyclic service and
    reachable from nothing. The daemon reported `zcyc1, zcyc2`.

    This is the constraint on any implementation: which cycle gets named
    depends on the order two names appear inside one YAML list, which is
    not a property of the graph. Reading `before`/`after` into a set, or
    sorting them, throws the deciding information away.
    """
    layer = ops.pebble.Layer({
        'services': {
            'alead': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['zcyc1', 'mcyc1'],
            },
            'mcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc2']},
            'mcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'zcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc2']},
            'zcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc1']},
            'bcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc2']},
            'bcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc1']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: zcyc1, zcyc2'
    )


def test_swapping_two_names_in_one_after_list_changes_the_named_cycle():
    """The twin of the shape above (probe #14 §42.4, `022`).

    The same services, the same declaration order, the same edges and the
    same graph, with `alead`'s list written `[mcyc1, zcyc1]`. The daemon
    reported `mcyc1, mcyc2`.

    A walk that sorted the list would give `mcyc1, mcyc2` for both shapes
    and be right here by coincidence, so the pair is what pins written
    order rather than either layer alone.
    """
    layer = ops.pebble.Layer({
        'services': {
            'alead': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['mcyc1', 'zcyc1'],
            },
            'mcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc2']},
            'mcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'zcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc2']},
            'zcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc1']},
            'bcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc2']},
            'bcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc1']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: mcyc1, mcyc2'
    )


def test_the_walk_passes_through_two_cycles_to_the_one_it_names():
    """The first completed cycle wins two levels down (probe #15 §43.3, `025`).

    Real-Pebble probe #15 built the shape §42.10 named as the obvious way
    the walk rule could still be wrong: `alead` leads into the
    `pone`/`ptwo` loop, that loop into `qone`/`qtwo`, and that one into
    `rone`/`rtwo`, which is a sink. The walk has to pass through two
    cycles to reach the one it completes first. The daemon reported
    `rone, rtwo`.

    It is also the shape that rules out ordering components by their
    *earliest* finishing member: `ptwo` is the first service to finish
    anywhere here, and its cycle completes last.
    """
    layer = ops.pebble.Layer({
        'services': {
            'qone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['qtwo', 'rone'],
            },
            'qtwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['qone']},
            'rone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['rtwo']},
            'rtwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['rone']},
            'pone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['ptwo', 'qone'],
            },
            'ptwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['pone']},
            'alead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['pone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: rone, rtwo'
    )


def test_the_named_cycle_is_the_sink_when_the_names_move_to_the_middle():
    """The twin of the shape above (probe #15 §43.3, `026`).

    The same three-cycle chain with the names permuted so that the
    alphabetically-first cyclic service sits in the middle loop rather
    than the first one. The daemon reported `rone, rtwo` again: the answer
    stayed where the structure put it while the names moved.
    """
    layer = ops.pebble.Layer({
        'services': {
            'pone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['ptwo', 'bmid1'],
            },
            'ptwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['pone']},
            'bmid1': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['bmid2', 'rone'],
            },
            'bmid2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bmid1']},
            'rone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['rtwo']},
            'rtwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['rone']},
            'alead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['pone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: rone, rtwo'
    )


def test_written_order_decides_from_inside_a_cycle_too():
    """A branching `after` list on a cyclic service (probe #15 §43.4, `027`).

    Probe #14 measured written order at an acyclic branch point (`021` and
    `022` above). This asks the same question where the branching list
    belongs to a service that is itself on a cycle: `cone` carries
    `[ctwo, zsink1, msink1]`, where `ctwo` closes its own loop and the
    other two lead into disjoint sink loops that nothing else reaches. The
    daemon reported `zsink1, zsink2`.

    So the written-order constraint is not confined to lead-in services,
    which matters because a cyclic service's own list is the awkward place
    to keep it.
    """
    layer = ops.pebble.Layer({
        'services': {
            'dec1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['dec2']},
            'dec2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['dec1']},
            'msink1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['msink2']},
            'msink2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['msink1']},
            'zsink1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zsink2']},
            'zsink2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zsink1']},
            'cone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['ctwo', 'zsink1', 'msink1'],
            },
            'ctwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['cone']},
            'alead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['cone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: zsink1, zsink2'
    )


def test_swapping_a_cyclic_services_branch_list_changes_the_named_cycle():
    """The twin of the shape above (probe #15 §43.4, `028`).

    Nine services, the same declaration order and the same edge set, with
    `cone`'s list written `[ctwo, msink1, zsink1]`. The daemon reported
    `msink1, msink2`.

    The mock at the previous tip could not tell this shape from its twin
    at all, because it held successors in a set: both gave `cone, ctwo`.
    A sorted walk gives `msink1, msink2` for both and is right here by
    coincidence, so it is the pair that discriminates.
    """
    layer = ops.pebble.Layer({
        'services': {
            'dec1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['dec2']},
            'dec2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['dec1']},
            'msink1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['msink2']},
            'msink2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['msink1']},
            'zsink1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zsink2']},
            'zsink2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zsink1']},
            'cone': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['ctwo', 'msink1', 'zsink1'],
            },
            'ctwo': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['cone']},
            'alead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['cone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: msink1, msink2'
    )


def test_the_walk_leaves_a_cycle_by_the_member_carrying_the_outgoing_edge():
    """Closing a cycle does not end that branch (probe #15 §43.3, `029`).

    `cone` is entered from `alead` and its only successor is `ctwo`, whose
    list is `[cone, zsink1]` -- so the back edge that closes
    `cone`/`ctwo` is explored before the edge leading out of it. An
    implementation that named a cycle the moment a back edge closed it
    would stop there. The daemon reported `zsink1, zsink2`, the loop
    reached through `cone`/`ctwo` and out the far side.

    `bdec1`/`bdec2`, which holds the alphabetically-first cyclic service
    and is reachable from nothing, is not named.
    """
    layer = ops.pebble.Layer({
        'services': {
            'cone': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['ctwo']},
            'ctwo': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['cone', 'zsink1'],
            },
            'zsink1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zsink2']},
            'zsink2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zsink1']},
            'bdec1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bdec2']},
            'bdec2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bdec1']},
            'alead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['cone']},
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: zsink1, zsink2'
    )


def test_the_named_cycle_is_the_first_completed_not_the_most_deeply_nested():
    """Depth does not decide (probe #15 §43.3, `030`).

    `alead` branches `[zcyc1, mlead]`: the first is one hop into a sink
    loop, the second a longer descent into a loop that itself reaches
    another. So the deepest component the walk reaches is `ncyc1`/`ncyc2`
    and the shallowest is `zcyc1`/`zcyc2`, and the daemon reported the
    shallow one.

    Every shape before this one had the first completed component also be
    the deepest the walk had got to, so the two had never been separated.
    """
    layer = ops.pebble.Layer({
        'services': {
            'ncyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['ncyc2']},
            'ncyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['ncyc1']},
            'mcyc1': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['mcyc2', 'ncyc1'],
            },
            'mcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'mlead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'zcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc2']},
            'zcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc1']},
            'alead': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['zcyc1', 'mlead'],
            },
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: zcyc1, zcyc2'
    )


def test_the_deep_branch_is_named_when_it_is_written_first():
    """The twin of the shape above (probe #15 §43.3, `031`).

    `alead`'s list written `[mlead, zcyc1]` and nothing else changed. The
    walk takes the deep branch first, so the deepest component is now also
    the first completed, and the daemon reported `ncyc1, ncyc2`.

    This is the second independent pair separating written order from
    sorted order, on a structure where the two branches differ in depth
    rather than being mirror images.
    """
    layer = ops.pebble.Layer({
        'services': {
            'ncyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['ncyc2']},
            'ncyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['ncyc1']},
            'mcyc1': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['mcyc2', 'ncyc1'],
            },
            'mcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'mlead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'zcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc2']},
            'zcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc1']},
            'alead': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['mlead', 'zcyc1'],
            },
        },
    })
    assert _add_layer_cycle_message(layer) == (
        '400 Bad Request: services in before/after loop: ncyc1, ncyc2'
    )


def test_the_mock_is_stable_where_real_pebble_is_not_both_edges_before():
    """The mock gives one answer where the daemon gives two (probe #14 §42.5, `023`).

    This is `021`'s graph spelt the other way round: `alead` carries no
    `after` at all and `zcyc1` and `mcyc1` each declare `before: [alead]`,
    so `alead`'s successors are assembled from two services' lists instead
    of one. The edge set is identical.

    **Real Pebble has no stable answer here.** Over 63 runs, each with a
    freshly created `$PEBBLE` and a freshly started daemon, it said
    `zcyc1, zcyc2` 49 times and `mcyc1, mcyc2` 14 times. So this test
    pins the mock's answer and not the daemon's: what is asserted is that
    repeated calls agree with each other, which is the decision recorded
    in §42.11 -- the mock is deterministic where Pebble is not, rather
    than nondeterministic to mirror it.
    """
    layer = ops.pebble.Layer({
        'services': {
            'alead': {'override': 'replace', 'command': '/bin/sleep 1000'},
            'zcyc1': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['zcyc2'],
                'before': ['alead'],
            },
            'zcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc1']},
            'mcyc1': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['mcyc2'],
                'before': ['alead'],
            },
            'mcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'bcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc2']},
            'bcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc1']},
        },
    })
    answers = {_add_layer_cycle_message(layer) for _ in range(10)}
    assert len(answers) == 1
    assert answers.pop().startswith('400 Bad Request: services in before/after loop: ')


def test_the_mock_is_stable_where_real_pebble_is_not_one_edge_each_way():
    """The same, with only one edge spelt `before` (probe #14 §42.5, `024`).

    `alead after: [zcyc1]` and `mcyc1 before: [alead]`, so the two edges
    still come from two lists but only one of them is a `before`. Over 30
    runs real Pebble said `zcyc1, zcyc2` 19 times and `mcyc1, mcyc2` 11
    times, which is what says the instability arrives as soon as a
    service's successors are assembled from more than one list rather than
    needing both edges spelt `before`.

    As above, this pins the mock's answer and not the daemon's.
    """
    layer = ops.pebble.Layer({
        'services': {
            'alead': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc1']},
            'zcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc2']},
            'zcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['zcyc1']},
            'mcyc1': {
                'override': 'replace',
                'command': '/bin/sleep 1000',
                'after': ['mcyc2'],
                'before': ['alead'],
            },
            'mcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['mcyc1']},
            'bcyc1': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc2']},
            'bcyc2': {'override': 'replace', 'command': '/bin/sleep 1000', 'after': ['bcyc1']},
        },
    })
    answers = {_add_layer_cycle_message(layer) for _ in range(10)}
    assert len(answers) == 1
    assert answers.pop().startswith('400 Bad Request: services in before/after loop: ')
