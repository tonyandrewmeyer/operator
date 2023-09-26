# Copyright 2019-2020 Canonical Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os
import shutil
import tempfile
from pathlib import Path

import ops
import ops.charm
from ops.model import _ModelBackend
from ops.storage import SQLiteStorage

from .test_helpers import FakeScriptTestCase


class TestCharm(FakeScriptTestCase):

    def setUp(self):
        def restore_env(env):
            os.environ.clear()
            os.environ.update(env)
        self.addCleanup(restore_env, os.environ.copy())

        os.environ['PATH'] = os.pathsep.join([
            str(Path(__file__).parent / 'bin'),
            os.environ['PATH']])
        os.environ['JUJU_UNIT_NAME'] = 'local/0'

        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.tmpdir))
        self.meta = ops.CharmMeta()

        class CustomEvent(ops.EventBase):
            pass

        class TestCharmEvents(ops.CharmEvents):
            custom = ops.EventSource(CustomEvent)

        # Relations events are defined dynamically and modify the class attributes.
        # We use a subclass temporarily to prevent these side effects from leaking.
        ops.CharmBase.on = TestCharmEvents()

        def cleanup():
            ops.CharmBase.on = ops.CharmEvents()
        self.addCleanup(cleanup)

    def create_framework(self):
        model = ops.Model(self.meta, _ModelBackend('local/0'))
        # we can pass foo_event as event_name because we're not actually testing dispatch
        framework = ops.Framework(SQLiteStorage(':memory:'), self.tmpdir, self.meta,
                                  model)
        self.addCleanup(framework.close)
        return framework

    def test_basic(self):

        class MyCharm(ops.CharmBase):

            def __init__(self, *args):
                super().__init__(*args)

                self.started = False
                framework.observe(self.on.start, self._on_start)

            def _on_start(self, event):
                self.started = True

        events = list(MyCharm.on.events())
        self.assertIn('install', events)
        self.assertIn('custom', events)

        framework = self.create_framework()
        charm = MyCharm(framework)
        charm.on.start.emit()

        self.assertEqual(charm.started, True)

        with self.assertRaisesRegex(TypeError, "observer methods must now be explicitly provided"):
            framework.observe(charm.on.start, charm)

    def test_observe_decorated_method(self):
        # we test that charm methods decorated with @functools.wraps(wrapper)
        # can be observed by Framework. Simpler decorators won't work because
        # Framework searches for __self__ and other method things; functools.wraps
        # is more careful and it still works, this test is here to ensure that
        # it keeps working in future releases, as this is presently the only
        # way we know of to cleanly decorate charm event observers.
        events = []

        def dec(fn):
            # simple decorator that appends to the nonlocal
            # `events` list all events it receives
            @functools.wraps(fn)
            def wrapper(charm, evt):
                events.append(evt)
                fn(charm, evt)
            return wrapper

        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                framework.observe(self.on.start, self._on_start)
                self.seen = None

            @dec
            def _on_start(self, event):
                self.seen = event

        framework = self.create_framework()
        charm = MyCharm(framework)
        charm.on.start.emit()
        # check that the event has been seen by the decorator
        self.assertEqual(1, len(events))
        # check that the event has been seen by the observer
        self.assertIsInstance(charm.seen, ops.StartEvent)

    def test_empty_action(self):
        meta = ops.CharmMeta.from_yaml('name: my-charm', '')
        self.assertEqual(meta.actions, {})

    def test_helper_properties(self):
        framework = self.create_framework()

        class MyCharm(ops.CharmBase):
            pass

        charm = MyCharm(framework)
        self.assertEqual(charm.app, framework.model.app)
        self.assertEqual(charm.unit, framework.model.unit)
        self.assertEqual(charm.meta, framework.meta)
        self.assertEqual(charm.charm_dir, framework.charm_dir)
        self.assertIs(charm.config, framework.model.config)

    def test_relation_events(self):

        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.seen = []
                for rel in ('req1', 'req-2', 'pro1', 'pro-2', 'peer1', 'peer-2'):
                    # Hook up relation events to generic handler.
                    self.framework.observe(self.on[rel].relation_joined, self.on_any_relation)
                    self.framework.observe(self.on[rel].relation_changed, self.on_any_relation)
                    self.framework.observe(self.on[rel].relation_departed, self.on_any_relation)
                    self.framework.observe(self.on[rel].relation_broken, self.on_any_relation)

            def on_any_relation(self, event):
                assert event.relation.name == 'req1'
                assert event.relation.app.name == 'remote'
                self.seen.append(type(event).__name__)

        # language=YAML
        self.meta = ops.CharmMeta.from_yaml(metadata='''
name: my-charm
requires:
 req1:
   interface: req1
 req-2:
   interface: req2
provides:
 pro1:
   interface: pro1
 pro-2:
   interface: pro2
peers:
 peer1:
   interface: peer1
 peer-2:
   interface: peer2
''')

        charm = MyCharm(self.create_framework())

        self.assertIn('pro_2_relation_broken', repr(charm.on))

        rel = charm.framework.model.get_relation('req1', 1)
        unit = charm.framework.model.get_unit('remote/0')
        charm.on['req1'].relation_joined.emit(rel, unit)
        charm.on['req1'].relation_changed.emit(rel, unit)
        charm.on['req-2'].relation_changed.emit(rel, unit)
        charm.on['pro1'].relation_departed.emit(rel, unit)
        charm.on['pro-2'].relation_departed.emit(rel, unit)
        charm.on['peer1'].relation_broken.emit(rel)
        charm.on['peer-2'].relation_broken.emit(rel)

        self.assertEqual(charm.seen, [
            'RelationJoinedEvent',
            'RelationChangedEvent',
            'RelationChangedEvent',
            'RelationDepartedEvent',
            'RelationDepartedEvent',
            'RelationBrokenEvent',
            'RelationBrokenEvent',
        ])

    def test_storage_events(self):
        this = self

        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.seen = []
                self.framework.observe(self.on['stor1'].storage_attached, self._on_stor1_attach)
                self.framework.observe(self.on['stor2'].storage_detaching, self._on_stor2_detach)
                self.framework.observe(self.on['stor3'].storage_attached, self._on_stor3_attach)
                self.framework.observe(self.on['stor-4'].storage_attached, self._on_stor4_attach)

            def _on_stor1_attach(self, event):
                self.seen.append(type(event).__name__)
                this.assertEqual(event.storage.location, Path("/var/srv/stor1/0"))

            def _on_stor2_detach(self, event):
                self.seen.append(type(event).__name__)

            def _on_stor3_attach(self, event):
                self.seen.append(type(event).__name__)

            def _on_stor4_attach(self, event):
                self.seen.append(type(event).__name__)

        # language=YAML
        self.meta = ops.CharmMeta.from_yaml('''
name: my-charm
storage:
  stor-4:
    multiple:
      range: 2-4
    type: filesystem
  stor1:
    type: filesystem
  stor2:
    multiple:
      range: "2"
    type: filesystem
  stor3:
    multiple:
      range: 2-
    type: filesystem
  stor-multiple-dashes:
    multiple:
      range: 2-
    type: filesystem
''')

        self.fake_script(
            "storage-get",
            """
            if [ "$1" = "-s" ]; then
                id=${2#*/}
                key=${2%/*}
                echo "\\"/var/srv/${key}/${id}\\"" # NOQA: test_quote_backslashes
            elif [ "$1" = '--help' ]; then
                printf '%s\\n' \\
                'Usage: storage-get [options] [<key>]' \\
                '   ' \\
                'Summary:' \\
                'print information for storage instance with specified id' \\
                '   ' \\
                'Options:' \\
                '--format  (= smart)' \\
                '    Specify output format (json|smart|yaml)' \\
                '-o, --output (= "")' \\
                '    Specify an output file' \\
                '-s  (= test-stor/0)' \\
                '    specify a storage instance by id' \\
                '   ' \\
                'Details:' \\
                'When no <key> is supplied, all keys values are printed.'
            else
                # Return the same path for all disks since `storage-get`
                # on attach and detach takes no parameters and is not
                # deterministically faked with fake_script
                exit 1
            fi
            """,
        )
        self.fake_script(
            "storage-list",
            """
            echo '["disks/0"]'
            """,
        )

        self.assertIsNone(self.meta.storages['stor1'].multiple_range)
        self.assertEqual(self.meta.storages['stor2'].multiple_range, (2, 2))
        self.assertEqual(self.meta.storages['stor3'].multiple_range, (2, None))
        self.assertEqual(self.meta.storages['stor-4'].multiple_range, (2, 4))

        charm = MyCharm(self.create_framework())

        charm.on['stor1'].storage_attached.emit(ops.Storage("stor1", 0, charm.model._backend))
        charm.on['stor2'].storage_detaching.emit(ops.Storage("stor2", 0, charm.model._backend))
        charm.on['stor3'].storage_attached.emit(ops.Storage("stor3", 0, charm.model._backend))
        charm.on['stor-4'].storage_attached.emit(ops.Storage("stor-4", 0, charm.model._backend))
        charm.on['stor-multiple-dashes'].storage_attached.emit(
            ops.Storage("stor-multiple-dashes", 0, charm.model._backend))

        self.assertEqual(charm.seen, [
            'StorageAttachedEvent',
            'StorageDetachingEvent',
            'StorageAttachedEvent',
            'StorageAttachedEvent',
        ])

    def test_workload_events(self):

        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.seen = []
                self.count = 0
                for workload in ('container-a', 'containerb'):
                    # Hook up relation events to generic handler.
                    self.framework.observe(
                        self.on[workload].pebble_ready,
                        self.on_any_pebble_ready)

            def on_any_pebble_ready(self, event):
                self.seen.append(type(event).__name__)
                self.count += 1

        # language=YAML
        self.meta = ops.CharmMeta.from_yaml(metadata='''
name: my-charm
containers:
  container-a:
  containerb:
''')

        charm = MyCharm(self.create_framework())

        self.assertIn('container_a_pebble_ready', repr(charm.on))
        self.assertIn('containerb_pebble_ready', repr(charm.on))

        charm.on['container-a'].pebble_ready.emit(
            charm.framework.model.unit.get_container('container-a'))
        charm.on['containerb'].pebble_ready.emit(
            charm.framework.model.unit.get_container('containerb'))

        self.assertEqual(charm.seen, [
            'PebbleReadyEvent',
            'PebbleReadyEvent'
        ])
        self.assertEqual(charm.count, 2)

    def test_relations_meta(self):

        # language=YAML
        self.meta = ops.CharmMeta.from_yaml('''
name: my-charm
requires:
  database:
    interface: mongodb
    limit: 1
    scope: container
  metrics:
    interface: prometheus-scraping
''')

        self.assertEqual(self.meta.requires['database'].interface_name, 'mongodb')
        self.assertEqual(self.meta.requires['database'].limit, 1)
        self.assertEqual(self.meta.requires['database'].scope, 'container')

        self.assertEqual(self.meta.requires['metrics'].interface_name, 'prometheus-scraping')
        self.assertIsNone(self.meta.requires['metrics'].limit)
        self.assertEqual(self.meta.requires['metrics'].scope, 'global')  # Default value

    def test_relations_meta_limit_type_validation(self):
        with self.assertRaisesRegex(TypeError, "limit should be an int, not <class 'str'>"):
            # language=YAML
            self.meta = ops.CharmMeta.from_yaml('''
name: my-charm
requires:
  database:
    interface: mongodb
    limit: foobar
''')

    def test_relations_meta_scope_type_validation(self):
        with self.assertRaisesRegex(TypeError,
                                    "scope should be one of 'global', 'container'; not 'foobar'"):
            # language=YAML
            self.meta = ops.CharmMeta.from_yaml('''
name: my-charm
requires:
  database:
    interface: mongodb
    scope: foobar
''')

    @classmethod
    def _get_action_test_meta(cls):
        # language=YAML
        return ops.CharmMeta.from_yaml(metadata='''
name: my-charm
''', actions='''
foo-bar:
  description: "Foos the bar."
  params:
    foo-name:
      description: "A foo name to bar"
      type: string
    silent:
      default: false
      description: ""
      type: boolean
  required: foo-bar
  title: foo-bar
start:
  description: "Start the unit."
''')

    def _setup_test_action(self):
        os.environ['JUJU_ACTION_NAME'] = 'foo-bar'
        self.fake_script('action-get', """echo '{"foo-name": "name", "silent": true}'""")
        self.fake_script('action-set', "")
        self.fake_script('action-log', "")
        self.fake_script('action-fail', "")
        self.meta = self._get_action_test_meta()

    def test_action_events(self):

        class MyCharm(ops.CharmBase):

            def __init__(self, *args):
                super().__init__(*args)
                framework.observe(self.on.foo_bar_action, self._on_foo_bar_action)
                framework.observe(self.on.start_action, self._on_start_action)

            def _on_foo_bar_action(self, event):
                self.seen_action_params = event.params
                event.log('test-log')
                event.set_results({'res': 'val with spaces'})
                event.fail('test-fail')

            def _on_start_action(self, event):
                pass

        self._setup_test_action()
        framework = self.create_framework()
        charm = MyCharm(framework)

        events = list(MyCharm.on.events())
        self.assertIn('foo_bar_action', events)
        self.assertIn('start_action', events)

        charm.on.foo_bar_action.emit()
        self.assertEqual(charm.seen_action_params, {"foo-name": "name", "silent": True})
        self.assertEqual(self.fake_script_calls(), [
            ['action-get', '--format=json'],
            ['action-log', "test-log"],
            ['action-set', "res=val with spaces"],
            ['action-fail', "test-fail"],
        ])

        # Make sure that action events that do not match the current context are
        # not possible to emit by hand.
        with self.assertRaises(RuntimeError):
            charm.on.start_action.emit()

    def test_invalid_action_results(self):

        class MyCharm(ops.CharmBase):

            def __init__(self, *args):
                super().__init__(*args)
                self.res = {}
                framework.observe(self.on.foo_bar_action, self._on_foo_bar_action)

            def _on_foo_bar_action(self, event):
                event.set_results(self.res)

        self._setup_test_action()
        framework = self.create_framework()
        charm = MyCharm(framework)

        for bad_res in (
                {'a': {'b': 'c'}, 'a.b': 'c'},
                {'a': {'B': 'c'}},
                {'a': {(1, 2): 'c'}},
                {'a': {None: 'c'}},
                {'aBc': 'd'}):
            charm.res = bad_res

            with self.assertRaises(ValueError):
                charm.on.foo_bar_action.emit()

    def _test_action_event_defer_fails(self, cmd_type):

        class MyCharm(ops.CharmBase):

            def __init__(self, *args):
                super().__init__(*args)
                framework.observe(self.on.start_action, self._on_start_action)

            def _on_start_action(self, event):
                event.defer()

        self.fake_script(f"{cmd_type}-get", """echo '{"foo-name": "name", "silent": true}'""")
        self.meta = self._get_action_test_meta()

        os.environ[f'JUJU_{cmd_type.upper()}_NAME'] = 'start'
        framework = self.create_framework()
        charm = MyCharm(framework)

        with self.assertRaises(RuntimeError):
            charm.on.start_action.emit()

    def test_action_event_defer_fails(self):
        self._test_action_event_defer_fails('action')

    def test_containers(self):
        meta = ops.CharmMeta.from_yaml("""
name: k8s-charm
containers:
  test1:
    k: v
  test2:
    k: v
""")
        self.assertIsInstance(meta.containers['test1'], ops.ContainerMeta)
        self.assertIsInstance(meta.containers['test2'], ops.ContainerMeta)
        self.assertEqual(meta.containers['test1'].name, 'test1')
        self.assertEqual(meta.containers['test2'].name, 'test2')

    def test_containers_storage(self):
        meta = ops.CharmMeta.from_yaml("""
name: k8s-charm
storage:
  data:
    type: filesystem
    location: /test/storage
  other:
    type: filesystem
    location: /test/other
containers:
  test1:
    mounts:
      - storage: data
        location: /test/storagemount
      - storage: other
        location: /test/otherdata
""")
        self.assertIsInstance(meta.containers['test1'], ops.ContainerMeta)
        self.assertIsInstance(meta.containers['test1'].mounts["data"], ops.ContainerStorageMeta)
        self.assertEqual(meta.containers['test1'].mounts["data"].location, '/test/storagemount')
        self.assertEqual(meta.containers['test1'].mounts["other"].location, '/test/otherdata')

    def test_containers_storage_multiple_mounts(self):
        meta = ops.CharmMeta.from_yaml("""
name: k8s-charm
storage:
  data:
    type: filesystem
    location: /test/storage
containers:
  test1:
    mounts:
      - storage: data
        location: /test/storagemount
      - storage: data
        location: /test/otherdata
""")
        self.assertIsInstance(meta.containers['test1'], ops.ContainerMeta)
        self.assertIsInstance(meta.containers['test1'].mounts["data"], ops.ContainerStorageMeta)
        self.assertEqual(
            meta.containers['test1'].mounts["data"].locations[0],
            '/test/storagemount')
        self.assertEqual(meta.containers['test1'].mounts["data"].locations[1], '/test/otherdata')

        with self.assertRaises(RuntimeError):
            meta.containers["test1"].mounts["data"].location

    def test_secret_events(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.seen = []
                self.framework.observe(self.on.secret_changed, self.on_secret_changed)
                self.framework.observe(self.on.secret_rotate, self.on_secret_rotate)
                self.framework.observe(self.on.secret_remove, self.on_secret_remove)
                self.framework.observe(self.on.secret_expired, self.on_secret_expired)

            def on_secret_changed(self, event):
                assert event.secret.id == 'secret:changed'
                assert event.secret.label is None
                self.seen.append(type(event).__name__)

            def on_secret_rotate(self, event):
                assert event.secret.id == 'secret:rotate'
                assert event.secret.label == 'rot'
                self.seen.append(type(event).__name__)

            def on_secret_remove(self, event):
                assert event.secret.id == 'secret:remove'
                assert event.secret.label == 'rem'
                assert event.revision == 7
                self.seen.append(type(event).__name__)

            def on_secret_expired(self, event):
                assert event.secret.id == 'secret:expired'
                assert event.secret.label == 'exp'
                assert event.revision == 42
                self.seen.append(type(event).__name__)

        self.meta = ops.CharmMeta.from_yaml(metadata='name: my-charm')
        charm = MyCharm(self.create_framework())

        charm.on.secret_changed.emit('secret:changed', None)
        charm.on.secret_rotate.emit('secret:rotate', 'rot')
        charm.on.secret_remove.emit('secret:remove', 'rem', 7)
        charm.on.secret_expired.emit('secret:expired', 'exp', 42)

        self.assertEqual(charm.seen, [
            'SecretChangedEvent',
            'SecretRotateEvent',
            'SecretRemoveEvent',
            'SecretExpiredEvent',
        ])

    def test_collect_app_status_leader(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.framework.observe(self.on.collect_app_status, self._on_collect_status)

            def _on_collect_status(self, event):
                event.add_status(ops.ActiveStatus())
                event.add_status(ops.BlockedStatus('first'))
                event.add_status(ops.WaitingStatus('waiting'))
                event.add_status(ops.BlockedStatus('second'))

        self.fake_script('is-leader', 'echo true')
        self.fake_script('status-set', 'exit 0')

        charm = MyCharm(self.create_framework())
        ops.charm._evaluate_status(charm)

        self.assertEqual(self.fake_script_calls(True), [
            ['is-leader', '--format=json'],
            ['status-set', '--application=True', 'blocked', 'first'],
        ])

    def test_collect_app_status_no_statuses(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.framework.observe(self.on.collect_app_status, self._on_collect_status)

            def _on_collect_status(self, event):
                pass

        self.fake_script('is-leader', 'echo true')

        charm = MyCharm(self.create_framework())
        ops.charm._evaluate_status(charm)

        self.assertEqual(self.fake_script_calls(True), [
            ['is-leader', '--format=json'],
        ])

    def test_collect_app_status_non_leader(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.framework.observe(self.on.collect_app_status, self._on_collect_status)

            def _on_collect_status(self, event):
                raise Exception  # shouldn't be called

        self.fake_script('is-leader', 'echo false')

        charm = MyCharm(self.create_framework())
        ops.charm._evaluate_status(charm)

        self.assertEqual(self.fake_script_calls(True), [
            ['is-leader', '--format=json'],
        ])

    def test_collect_unit_status(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.framework.observe(self.on.collect_unit_status, self._on_collect_status)

            def _on_collect_status(self, event):
                event.add_status(ops.ActiveStatus())
                event.add_status(ops.BlockedStatus('first'))
                event.add_status(ops.WaitingStatus('waiting'))
                event.add_status(ops.BlockedStatus('second'))

        self.fake_script('is-leader', 'echo false')  # called only for collecting app statuses
        self.fake_script('status-set', 'exit 0')

        charm = MyCharm(self.create_framework())
        ops.charm._evaluate_status(charm)

        self.assertEqual(self.fake_script_calls(True), [
            ['is-leader', '--format=json'],
            ['status-set', '--application=False', 'blocked', 'first'],
        ])

    def test_collect_unit_status_no_statuses(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.framework.observe(self.on.collect_unit_status, self._on_collect_status)

            def _on_collect_status(self, event):
                pass

        self.fake_script('is-leader', 'echo false')  # called only for collecting app statuses

        charm = MyCharm(self.create_framework())
        ops.charm._evaluate_status(charm)

        self.assertEqual(self.fake_script_calls(True), [
            ['is-leader', '--format=json'],
        ])

    def test_collect_app_and_unit_status(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.framework.observe(self.on.collect_app_status, self._on_collect_app_status)
                self.framework.observe(self.on.collect_unit_status, self._on_collect_unit_status)

            def _on_collect_app_status(self, event):
                event.add_status(ops.ActiveStatus())

            def _on_collect_unit_status(self, event):
                event.add_status(ops.WaitingStatus('blah'))

        self.fake_script('is-leader', 'echo true')
        self.fake_script('status-set', 'exit 0')

        charm = MyCharm(self.create_framework())
        ops.charm._evaluate_status(charm)

        self.assertEqual(self.fake_script_calls(True), [
            ['is-leader', '--format=json'],
            ['status-set', '--application=True', 'active', ''],
            ['status-set', '--application=False', 'waiting', 'blah'],
        ])

    def test_add_status_type_error(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.framework.observe(self.on.collect_app_status, self._on_collect_status)

            def _on_collect_status(self, event):
                event.add_status('active')

        self.fake_script('is-leader', 'echo true')

        charm = MyCharm(self.create_framework())
        with self.assertRaises(TypeError):
            ops.charm._evaluate_status(charm)

    def test_collect_status_priority(self):
        class MyCharm(ops.CharmBase):
            def __init__(self, *args, statuses=None):
                super().__init__(*args)
                self.framework.observe(self.on.collect_app_status, self._on_collect_status)
                self.statuses = statuses

            def _on_collect_status(self, event):
                for status in self.statuses:
                    event.add_status(ops.StatusBase.from_name(status, ''))

        self.fake_script('is-leader', 'echo true')
        self.fake_script('status-set', 'exit 0')

        charm = MyCharm(self.create_framework(), statuses=['blocked', 'error'])
        ops.charm._evaluate_status(charm)

        charm = MyCharm(self.create_framework(), statuses=['waiting', 'blocked'])
        ops.charm._evaluate_status(charm)

        charm = MyCharm(self.create_framework(), statuses=['waiting', 'maintenance'])
        ops.charm._evaluate_status(charm)

        charm = MyCharm(self.create_framework(), statuses=['active', 'waiting'])
        ops.charm._evaluate_status(charm)

        charm = MyCharm(self.create_framework(), statuses=['active', 'unknown'])
        ops.charm._evaluate_status(charm)

        charm = MyCharm(self.create_framework(), statuses=['unknown'])
        ops.charm._evaluate_status(charm)

        status_set_calls = [call for call in self.fake_script_calls(True)
                            if call[0] == 'status-set']
        self.assertEqual(status_set_calls, [
            ['status-set', '--application=True', 'error', ''],
            ['status-set', '--application=True', 'blocked', ''],
            ['status-set', '--application=True', 'maintenance', ''],
            ['status-set', '--application=True', 'waiting', ''],
            ['status-set', '--application=True', 'active', ''],
            ['status-set', '--application=True', 'unknown', ''],
        ])
