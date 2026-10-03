# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the default mocks that fake libraries and give each unit its own filesystem.

The libraries are faked with small modules shaped like the real ones, written
into each charm's ``lib/`` as a charm would vendor them, so that they import
the same way in the test process and in a worker. Where the real package is
installed, it is tested too.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import textwrap
import uuid
from collections.abc import Generator
from typing import Any

import pytest

from ops import testing

# Fake libraries, shaped like the real ones: everything that would reach a
# cluster or snapd raises instead.

FAKE_LIGHTKUBE = {
    'lib/lightkube/__init__.py': """
        from .core.async_client import AsyncClient
        from .core.client import Client
        from .core.exceptions import ApiError, ConfigError
    """,
    'lib/lightkube/core/__init__.py': '',
    'lib/lightkube/core/exceptions.py': """
        class ConfigError(Exception):
            pass


        class ApiError(Exception):
            def __init__(self, request=None, response=None, status=None):
                self.status = status
                super().__init__(getattr(status, 'message', ''))
    """,
    'lib/lightkube/core/client.py': """
        from .exceptions import ConfigError


        class Client:
            def __init__(self, config=None, namespace=None, **kwargs):
                raise ConfigError('Configuration file ~/.kube/config not found')

            @property
            def namespace(self):
                return self._client.namespace

            def get(self, res, name, *, namespace=None):
                raise RuntimeError('reached a cluster')

            def list(self, res, **kwargs):
                raise RuntimeError('reached a cluster')

            def create(self, obj, **kwargs):
                raise RuntimeError('reached a cluster')

            def patch(self, res, name, obj, **kwargs):
                raise RuntimeError('reached a cluster')

            def delete(self, res, name, **kwargs):
                raise RuntimeError('reached a cluster')
    """,
    'lib/lightkube/core/async_client.py': """
        from .exceptions import ConfigError


        class AsyncClient:
            def __init__(self, config=None, namespace=None, **kwargs):
                raise ConfigError('Configuration file ~/.kube/config not found')

            async def get(self, res, name, *, namespace=None):
                raise RuntimeError('reached a cluster')

            def list(self, res, **kwargs):
                raise RuntimeError('reached a cluster')
    """,
}

_NAMESPACE_PROPERTY = """
    @property
    def _namespace(self):
        with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace') as f:
            return f.read().strip()
"""

FAKE_K8S_PATCH_LIBS = {
    'lib/charms/observability_libs/v1/kubernetes_service_patch.py': ''.join((
        textwrap.dedent("""
        import ops


        class KubernetesServicePatch(ops.Object):
            def __init__(self, charm, ports, service_name=None):
                super().__init__(charm, 'kubernetes-service-patch')
                self.charm = charm
                self.framework.observe(charm.on.install, self._patch)
                self.framework.observe(charm.on.remove, self._remove_service)

            def _patch(self, _):
                raise RuntimeError('reached a cluster')

            def _remove_service(self, _):
                raise RuntimeError('reached a cluster')

            def is_patched(self):
                raise RuntimeError('reached a cluster')
    """),
        _NAMESPACE_PROPERTY,
    )),
    'lib/charms/observability_libs/v0/kubernetes_compute_resources_patch.py': ''.join((
        textwrap.dedent("""
        import ops


        class ResourcePatcher:
            def __init__(self, namespace, statefulset_name, container_name):
                raise RuntimeError('reached a cluster')

            def is_ready(self, pod_name, resource_reqs):
                raise RuntimeError('reached a cluster')


        class KubernetesComputeResourcesPatch(ops.Object):
            def __init__(self, charm, container_name, *, resource_reqs_func):
                super().__init__(charm, f'KubernetesComputeResourcesPatch_{container_name}')
                self._charm = charm
                self.patcher = ResourcePatcher(self._namespace, charm.app.name, container_name)
                self.framework.observe(charm.on.config_changed, self._on_config_changed)

            def _on_config_changed(self, _):
                self._patch()

            def _patch(self):
                raise RuntimeError('reached a cluster')

            def is_ready(self):
                return self.patcher.is_ready('pod', None)

            def get_status(self):
                raise RuntimeError('reached a cluster')
    """),
        _NAMESPACE_PROPERTY,
    )),
    'lib/charms/comsys_libs/v0/kubernetes_statefulset_patch.py': """
        import ops


        class KubernetesStatefulsetPatch(ops.Object):
            def __init__(self, charm, resources):
                super().__init__(charm, 'kubernetes-statefulset-patch')
                self.framework.observe(charm.on.install, self._patch_statefulset)

            def _patch_statefulset(self, _):
                raise RuntimeError('reached a cluster')

            def _is_patched(self, container, desired_resources):
                raise RuntimeError('reached a cluster')
    """,
}

_SNAP_LIB = """
    import enum
    import subprocess


    class SnapState(enum.Enum):
        Latest = 'latest'
        Present = 'present'
        Absent = 'absent'
        Available = 'available'


    class SnapError(Exception):
        pass


    class SnapClient:
        def _request(self, method, path, query=None, body=None):
            raise SnapError('reached snapd')

        def get_installed_snap_apps(self, name):
            return self._request('GET', 'apps', {'names': name})


    class Snap:
        def __init__(self, name, state, channel, revision, confinement, apps=None, cohort=''):
            self._name = name
            self._state = state
            self._channel = channel
            self._revision = revision
            self._confinement = confinement
            self._cohort = cohort
            self._apps = apps or []
            self._snap_client = SnapClient()

        def ensure(self, state, classic=False, devmode=False, channel=None, cohort=None,
                   revision=None):
            subprocess.check_output(['snap', 'install', self._name])
            self._update_snap_apps()
            raise SnapError('reached snapd')

        def _update_snap_apps(self):
            self._apps = self._snap_client.get_installed_snap_apps(self._name)

        @property
        def present(self):
            return self._state in (SnapState.Present, SnapState.Latest)

        @property
        def channel(self):
            return self._channel

        @property
        def revision(self):
            return self._revision


    class SnapCache:
        def __init__(self):
            self._snap_client = SnapClient()
            self._snap_map = {}
            for info in self._snap_client._request('GET', 'snaps'):
                pass

        def __getitem__(self, name):
            raise SnapError('reached snapd')

        def __contains__(self, name):
            return name in self._snap_map

        def __len__(self):
            return len(self._snap_map)


    class _MetaCache(type):
        @property
        def cache(cls):
            if cls._cache is None:
                cls._cache = SnapCache()
            return cls._cache

        def __getitem__(cls, name):
            return cls.cache[name]


    class _Cache(metaclass=_MetaCache):
        _cache = None


    def add(snap_name, state=SnapState.Latest, channel=None):
        snap = _Cache[snap_name]
        snap.ensure(state=state, channel=channel)
        return snap
"""

FAKE_OPERATOR_LIBS_LINUX_SNAP = {'lib/charms/operator_libs_linux/v2/snap.py': _SNAP_LIB}

#: charmlibs.snap before 2.0 was the same SnapCache API, in _snap.
FAKE_CHARMLIBS_SNAP_1 = {
    'lib/charmlibs/snap/__init__.py': """
        from ._snap import Snap, SnapCache, SnapError, SnapState, add
    """,
    'lib/charmlibs/snap/_snap.py': _SNAP_LIB,
}

#: charmlibs.snap 2.x is functions over a snapd client.
FAKE_CHARMLIBS_SNAP_2 = {
    'lib/charmlibs/snap/__init__.py': """
        from ._errors import NotInstalledError
        from ._snapd_snaps import ensure_installed, list_one
    """,
    'lib/charmlibs/snap/_errors.py': """
        class APIError(Exception):
            def __init__(self, message, *, kind='', value='', status_code=None, status=None):
                super().__init__(message)
                self.kind = kind


        class NotInstalledError(APIError):
            pass
    """,
    'lib/charmlibs/snap/_client.py': """
        from . import _errors


        def get(path, query=None):
            raise ConnectionError('reached snapd')


        def post(path, body=None):
            raise ConnectionError('reached snapd')


        def _make_error(response):
            result = response['result']
            error_type = (
                _errors.NotInstalledError
                if result['kind'] == 'snap-not-found'
                else _errors.APIError
            )
            return error_type(result['message'], kind=result['kind'])
    """,
    'lib/charmlibs/snap/_snapd_snaps.py': """
        from . import _client, _errors


        def list_one(snap):
            return _client.get(f'/v2/snaps/{snap}')


        def ensure_installed(snap, channel=None):
            try:
                list_one(snap)
            except _errors.NotInstalledError:
                _client.post(f'/v2/snaps/{snap}', body={'action': 'install', 'channel': channel})
                return True
            return False
    """,
}


def write_charm(
    root: pathlib.Path,
    charm: str,
    *,
    name: str = 'probe',
    meta: str | None = None,
    files: dict[str, str] | None = None,
) -> pathlib.Path:
    """Write a charm's source tree under ``root`` and return its directory."""
    all_files = {
        'charmcraft.yaml': 'type: charm\nsummary: A charm.\ndescription: A charm.\n'
        + (meta if meta is not None else f'name: {name}\n'),
        'src/charm.py': charm,
        **(files or {}),
    }
    for path, content in all_files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content))
    return root


def deploy_kwargs(isolated: bool) -> dict[str, Any]:
    return {'python_executable': sys.executable} if isolated else {}


@pytest.fixture(params=[False, True], ids=['in-process', 'isolated'])
def isolated(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(autouse=True)
def forget_charm_libraries() -> Generator[None]:
    """Unload the fake libraries afterwards, so the next test imports its own (or the real ones).

    An in-process charm's ``lib/`` goes on ``sys.path``, and the libraries it
    imports stay in ``sys.modules``.
    """
    path = list(sys.path)
    modules = set(sys.modules)
    yield
    sys.path[:] = path
    prefixes = ('lightkube', 'charms', 'charmlibs')
    for name in set(sys.modules) - modules:
        if name.split('.')[0] in prefixes:
            del sys.modules[name]


def status(unit: testing.Unit) -> Any:
    return json.loads(unit.state.unit_status.message)


# A machine charm


ETCD_CHARM = """
    import json
    import pathlib
    import subprocess

    import ops
    from charms.operator_libs_linux.v2 import snap


    class EtcdCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)
            framework.observe(self.on.config_changed, self._on_config_changed)

        def _on_install(self, _):
            snap.SnapCache()['charmed-etcd'].ensure(snap.SnapState.Latest, channel='3.5/stable')
            subprocess.run(['snap', 'set', 'charmed-etcd', 'name=x'], check=True)
            subprocess.run(['systemctl', 'restart', 'snap.charmed-etcd.etcd'], check=True)
            conf = pathlib.Path('/etc/ops-testing-etcd/etcd.conf')
            conf.parent.mkdir(parents=True, exist_ok=True)
            conf.write_text(f'name: {self.unit.name}\\n')

        def _on_config_changed(self, _):
            etcd = snap.SnapCache()['charmed-etcd']
            status = {'present': etcd.present, 'channel': etcd.channel}
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""


def test_a_machine_charm_s_install_stays_off_the_host(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(
        tmp_path / 'etcd',
        ETCD_CHARM,
        meta='name: etcd\nbase: ubuntu@22.04\n',
        files=FAKE_OPERATOR_LIBS_LINUX_SNAP,
    )
    with testing.Juju(type='lxd') as juju:
        app = juju._deploy(root, num_units=2, **deploy_kwargs(isolated))
        juju.settle()
        for unit in app.units:
            # The snap installed in install is still there in config-changed.
            assert status(unit) == {'present': True, 'channel': '3.5/stable'}
            conf = unit.filesystem / 'etc/ops-testing-etcd/etcd.conf'
            assert conf.read_text() == f'name: {unit.name}\n'
        assert app.units[0].filesystem != app.units[1].filesystem
    assert not pathlib.Path('/etc/ops-testing-etcd').exists()


# The filesystem


def test_reads_fall_through_to_the_host_and_writes_stay_in_the_root(
    tmp_path: pathlib.Path, isolated: bool
):
    host = tmp_path / 'host'
    host.mkdir()
    (host / 'shared.txt').write_text('from the host')
    (host / 'doomed.txt').write_text('removed by the charm')
    charm = f"""
        import json
        import os
        import pathlib
        import shutil

        import ops

        HOST = pathlib.Path({str(host)!r})


        class FilesCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on.install, self._on_install)
                framework.observe(self.on.start, self._on_start)

            def _on_install(self, _):
                before = (HOST / 'shared.txt').read_text()
                with open(HOST / 'shared.txt', 'a') as f:
                    f.write(', and the charm')
                (HOST / 'new' / 'deep').mkdir(parents=True)
                shutil.copy2(HOST / 'shared.txt', HOST / 'new' / 'deep' / 'copy.txt')
                os.rename(HOST / 'new' / 'deep' / 'copy.txt', HOST / 'new' / 'moved.txt')
                os.remove(HOST / 'doomed.txt')
                self.unit.status = ops.MaintenanceStatus(before)

            def _on_start(self, _):
                status = {{
                    'shared': (HOST / 'shared.txt').read_text(),
                    'listing': sorted(os.listdir(HOST)),
                    'walked': sorted(
                        os.path.relpath(os.path.join(d, f), HOST)
                        for d, _, files in os.walk(HOST)
                        for f in files
                    ),
                    'doomed': (HOST / 'doomed.txt').exists(),
                }}
                self.unit.status = ops.ActiveStatus(json.dumps(status))
    """
    root = write_charm(tmp_path / 'files', charm)
    with testing.Juju() as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        trace = juju.settle()
        unit = app.leader
        assert trace[0].state_out.unit_status.message == 'from the host'
        assert status(unit) == {
            'shared': 'from the host, and the charm',
            'listing': ['new', 'shared.txt'],
            'walked': ['new/moved.txt', 'shared.txt'],
            'doomed': False,
        }
        in_root = unit.filesystem / host.relative_to('/')
        assert (in_root / 'shared.txt').read_text() == 'from the host, and the charm'
        assert (in_root / 'new' / 'moved.txt').read_text() == 'from the host, and the charm'
    # The host is as the test left it.
    assert sorted(p.name for p in host.iterdir()) == ['doomed.txt', 'shared.txt']
    assert (host / 'shared.txt').read_text() == 'from the host'


TEMP_CHARM = """
    import json
    import os
    import tempfile

    import ops


    class TempCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            fd, made = tempfile.mkstemp()
            os.close(fd)
            with open('/tmp/ops-testing-literal.txt', 'w') as f:
                f.write('literal')
            with open('/etc/os-release') as f:
                release = dict(line.split('=', 1) for line in f.read().splitlines())
            status = {
                'tempdir': tempfile.gettempdir(),
                'made': made,
                'codename': release['VERSION_CODENAME'],
                'charm_dir': str(self.charm_dir),
            }
            (self.charm_dir / 'written-by-the-charm').write_text('x')
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""


@pytest.mark.parametrize(
    ('meta', 'codename'),
    [
        ('base: ubuntu@22.04', 'jammy'),
        ('bases:\n  - name: ubuntu\n    channel: "20.04"', 'focal'),
        (
            'bases:\n  - build-on: [{name: ubuntu, channel: "22.04"}]\n'
            '    run-on: [{name: ubuntu, channel: "24.04"}]',
            'noble',
        ),
        ('', 'noble'),
    ],
)
def test_temp_files_os_release_and_the_charm_directory_are_the_unit_s(
    tmp_path: pathlib.Path, isolated: bool, meta: str, codename: str
):
    root = write_charm(tmp_path / 'temp', TEMP_CHARM, meta=f'name: temp\n{meta}\n')
    with testing.Juju() as juju:
        app = juju._deploy(root, num_units=2, **deploy_kwargs(isolated))
        juju.settle()
        for unit in app.units:
            result = status(unit)
            assert result['tempdir'] == str(unit.filesystem / 'tmp')
            assert pathlib.Path(result['made']).parent == unit.filesystem / 'tmp'
            assert (unit.filesystem / 'tmp/ops-testing-literal.txt').read_text() == 'literal'
            assert result['codename'] == codename
            charm_dir = unit.filesystem / f'var/lib/juju/agents/unit-temp-{unit.id}/charm'
            assert result['charm_dir'] == str(charm_dir)
            assert (charm_dir / 'src' / 'charm.py').exists()
            assert (charm_dir / 'written-by-the-charm').exists()
    assert not (root / 'written-by-the-charm').exists()
    assert sorted(p.name for p in root.iterdir()) == ['charmcraft.yaml', 'src']


def test_container_files_and_mounts_still_reach_the_framework(
    tmp_path: pathlib.Path, isolated: bool
):
    mounted = tmp_path / 'mounted'
    mounted.mkdir()
    (mounted / 'config.yaml').write_text('from the mount')
    charm = """
        import json

        import ops


        class WorkloadCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on.workload_pebble_ready, self._on_ready)

            def _on_ready(self, event):
                container = event.workload
                container.push('/etc/workload/settings', 'pushed', make_dirs=True)
                with container.pull('/etc/workload/settings') as f:
                    pushed = f.read()
                with container.pull('/srv/config.yaml') as f:
                    mounted = f.read()
                status = {'pushed': pushed, 'mounted': mounted}
                self.unit.status = ops.ActiveStatus(json.dumps(status))
    """
    root = write_charm(
        tmp_path / 'workload',
        charm,
        meta='name: workload\ncontainers:\n  workload: {}\n',
    )
    template = testing.State(
        containers={
            testing.Container(
                'workload',
                can_connect=True,
                mounts={'srv': testing.Mount(location='/srv', source=mounted)},
            )
        }
    )
    with testing.Juju() as juju:
        app = juju._deploy(root, state_template=template, **deploy_kwargs(isolated))
        juju.settle()
        assert status(app.leader) == {'pushed': 'pushed', 'mounted': 'from the mount'}
        # The workload's files are the container's, not the charm's.
        assert not (app.leader.filesystem / 'etc/workload').exists()


def test_the_roots_go_when_the_juju_closes(tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'temp', TEMP_CHARM, meta='name: temp\n')
    with testing.Juju() as juju:
        app = juju.deploy(root, num_units=2)
        juju.settle()
        removed = app.units[1]
        juju.remove_unit(app, num_units=1)
        juju.settle()
        # A removed unit's filesystem is still there to assert on.
        assert (removed.filesystem / 'tmp/ops-testing-literal.txt').exists()
        filesystems = [u.filesystem for u in (app.units[0], removed)]
    assert not any(f.exists() for f in filesystems)


TEMPLATE_CHARM = """
    import json
    import os

    import ops


    class TemplateCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.config_changed, self._on_config_changed)

        def _on_config_changed(self, _):
            with open('templates/route.j2') as f:
                template = f.read()
            status = {'template': template, 'cwd': os.getcwd(), 'charm_dir': str(self.charm_dir)}
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""


def test_each_dispatch_runs_in_the_unit_s_charm_directory(
    tmp_path: pathlib.Path, isolated: bool, monkeypatch: pytest.MonkeyPatch
):
    root = write_charm(
        tmp_path / 'templated',
        TEMPLATE_CHARM,
        files={'templates/route.j2': 'route {{ host }}'},
    )
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with testing.Juju() as juju:
        app = juju._deploy(root, num_units=2, **deploy_kwargs(isolated))
        juju.settle()
        charm_dirs: set[str] = set()
        for unit in app.units:
            result = status(unit)
            assert result['template'] == 'route {{ host }}'
            assert result['cwd'] == result['charm_dir']
            charm_dirs.add(result['cwd'])
        # Each unit runs in its own copy of the charm.
        assert len(charm_dirs) == 2
        assert pathlib.Path.cwd() == elsewhere
    assert pathlib.Path.cwd() == elsewhere


RESTART_CHARM = """
    import hashlib
    import json

    import ops

    CONFIG = '/etc/workload/workload.ini'


    class RestartCharm(ops.CharmBase):
        _stored = ops.StoredState()

        def __init__(self, framework):
            super().__init__(framework)
            self._stored.set_default(restarts=0)
            framework.observe(self.on.config_changed, self._reconcile)
            framework.observe(self.on.workload_pebble_ready, self._reconcile)
            framework.observe(self.on.update_status, self._reconcile)

        def _reconcile(self, _):
            container = self.unit.get_container('workload')
            if not container.can_connect():
                return
            wanted = f'[server]\\nport = {self.config["port"]}\\nunit = {self.unit.name}\\n'
            # Only restart when the file changes, as Grafana does.
            current = None
            if container.exists(CONFIG):
                with container.pull(CONFIG) as f:
                    current = f.read()
            if current != wanted:
                container.push(CONFIG, wanted, make_dirs=True)
                self._stored.restarts += 1
            data = ''
            if container.exists('/var/lib/workload/data'):
                with container.pull('/var/lib/workload/data') as f:
                    data = f.read()
            container.push('/var/lib/workload/data', data + 'x')
            status = {'restarts': self._stored.restarts, 'data': len(data) + 1}
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""

RESTART_META = """
    name: restarter
    containers:
      workload:
        resource: workload-image
        mounts:
          - storage: data
            location: /var/lib/workload
    storage:
      data:
        type: filesystem
    config:
      options:
        port: {type: int, default: 3000}
        colour: {type: string, default: blue}
"""


def test_container_files_persist_per_unit_and_can_be_read(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(tmp_path / 'restarter', RESTART_CHARM, meta=textwrap.dedent(RESTART_META))
    template = testing.State(storages={testing.Storage('data')})
    with testing.Juju() as juju:
        app = juju._deploy(root, num_units=2, state_template=template, **deploy_kwargs(isolated))
        juju.settle()
        # config-changed and pebble-ready both reconcile, but only the first writes.
        assert [status(unit)['restarts'] for unit in app.units] == [1, 1]

        juju.config(app, {'colour': 'red'})
        juju.settle()
        assert [status(unit)['restarts'] for unit in app.units] == [1, 1]

        juju.config(app, {'port': 3001})
        juju.settle()
        assert [status(unit)['restarts'] for unit in app.units] == [2, 2]

        roots = [unit.container_filesystem('workload') for unit in app.units]
        assert roots[0] != roots[1]
        for unit, container_root in zip(app.units, roots, strict=True):
            config = (container_root / 'etc/workload/workload.ini').read_text()
            assert config == f'[server]\nport = 3001\nunit = {unit.name}\n'
            # The storage that the metadata mounts in the container is kept too.
            data = (container_root / 'var/lib/workload/data').read_text()
            assert len(data) == status(unit)['data'] > 1
            # The workload's files are the container's, not the charm's.
            assert not (unit.filesystem / 'etc/workload').exists()

        removed = app.units[1]
        juju.remove_unit(app, num_units=1)
        juju.settle()
        assert (removed.container_filesystem('workload') / 'etc/workload/workload.ini').exists()
    assert not any(r.exists() for r in roots)


def test_container_filesystem_needs_a_container_the_charm_has(tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'temp', TEMP_CHARM, meta='name: temp\n')
    with testing.Juju() as juju:
        app = juju.deploy(root)
        with pytest.raises(testing.errors.JujuError, match="no container 'workload'"):
            app.leader.container_filesystem('workload')


# lightkube


LIGHTKUBE_CHARM = """
    import asyncio
    import json

    import ops
    import lightkube
    import lightkube.core.async_client
    from lightkube import ApiError
    from lightkube.core.client import Client as DirectClient
    {resources}


    class KubeCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            self.client = lightkube.Client()
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            direct = DirectClient(namespace='other')
            probe = {{
                'list': list(self.client.list(Pod)),
                'namespace': direct.namespace,
                'created': self.client.create(make_pod('web-0')).metadata.name,
                'deleted': self.client.delete(Pod, 'web-0'),
            }}
            try:
                self.client.get(Pod, 'missing')
            except ApiError as e:
                probe['get'] = e.status.code

            async def run_async():
                client = lightkube.core.async_client.AsyncClient()
                items = [item async for item in client.list(Pod)]
                try:
                    await client.get(Pod, 'missing')
                except ApiError as e:
                    return [items, e.status.code]

            probe['async'] = asyncio.run(run_async())
            self.unit.status = ops.ActiveStatus(json.dumps(probe))
"""

FAKE_RESOURCES = """
import types


class Pod:
    def __init__(self, metadata):
        self.metadata = metadata


def make_pod(name):
    return Pod(types.SimpleNamespace(name=name))
"""

REAL_RESOURCES = """
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Pod


def make_pod(name):
    return Pod(metadata=ObjectMeta(name=name))
"""

LIGHTKUBE_EXPECTED: dict[str, Any] = {
    'list': [],
    'namespace': 'other',
    'created': 'web-0',
    'deleted': None,
    'get': 404,
    'async': [[], 404],
}


def test_lightkube_has_no_cluster_to_reach(tmp_path: pathlib.Path, isolated: bool):
    charm = LIGHTKUBE_CHARM.format(resources=textwrap.indent(FAKE_RESOURCES, '    '))
    root = write_charm(tmp_path / 'kube', charm, files=FAKE_LIGHTKUBE)
    with testing.Juju() as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert status(app.leader) == LIGHTKUBE_EXPECTED


def test_the_real_lightkube_has_no_cluster_to_reach(tmp_path: pathlib.Path, isolated: bool):
    pytest.importorskip('lightkube')
    charm = LIGHTKUBE_CHARM.format(resources=textwrap.indent(REAL_RESOURCES, '    '))
    root = write_charm(tmp_path / 'kube', charm)
    with testing.Juju() as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert status(app.leader) == LIGHTKUBE_EXPECTED


def test_the_charm_s_own_lightkube_patches_win(tmp_path: pathlib.Path):
    charm = """
        import ops
        import lightkube


        class KubeCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on.install, self._on_install)

            def _on_install(self, _):
                self.unit.status = ops.ActiveStatus(lightkube.Client().get('Pod', 'web-0'))
    """
    mocking = """
        import contextlib
        from unittest import mock


        @contextlib.contextmanager
        def mocked():
            with mock.patch('lightkube.core.client.Client.get', return_value='"from the charm"'):
                yield
    """
    root = write_charm(
        tmp_path / 'kube', charm, files={**FAKE_LIGHTKUBE, 'tests/unit/mocking.py': mocking}
    )
    with testing.Juju() as juju:
        app = juju.deploy(root)
        juju.settle()
        assert status(app.leader) == 'from the charm'


# The Kubernetes patch libraries


K8S_PATCH_CHARM = """
    import json

    import ops
    from charms.comsys_libs.v0.kubernetes_statefulset_patch import KubernetesStatefulsetPatch
    from charms.observability_libs.v0.kubernetes_compute_resources_patch import (
        KubernetesComputeResourcesPatch,
    )
    from charms.observability_libs.v1.kubernetes_service_patch import KubernetesServicePatch


    class PatchingCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            self.service = KubernetesServicePatch(self, [('http', 80)])
            self.resources = KubernetesComputeResourcesPatch(
                self, 'workload', resource_reqs_func=lambda: None
            )
            self.statefulset = KubernetesStatefulsetPatch(self, {})
            framework.observe(self.on.start, self._on_start)

        def _on_start(self, _):
            status = {
                'service': self.service.is_patched(),
                'namespace': self.service._namespace,
                'resources': self.resources.is_ready(),
                'status': self.resources.get_status().name,
            }
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""


def test_the_k8s_patch_libraries_do_nothing(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(tmp_path / 'patching', K8S_PATCH_CHARM, files=FAKE_K8S_PATCH_LIBS)
    with testing.Juju('mymodel') as juju:
        app = juju._deploy(root, num_units=2, **deploy_kwargs(isolated))
        # install, config-changed and remove all reach the libraries' handlers.
        juju.settle()
        assert status(app.leader) == {
            'service': True,
            'namespace': 'mymodel',
            'resources': True,
            'status': 'active',
        }
        juju.remove_unit(app, num_units=1)
        juju.settle()


def test_the_charm_s_own_k8s_patches_win(tmp_path: pathlib.Path):
    mocking = """
        import contextlib
        from unittest import mock

        from charms.observability_libs.v0 import kubernetes_compute_resources_patch as lib


        @contextlib.contextmanager
        def mocked():
            patch = lib.KubernetesComputeResourcesPatch
            with mock.patch.object(patch, 'is_ready', lambda _: False):
                yield
    """
    root = write_charm(
        tmp_path / 'patching',
        K8S_PATCH_CHARM,
        files={**FAKE_K8S_PATCH_LIBS, 'tests/unit/mocking.py': mocking},
    )
    with testing.Juju() as juju:
        app = juju.deploy(root)
        juju.settle()
        assert status(app.leader)['resources'] is False


# snap


SNAP_CACHE_CHARM = """
    import json

    import ops
    from {module} import snap


    class SnapCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)
            framework.observe(self.on.start, self._on_start)

        def _on_install(self, _):
            cache = snap.SnapCache()
            assert len(cache) == 0
            cache['lxd'].ensure(snap.SnapState.Present, classic=True)
            snap.add('juju', channel='3.6/stable')
            snap.SnapCache()['lxd'].ensure(snap.SnapState.Latest, revision=35000)

        def _on_start(self, _):
            cache = snap.SnapCache()
            status = {{
                name: [cache[name].present, cache[name].channel, cache[name].revision]
                for name in ('lxd', 'juju', 'microk8s')
            }}
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""

SNAP_CACHE_EXPECTED = {
    'lxd': [True, 'latest/stable', '35000'],
    'juju': [True, '3.6/stable', '1'],
    'microk8s': [False, '', ''],
}


@pytest.mark.parametrize(
    ('module', 'files'),
    [
        ('charms.operator_libs_linux.v2', FAKE_OPERATOR_LIBS_LINUX_SNAP),
        ('charmlibs', FAKE_CHARMLIBS_SNAP_1),
    ],
    ids=['operator_libs_linux', 'charmlibs-1'],
)
def test_the_snap_cache_is_empty_and_ensure_succeeds(
    tmp_path: pathlib.Path, isolated: bool, module: str, files: dict[str, str]
):
    root = write_charm(tmp_path / 'snapper', SNAP_CACHE_CHARM.format(module=module), files=files)
    with testing.Juju(type='lxd') as juju:
        app = juju._deploy(root, num_units=2, **deploy_kwargs(isolated))
        juju.settle()
        for unit in app.units:
            assert status(unit) == SNAP_CACHE_EXPECTED


SNAPD_CLIENT_CHARM = """
    import json

    import ops
    from charmlibs import snap


    class SnapCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)
            framework.observe(self.on.start, self._on_start)

        def _on_install(self, _):
            try:
                snap.list_one('lxd')
            except snap.NotInstalledError:
                pass
            else:
                raise AssertionError('lxd is installed')
            self.installed = snap.ensure_installed('lxd', channel='5.21/stable')

        def _on_start(self, _):
            info = snap.list_one('lxd')
            status = {
                'again': snap.ensure_installed('lxd', channel='5.21/stable'),
                'tracking': info.tracking,
            }
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""


def test_charmlibs_snap_2_has_an_empty_snapd(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(
        tmp_path / 'snapper',
        SNAPD_CLIENT_CHARM.replace('info.tracking', 'info["tracking-channel"]'),
        files=FAKE_CHARMLIBS_SNAP_2,
    )
    with testing.Juju(type='lxd') as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert status(app.leader) == {'again': False, 'tracking': '5.21/stable'}


def _real_charmlibs_snap(major: int) -> None:
    snap = pytest.importorskip('charmlibs.snap')
    version = getattr(snap, '__version__', '0')
    if int(version.split('.')[0]) != major:
        pytest.skip(f'charmlibs-snap {version} is installed, not {major}.x')


def test_the_real_charmlibs_snap_1_has_an_empty_cache(tmp_path: pathlib.Path, isolated: bool):
    _real_charmlibs_snap(1)
    root = write_charm(tmp_path / 'snapper', SNAP_CACHE_CHARM.format(module='charmlibs'))
    with testing.Juju(type='lxd') as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert status(app.leader) == SNAP_CACHE_EXPECTED


def test_the_real_charmlibs_snap_2_has_an_empty_snapd(tmp_path: pathlib.Path, isolated: bool):
    _real_charmlibs_snap(2)
    root = write_charm(tmp_path / 'snapper', SNAPD_CLIENT_CHARM)
    with testing.Juju(type='lxd') as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert status(app.leader) == {'again': True, 'tracking': '5.21/stable'}


# apt


_APT_LIB = """
    import enum
    import re
    import subprocess

    _GPG_KEY_DIR = '/etc/apt/trusted.gpg.d/'


    class Error(Exception):
        pass


    class PackageError(Error):
        pass


    class PackageNotFoundError(Error):
        pass


    class GPGKeyError(Error):
        pass


    class PackageState(enum.Enum):
        Present = 'present'
        Absent = 'absent'
        Latest = 'latest'
        Available = 'available'


    class Version:
        def __init__(self, version, epoch):
            self._version = version
            self._epoch = epoch

        def __str__(self):
            return f'{self._epoch}:{self._version}' if self._epoch else self._version


    class DebianPackage:
        def __init__(self, name, version, epoch, arch, state):
            self._name = name
            self._arch = arch
            self._state = state
            self._version = Version(version, epoch)

        @staticmethod
        def _apt(command, package_names, optargs=None):
            subprocess.run(['apt-get', '-y', command, package_names], check=True)
            raise PackageError('reached the host')

        def _add(self):
            self._apt('install', f'{self.name}={self.version}')

        def _remove(self):
            self._apt('remove', f'{self.name}={self.version}')

        @property
        def name(self):
            return self._name

        @property
        def version(self):
            return self._version

        @property
        def arch(self):
            return self._arch

        @property
        def present(self):
            return self._state in (PackageState.Present, PackageState.Latest)

        def ensure(self, state):
            if self._state is not state:
                if state not in (PackageState.Present, PackageState.Latest):
                    self._remove()
                else:
                    self._add()
            self._state = state

        @staticmethod
        def _get_epoch_from_version(version):
            matches = re.search(r'^((?P<epoch>\\d+):)?(?P<version>.*)', version).groupdict()
            return matches.get('epoch', ''), matches['version']

        @classmethod
        def from_system(cls, package, version='', arch=''):
            try:
                return cls.from_installed_package(package, version, arch)
            except PackageNotFoundError:
                pass
            try:
                return cls.from_apt_cache(package, version, arch)
            except (PackageNotFoundError, PackageError):
                raise PackageNotFoundError(f'{package} could not be found') from None

        @classmethod
        def from_installed_package(cls, package, version='', arch=''):
            output = subprocess.check_output(['dpkg', '-l', package], text=True)
            for line in output.splitlines()[5:]:
                raise PackageError('parsed the host')
            raise PackageNotFoundError(f'Package {package} is not installed!')

        @classmethod
        def from_apt_cache(cls, package, version='', arch=''):
            output = subprocess.check_output(['apt-cache', 'show', package], text=True)
            vals = {}
            for line in output.splitlines():
                if line.startswith(('Package', 'Architecture', 'Version')):
                    key, value = line.split(':', 1)
                    vals[key] = value.strip()
            epoch, number = cls._get_epoch_from_version(vals['Version'])
            available = PackageState.Available
            return cls(vals['Package'], number, epoch, vals['Architecture'], available)


    def add_package(package_names, version='', arch='', update_cache=False):
        if update_cache:
            update()
        names = [package_names] if isinstance(package_names, str) else package_names
        added = []
        for name in names:
            package = DebianPackage.from_system(name, version, arch)
            package.ensure(state=PackageState.Present)
            added.append(package)
        return added[0] if len(added) == 1 else added


    def remove_package(package_names):
        names = [package_names] if isinstance(package_names, str) else package_names
        removed = []
        for name in names:
            try:
                package = DebianPackage.from_installed_package(name)
            except PackageNotFoundError:
                continue
            package.ensure(state=PackageState.Absent)
            removed.append(package)
        return removed[0] if len(removed) == 1 else removed


    def update():
        subprocess.run(['apt-get', 'update'], check=True)
        raise PackageError('reached the host')


    def import_key(key):
        raise GPGKeyError('reached the host')


    class DebianRepository:
        def __init__(self, enabled, repotype, uri, release, groups, filename=''):
            self.enabled = enabled
            self.repotype = repotype
            self.uri = uri
            self.release = release
            self.groups = groups
            self.filename = filename

        @staticmethod
        def from_repo_line(repo_line, write_file=True):
            repo = RepositoryMapping._parse(repo_line, 'UserInput')
            if write_file:
                _add_repository(repo)
            return repo

        def disable(self):
            raise OSError('reached the host')


    def _repo_to_identifier(repo):
        return f'{repo.repotype}-{repo.uri}-{repo.release}'


    def _repo_to_line(repo, include_signed_by=True):
        return f"{repo.repotype} {repo.uri} {repo.release} {' '.join(repo.groups)}"


    def _add_repository(repo, remove=False, update_cache=False):
        subprocess.run(['add-apt-repository', '--yes', _repo_to_line(repo)], check=True)
        raise OSError('reached the host')


    class RepositoryMapping:
        def __init__(self):
            with open('/etc/apt/sources.list') as f:
                f.read()
            raise OSError('reached the host')

        def __len__(self):
            return len(self._repository_map)

        def __contains__(self, key):
            return key in self._repository_map

        def add(self, repo):
            _add_repository(repo)
            self._repository_map[_repo_to_identifier(repo)] = repo

        @staticmethod
        def _parse(line, filename):
            repotype, uri, release, *groups = line.split()
            return DebianRepository(True, repotype, uri, release, groups, filename)
"""

FAKE_OPERATOR_LIBS_LINUX_APT = {'lib/charms/operator_libs_linux/v0/apt.py': _APT_LIB}

FAKE_CHARMLIBS_APT = {'lib/charmlibs/apt/__init__.py': _APT_LIB}

APT_CHARM = """
    import json

    import ops
    from {module} import apt

    ARMORED = '-----BEGIN PGP PUBLIC KEY BLOCK-----\\\\nabc\\\\n-----END PGP PUBLIC KEY BLOCK-----'


    class AptCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)
            framework.observe(self.on.config_changed, self._on_config_changed)

        def _on_install(self, _):
            apt.update()
            apt.import_key(ARMORED)
            repositories = apt.RepositoryMapping()
            assert len(repositories) == 0
            line = 'deb http://ppa.example.com/ops jammy main'
            repositories.add(apt.DebianRepository.from_repo_line(line, write_file=False))
            apt.add_package(['nginx', 'git'])
            apt.add_package('redis-server', version='7.0.15', update_cache=True)
            apt.add_package('telnet')
            apt.remove_package('telnet')

        def _on_config_changed(self, _):
            status = {{}}
            for name in ('nginx', 'git', 'redis-server', 'telnet'):
                try:
                    package = apt.DebianPackage.from_installed_package(name)
                except apt.PackageNotFoundError:
                    status[name] = 'not installed'
                else:
                    status[name] = [package.present, str(package.version), package.arch]
            status['repositories'] = len(apt.RepositoryMapping())
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""

APT_EXPECTED = {
    'nginx': [True, '1.0', 'amd64'],
    'git': [True, '1.0', 'amd64'],
    'redis-server': [True, '7.0.15', 'amd64'],
    'telnet': 'not installed',
    'repositories': 1,
}

APT_HISTORY = [
    'update',
    'import-key /etc/apt/trusted.gpg.d/{key}.gpg',
    'add-repository deb http://ppa.example.com/ops jammy main',
    'install nginx=1.0',
    'install git=1.0',
    'update',
    'install redis-server=7.0.15',
    'install telnet=1.0',
    'remove telnet',
]


def _apt_history(app: testing.App, unit: testing.Unit) -> list[str]:
    runner: Any = app._runner
    return runner._mocking._apt[unit.name].history


@pytest.mark.parametrize(
    ('module', 'files'),
    [
        ('charms.operator_libs_linux.v0', FAKE_OPERATOR_LIBS_LINUX_APT),
        ('charmlibs', FAKE_CHARMLIBS_APT),
    ],
    ids=['operator_libs_linux', 'charmlibs'],
)
def test_a_machine_charm_s_apt_installs_succeed_and_are_recorded(
    tmp_path: pathlib.Path, isolated: bool, module: str, files: dict[str, str]
):
    root = write_charm(
        tmp_path / 'apt-user',
        APT_CHARM.format(module=module),
        meta='name: apt-user\nbase: ubuntu@22.04\n',
        files=files,
    )
    with testing.Juju(type='lxd') as juju:
        app = juju._deploy(root, num_units=2, **deploy_kwargs(isolated))
        juju.settle()
        for unit in app.units:
            # What install asked for is still there in config-changed.
            assert status(unit) == APT_EXPECTED
            if not isolated:
                history = _apt_history(app, unit)
                key = history[1].split('/')[-1].removesuffix('.gpg')
                assert history == [line.format(key=key) for line in APT_HISTORY]
    assert not pathlib.Path('/etc/apt/sources.list.d/ppa.example.com-ops-jammy.list').exists()


def test_the_real_charmlibs_apt_installs_nothing(tmp_path: pathlib.Path, isolated: bool):
    pytest.importorskip('opentelemetry.trace')
    pytest.importorskip('charmlibs.apt')
    root = write_charm(tmp_path / 'apt-user', APT_CHARM.format(module='charmlibs'))
    with testing.Juju(type='lxd') as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert status(app.leader) == APT_EXPECTED


def test_the_charm_s_own_apt_patches_win(tmp_path: pathlib.Path):
    charm = """
        import ops
        from charms.operator_libs_linux.v0 import apt


        class AptCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on.install, self._on_install)

            def _on_install(self, _):
                package = apt.add_package('nginx')
                self.unit.status = ops.ActiveStatus(f'"{package.version}"')
    """
    mocking = """
        import contextlib
        from unittest import mock

        from charms.operator_libs_linux.v0 import apt


        @contextlib.contextmanager
        def mocked():
            package = apt.DebianPackage('nginx', '1.24.0', '', 'amd64', apt.PackageState.Present)
            with mock.patch.object(apt.DebianPackage, 'from_system', return_value=package):
                yield
    """
    root = write_charm(
        tmp_path / 'apt-user',
        charm,
        files={**FAKE_OPERATOR_LIBS_LINUX_APT, 'tests/unit/mocking.py': mocking},
    )
    with testing.Juju(type='lxd') as juju:
        app = juju.deploy(root)
        juju.settle()
        assert status(app.leader) == '1.24.0'


# Switching the defaults off


DISABLE_CHARM = """
    import json
    import pathlib

    import ops
    import lightkube
    from charms.observability_libs.v1.kubernetes_service_patch import KubernetesServicePatch
    from charms.operator_libs_linux.v0 import apt
    from charms.operator_libs_linux.v2 import snap

    HOST_FILE = pathlib.Path({host_file!r})


    def attempt(f):
        try:
            return f()
        except Exception as e:
            return type(e).__name__


    class DisablingCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            self.service = KubernetesServicePatch(self, [('http', 80)])
            framework.observe(self.on.start, self._on_start)

        def _on_start(self, _):
            HOST_FILE.write_text('written')
            status = {{
                'lightkube': attempt(lambda: lightkube.Client() and 'faked'),
                'k8s-patch-libs': attempt(lambda: self.service.is_patched()),
                'snap': attempt(lambda: len(snap.SnapCache())),
                'apt': attempt(lambda: str(apt.add_package('nginx').version)),
            }}
            self.unit.status = ops.ActiveStatus(json.dumps(status))
"""

FAKED = {'lightkube': 'faked', 'k8s-patch-libs': True, 'snap': 0, 'apt': '1.0'}
REAL = {
    'lightkube': 'ConfigError',
    'k8s-patch-libs': 'RuntimeError',
    'snap': 'SnapError',
    'apt': 'KeyError',
}


@pytest.mark.parametrize(
    'disable', ['lightkube', 'k8s-patch-libs', 'snap', 'apt', 'filesystem', 'all'], ids=str
)
def test_each_default_can_be_disabled(tmp_path: pathlib.Path, isolated: bool, disable: str):
    host_file = tmp_path / f'host-{uuid.uuid4().hex}'
    disabled = '"all"' if disable == 'all' else f'["{disable}"]'
    # The service patch's install and remove handlers would fail with it
    # switched off, so the charm only uses it on start.
    charm = DISABLE_CHARM.format(host_file=str(host_file)).replace(
        "self.service = KubernetesServicePatch(self, [('http', 80)])",
        'self.service = KubernetesServicePatch.__new__(KubernetesServicePatch)',
    )
    root = write_charm(
        tmp_path / 'disabling',
        charm,
        files={
            **FAKE_LIGHTKUBE,
            **FAKE_K8S_PATCH_LIBS,
            **FAKE_OPERATOR_LIBS_LINUX_SNAP,
            **FAKE_OPERATOR_LIBS_LINUX_APT,
            'pyproject.toml': f'[tool.ops.testing.mocking]\ndisable = {disabled}\n',
        },
    )
    with testing.Juju() as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        result = status(app.leader)
        in_root = (app.leader.filesystem / host_file.relative_to('/')).exists()
    expected = {name: REAL[name] if disable in (name, 'all') else FAKED[name] for name in FAKED}
    if disable == 'all':
        # With subprocess switched off too, the fake library's dpkg call reaches
        # the host, and how that fails depends on the host.
        assert result.pop('apt') != FAKED['apt']
        del expected['apt']
    assert result == expected
    reached_host = disable in ('filesystem', 'all')
    assert host_file.exists() is reached_host
    assert in_root is not reached_host


# Missing dependency groups


def test_a_missing_dependency_group_package_is_reported_from_the_worker(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path / 'probe',
        'import ops\n\nclass ProbeCharm(ops.CharmBase):\n    pass\n',
        files={
            'tests/unit/mocking.py': (
                'import contextlib\n\n@contextlib.contextmanager\ndef mocked():\n    yield\n'
            ),
            'pyproject.toml': """
                [dependency-groups]
                unit = ["definitely-not-an-installed-package>=1"]

                [tool.ops.testing.mocking]
                dependency-groups = ["unit"]
            """,
        },
    )
    with testing.Juju() as juju:
        juju._deploy(root, python_executable=sys.executable)
        with pytest.raises(
            testing.errors.IsolationError,
            match=r"definitely-not-an-installed-package.*'unit' dependency group",
        ):
            juju.settle()


@pytest.mark.skipif(os.geteuid() == 0, reason='root can give files away')
def test_chown_in_the_root_succeeds(tmp_path: pathlib.Path):
    charm = """
        import os
        import shutil

        import ops


        class OwnerCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on.install, self._on_install)

            def _on_install(self, _):
                with open('/etc/ops-testing-owned', 'w') as f:
                    f.write('mine')
                os.chown('/etc/ops-testing-owned', 0, 0)
                self.unit.status = ops.ActiveStatus('"owned"')
    """
    root = write_charm(tmp_path / 'owner', charm)
    with testing.Juju() as juju:
        app = juju.deploy(root)
        juju.settle()
        assert status(app.leader) == 'owned'
