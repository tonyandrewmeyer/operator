# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the charmlibs default: each library's mocked(), found from the charm's imports."""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import sys
import textwrap
from collections.abc import Iterator

import pytest
from scenario import _charmlibs_mocking

from ops import testing

LIBRARY = """
    def value():
        return 'real'
"""

TESTING = """
    import contextlib
    from unittest import mock

    import charmlibs.fakelib


    @contextlib.contextmanager
    def mocked():
        with mock.patch.object(charmlibs.fakelib, 'value', lambda: 'mocked'):
            yield
"""

CHARM = """
    import ops

    from charmlibs import fakelib


    class FakelibCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            self.unit.status = ops.ActiveStatus(fakelib.value())
"""


ENTRY_POINTS = """
[ops.testing.mocking]
charmlibs.fakelib = charmlibs.fakelib_testing:mocked
"""


def write(root: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content).lstrip())
    return root


def distribution(
    site: pathlib.Path,
    name: str,
    version: str,
    package_files: dict[str, str],
    *,
    entry_points: str | None = None,
    record: bool = True,
):
    """Install a distribution into ``site`` by hand: its files and its .dist-info."""
    write(site, package_files)
    info = f'{name.replace("-", "_")}-{version}.dist-info'
    files = {f'{info}/METADATA': f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'}
    if entry_points is not None:
        files[f'{info}/entry_points.txt'] = entry_points
    if record:
        files[f'{info}/RECORD'] = ''.join(f'{p},,\n' for p in [*package_files, *files])
    write(site, files)


@pytest.fixture
def site(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """A directory on sys.path (and on the worker's PYTHONPATH) to install fake packages into."""
    site = tmp_path / 'site'
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))  # type: ignore[reportUnknownMemberType]
    existing = os.environ.get('PYTHONPATH')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(filter(None, [str(site), existing])))
    yield site
    for name in list(sys.modules):
        if name == 'charmlibs' or name.startswith('charmlibs.'):
            del sys.modules[name]
    importlib.invalidate_caches()


def install_library(site: pathlib.Path, version: str = '1.2.0', *, record: bool = True):
    distribution(
        site,
        'charmlibs-fakelib',
        version,
        {'charmlibs/fakelib/__init__.py': LIBRARY},
        record=record,
    )


def install_testing(site: pathlib.Path, version: str = '1.2.0'):
    distribution(
        site,
        'charmlibs-fakelib-testing',
        version,
        {'charmlibs/fakelib_testing/__init__.py': TESTING},
        entry_points=ENTRY_POINTS,
    )


def charm(tmp_path: pathlib.Path, pyproject: str | None = None) -> pathlib.Path:
    files = {'metadata.yaml': 'name: fakelib-user\n', 'src/charm.py': CHARM}
    if pyproject is not None:
        files['pyproject.toml'] = pyproject
    return write(tmp_path / 'charm', files)


@pytest.fixture(params=[False, True], ids=['in-process', 'isolated'])
def isolated(request: pytest.FixtureRequest) -> bool:
    return request.param


def deploy(juju: testing.Juju, root: pathlib.Path, isolated: bool) -> testing.App:
    if isolated:
        return juju._deploy(root, python_executable=sys.executable)
    return juju.deploy(root)


def test_the_library_s_mocking_is_opened_around_the_charm(
    tmp_path: pathlib.Path, site: pathlib.Path, isolated: bool
):
    install_library(site)
    install_testing(site)
    with testing.Juju() as juju:
        app = deploy(juju, charm(tmp_path), isolated)
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('mocked')


def test_the_charm_can_switch_the_default_off(
    tmp_path: pathlib.Path, site: pathlib.Path, isolated: bool
):
    install_library(site)
    root = charm(tmp_path, '[tool.ops.testing.mocking]\ndisable = ["charmlibs"]\n')
    with testing.Juju() as juju:
        app = deploy(juju, root, isolated)
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('real')


def test_a_library_with_no_testing_package_runs_unmocked(
    tmp_path: pathlib.Path, site: pathlib.Path, isolated: bool
):
    install_library(site)
    with testing.Juju() as juju:
        app = deploy(juju, charm(tmp_path), isolated)
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('real')


def test_a_testing_package_at_another_version_is_reported_from_the_worker(
    tmp_path: pathlib.Path, site: pathlib.Path
):
    install_library(site, '1.2.0')
    install_testing(site, '1.1.0')
    with testing.Juju() as juju:
        juju._deploy(charm(tmp_path), python_executable=sys.executable)
        with pytest.raises(testing.errors.IsolationError, match=r'is 1\.1\.0'):
            juju.settle()


def test_a_testing_package_at_another_version_is_an_error(
    tmp_path: pathlib.Path, site: pathlib.Path
):
    install_library(site, '1.2.0')
    install_testing(site, '1.1.0')
    with testing.Juju() as juju:
        with pytest.raises(
            testing.errors.JujuError, match=r'charmlibs\.fakelib 1\.2\.0.*is 1\.1\.0'
        ):
            juju.deploy(charm(tmp_path))


def test_an_editable_library_is_found_from_its_name(tmp_path: pathlib.Path, site: pathlib.Path):
    # An editable install's RECORD lists a .pth file rather than the package.
    install_library(site, record=False)
    install_testing(site)
    with testing.Juju() as juju:
        app = juju.deploy(charm(tmp_path))
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('mocked')


def test_a_charm_spec_gets_the_default_for_its_class_s_module(
    tmp_path: pathlib.Path, site: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    install_library(site)
    install_testing(site)
    module = write(tmp_path / 'mod', {'fakelib_charm.py': CHARM}) / 'fakelib_charm.py'
    spec = importlib.util.spec_from_file_location('_fakelib_charm_under_test', module)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    spec.loader.exec_module(loaded)
    with testing.Juju() as juju:
        app = juju.deploy(testing.CharmSpec(loaded.FakelibCharm, meta={'name': 'fakelib-user'}))
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('mocked')


def test_charmlibs_imports_in_every_form(tmp_path: pathlib.Path):
    source = write(
        tmp_path,
        {
            'src/charm.py': """
                import charmlibs.pathops
                import os
                from charmlibs import apt, snap as snaps
                from charmlibs.interfaces.tls_certificates import TLSCertificatesRequiresV4
                from charmlibs.interfaces import tracing
                from . import relative
                from charmlibs.star import *
            """,
            'lib/charms/old/v0/lib.py': 'import charmlibs.passwd\n',
        },
    )
    assert _charmlibs_mocking.charmlibs_imports([source / 'src', source / 'lib']) == {
        'charmlibs.pathops',
        'charmlibs',
        'charmlibs.apt',
        'charmlibs.snap',
        'charmlibs.interfaces.tls_certificates',
        'charmlibs.interfaces.tls_certificates.TLSCertificatesRequiresV4',
        'charmlibs.interfaces',
        'charmlibs.interfaces.tracing',
        'charmlibs.star',
        'charmlibs.passwd',
    }


def test_a_charm_with_no_charmlibs_imports_needs_nothing(tmp_path: pathlib.Path):
    source = write(tmp_path, {'src/charm.py': 'import ops\n'})
    assert _charmlibs_mocking.find('app', [source / 'src']) == []
