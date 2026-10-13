# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for the mocking a charm provides for itself, and the defaults around it."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import textwrap
from collections.abc import Generator
from typing import Any

import pytest
from scenario import _charm_mocking

import ops
from ops import testing

VAULT_CHARM = """
    import ops

    import vault


    class VaultCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            self.unit.status = ops.ActiveStatus('sealed' if vault.is_sealed() else 'unsealed')
"""

VAULT_MODULE = """
    def is_sealed():
        raise RuntimeError('the real vault is not here')
"""

VAULT_MOCKING = """
    import contextlib
    from unittest import mock


    @contextlib.contextmanager
    def mocked(*, sealed=False):
        with mock.patch('vault.is_sealed', return_value=sealed):
            yield
"""


def write_charm(
    root: pathlib.Path,
    *,
    name: str = 'vault',
    charm: str = VAULT_CHARM,
    files: dict[str, str] | None = None,
    pyproject: str | None = None,
) -> pathlib.Path:
    """Write a charm's source tree under ``root`` and return its directory."""
    all_files = {
        'metadata.yaml': f'name: {name}\n',
        'src/charm.py': charm,
        'src/vault.py': VAULT_MODULE,
        **(files or {}),
    }
    if pyproject is not None:
        all_files['pyproject.toml'] = pyproject
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


# The charm's own mocking


def test_the_charm_s_own_mocking_is_applied(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(tmp_path / 'vault', files={'tests/unit/mocking.py': VAULT_MOCKING})
    with testing.Juju() as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('unsealed')


def test_mocking_varies_the_charm_s_mocks(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(tmp_path / 'vault', files={'tests/unit/mocking.py': VAULT_MOCKING})
    with testing.Juju() as juju:
        app = juju._deploy(root, mocked={'sealed': True}, **deploy_kwargs(isolated))
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('sealed')


def test_the_path_and_function_come_from_pyproject(tmp_path: pathlib.Path, isolated: bool):
    mocking = VAULT_MOCKING.replace('def mocked(', 'def fake_vault(')
    root = write_charm(
        tmp_path / 'vault',
        files={'testing/fakes.py': mocking},
        pyproject="""
            [tool.ops.testing.mocking]
            path = "testing/fakes.py"
            function = "fake_vault"
        """,
    )
    with testing.Juju() as juju:
        app = juju._deploy(root, mocked={'sealed': True}, **deploy_kwargs(isolated))
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('sealed')


def test_the_mocking_can_be_an_importable_module(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(
        tmp_path / 'vault',
        files={'src/vault_mocks.py': VAULT_MOCKING},
        pyproject="""
            [tool.ops.testing.mocking]
            module = "vault_mocks"
        """,
    )
    with testing.Juju() as juju:
        app = juju._deploy(root, mocked={'sealed': True}, **deploy_kwargs(isolated))
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('sealed')


def test_the_mocking_loads_before_the_charm(tmp_path: pathlib.Path, isolated: bool):
    # A mocking module can replace a whole module before the charm imports it.
    charm = """
        import ops

        import unavailable_client


        class ClientCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on.install, self._on_install)

            def _on_install(self, _):
                self.unit.status = ops.ActiveStatus(unavailable_client.ping())
    """
    mocking = """
        import contextlib
        import sys
        import types

        fake = types.ModuleType('unavailable_client')
        fake.ping = lambda: 'pong'
        sys.modules['unavailable_client'] = fake


        @contextlib.contextmanager
        def mocked():
            yield
    """
    root = write_charm(tmp_path / 'client', charm=charm, files={'tests/unit/mocking.py': mocking})
    with testing.Juju() as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('pong')


COUNTING_CHARM = """
    import ops

    import vault


    class CountingCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_event)
            framework.observe(self.on.start, self._on_event)

        def _on_event(self, _):
            self.unit.status = ops.ActiveStatus(str(vault.count()))
"""

COUNTING_MOCKING = """
    import contextlib
    from unittest import mock

    calls = 0


    def _count():
        global calls
        calls += 1
        return calls


    @contextlib.contextmanager
    def mocked():
        with mock.patch('vault.count', _count, create=True):
            yield
"""


def test_module_level_fake_state_is_per_application(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path / 'counter',
        charm=COUNTING_CHARM,
        files={'tests/unit/mocking.py': COUNTING_MOCKING},
    )
    with testing.Juju() as juju:
        first = juju.deploy(root, app='first')
        second = juju.deploy(root, app='second')
        juju.settle()
        # Each application's copy of the module counted only its own calls:
        # install and start.
        assert first.leader.state.unit_status == testing.ActiveStatus('2')
        assert second.leader.state.unit_status == testing.ActiveStatus('2')


PLAIN_VAULT_CHARM = """
    import ops

    import vault


    class PlainCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            try:
                vault.is_sealed()
            except RuntimeError:
                self.unit.status = ops.ActiveStatus('real vault')
            else:
                self.unit.status = ops.ActiveStatus('mocked vault')
"""


def test_one_charm_s_mocking_does_not_reach_another(tmp_path: pathlib.Path):
    mocked_root = write_charm(tmp_path / 'mocked', files={'tests/unit/mocking.py': VAULT_MOCKING})
    plain_root = write_charm(tmp_path / 'plain', name='plain', charm=PLAIN_VAULT_CHARM)
    with testing.Juju() as juju:
        mocked = juju.deploy(mocked_root, mocked={'sealed': True})
        plain = juju.deploy(plain_root)
        juju.settle()
        # Both charms run in this process and share the vault module, but the
        # patch is only open around the mocked charm's dispatches.
        assert mocked.leader.state.unit_status == testing.ActiveStatus('sealed')
        assert plain.leader.state.unit_status == testing.ActiveStatus('real vault')


# Problems with the mocking are reported as such


def test_mocking_must_be_json(tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'vault', files={'tests/unit/mocking.py': VAULT_MOCKING})
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match=r"'sealed'.*JSON"):
            juju.deploy(root, mocked={'sealed': object()})


def test_a_misspelt_keyword_names_the_charm_and_the_keyword(tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'vault', files={'tests/unit/mocking.py': VAULT_MOCKING})
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match=r"vault: mocked=.*'seeled'"):
            juju.deploy(root, mocked={'seeled': True})


def test_a_misspelt_keyword_is_reported_from_the_worker(tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'vault', files={'tests/unit/mocking.py': VAULT_MOCKING})
    with testing.Juju() as juju:
        juju._deploy(root, mocked={'seeled': True}, python_executable=sys.executable)
        with pytest.raises(testing.errors.IsolationError, match=r"vault: mocked=.*'seeled'") as e:
            juju.settle()
        assert 'Traceback' not in str(e.value)


@pytest.mark.parametrize('mocked', [{'sealed': True}, {}])
def test_mocked_is_rejected_for_an_isolated_charm(tmp_path: pathlib.Path, mocked: dict[str, Any]):
    # In this version mocked= is only for charms in the test process. The
    # worker still handles mocking keywords (see the tests above, which reach
    # it through _deploy), but deploy() never sends any.
    root = write_charm(tmp_path / 'vault', files={'tests/unit/mocking.py': VAULT_MOCKING})
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match=r'mocked=.*test process'):
            juju.deploy(root, isolated=True, mocked=mocked)
        assert not juju._state.apps


def test_mocking_for_a_charm_without_its_own_is_an_error(tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'vault')
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match='no mocking of its own'):
            juju.deploy(root, mocked={'sealed': True})


def test_a_charm_without_its_own_mocking_gets_the_defaults(tmp_path: pathlib.Path):
    charm = VAULT_CHARM.replace(
        "'sealed' if vault.is_sealed() else 'unsealed'",
        'str(subprocess.run(["false"]).returncode)',
    ).replace('import vault', 'import subprocess')
    root = write_charm(tmp_path / 'plain', charm=charm)
    with testing.Juju() as juju:
        app = juju.deploy(root)
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('0')


@pytest.mark.parametrize('isolated', [False, True], ids=['in-process', 'isolated'])
def test_a_configured_mocking_file_must_exist(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(
        tmp_path / 'vault',
        pyproject="""
            [tool.ops.testing.mocking]
            path = "tests/unit/missing.py"
        """,
    )
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match=r'missing.py.*does not exist'):
            juju._deploy(root, **deploy_kwargs(isolated))


def test_a_mocking_file_that_fails_to_import_says_why(tmp_path: pathlib.Path):
    mocking = VAULT_MOCKING.replace(
        '    import contextlib',
        '    from tests.unit.helpers import fake_vault\n    import contextlib',
    )
    root = write_charm(tmp_path / 'vault', files={'tests/unit/mocking.py': mocking})
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match="not from elsewhere in the charm's"):
            juju.deploy(root)


@pytest.mark.parametrize(
    ('table', 'message'),
    [
        ('pth = "x.py"', "unknown key 'pth'"),
        ('disable = ["no-such-default"]', "'no-such-default'.*not one of the default mocks"),
        ('path = "a.py"\nmodule = "b"', 'both path and module'),
    ],
)
def test_the_configuration_is_checked(tmp_path: pathlib.Path, table: str, message: str):
    root = write_charm(tmp_path / 'vault', pyproject=f'[tool.ops.testing.mocking]\n{table}\n')
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match=message):
            juju.deploy(root)


def test_dependency_groups_must_be_installed(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path / 'vault',
        files={'tests/unit/mocking.py': VAULT_MOCKING},
        pyproject="""
            [dependency-groups]
            unit = ["pytest", "definitely-not-an-installed-package>=1"]

            [tool.ops.testing.mocking]
            dependency-groups = ["unit"]
        """,
    )
    with testing.Juju() as juju:
        with pytest.raises(
            testing.errors.JujuError, match=r"definitely-not-an-installed-package.*'unit'"
        ):
            juju.deploy(root)


def test_installed_dependency_groups_are_accepted(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path / 'vault',
        files={'tests/unit/mocking.py': VAULT_MOCKING},
        pyproject="""
            [dependency-groups]
            base = ["pytest"]
            unit = [{include-group = "base"}, "PyYAML"]

            [tool.ops.testing.mocking]
            dependency-groups = ["unit"]
        """,
    )
    with testing.Juju() as juju:
        app = juju.deploy(root)
        juju.settle()
        assert app.leader.state.unit_status == testing.ActiveStatus('unsealed')


# The defaults


PROBE_CHARM = """
    import json
    import os
    import socket
    import subprocess

    import ops


    class ProbeCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.install, self._on_install)

        def _on_install(self, _):
            probe = {}
            result = subprocess.run(['false'], capture_output=True, text=True)
            probe['subprocess'] = [result.returncode, result.stdout]
            probe['system'] = os.system('false')
            probe['fqdn'] = socket.getfqdn()
            probe['hostname'] = socket.gethostname()
            probe['address'] = socket.gethostbyname(socket.gethostname())
            probe['home'] = os.environ.get('HOME')
            probe['unit'] = os.environ.get('JUJU_UNIT_NAME')
            try:
                socket.create_connection(('127.0.0.1', 9))
            except OSError as e:
                probe['network'] = str(e)
            self.unit.status = ops.ActiveStatus(json.dumps(probe))
"""


def probe(app: testing.App, unit: int = 0) -> dict[str, Any]:
    return json.loads(app.units[unit].state.unit_status.message)


def test_the_defaults_keep_the_charm_off_the_host(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(tmp_path / 'probe', name='probe', charm=PROBE_CHARM)
    with testing.Juju(model_name='mymodel') as juju:
        app = juju._deploy(root, num_units=2, **deploy_kwargs(isolated))
        juju.settle()
        result = probe(app, 1)
    assert result['subprocess'] == [0, '']
    assert result['system'] == 0
    assert result['hostname'] == 'probe-1'
    assert result['fqdn'] == 'probe-1.mymodel'
    assert result['address'] == _charm_mocking.unit_address('mymodel', 'probe/1')
    assert result['home'] is None
    assert result['unit'] == 'probe/1'
    assert '127.0.0.1:9' in result['network']


def test_the_defaults_are_only_open_around_dispatches(tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'probe', name='probe', charm=PROBE_CHARM)
    home = os.environ.get('HOME')
    with testing.Juju() as juju:
        juju.deploy(root)
        juju.settle()
    assert subprocess.run([sys.executable, '-c', 'raise SystemExit(1)']).returncode == 1
    assert os.environ.get('HOME') == home
    assert socket.gethostname() != 'probe-0'


def test_a_charm_can_disable_defaults(tmp_path: pathlib.Path, isolated: bool):
    root = write_charm(
        tmp_path / 'probe',
        name='probe',
        charm=PROBE_CHARM,
        pyproject="""
            [tool.ops.testing.mocking]
            disable = ["subprocess", "env"]
        """,
    )
    with testing.Juju() as juju:
        app = juju._deploy(root, **deploy_kwargs(isolated))
        juju.settle()
        result = probe(app)
    assert result['subprocess'] == [1, '']
    assert result['home'] == os.environ.get('HOME')
    # The rest of the defaults still apply.
    assert result['hostname'] == 'probe-0'


def test_a_charm_can_disable_every_default(tmp_path: pathlib.Path):
    root = write_charm(
        tmp_path / 'probe',
        name='probe',
        charm=PROBE_CHARM,
        pyproject="""
            [tool.ops.testing.mocking]
            disable = "all"
        """,
    )
    with testing.Juju() as juju:
        app = juju.deploy(root)
        juju.settle()
        result = probe(app)
    assert result['subprocess'] == [1, '']
    assert result['hostname'] == socket.gethostname()


# A CharmSpec's mocking


class SpecCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_install)

    def _on_install(self, _: ops.EventBase):
        result = subprocess.run(['false'], capture_output=True)  # ruff: ignore[start-process-with-partial-path]
        self.unit.status = ops.ActiveStatus(f'{_SPEC_FLAGS["sealed"]} rc={result.returncode}')


_SPEC_FLAGS: dict[str, Any] = {'sealed': None}


@contextlib.contextmanager
def spec_mocked(*, sealed: bool = False) -> Generator[None]:
    _SPEC_FLAGS['sealed'] = sealed
    try:
        yield
    finally:
        _SPEC_FLAGS['sealed'] = None


def test_a_charm_spec_s_mocking_gets_mocked_as_keyword_arguments():
    spec = testing.CharmSpec(SpecCharm, meta={'name': 'spec'}, mocking=spec_mocked)
    with testing.Juju() as juju:
        sealed = juju.deploy(spec, app='sealed', mocked={'sealed': True})
        unsealed = juju.deploy(spec, app='unsealed')
        juju.settle()
    # The subprocess default is around the charm's own mocking, so 'false'
    # succeeds.
    assert sealed.leader.state.unit_status == testing.ActiveStatus('True rc=0')
    assert unsealed.leader.state.unit_status == testing.ActiveStatus('False rc=0')
    assert _SPEC_FLAGS['sealed'] is None


def test_a_charm_spec_without_mocking_still_gets_the_defaults():
    with testing.Juju() as juju:
        app = juju.deploy(testing.CharmSpec(SpecCharm, meta={'name': 'spec'}))
        juju.settle()
    assert app.leader.state.unit_status == testing.ActiveStatus('None rc=0')


def test_a_charm_spec_s_mocking_rejects_a_misspelt_keyword():
    spec = testing.CharmSpec(SpecCharm, meta={'name': 'spec'}, mocking=spec_mocked)
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match=r"spec: mocked=.*'seeled'"):
            juju.deploy(spec, mocked={'seeled': True})


def test_mocked_needs_a_charm_spec_with_mocking():
    with testing.Juju() as juju:
        with pytest.raises(testing.errors.JujuError, match='no mocking of its own'):
            juju.deploy(testing.CharmSpec(SpecCharm, meta={'name': 'spec'}), mocked={'x': 1})
