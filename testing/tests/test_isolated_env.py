# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Building the virtual environment an ``isolated=True`` charm runs in.

The tests that build environments run ``uv`` against a local directory of
wheels (see ``local_index``), so they don't need the network, and are skipped
when ``uv`` isn't on ``PATH``.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import textwrap
import time
from collections.abc import Generator

import pytest
from scenario import _environment

import ops
from ops import testing

from . import local_index

_ISOLATION = pathlib.Path(__file__).parent / 'test_isolation'
_CONFDEP = {
    version: (
        _ISOLATION / 'deps' / f'confdep_v{version[0]}' / 'confdep' / '__init__.py'
    ).read_text()
    for version in ('1.0', '2.0')
}

needs_uv = pytest.mark.skipif(shutil.which('uv') is None, reason='needs uv on PATH')


@pytest.fixture(scope='session')
def index(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp('index')
    local_index.ops_dependency_wheels(directory)
    for version, source in _CONFDEP.items():
        local_index.write_wheel(directory, 'confdep', version, {'confdep/__init__.py': source})
    local_index.write_wheel(
        directory, 'mockhelper', '1.0', {'mockhelper/__init__.py': "VALUE = 'from mockhelper'\n"}
    )
    local_index.write_wheel(
        directory, 'extradep', '1.0', {'extradep/__init__.py': "VALUE = 'from extradep'\n"}
    )
    return directory


@pytest.fixture
def cache(index: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``uv`` at the local index, and the environment cache into ``tmp_path``."""
    directory = tmp_path / 'cache'
    for name, value in local_index.uv_environ(index, directory).items():
        monkeypatch.setenv(name, value)
    return directory


@pytest.fixture
def juju() -> Generator[testing.Juju]:
    with testing.Juju() as juju:
        yield juju


def write_charm(
    root: pathlib.Path, files: dict[str, str], *, source: str | None = None
) -> pathlib.Path:
    """Write a charm: ``src/charm.py`` from ``source`` (or alpha's), and ``files``."""
    if source is None:
        shutil.copytree(_ISOLATION / 'charms' / 'alpha', root)
    else:
        (root / 'src').mkdir(parents=True)
        (root / 'src' / 'charm.py').write_text(textwrap.dedent(source))
        (root / 'metadata.yaml').write_text(f'name: {root.name}\n')
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content))
    return root


def uv_charm(root: pathlib.Path, dependencies: list[str], **pyproject: str) -> pathlib.Path:
    """Give a charm a uv plugin part and a ``pyproject.toml``, and lock it."""
    extra = ''.join(f'\n{text}\n' for text in pyproject.values())
    write_charm(
        root,
        {
            'charmcraft.yaml': 'type: charm\nparts:\n  charm:\n    plugin: uv\n    source: .\n',
            'pyproject.toml': (
                f'[project]\nname = "{root.name}"\nversion = "0"\n'
                f'requires-python = ">=3.10"\ndependencies = {dependencies!r}\n{extra}'
            ).replace("'", '"'),
        },
    )
    import subprocess

    subprocess.run([shutil.which('uv') or 'uv', 'lock', '--quiet'], cwd=root, check=True)
    return root


STATUS_CHARM = """
    import importlib

    import ops

    MODULE, ATTRIBUTE = {target!r}


    class StatusCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.start, self._on_start)

        def _on_start(self, event):
            value = getattr(importlib.import_module(MODULE), ATTRIBUTE)
            self.unit.status = ops.ActiveStatus(str(value))
"""


def status_charm(root: pathlib.Path, module: str, attribute: str, files: dict[str, str]):
    """A charm whose status, after ``start``, is ``module.attribute``."""
    return write_charm(root, files, source=STATUS_CHARM.format(target=(module, attribute)))


# Detecting the build plugin


def _charmcraft(root: pathlib.Path, text: str) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'charmcraft.yaml').write_text(textwrap.dedent(text))
    return root


@pytest.mark.parametrize('plugin', ['charm', 'python', 'uv'])
def test_detects_the_plugin_from_charmcraft_yaml(tmp_path: pathlib.Path, plugin: str):
    root = _charmcraft(
        tmp_path,
        f"""
        type: charm
        parts:
          deps:
            plugin: nil
          charm:
            plugin: {plugin}
            source: .
            {plugin}-extra: x
        """,
    )
    detected, part = _environment.detect_plugin(root, 'app')
    assert detected == plugin
    assert part[f'{plugin}-extra'] == 'x'


def test_a_part_named_charm_with_no_plugin_is_the_charm_plugin(tmp_path: pathlib.Path):
    root = _charmcraft(tmp_path, 'parts:\n  charm:\n    charm-requirements: [reqs.txt]\n')
    assert _environment.detect_plugin(root, 'app') == (
        'charm',
        {'charm-requirements': ['reqs.txt']},
    )


def test_charmcraft_yaml_with_no_parts_is_the_charm_plugin(tmp_path: pathlib.Path):
    root = _charmcraft(tmp_path, 'type: charm\nname: x\n')
    (root / 'uv.lock').write_text('')
    assert _environment.detect_plugin(root, 'app') == ('charm', {})


def test_with_no_charmcraft_yaml_a_uv_lock_means_uv(tmp_path: pathlib.Path):
    assert _environment.detect_plugin(tmp_path, 'app') == ('charm', {})
    (tmp_path / 'uv.lock').write_text('')
    assert _environment.detect_plugin(tmp_path, 'app') == ('uv', {})


@pytest.mark.parametrize('plugin', ['poetry', 'reactive'])
def test_unsupported_plugins_suggest_requirements(tmp_path: pathlib.Path, plugin: str):
    root = _charmcraft(tmp_path, f'parts:\n  charm:\n    plugin: {plugin}\n')
    with pytest.raises(testing.errors.JujuError, match=rf'{plugin} plugin.*requirements='):
        _environment.detect_plugin(root, 'app')


def test_a_poetry_lock_without_charmcraft_yaml_suggests_requirements(tmp_path: pathlib.Path):
    (tmp_path / 'poetry.lock').write_text('')
    with pytest.raises(testing.errors.JujuError, match=r'poetry.*requirements='):
        _environment.detect_plugin(tmp_path, 'app')


def test_no_python_part_suggests_requirements(tmp_path: pathlib.Path):
    root = _charmcraft(
        tmp_path, 'parts:\n  web:\n    plugin: flask-framework\n  files:\n    plugin: dump\n'
    )
    with pytest.raises(testing.errors.JujuError, match=r'web \(flask-framework\).*requirements='):
        _environment.detect_plugin(root, 'app')


def test_an_unreadable_charmcraft_yaml_suggests_requirements(tmp_path: pathlib.Path):
    root = _charmcraft(tmp_path, 'parts: [\n')
    with pytest.raises(testing.errors.JujuError, match=r'(?s)could not read.*requirements='):
        _environment.detect_plugin(root, 'app')


# Extracting requirements


def test_read_requirements_flattens_a_requirements_file(tmp_path: pathlib.Path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'more.txt').write_text('included==1  # A comment.\n-r ../base.txt\n')
    (tmp_path / 'base.txt').write_text('base>=2\n')
    (tmp_path / 'constraints.txt').write_text('')
    (tmp_path / 'requirements.txt').write_text(
        textwrap.dedent("""\
            # Exported.
            pinned==1.0 \\
                --hash=sha256:abc \\
                --hash=sha256:def
            marked==2.0 ; python_version >= "3.8"
            -r sub/more.txt
            -c constraints.txt
            ./local/package
            named @ ./local/named
            named-url @ https://example.com/x.whl
            --extra-index-url https://example.com/simple

            """)
    )
    assert _environment.read_requirements(tmp_path / 'requirements.txt') == [
        'pinned==1.0',
        'marked==2.0 ; python_version >= "3.8"',
        'included==1',
        'base>=2',
        f'-c {tmp_path / "constraints.txt"}',
        str(tmp_path / 'local' / 'package'),
        f'named @ {(tmp_path / "local" / "named").as_uri()}',
        'named-url @ https://example.com/x.whl',
        '--extra-index-url https://example.com/simple',
    ]


def test_pydeps_are_harvested_from_lib_without_importing(tmp_path: pathlib.Path):
    lib = tmp_path / 'lib' / 'charms'
    (lib / 'one' / 'v0').mkdir(parents=True)
    (lib / 'two' / 'v1').mkdir(parents=True)
    (lib / 'one' / 'v0' / 'a.py').write_text(
        'raise RuntimeError("never imported")\nPYDEPS = ["first>=1", "second"]\n'
    )
    (lib / 'two' / 'v1' / 'b.py').write_text('PYDEPS: list[str] = ("third==3",)\n')
    (lib / 'two' / 'v1' / 'c.py').write_text('PYDEPS = [f"{x}" for x in ()]\nOTHER = ["no"]\n')
    (lib / 'two' / 'v1' / 'd.py').write_text('def f(:\n')
    assert _environment.harvest_pydeps(tmp_path) == ['first>=1', 'second', 'third==3']
    assert _environment.harvest_pydeps(tmp_path / 'nothing') == []


def _plan(root: pathlib.Path) -> _environment._Plan:
    return _environment._plan_from_charm(root, 'app', 'uv-is-not-used')


def _without_ops(plan: _environment._Plan) -> list[str]:
    return [r for r in plan.requirements if r not in _environment._ops_requirements()]


def test_charm_plugin_requirements(tmp_path: pathlib.Path):
    root = _charmcraft(
        tmp_path,
        """
        parts:
          charm:
            charm-python-packages: [setuptools]
            charm-binary-python-packages: [cryptography==42.0]
        """,
    )
    (root / 'requirements.txt').write_text('ops==2.17.0\nops-scenario==7.0\nrequests==2.32\n')
    (root / 'lib').mkdir()
    (root / 'lib' / 'lib.py').write_text('PYDEPS = ["pydantic>=2"]\n')
    plan = _plan(root)
    assert plan.plugin == 'charm'
    # The charm's own ops and ops.testing are left out: the test's are used.
    assert _without_ops(plan) == [
        'requests==2.32',
        'setuptools',
        'cryptography==42.0',
        'pydantic>=2',
    ]
    assert plan.requirements[-len(_environment._ops_requirements()) :] == tuple(
        _environment._ops_requirements()
    )


def test_charm_plugin_requirements_files_from_charmcraft_yaml(tmp_path: pathlib.Path):
    root = _charmcraft(tmp_path, 'parts:\n  charm:\n    charm-requirements: [a.txt, b.txt]\n')
    (root / 'requirements.txt').write_text('ignored\n')
    (root / 'a.txt').write_text('a\n')
    (root / 'b.txt').write_text('b\n')
    assert _without_ops(_plan(root)) == ['a', 'b']


def test_a_missing_requirements_file_suggests_requirements(tmp_path: pathlib.Path):
    root = _charmcraft(tmp_path, 'parts:\n  charm:\n    charm-requirements: [missing.txt]\n')
    with pytest.raises(testing.errors.JujuError, match=r'missing.txt.*requirements='):
        _plan(root)


def test_python_plugin_requirements(tmp_path: pathlib.Path):
    root = _charmcraft(
        tmp_path,
        """
        parts:
          charm:
            plugin: python
            source: .
            python-requirements: [requirements.txt]
            python-packages: [extra==1]
            python-constraints: [constraints.txt]
        """,
    )
    (root / 'requirements.txt').write_text('requests==2.32\n')
    (root / 'constraints.txt').write_text('urllib3<3\n')
    (root / 'lib').mkdir()
    (root / 'lib' / 'lib.py').write_text('PYDEPS = ["pydantic>=2"]\n')
    plan = _plan(root)
    assert plan.plugin == 'python'
    assert _without_ops(plan) == [
        'requests==2.32',
        'extra==1',
        f'-c {root / "constraints.txt"}',
        'pydantic>=2',
    ]


def test_mocking_dependency_groups_are_added(tmp_path: pathlib.Path):
    root = _charmcraft(tmp_path, 'type: charm\n')
    (root / 'pyproject.toml').write_text(
        textwrap.dedent("""
            [dependency-groups]
            base = ["pytest-mock"]
            unit = ["mockhelper==1.0", {include-group = "base"}]
            lint = ["ruff"]

            [tool.ops.testing.mocking]
            dependency-groups = ["unit"]
        """)
    )
    plan = _plan(root)
    assert _without_ops(plan) == ['mockhelper==1.0', 'pytest-mock']
    assert 'pyproject.toml' in plan.files


def test_an_undeclared_mocking_dependency_group_is_an_error(tmp_path: pathlib.Path):
    root = _charmcraft(tmp_path, 'type: charm\n')
    (root / 'pyproject.toml').write_text(
        '[tool.ops.testing.mocking]\ndependency-groups = ["unit"]\n'
    )
    with pytest.raises(testing.errors.JujuError, match=r"'unit'.*does not declare"):
        _plan(root)


def test_ops_requirements_are_what_ops_needs():
    requirements = _environment._ops_requirements()
    names = {_environment._canonical_name(r) for r in requirements}
    assert 'pyyaml' in names
    assert 'ops' not in names
    assert 'ops-scenario' not in names
    assert not any('extra ==' in r for r in requirements)


# The cache key


def test_the_cache_key_changes_with_each_input(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    plan = _environment._Plan('charm', ('a==1', 'b==2'), {'pyproject.toml': b'x'})
    key = _environment._cache_key(plan)
    assert key == _environment._cache_key(
        _environment._Plan('charm', ('b==2', 'a==1'), {'pyproject.toml': b'x'})
    )
    others = [
        _environment._Plan('python', ('a==1', 'b==2'), {'pyproject.toml': b'x'}),
        _environment._Plan('charm', ('a==1', 'b==3'), {'pyproject.toml': b'x'}),
        _environment._Plan('charm', ('a==1', 'b==2'), {'pyproject.toml': b'y'}),
        _environment._Plan('charm', ('a==1', 'b==2'), {'pyproject.toml': b'x', 'uv.lock': b''}),
    ]
    assert len({key, *(_environment._cache_key(p) for p in others)}) == 5
    monkeypatch.setattr(ops.version, 'version', '0.0.1')
    assert _environment._cache_key(plan) != key
    monkeypatch.undo()
    monkeypatch.setattr(sys, 'version', sys.version + ' (another build)')
    assert _environment._cache_key(plan) != key


def test_the_cache_is_under_xdg_cache_home(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    assert _environment._cache_root() == tmp_path / 'ops-testing' / 'environments'
    monkeypatch.delenv('XDG_CACHE_HOME')
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    assert _environment._cache_root() == (
        tmp_path / 'home' / '.cache' / 'ops-testing' / 'environments'
    )


# Errors from deploy()


def test_isolated_without_uv_is_an_error(
    juju: testing.Juju, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    root = write_charm(tmp_path / 'alpha', {})
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    with pytest.raises(testing.errors.JujuError, match='uv, which is not on PATH'):
        juju.deploy(root, isolated=True)
    assert not juju._state.apps


def test_a_missing_requirements_file_is_an_error(juju: testing.Juju, tmp_path: pathlib.Path):
    root = write_charm(tmp_path / 'alpha', {})
    with pytest.raises(testing.errors.JujuError, match='no requirements file'):
        juju.deploy(root, isolated=True, requirements=tmp_path / 'missing.txt')


def test_a_poetry_charm_suggests_requirements(juju: testing.Juju, tmp_path: pathlib.Path):
    if shutil.which('uv') is None:
        pytest.skip('needs uv on PATH')
    root = write_charm(
        tmp_path / 'alpha', {'charmcraft.yaml': 'parts:\n  c:\n    plugin: poetry\n'}
    )
    with pytest.raises(testing.errors.JujuError, match=r'poetry plugin.*requirements='):
        juju.deploy(root, isolated=True)


# Building environments


@needs_uv
def test_conflicting_dependencies_settle_together(
    juju: testing.Juju,
    tmp_path: pathlib.Path,
    cache: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The acceptance test: one charm isolated, one in-process, needing different confdeps."""
    # The test process has confdep 2.0, which beta needs.
    monkeypatch.delitem(sys.modules, 'confdep', raising=False)
    monkeypatch.setattr(sys, 'path', [str(_ISOLATION / 'deps' / 'confdep_v2'), *sys.path])
    # alpha needs confdep 1.0, and says so in its requirements.txt.
    alpha_root = write_charm(tmp_path / 'alpha', {'requirements.txt': 'confdep==1.0\n'})
    alpha = juju.deploy(alpha_root, isolated=True, num_units=2)
    beta = juju.deploy(_ISOLATION / 'charms' / 'beta')
    juju.settle()
    for unit in alpha.units:
        assert unit.state.unit_status == testing.ActiveStatus(
            'confdep=1.0 legacy=alpha-only-name compute=1'
        )
    assert beta.leader.state.unit_status == testing.ActiveStatus(
        'confdep=2.0 new=beta-only-name compute=2'
    )
    import confdep  # type: ignore

    assert confdep.VERSION == '2.0'  # pyright: ignore[reportUnknownMemberType]


@needs_uv
def test_two_isolated_charms_with_conflicting_dependencies(
    juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path
):
    alpha_root = write_charm(tmp_path / 'alpha', {'requirements.txt': 'confdep==1.0\n'})
    beta_root = uv_charm(tmp_path / 'beta', ['confdep==2.0'])
    shutil.copy(
        _ISOLATION / 'charms' / 'beta' / 'src' / 'charm.py', beta_root / 'src' / 'charm.py'
    )
    alpha = juju.deploy(alpha_root, isolated=True)
    beta = juju.deploy(beta_root, 'beta', isolated=True)
    juju.settle()
    assert alpha.leader.state.unit_status.message.startswith('confdep=1.0')
    assert beta.leader.state.unit_status.message.startswith('confdep=2.0')


@needs_uv
def test_a_second_deploy_uses_the_cached_environment(
    tmp_path: pathlib.Path, cache: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    root = write_charm(tmp_path / 'alpha', {'requirements.txt': 'confdep==1.0\n'})
    started = time.monotonic()
    cold = _environment.build(root, 'alpha')
    cold_seconds = time.monotonic() - started
    assert not cold.cached
    assert cold.path.parent == cache / 'ops-testing' / 'environments'
    assert (cold.path / _environment._MARKER).exists()

    def no_install(*args: object, **kwargs: object):
        raise AssertionError('a cache hit should not build anything')

    with monkeypatch.context() as patch:
        patch.setattr(_environment, '_create', no_install)
        started = time.monotonic()
        warm = _environment.build(root, 'alpha')
        warm_seconds = time.monotonic() - started
        assert warm.cached
        assert warm.path == cold.path
        assert warm_seconds < cold_seconds
        # The charm still runs from the cached environment.
        with testing.Juju() as juju:
            app = juju.deploy(root, isolated=True)
            juju.settle()
            assert app.leader.state.unit_status.message.startswith('confdep=1.0')

    # A change to the requirements is a new environment.
    (root / 'requirements.txt').write_text('confdep==2.0\n')
    other = _environment.build(root, 'alpha')
    assert not other.cached
    assert other.path != cold.path


@needs_uv
def test_the_worker_uses_the_tests_own_ops(
    juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path
):
    # The local index has no ops, so this only builds because the charm's own
    # ops requirement is left out.
    root = status_charm(
        tmp_path / 'opscheck', 'ops', '__file__', {'requirements.txt': 'ops==2.17.0\n'}
    )
    app = juju.deploy(root, isolated=True)
    juju.settle()
    worker_ops = pathlib.Path(app.leader.state.unit_status.message)
    assert worker_ops.resolve() == pathlib.Path(ops.__file__).resolve()
    environment = _environment.build(root, 'opscheck')
    assert environment.cached
    assert not list(environment.path.rglob('ops/__init__.py'))


@needs_uv
def test_pydeps_are_installed(juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path):
    root = status_charm(
        tmp_path / 'pydeps',
        'confdep',
        'VERSION',
        {'lib/charms/thing/v0/thing.py': 'PYDEPS = ["confdep==1.0"]\n'},
    )
    app = juju.deploy(root, isolated=True)
    juju.settle()
    assert app.leader.state.unit_status == testing.ActiveStatus('1.0')


@needs_uv
def test_python_plugin_charm(juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path):
    root = status_charm(
        tmp_path / 'pyplugin',
        'confdep',
        'VERSION',
        {
            'charmcraft.yaml': """
                parts:
                  charm:
                    plugin: python
                    source: .
                    python-requirements: [requirements.txt]
                    python-packages: [extradep==1.0]
            """,
            'requirements.txt': 'confdep==2.0\n',
        },
    )
    app = juju.deploy(root, isolated=True)
    juju.settle()
    assert app.leader.state.unit_status == testing.ActiveStatus('2.0')
    environment = _environment.build(root, 'pyplugin')
    assert list(environment.path.rglob('extradep/__init__.py'))


@needs_uv
def test_uv_plugin_forwards_extras_and_groups(
    juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path
):
    root = uv_charm(
        tmp_path / 'uvplugin',
        ['confdep==1.0'],
        extras='[project.optional-dependencies]\nx = ["extradep==1.0"]',
        groups='[dependency-groups]\ng = ["mockhelper==1.0"]',
    )
    shutil.copytree(
        status_charm(tmp_path / 'source', 'extradep', 'VALUE', {}) / 'src',
        root / 'src',
        dirs_exist_ok=True,
    )
    (root / 'charmcraft.yaml').write_text(
        'parts:\n  charm:\n    plugin: uv\n    uv-extras: [x]\n    uv-groups: [g]\n'
    )
    app = juju.deploy(root, isolated=True)
    juju.settle()
    assert app.leader.state.unit_status == testing.ActiveStatus('from extradep')
    environment = _environment.build(root, 'uvplugin')
    assert environment.cached
    assert list(environment.path.rglob('mockhelper/__init__.py'))
    assert list(environment.path.rglob('confdep/__init__.py'))


@needs_uv
def test_mocking_dependency_groups_are_installed(
    juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path
):
    """The charm's mocking imports a package that only its dependency group has."""
    root = status_charm(
        tmp_path / 'mocked',
        'mockhelper',
        'VALUE',
        {
            'pyproject.toml': """
                [dependency-groups]
                unit = ["mockhelper==1.0"]

                [tool.ops.testing.mocking]
                dependency-groups = ["unit"]
            """,
            'tests/unit/mocking.py': """
                import contextlib

                import mockhelper


                @contextlib.contextmanager
                def mocked():
                    assert mockhelper.VALUE
                    yield
            """,
        },
    )
    app = juju.deploy(root, isolated=True)
    juju.settle()
    assert app.leader.state.unit_status == testing.ActiveStatus('from mockhelper')


@needs_uv
def test_uv_plugin_with_a_stale_lockfile(
    juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path
):
    root = uv_charm(tmp_path / 'stale', ['confdep==1.0'])
    pyproject = root / 'pyproject.toml'
    pyproject.write_text(pyproject.read_text().replace('confdep==1.0', 'confdep==2.0'))
    with pytest.raises(testing.errors.JujuError, match=r'uv.lock is out of date.*uv lock'):
        juju.deploy(root, isolated=True)
    assert not juju._state.apps


@needs_uv
def test_uv_plugin_without_a_lockfile(
    juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path
):
    root = uv_charm(tmp_path / 'unlocked', ['confdep==1.0'])
    (root / 'uv.lock').unlink()
    with pytest.raises(testing.errors.JujuError, match=r'no uv.lock.*requirements='):
        juju.deploy(root, isolated=True)


@needs_uv
def test_an_unresolvable_requirement(
    juju: testing.Juju, tmp_path: pathlib.Path, cache: pathlib.Path
):
    root = write_charm(tmp_path / 'alpha', {'requirements.txt': 'confdep==9.9\n'})
    with pytest.raises(testing.errors.JujuError, match="requirements can't be resolved"):
        juju.deploy(root, isolated=True)
    assert not list((cache / 'ops-testing' / 'environments').iterdir())


@needs_uv
def test_requirements_skips_detection(
    juju: testing.Juju,
    tmp_path: pathlib.Path,
    cache: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # A poetry charm can't be detected, but a requirements file works.
    write_charm(
        tmp_path / 'alpha',
        {
            'charmcraft.yaml': 'parts:\n  charm:\n    plugin: poetry\n',
            'requirements.txt': 'confdep==2.0\n',
        },
    )
    (tmp_path / 'test-requirements.txt').write_text('confdep==1.0\n')
    # A relative path is relative to the working directory, as charm= is.
    monkeypatch.chdir(tmp_path)
    app = juju.deploy('alpha', isolated=True, requirements='test-requirements.txt')
    juju.settle()
    assert app.leader.state.unit_status.message.startswith('confdep=1.0')


@needs_uv
def test_the_ops_shim_holds_only_ops_and_scenario(cache: pathlib.Path):
    shim = _environment._ops_shim()
    assert sorted(p.name for p in shim.iterdir()) == ['ops', 'scenario']
    assert (shim / 'ops').resolve() == pathlib.Path(ops.__file__).resolve().parent
    assert _environment._ops_shim() == shim
