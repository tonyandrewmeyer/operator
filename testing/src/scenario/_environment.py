# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Build the virtual environment an isolated charm runs in.

``Juju.deploy(..., isolated=True)`` runs a charm in a worker process, with the
interpreter from a virtual environment built from the charm's own declared
dependencies. This module finds those dependencies from the charm's build
plugin, the same way ``charmcraft pack`` would, and installs them with
``uv pip install`` into a cached environment.

Extraction, by plugin:

* ``charm``: the ``charm-requirements`` files (``requirements.txt`` if the
  part doesn't list any and that file exists), ``charm-python-packages``,
  ``charm-binary-python-packages``, and the ``PYDEPS`` of every charm library
  in ``lib/``.
* ``python``: the ``python-requirements`` files, ``python-packages``,
  ``python-constraints``, and ``PYDEPS``.
* ``uv``: ``uv export`` from the charm's ``uv.lock``, with the part's
  ``uv-extras`` and ``uv-groups``.

Everything is normalised to one flat requirements file. Added to it are the
dependency groups the charm's mocking configuration names, and the
dependencies of ``ops`` and ``ops.testing`` themselves. ``ops`` and
``ops.testing`` are not installed: the worker imports the test process's own
copies, through a directory that holds only those two packages, so the
versions always match and a development install is used as it is.

Environments are cached under ``$XDG_CACHE_HOME/ops-testing/environments``
(``~/.cache/ops-testing/environments`` by default), keyed by a SHA-256 of
everything that goes into one.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import yaml

import ops
import ops.version

from . import _charm_mocking
from .errors import JujuError

#: The build plugins that environments can be built for.
_SUPPORTED_PLUGINS = ('charm', 'python', 'uv')
#: Plugins that are charmcraft's, but that this module can't extract from.
_UNSUPPORTED_PLUGINS = ('poetry', 'reactive')
#: Plugins that don't install Python dependencies, so are skipped when looking
#: for the part that does.
_OTHER_PLUGINS = ('nil', 'dump', 'make', 'autotools', 'cmake', 'go', 'rust', 'npm')

#: The worker gets these from the test process, never from the environment.
_OPS_DISTRIBUTIONS = ('ops', 'ops-scenario')

#: Written into an environment once it is complete.
_MARKER = 'ops-testing-environment.json'

_REQUIREMENT_NAME = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)')
_HASH_OPTION = re.compile(r'\s--hash[=\s]\S+')


@dataclasses.dataclass(frozen=True)
class Environment:
    """A built environment, ready for a worker to run in."""

    #: The environment's interpreter.
    python_executable: str
    #: Directories the worker needs at the front of its ``PYTHONPATH``.
    python_path: tuple[str, ...]
    #: Where the environment is.
    path: pathlib.Path
    #: Whether the environment came from the cache.
    cached: bool


@dataclasses.dataclass(frozen=True)
class _Plan:
    """What goes into an environment: the inputs to its cache key, and the install."""

    plugin: str
    #: The flat requirement list, one requirement or option per entry.
    requirements: tuple[str, ...]
    #: Files whose content goes into the cache key, by name.
    files: Mapping[str, bytes]
    #: A requirements file installed as it is (``requirements=``).
    requirements_file: pathlib.Path | None = None


def build(
    charm_root: pathlib.Path,
    app_name: str,
    requirements: pathlib.Path | None = None,
) -> Environment:
    """Build (or find in the cache) the environment for a charm on disk.

    Args:
        charm_root: The charm's source directory.
        app_name: The application name, for error messages.
        requirements: A requirements file to install as it is, instead of
            finding the charm's dependencies from its build plugin.

    Raises:
        JujuError: if ``uv`` isn't on ``PATH``, the build plugin can't be
            found or isn't supported, the dependencies can't be extracted,
            the charm's ``uv.lock`` is out of date, or a requirement can't be
            resolved.
    """
    if requirements is not None:
        plan = _plan_from_file(requirements, app_name)
        uv = _uv(app_name)
    else:
        uv = _uv(app_name)
        plan = _plan_from_charm(charm_root, app_name, uv)
    key = _cache_key(plan)
    path = _cache_root() / key
    shim = _ops_shim()
    if (path / _MARKER).exists():
        return Environment(_interpreter(path), (str(shim),), path, cached=True)
    _create(path, plan, charm_root, app_name, uv)
    return Environment(_interpreter(path), (str(shim),), path, cached=False)


# Detection and extraction


def _uv(app_name: str) -> str:
    uv = shutil.which('uv')
    if uv is None:
        raise JujuError(
            f'{app_name}: isolated=True builds a virtual environment for the charm with '
            'uv, which is not on PATH. Install uv (https://docs.astral.sh/uv/), or '
            'deploy the charm without isolated=, to run it in the test process.'
        )
    return uv


def _suggest_requirements(message: str) -> JujuError:
    return JujuError(
        f'{message} Pass requirements= with a requirements file for the charm to skip '
        'detecting its dependencies.'
    )


def detect_plugin(charm_root: pathlib.Path, app_name: str) -> tuple[str, Mapping[str, Any]]:
    """Find the charm's build plugin, and the ``charmcraft.yaml`` part that uses it.

    The first part whose plugin installs Python dependencies is the one
    used. With no ``charmcraft.yaml``, or one with no ``parts``, the plugin
    is ``charm`` (charmcraft's default), or ``uv`` if there is a ``uv.lock``
    and no ``charmcraft.yaml``.

    Raises:
        JujuError: if ``charmcraft.yaml`` can't be read, or the plugin isn't
            one that environments can be built for.
    """
    path = charm_root / 'charmcraft.yaml'
    if not path.exists():
        if (charm_root / 'uv.lock').exists():
            return 'uv', {}
        if (charm_root / 'poetry.lock').exists():
            raise _suggest_requirements(
                f'{app_name}: the charm has a poetry.lock, and building environments for '
                'the poetry plugin is not supported yet.'
            )
        return 'charm', {}
    try:
        charmcraft: Any = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise _suggest_requirements(f'{app_name}: could not read {path}: {e}.') from None
    parts: Any = (
        cast('dict[str, Any]', charmcraft).get('parts') if isinstance(charmcraft, dict) else None
    )
    if not parts:
        return 'charm', {}
    if not isinstance(parts, dict):
        raise _suggest_requirements(f'{app_name}: parts in {path} is not a mapping.')
    unknown: list[str] = []
    for name, part in cast('dict[str, Any]', parts).items():
        part = cast('dict[str, Any]', part or {})
        plugin = part.get('plugin', name)
        if plugin in _SUPPORTED_PLUGINS:
            return plugin, part
        if plugin in _UNSUPPORTED_PLUGINS:
            raise _suggest_requirements(
                f'{app_name}: the charm is built with the {plugin} plugin (part {name!r} '
                f'in {path}), and building environments for it is not supported yet.'
            )
        if plugin not in _OTHER_PLUGINS:
            unknown.append(f'{name} ({plugin})')
    found = f' Its parts use: {", ".join(unknown)}.' if unknown else ''
    raise _suggest_requirements(
        f'{app_name}: no part in {path} uses the charm, python or uv plugin, so the '
        f"charm's dependencies can't be found.{found}"
    )


def _plan_from_charm(charm_root: pathlib.Path, app_name: str, uv: str) -> _Plan:
    plugin, part = detect_plugin(charm_root, app_name)
    files: dict[str, bytes] = {}
    for name in ('pyproject.toml', 'uv.lock'):
        if (charm_root / name).exists():
            files[name] = (charm_root / name).read_bytes()
    groups = _mocking_groups(charm_root, app_name)
    if plugin == 'uv':
        requirements = _uv_export(charm_root, part, groups, app_name, uv)
    else:
        requirements = _plugin_requirements(charm_root, plugin, part, app_name)
        requirements += _group_requirements(charm_root, groups, app_name)
    requirements += _ops_requirements()
    return _Plan(plugin, tuple(_deduplicated(requirements)), files)


def _plan_from_file(path: pathlib.Path, app_name: str) -> _Plan:
    if not path.is_file():
        raise JujuError(f'{app_name}: no requirements file at {str(path)!r}.')
    return _Plan(
        'requirements',
        tuple(_ops_requirements()),
        {'requirements': path.read_bytes(), 'cwd': os.getcwd().encode()},
        requirements_file=path.resolve(),
    )


def _plugin_requirements(
    charm_root: pathlib.Path, plugin: str, part: Mapping[str, Any], app_name: str
) -> list[str]:
    """The requirements the ``charm`` or ``python`` plugin would install."""
    requirements: list[str] = []
    if plugin == 'charm':
        files = part.get('charm-requirements')
        if files is None:
            files = ['requirements.txt'] if (charm_root / 'requirements.txt').exists() else []
        packages = [
            *part.get('charm-python-packages', ()),
            *part.get('charm-binary-python-packages', ()),
        ]
        constraints: list[str] = []
    else:
        files = part.get('python-requirements', [])
        packages = list(part.get('python-packages', ()))
        constraints = list(part.get('python-constraints', ()))
    for name in files:
        path = charm_root / name
        if not path.is_file():
            raise _suggest_requirements(
                f'{app_name}: the {plugin} plugin lists {name} as a requirements file, but '
                f'there is no {path}.'
            )
        requirements.extend(read_requirements(path))
    requirements.extend(str(package) for package in packages)
    for name in constraints:
        path = charm_root / name
        if not path.is_file():
            raise _suggest_requirements(
                f'{app_name}: the python plugin lists {name} as a constraints file, but '
                f'there is no {path}.'
            )
        requirements.append(f'-c {path.resolve()}')
    requirements.extend(harvest_pydeps(charm_root))
    return requirements


def _uv_export(
    charm_root: pathlib.Path,
    part: Mapping[str, Any],
    groups: Sequence[str],
    app_name: str,
    uv: str,
) -> list[str]:
    """The requirements ``uv export`` gives from the charm's lockfile."""
    if not (charm_root / 'uv.lock').exists():
        raise _suggest_requirements(
            f'{app_name}: the charm is built with the uv plugin but has no uv.lock. Run '
            '`uv lock` in the charm.'
        )
    cmd = [
        uv,
        'export',
        '--locked',
        '--no-emit-project',
        '--no-hashes',
        '--no-header',
        '--format',
        'requirements-txt',
        '--directory',
        str(charm_root),
    ]
    for extra in part.get('uv-extras', ()):
        cmd += ['--extra', str(extra)]
    for group in [*part.get('uv-groups', ()), *groups]:
        cmd += ['--group', str(group)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=charm_root)
    if result.returncode != 0:
        output = result.stderr.strip()
        if '--locked' in output or 'needs to be updated' in output:
            raise _suggest_requirements(
                f"{app_name}: the charm's uv.lock is out of date with its pyproject.toml. "
                'Run `uv lock` in the charm.'
            )
        raise _suggest_requirements(f'{app_name}: `uv export` failed:\n{output}\n')
    with tempfile.TemporaryDirectory(prefix='ops-testing-export-') as tmp:
        exported = pathlib.Path(tmp) / 'requirements.txt'
        exported.write_text(result.stdout)
        # uv writes path dependencies relative to the project.
        return read_requirements(exported, base=charm_root)


def _mocking_groups(charm_root: pathlib.Path, app_name: str) -> tuple[str, ...]:
    return _charm_mocking._read_mocking_config(charm_root, app_name).dependency_groups


def _group_requirements(
    charm_root: pathlib.Path, groups: Sequence[str], app_name: str
) -> list[str]:
    """The requirements in the mocking's dependency groups, from ``pyproject.toml``."""
    if not groups:
        return []
    pyproject = _charm_mocking._load_toml(charm_root / 'pyproject.toml')
    pairs = _charm_mocking.group_requirements(pyproject, groups, app_name)
    return [requirement for _, requirement in pairs]


def read_requirements(
    path: pathlib.Path,
    *,
    base: pathlib.Path | None = None,
    seen: frozenset[pathlib.Path] = frozenset(),
) -> list[str]:
    """Read a requirements file into a flat list, one requirement or option per entry.

    Included files (``-r``) are read in place. Hashes are dropped, because
    requirements from several files are installed together and ``uv``
    requires either every requirement to have one or none to. Relative paths
    are made absolute against ``base``, which defaults to the file's
    directory, so that the list can be installed from anywhere.
    """
    path = path.resolve()
    base = base or path.parent
    lines = path.read_text().replace('\\\n', ' ').splitlines()
    requirements: list[str] = []
    for line in lines:
        line = re.sub(r'(^|\s)#.*$', '', line).strip()
        line = _HASH_OPTION.sub('', f' {line}').strip()
        if not line:
            continue
        option, _, value = line.partition(' ')
        if '=' in option and option.startswith('--'):
            option, _, value = option.partition('=')
        value = value.strip()
        if option in ('-r', '--requirement'):
            included = (base / value).resolve()
            if included not in seen:
                requirements.extend(read_requirements(included, seen=seen | {path}))
        elif option in ('-c', '--constraint', '-e', '--editable', '-f', '--find-links'):
            requirements.append(f'{option} {_absolute_location(value, base)}')
        elif line.startswith('-'):
            requirements.append(line)
        else:
            requirements.append(_absolute(line, base))
    return requirements


def _absolute_location(location: str, base: pathlib.Path) -> str:
    """Make an option's value absolute if it's a relative path rather than a URL."""
    if ':' in location or pathlib.PurePath(location).is_absolute():
        return location
    return str((base / location).resolve())


def _absolute(requirement: str, base: pathlib.Path) -> str:
    """Make a requirement that is a relative path, or names one, absolute."""
    if requirement.startswith(('./', '../', '.\\', '..\\')) or requirement == '.':
        return str((base / requirement).resolve())
    name, at, location = requirement.partition(' @ ')
    if at and location.startswith(('./', '../')):
        return f'{name} @ {(base / location).resolve().as_uri()}'
    return requirement


def harvest_pydeps(charm_root: pathlib.Path) -> list[str]:
    """The ``PYDEPS`` of every charm library under ``lib/``, without importing them.

    Only a list or tuple of string literals assigned to ``PYDEPS`` at module
    level is read.
    """
    lib = charm_root / 'lib'
    if not lib.is_dir():
        return []
    requirements: list[str] = []
    for path in sorted(lib.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == 'PYDEPS' for t in targets):
                continue
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                requirements.extend(
                    element.value
                    for element in value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                )
    return requirements


def _ops_requirements() -> list[str]:
    """What ``ops`` and ``ops.testing`` need installed, without themselves."""
    requirements: list[str] = []
    for name in _OPS_DISTRIBUTIONS:
        try:
            requires = importlib.metadata.requires(name) or []
        except importlib.metadata.PackageNotFoundError:
            raise JujuError(
                f'isolated=True needs {name} installed in the test environment, to find '
                'what the charm environment needs for it.'
            ) from None
        for requirement in requires:
            if 'extra ==' in requirement.replace('extra==', 'extra =='):
                continue
            if _canonical_name(requirement) in _OPS_DISTRIBUTIONS:
                continue
            requirements.append(requirement)
    return requirements


def _canonical_name(requirement: str) -> str | None:
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        return None
    return re.sub(r'[-_.]+', '-', match.group(1)).lower()


def _deduplicated(requirements: Iterable[str]) -> list[str]:
    """Drop repeats, and the charm's own ``ops`` and ``ops.testing`` requirements.

    The worker always uses the test process's ``ops``, so installing the
    charm's would only be shadowed.
    """
    seen: set[str] = set()
    result: list[str] = []
    for requirement in requirements:
        if requirement in seen:
            continue
        seen.add(requirement)
        if not requirement.startswith('-') and _canonical_name(requirement) in _OPS_DISTRIBUTIONS:
            continue
        result.append(requirement)
    return result


# The cache


def _cache_root() -> pathlib.Path:
    base = os.environ.get('XDG_CACHE_HOME') or str(pathlib.Path.home() / '.cache')
    return pathlib.Path(base) / 'ops-testing' / 'environments'


def _scenario_version() -> str:
    try:
        return importlib.metadata.version('ops-scenario')
    except importlib.metadata.PackageNotFoundError:
        return ''


def _cache_key(plan: _Plan) -> str:
    inputs = {
        'plugin': plan.plugin,
        'requirements': sorted(plan.requirements),
        'files': {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(plan.files.items())
        },
        'python': [sys.implementation.name, sys.version, sys.base_prefix],
        'ops': ops.version.version,
        'ops-scenario': _scenario_version(),
    }
    encoded = json.dumps(inputs, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _interpreter(venv: pathlib.Path) -> str:
    if sys.platform == 'win32':
        return str(venv / 'Scripts' / 'python.exe')
    return str(venv / 'bin' / 'python')


def _create(
    path: pathlib.Path,
    plan: _Plan,
    charm_root: pathlib.Path,
    app_name: str,
    uv: str,
) -> None:
    """Build the environment in a scratch directory, then move it into place.

    Two test processes building the same environment at once both succeed;
    the second to finish throws its copy away.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / f'.{path.name}-{uuid.uuid4().hex}'
    try:
        _run(
            [uv, 'venv', '--quiet', '--python', sys.executable, str(scratch)],
            app_name,
            cwd=path.parent,
        )
        requirements = scratch / 'ops-testing-requirements.txt'
        requirements.write_text(''.join(f'{line}\n' for line in plan.requirements))
        cmd = [
            uv,
            'pip',
            'install',
            '--quiet',
            '--python',
            _interpreter(scratch),
            '-r',
            str(requirements),
        ]
        if plan.requirements_file is not None:
            cmd += ['-r', str(plan.requirements_file)]
        # A requirements= file is installed as it is, so its relative paths
        # are relative to the working directory, as with pip. Extracted
        # requirements are already absolute.
        cwd = pathlib.Path.cwd() if plan.requirements_file is not None else charm_root
        _run(cmd, app_name, cwd=cwd, installing=True)
        (scratch / _MARKER).write_text(
            json.dumps({'plugin': plan.plugin, 'requirements': list(plan.requirements)}, indent=2)
        )
        try:
            scratch.rename(path)
        except OSError:
            if not (path / _MARKER).exists():
                raise
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)


def _run(
    cmd: Sequence[str], app_name: str, *, cwd: pathlib.Path, installing: bool = False
) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode == 0:
        return
    output = (result.stderr or result.stdout).strip()
    if installing and ('No solution found' in output or 'not found' in output):
        raise JujuError(
            f"{app_name}: the charm's requirements can't be resolved, so its isolated "
            f"environment can't be built:\n{output}\n"
        )
    raise JujuError(
        f"{app_name}: building the charm's isolated environment failed "
        f'(`{" ".join(cmd[:3])} ...`):\n{output}\n'
    )


# ops and ops.testing, from the test process


def _ops_shim() -> pathlib.Path:
    """A directory holding only the test process's ``ops`` and ``scenario`` packages.

    The worker puts it at the front of its ``PYTHONPATH``, so it imports the
    same ``ops`` and ``ops.testing`` as the test, whatever the charm's
    environment has installed, without the rest of the test's packages.
    """
    import scenario

    packages = {
        'ops': pathlib.Path(ops.__file__).resolve().parent,
        'scenario': pathlib.Path(scenario.__file__).resolve().parent,
    }
    digest = hashlib.sha256(
        json.dumps({name: str(p) for name, p in packages.items()}, sort_keys=True).encode()
    ).hexdigest()
    shim = _cache_root().parent / 'shims' / digest
    if shim.is_dir():
        return shim
    shim.parent.mkdir(parents=True, exist_ok=True)
    scratch = shim.parent / f'.{digest}-{uuid.uuid4().hex}'
    scratch.mkdir()
    try:
        for name, source in packages.items():
            _link_or_copy(source, scratch / name)
        try:
            scratch.rename(shim)
        except OSError:
            if not shim.is_dir():
                raise
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
    return shim


def _link_or_copy(source: pathlib.Path, target: pathlib.Path) -> None:
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:  # Windows without the symlink privilege.
        shutil.copytree(source, target)
