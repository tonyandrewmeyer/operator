# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""The ``charmlibs`` default: each library's own ``mocked()``, from its testing package.

The libraries a charm uses are found from its source: every ``charmlibs``
import in it, matched to the installed library distribution that provides
it. Each library's testing package registers the library's ``mocked`` in
the ``ops.testing.mocking`` entry-point group, under the library's import
package, for example::

    [project.entry-points."ops.testing.mocking"]
    "charmlibs.interfaces.tracing" = "charmlibs.interfaces.tracing_testing:mocked"

A testing package installed at a different version from its library is an
error, since the two are released in lockstep. A library with no testing
package installed runs without its mocking, because most libraries don't
publish one yet. A testing package reaches an isolated charm's environment
the way any test dependency does: through the charm's ``dependency-groups``,
or the ``requirements=`` file.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.metadata
import pathlib
import re
from collections.abc import Callable, Iterable, Mapping
from typing import cast

from .errors import JujuError

#: The entry-point group that testing packages register their library's ``mocked`` in.
ENTRY_POINT_GROUP = 'ops.testing.mocking'


@dataclasses.dataclass(frozen=True)
class LibraryMocking:
    """One library's ``mocked()``, ready to open around a dispatch."""

    package: str
    """The library's import package, such as ``charmlibs.interfaces.tls_certificates``."""

    mocked: Callable[[], contextlib.AbstractContextManager[object]]


def find(app_name: str, sources: Iterable[pathlib.Path]) -> list[LibraryMocking]:
    """The ``mocked()`` of each ``charmlibs`` library imported in ``sources``.

    ``sources`` are files, or directories to search for ``.py`` files. The
    libraries come back sorted by import package, so they are opened in the
    same order on every run.

    A library with no testing package installed is left out.

    Raises:
        JujuError: naming the library and the testing package, if the
            testing package is at a different version from the library, or
            its registered ``mocked`` can't be loaded.
    """
    imported = charmlibs_imports(sources)
    if not imported:
        return []
    libraries = _installed_libraries()
    used: dict[str, importlib.metadata.Distribution] = {}
    for name in imported:
        package = _owning_package(name, libraries)
        if package is not None:
            used[package] = libraries[package]
    registered = {ep.name: ep for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
    found: list[LibraryMocking] = []
    for package in sorted(used):
        library = used[package]
        version = library.version
        entry_point = registered.get(package)
        if entry_point is None:
            continue
        testing = entry_point.dist
        if testing is not None and testing.version != version:
            raise JujuError(
                f'{app_name}: the charm uses {package} {version}, but the installed '
                f'testing package, {testing.metadata["Name"]}, is {testing.version}. '
                f'Install {testing.metadata["Name"]}=={version}.'
            )
        try:
            mocked = entry_point.load()
        except Exception as e:
            raise JujuError(
                f'{app_name}: cannot load the mocking for {package} ({entry_point.value}): {e!r}'
            ) from e
        if not callable(mocked):
            raise JujuError(
                f'{app_name}: the mocking registered for {package} ({entry_point.value}) '
                'is not callable.'
            )
        found.append(
            LibraryMocking(
                package,
                cast('Callable[[], contextlib.AbstractContextManager[object]]', mocked),
            )
        )
    return found


def charmlibs_imports(sources: Iterable[pathlib.Path]) -> set[str]:
    """Every module under ``charmlibs`` imported in the given files and directories.

    ``from charmlibs.interfaces import tls_certificates`` counts as an import
    of ``charmlibs.interfaces.tls_certificates``, since the name imported may
    be a module. Relative imports and imports inside strings aren't seen.
    """
    names: set[str] = set()
    for path in _python_files(sources):
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError, OSError):
            continue  # Importing the charm will report it, if it matters.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names if _is_charmlibs(a.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if not _is_charmlibs(node.module):
                    continue
                names.add(node.module)
                names.update(f'{node.module}.{a.name}' for a in node.names if a.name != '*')
    return names


def _is_charmlibs(name: str) -> bool:
    return name == 'charmlibs' or name.startswith('charmlibs.')


def _python_files(sources: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(sorted(source.rglob('*.py')))
        elif source.suffix == '.py' and source.is_file():
            files.append(source)
    return files


def _installed_libraries() -> dict[str, importlib.metadata.Distribution]:
    """Installed ``charmlibs`` library distributions, keyed by import package.

    Testing packages are left out: they're found through their entry points.
    """
    libraries: dict[str, importlib.metadata.Distribution] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata['Name'] or ''
        normalised = _normalise(name)
        if not normalised.startswith('charmlibs-') or normalised.endswith('-testing'):
            continue
        for package in _import_packages(dist):
            libraries.setdefault(package, dist)
    return libraries


def _import_packages(dist: importlib.metadata.Distribution) -> set[str]:
    """The import packages a ``charmlibs`` distribution provides.

    These are the outermost directories under ``charmlibs/`` that have an
    ``__init__.py`` (``charmlibs`` and ``charmlibs.interfaces`` are namespace
    packages). An editable install lists no package files, so then the
    package is taken from the distribution name, which ``charmlibs`` keeps the
    same as the import path: ``charmlibs-interfaces-tls_certificates`` is
    ``charmlibs.interfaces.tls_certificates``.
    """
    inits = [
        pathlib.PurePosixPath(str(f)).parts[:-1]
        for f in dist.files or ()
        if f.name == '__init__.py' and pathlib.PurePosixPath(str(f)).parts[:1] == ('charmlibs',)
    ]
    packages = {'.'.join(p) for p in inits if p}
    outermost = {p for p in packages if not any(p.startswith(f'{q}.') for q in packages)}
    if outermost:
        return outermost
    return {(dist.metadata['Name'] or '').replace('-', '.')}


def _owning_package(
    name: str, libraries: Mapping[str, importlib.metadata.Distribution]
) -> str | None:
    """The library import package that ``name`` is, or is inside, if any."""
    parts = name.split('.')
    for end in range(len(parts), 0, -1):
        candidate = '.'.join(parts[:end])
        if candidate in libraries:
            return candidate
    return None


def _normalise(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()
