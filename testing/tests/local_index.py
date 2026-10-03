# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""A local package index, so that tests can build isolated environments offline.

``uv`` is pointed at a directory of wheels with ``UV_FIND_LINKS`` and
``UV_NO_INDEX``, instead of PyPI. The directory holds wheels made from the
packages ``ops`` needs, as installed in the test environment, and whatever
small packages a test writes into it.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import pathlib
import re
import zipfile
from collections.abc import Iterable, Mapping

#: A fixed timestamp, so that the same wheel is the same bytes each time.
_DATE_TIME = (2026, 1, 1, 0, 0, 0)

_SKIPPED_METADATA = {'RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json'}


def _normalised(name: str) -> str:
    return re.sub(r'[-_.]+', '_', name).lower()


def _record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
    return f'{path},sha256={digest},{len(data)}'


def _write(
    wheel: pathlib.Path, files: Mapping[str, bytes], dist_info: str, metadata: Mapping[str, bytes]
) -> pathlib.Path:
    record: list[str] = []
    with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED) as zf:
        entries = {**files, **{f'{dist_info}/{name}': data for name, data in metadata.items()}}
        for path, data in sorted(entries.items()):
            zf.writestr(zipfile.ZipInfo(path, _DATE_TIME), data)
            record.append(_record_line(path, data))
        record.append(f'{dist_info}/RECORD,,')
        zf.writestr(zipfile.ZipInfo(f'{dist_info}/RECORD', _DATE_TIME), '\n'.join(record) + '\n')
    return wheel


def write_wheel(
    directory: pathlib.Path,
    name: str,
    version: str,
    files: Mapping[str, str],
    requires: Iterable[str] = (),
) -> pathlib.Path:
    """Write a pure-Python wheel holding ``files`` (path in the wheel to content)."""
    dist_info = f'{_normalised(name)}-{version}.dist-info'
    metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n' + ''.join(
        f'Requires-Dist: {r}\n' for r in requires
    )
    wheel_file = 'Wheel-Version: 1.0\nGenerator: ops-testing-tests\nRoot-Is-Purelib: true\n'
    wheel_file += 'Tag: py3-none-any\n'
    return _write(
        directory / f'{_normalised(name)}-{version}-py3-none-any.whl',
        {path: content.encode() for path, content in files.items()},
        dist_info,
        {'METADATA': metadata.encode(), 'WHEEL': wheel_file.encode()},
    )


def wheel_from_installed(directory: pathlib.Path, name: str) -> pathlib.Path:
    """Pack an installed distribution back into a wheel."""
    dist = importlib.metadata.distribution(name)
    files: dict[str, bytes] = {}
    metadata: dict[str, bytes] = {}
    dist_info = ''
    for path in dist.files or ():
        parts = path.parts
        if parts[0] == '..' or '__pycache__' in parts or path.suffix == '.pyc':
            continue
        if parts[0].endswith('.dist-info'):
            dist_info = parts[0]
            if path.name not in _SKIPPED_METADATA and len(parts) == 2:
                metadata[path.name] = pathlib.Path(str(dist.locate_file(path))).read_bytes()
            continue
        files[path.as_posix()] = pathlib.Path(str(dist.locate_file(path))).read_bytes()
    wheel_meta = metadata['WHEEL'].decode()
    tag = next(
        line.split(':', 1)[1].strip()
        for line in wheel_meta.splitlines()
        if line.startswith('Tag:')
    )
    version = dist.version
    return _write(
        directory / f'{_normalised(dist.metadata["Name"])}-{version}-{tag}.whl',
        files,
        dist_info,
        metadata,
    )


def _requirement_name(requirement: str) -> str:
    match = re.match(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)', requirement)
    assert match is not None
    return match.group(1)


def ops_dependency_wheels(directory: pathlib.Path) -> None:
    """Write wheels for everything ``ops`` and ``ops.testing`` depend on."""
    pending = ['ops', 'ops-scenario']
    seen = {'ops', 'ops_scenario'}
    while pending:
        name = pending.pop()
        for requirement in importlib.metadata.requires(name) or ():
            if 'extra' in requirement.partition(';')[2]:
                continue
            dependency = _requirement_name(requirement)
            if _normalised(dependency) in seen:
                continue
            seen.add(_normalised(dependency))
            try:
                wheel_from_installed(directory, dependency)
            except importlib.metadata.PackageNotFoundError:
                continue  # Not needed on this Python.
            pending.append(dependency)


def uv_environ(index: pathlib.Path, cache: pathlib.Path) -> dict[str, str]:
    """Environment variables that keep ``uv`` to the local index, and the cache in ``cache``."""
    return {
        'UV_NO_INDEX': '1',
        'UV_FIND_LINKS': str(index),
        'UV_OFFLINE': '1',
        'XDG_CACHE_HOME': str(cache),
    }
