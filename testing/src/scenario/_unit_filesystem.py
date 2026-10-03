# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Each unit's own filesystem, at the Python level.

A charm that writes to ``/etc/nginx/ssl`` in a unit test would otherwise
write to the test machine. Inside :func:`translated`, file access through
``open``, ``os`` and ``shutil`` (and so ``pathlib``, which is built on them) is
translated into a per-unit root:

* Writes land under the root: ``/etc/nginx/ssl/x`` becomes
  ``<root>/etc/nginx/ssl/x``. A write creates the directories above it in the
  root, because the packages that would have created them on a real machine
  aren't installed here.
* Reads check the root first, then the host. Removing something that is only
  on the host hides it from the unit, without touching the host.
* Some paths pass through untranslated: the root itself, the Python
  installation, ``/dev``, and the paths the caller allows (the framework's
  own temporary directories, for example).

A chroot would need root, and unprivileged user namespaces are often
restricted, so this works at the Python level only. A C extension that does
its own file I/O isn't translated, and neither is a module that bound
``open`` to a name of its own at import time (``tarfile``, for example). An
operation relative to a directory file descriptor (``dir_fd=``, or a
descriptor from ``os.open`` on a directory) isn't translated either.
"""

from __future__ import annotations

import builtins
import contextlib
import errno
import io
import os
import pathlib
import shutil
import site
import stat
import sys
import tempfile
from collections.abc import Callable, Collection, Generator, Mapping
from typing import Any, AnyStr, cast

import yaml

# The real functions, captured before anything is patched.
_open = builtins.open
_os_open = os.open
_stat = os.stat
_lstat = os.lstat
_listdir = os.listdir
_scandir = os.scandir
_mkdir = os.mkdir
_rmdir = os.rmdir
_unlink = os.unlink
_rename = os.rename
_replace = os.replace
_symlink = os.symlink
_link = os.link
_readlink = os.readlink
_chmod = os.chmod
_chown = os.chown
_lchown = os.lchown
_utime = os.utime
_truncate = os.truncate
_access = os.access
_chdir = os.chdir
_getcwd = os.getcwd
_rmtree = shutil.rmtree
_statvfs = os.statvfs
_mkfifo = os.mkfifo
# Linux only.
_getxattr = getattr(os, 'getxattr', None)
_listxattr = getattr(os, 'listxattr', None)
_setxattr = getattr(os, 'setxattr', None)
_removexattr = getattr(os, 'removexattr', None)

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _python_paths() -> tuple[str, ...]:
    """The Python installation or virtual environment the charm runs in."""
    paths = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    with contextlib.suppress(AttributeError):
        paths.update(site.getsitepackages())
    with contextlib.suppress(AttributeError):
        paths.add(site.getusersitepackages())
    return tuple(sorted(p for p in paths if p))


def _lexists(path: str) -> bool:
    try:
        _lstat(path)
    except (OSError, ValueError):
        return False
    return True


def _isdir(path: str) -> bool:
    try:
        return stat.S_ISDIR(_stat(path).st_mode)
    except (OSError, ValueError):
        return False


def _makedirs(path: str) -> None:
    if _isdir(path):
        return
    _makedirs(os.path.dirname(path))
    with contextlib.suppress(FileExistsError):
        _mkdir(path)


def _fspath(path: Any) -> str | bytes:
    return cast('str | bytes', os.fspath(path))


def _text(path: Any) -> str:
    return os.fsdecode(_fspath(path))


def _not_found(path: object) -> FileNotFoundError:
    return FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


class _Entry:
    """A :class:`os.DirEntry` from the root or the host, with the path the charm asked for."""

    def __init__(self, entry: os.DirEntry[Any], path: Any):
        self._entry = entry
        self.name = entry.name
        self.path = path

    def inode(self) -> int:
        return self._entry.inode()

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        return self._entry.is_dir(follow_symlinks=follow_symlinks)

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        return self._entry.is_file(follow_symlinks=follow_symlinks)

    def is_symlink(self) -> bool:
        return self._entry.is_symlink()

    def is_junction(self) -> bool:
        return False

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        return self._entry.stat(follow_symlinks=follow_symlinks)

    def __fspath__(self) -> Any:
        return self.path

    def __repr__(self) -> str:
        return f'<DirEntry {self.name!r}>'


class _ScandirIterator:
    def __init__(self, entries: list[Any]):
        self._entries = iter(entries)

    def __iter__(self) -> _ScandirIterator:
        return self

    def __next__(self) -> Any:
        return next(self._entries)

    def __enter__(self) -> _ScandirIterator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._entries = iter(())


class UnitFilesystem:
    """One unit's root, and the host paths it hides.

    The same object is used for every dispatch to the unit, so what the unit
    removed from the host stays removed.
    """

    def __init__(self, root: str | os.PathLike[str]):
        self.root = os.path.realpath(root)
        #: Host paths the unit has removed: they, and everything under them,
        #: are no longer visible to the unit unless it writes them again.
        self.hidden: set[str] = set()


class _Translator:
    def __init__(self, filesystem: UnitFilesystem, allow: Collection[str]):
        self._fs = filesystem
        self._root = filesystem.root
        self._hidden = filesystem.hidden
        passthrough = {self._root, '/dev', *_python_paths()}
        passthrough.update(os.path.abspath(p) for p in allow)
        self._passthrough = tuple(sorted(p.rstrip('/') or '/' for p in passthrough))

    # Classifying paths

    def _virtual(self, path: str) -> str:
        if not os.path.isabs(path):
            path = os.path.join(_getcwd(), path)
        path = os.path.normpath(path)
        # normpath keeps a leading '//', which POSIX allows to mean something else.
        return '/' + path.lstrip('/')

    def _passes(self, path: str) -> bool:
        for allowed in self._passthrough:
            if path == allowed or path.startswith(allowed + '/'):
                return True
        return False

    def _is_hidden(self, path: str) -> bool:
        if not self._hidden:
            return False
        current = path
        while True:
            if current in self._hidden:
                return True
            parent: str = os.path.dirname(current)
            if parent == current:
                return False
            current = parent

    def _on_host(self, path: str) -> bool:
        return not self._is_hidden(path) and _lexists(path)

    def _exists(self, path: str) -> bool:
        if self._passes(path):
            return _lexists(path)
        return _lexists(self._root + path) or self._on_host(path)

    def _hide(self, path: str) -> None:
        self._hidden.add(path)

    # Translating

    def _read(self, path: str) -> str:
        if self._passes(path):
            return path
        mapped = self._root + path
        if _lexists(mapped) or self._is_hidden(path):
            return mapped
        return path

    def _write(self, path: str, *, copy_up: bool = False) -> str:
        if self._passes(path):
            return path
        mapped = self._root + path
        _makedirs(os.path.dirname(mapped))
        if copy_up and not _lexists(mapped) and self._on_host(path):
            self._copy_up(path, mapped)
        return mapped

    def _copy_up(self, path: str, mapped: str) -> None:
        """Copy a host file into the root, so a change to it stays in the root."""
        info = _stat(path)
        if stat.S_ISDIR(info.st_mode):
            _mkdir(mapped, stat.S_IMODE(info.st_mode))
            return
        with _open(path, 'rb') as source, _open(mapped, 'wb') as target:
            shutil.copyfileobj(source, target)
        _chmod(mapped, stat.S_IMODE(info.st_mode))

    def _convert(self, path: Any, translate: Callable[[str], str]) -> Any:
        """Translate a path argument, keeping its type; leave descriptors alone."""
        if isinstance(path, int):
            return path
        try:
            value = _fspath(path)
        except TypeError:
            return path  # Let the real function report it.
        if isinstance(value, bytes):
            return os.fsencode(translate(self._virtual(os.fsdecode(value))))
        return translate(self._virtual(value))

    def _exists_arg(self, path: Any) -> bool:
        """Whether a path argument exists, in the root or visibly on the host."""
        return self._exists(self._virtual(_text(path)))

    def read(self, path: Any) -> Any:
        return self._convert(path, self._read)

    def write(self, path: Any, *, copy_up: bool = False) -> Any:
        return self._convert(path, lambda p: self._write(p, copy_up=copy_up))

    def _untranslate(self, value: AnyStr) -> AnyStr:
        root: Any = os.fsencode(self._root) if isinstance(value, bytes) else self._root
        sep: Any = b'/' if isinstance(value, bytes) else '/'
        if value.startswith(root + sep):
            return value[len(root) :]
        return value

    # The replacements

    def open(self, file: Any, mode: str = 'r', *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, int):
            return _open(file, mode, *args, **kwargs)
        if not any(c in mode for c in 'wax+'):
            return _open(self.read(file), mode, *args, **kwargs)
        if 'x' in mode and self._exists_arg(file):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), file)
        copy_up = 'w' not in mode
        return _open(self.write(file, copy_up=copy_up), mode, *args, **kwargs)

    def os_open(
        self, path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if dir_fd is not None:
            return _os_open(path, flags, mode, dir_fd=dir_fd)
        if not flags & _WRITE_FLAGS:
            return _os_open(self.read(path), flags, mode)
        exclusive = flags & os.O_CREAT and flags & os.O_EXCL
        if exclusive and self._exists_arg(path):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
        return _os_open(self.write(path, copy_up=not flags & os.O_TRUNC), flags, mode)

    def stat(self, path: Any, *, dir_fd: int | None = None, follow_symlinks: bool = True) -> Any:
        if dir_fd is not None:
            return _stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        return _stat(self.read(path), follow_symlinks=follow_symlinks)

    def lstat(self, path: Any, *, dir_fd: int | None = None) -> Any:
        if dir_fd is not None:
            return _lstat(path, dir_fd=dir_fd)
        return _lstat(self.read(path))

    def access(self, path: Any, mode: int, **kwargs: Any) -> bool:
        if kwargs.get('dir_fd') is not None:
            return _access(path, mode, **kwargs)
        return _access(self.read(path), mode, **kwargs)

    def readlink(self, path: Any, *, dir_fd: int | None = None) -> Any:
        if dir_fd is not None:
            return _readlink(path, dir_fd=dir_fd)
        return self._untranslate(_readlink(self.read(path)))

    def chdir(self, path: Any) -> None:
        _chdir(self.read(path))

    def _merged(self, path: Any) -> tuple[str | None, str | None, bool] | None:
        """The root and host directories to list for ``path``, and whether it is bytes.

        ``None`` if the path passes through untranslated.
        """
        value = _fspath(path)
        is_bytes = isinstance(value, bytes)
        virtual = self._virtual(os.fsdecode(value))
        if self._passes(virtual):
            return None
        mapped = self._root + virtual
        in_root = mapped if _isdir(mapped) else None
        on_host = virtual if not self._is_hidden(virtual) and _isdir(virtual) else None
        if in_root is None and on_host is None:
            raise _not_found(path)
        return in_root, on_host, is_bytes

    def _host_names(self, directory: str) -> list[str]:
        return [n for n in _listdir(directory) if not self._is_hidden(f'{directory}/{n}')]

    def listdir(self, path: Any = '.') -> list[Any]:
        merged = None if isinstance(path, int) else self._merged(path)
        if merged is None:
            return _listdir(path)
        in_root, on_host, is_bytes = merged
        names: set[str] = set()
        if in_root is not None:
            names.update(_listdir(in_root))
        if on_host is not None:
            names.update(self._host_names(on_host))
        listed = sorted(names)
        return [os.fsencode(n) for n in listed] if is_bytes else listed

    def scandir(self, path: Any = '.') -> Any:
        merged = None if isinstance(path, int) else self._merged(path)
        if merged is None:
            return _scandir(path)
        in_root, on_host, is_bytes = merged
        found: dict[str, os.DirEntry[str]] = {}
        if on_host is not None:
            visible = set(self._host_names(on_host))
            with _scandir(on_host) as scanned:
                found.update((e.name, e) for e in scanned if e.name in visible)
        if in_root is not None:
            with _scandir(in_root) as scanned:
                found.update((e.name, e) for e in scanned)
        given: Any = _fspath(path)
        entries: list[Any] = []
        for name in sorted(found):
            entry_name: Any = os.fsencode(name) if is_bytes else name
            entry = _Entry(found[name], os.path.join(given, entry_name))
            entry.name = entry_name
            entries.append(entry)
        return _ScandirIterator(entries)

    def mkdir(self, path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        if dir_fd is not None:
            return _mkdir(path, mode, dir_fd=dir_fd)
        if self._exists_arg(path):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
        _mkdir(self.write(path), mode)

    def _remove(self, path: Any, remover: Callable[[Any], None], **kwargs: Any) -> None:
        if kwargs.get('dir_fd') is not None:
            return remover(path, **kwargs)
        virtual = self._virtual(_text(path))
        if self._passes(virtual):
            return remover(path)
        removed = False
        mapped = self._root + virtual
        if _lexists(mapped):
            remover(mapped)
            removed = True
        if self._on_host(virtual):
            self._hide(virtual)
            removed = True
        if not removed:
            raise _not_found(path)

    def unlink(self, path: Any, *, dir_fd: int | None = None) -> None:
        self._remove(path, _unlink, dir_fd=dir_fd)

    def rmdir(self, path: Any, *, dir_fd: int | None = None) -> None:
        self._remove(path, _rmdir, dir_fd=dir_fd)

    def rmtree(self, path: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get('dir_fd') is not None:
            return _rmtree(path, *args, **kwargs)
        virtual = self._virtual(_text(path))
        if self._passes(virtual):
            return _rmtree(path, *args, **kwargs)
        mapped = self._root + virtual
        on_host = self._on_host(virtual)
        if _lexists(mapped) or not on_host:
            # Missing everywhere: the real function reports it (or ignores it).
            _rmtree(mapped, *args, **kwargs)
        if on_host:
            self._hide(virtual)

    def _move_source(self, virtual: str) -> tuple[str, bool]:
        """Where to move ``virtual`` from, and whether to hide it on the host afterwards."""
        if self._passes(virtual):
            return virtual, False
        mapped = self._root + virtual
        on_host = self._on_host(virtual)
        if not _lexists(mapped) and on_host:
            _makedirs(os.path.dirname(mapped))
            if _isdir(virtual):
                shutil.copytree(virtual, mapped, symlinks=True)
            else:
                self._copy_up(virtual, mapped)
        return mapped, on_host

    def _move(self, mover: Callable[..., None], src: Any, dst: Any, **kwargs: Any) -> None:
        if kwargs.get('src_dir_fd') is not None or kwargs.get('dst_dir_fd') is not None:
            return mover(src, dst, **kwargs)
        virtual = self._virtual(_text(src))
        source, hide = self._move_source(virtual)
        mover(source, self.write(dst))
        if hide:
            self._hide(virtual)

    def rename(self, src: Any, dst: Any, **kwargs: Any) -> None:
        self._move(_rename, src, dst, **kwargs)

    def replace(self, src: Any, dst: Any, **kwargs: Any) -> None:
        self._move(_replace, src, dst, **kwargs)

    def symlink(
        self, src: Any, dst: Any, target_is_directory: bool = False, **kwargs: Any
    ) -> None:
        if kwargs.get('dir_fd') is not None:
            return _symlink(src, dst, target_is_directory, **kwargs)
        target = src
        if os.path.isabs(_fspath(src)):
            # The link has to point into the root, or a write through it would
            # reach the host.
            target = self._convert(src, lambda p: p if self._passes(p) else self._root + p)
        _symlink(target, self.write(dst), target_is_directory)

    def link(self, src: Any, dst: Any, **kwargs: Any) -> None:
        if kwargs.get('src_dir_fd') is not None or kwargs.get('dst_dir_fd') is not None:
            return _link(src, dst, **kwargs)
        # A hard link to a host file would let writes through it reach the
        # host, so the root gets its own copy to link to.
        _link(self.write(src, copy_up=True), self.write(dst), **kwargs)

    def chmod(self, path: Any, mode: int, **kwargs: Any) -> None:
        if isinstance(path, int) or kwargs.get('dir_fd') is not None:
            return _chmod(path, mode, **kwargs)
        _chmod(self.write(path, copy_up=True), mode, **kwargs)

    def _chown(
        self, real: Callable[..., None], path: Any, uid: int, gid: int, **kwargs: Any
    ) -> None:
        if isinstance(path, int) or kwargs.get('dir_fd') is not None:
            return real(path, uid, gid, **kwargs)
        target = self.write(path, copy_up=True)
        value = os.fsdecode(target)
        if value == self._root or value.startswith(self._root + '/'):
            # The test doesn't run as root, so it can't give files away. The
            # owner isn't part of what a test can assert on, so this succeeds
            # as it would for a charm, which does run as root.
            if not _lexists(value):
                raise _not_found(path)
            return
        real(target, uid, gid, **kwargs)

    def chown(self, path: Any, uid: int, gid: int, **kwargs: Any) -> None:
        self._chown(_chown, path, uid, gid, **kwargs)

    def lchown(self, path: Any, uid: int, gid: int) -> None:
        self._chown(_lchown, path, uid, gid)

    def utime(self, path: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(path, int) or kwargs.get('dir_fd') is not None:
            return _utime(path, *args, **kwargs)
        _utime(self.write(path, copy_up=True), *args, **kwargs)

    def truncate(self, path: Any, length: int) -> None:
        if isinstance(path, int):
            return _truncate(path, length)
        _truncate(self.write(path, copy_up=True), length)

    def statvfs(self, path: Any) -> os.statvfs_result:
        return _statvfs(self.read(path))

    def mkfifo(self, path: Any, mode: int = 0o666, *, dir_fd: int | None = None) -> None:
        if dir_fd is not None:
            return _mkfifo(path, mode, dir_fd=dir_fd)
        _mkfifo(self.write(path), mode)

    # shutil.copystat copies extended attributes, by path, onto the copy.

    def getxattr(self, path: Any, *args: Any, **kwargs: Any) -> bytes:
        assert _getxattr is not None
        return _getxattr(self.read(path), *args, **kwargs)

    def listxattr(self, path: Any = None, *args: Any, **kwargs: Any) -> list[str]:
        assert _listxattr is not None
        return _listxattr(None if path is None else self.read(path), *args, **kwargs)

    def setxattr(self, path: Any, *args: Any, **kwargs: Any) -> None:
        assert _setxattr is not None
        _setxattr(self.write(path, copy_up=True), *args, **kwargs)

    def removexattr(self, path: Any, *args: Any, **kwargs: Any) -> None:
        assert _removexattr is not None
        _removexattr(self.write(path, copy_up=True), *args, **kwargs)


def _replacements(t: _Translator) -> list[tuple[Any, str, Any]]:
    rmtree = cast('Any', t.rmtree)
    patches: list[tuple[Any, str, Any]] = [
        (builtins, 'open', t.open),
        (io, 'open', t.open),
        (os, 'open', t.os_open),
        (os, 'stat', t.stat),
        (os, 'lstat', t.lstat),
        (os, 'access', t.access),
        (os, 'readlink', t.readlink),
        (os, 'chdir', t.chdir),
        (os, 'listdir', t.listdir),
        (os, 'scandir', t.scandir),
        (os, 'mkdir', t.mkdir),
        (os, 'unlink', t.unlink),
        (os, 'remove', t.unlink),
        (os, 'rmdir', t.rmdir),
        (os, 'rename', t.rename),
        (os, 'replace', t.replace),
        (os, 'symlink', t.symlink),
        (os, 'link', t.link),
        (os, 'chmod', t.chmod),
        (os, 'chown', t.chown),
        (os, 'lchown', t.lchown),
        (os, 'utime', t.utime),
        (os, 'truncate', t.truncate),
        (shutil, 'rmtree', rmtree),
        (os, 'statvfs', t.statvfs),
        (os, 'mkfifo', t.mkfifo),
    ]
    if _setxattr is not None:
        patches += [
            (os, 'getxattr', t.getxattr),
            (os, 'listxattr', t.listxattr),
            (os, 'setxattr', t.setxattr),
            (os, 'removexattr', t.removexattr),
        ]
    # Python 3.10's pathlib bound the os functions when it was imported.
    accessor = getattr(pathlib, '_normal_accessor', None)
    if accessor is not None:
        bound = {
            'open': t.open,
            'stat': t.stat,
            'lstat': t.lstat,
            'listdir': t.listdir,
            'scandir': t.scandir,
            'chmod': t.chmod,
            'mkdir': t.mkdir,
            'unlink': t.unlink,
            'link': t.link,
            'rmdir': t.rmdir,
            'rename': t.rename,
            'replace': t.replace,
            'readlink': t.readlink,
        }
        patches.extend((accessor, name, f) for name, f in bound.items() if hasattr(accessor, name))
        if hasattr(accessor, 'symlink'):
            patches.append((accessor, 'symlink', _accessor_symlink(t)))
    return patches


def _accessor_symlink(t: _Translator) -> Callable[..., None]:
    def symlink(a: Any, b: Any, target_is_directory: bool = False) -> None:
        t.symlink(a, b, target_is_directory)

    return symlink


@contextlib.contextmanager
def translated(filesystem: UnitFilesystem, allow: Collection[str] = ()) -> Generator[None]:
    """Translate file access into the unit's root for the duration of the block.

    ``tempfile`` makes its files under ``<root>/tmp``, and the working
    directory is restored on the way out.
    """
    t = _Translator(filesystem, allow)
    temp = os.path.join(filesystem.root, 'tmp')
    _makedirs(temp)
    patches = _replacements(t)
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]
    saved_tempdir = tempfile.tempdir
    cwd = _getcwd()
    for owner, name, replacement in patches:
        setattr(owner, name, replacement)
    tempfile.tempdir = temp
    try:
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)
        tempfile.tempdir = saved_tempdir
        with contextlib.suppress(OSError):
            _chdir(cwd)


def framework_paths(ctx: Any, state: Any) -> list[str]:
    """The paths outside a unit's root that the framework writes to during a dispatch.

    ``Context`` keeps container filesystems and storage under a temporary
    directory of its own, and a mount's source is where the test put it.
    """
    paths: list[str] = [os.fspath(ctx._tmp_path)]
    for container in state.containers:
        paths.extend(str(mount.source) for mount in container.mounts.values())
    return paths


# Seeding the root

_UBUNTU_CODENAMES = {
    '20.04': ('focal', 'Focal Fossa'),
    '22.04': ('jammy', 'Jammy Jellyfish'),
    '24.04': ('noble', 'Noble Numbat'),
    '25.04': ('plucky', 'Plucky Puffin'),
    '25.10': ('questing', 'Questing Quokka'),
    '26.04': ('resolute', 'Resolute Raccoon'),
}

#: The base for a charm that doesn't name one.
DEFAULT_BASE = ('ubuntu', '24.04')


def _as_dict(value: Any) -> dict[str, Any]:
    return cast('dict[str, Any]', value) if isinstance(value, dict) else {}


def charm_base(meta: Mapping[str, Any]) -> tuple[str, str]:
    """The base a charm runs on, from its ``charmcraft.yaml``-shaped metadata.

    ``base: ubuntu@24.04`` is the current form; ``bases:`` (with or without
    ``run-on``) is the older one, and the first entry wins. A charm that
    doesn't say gets :data:`DEFAULT_BASE`.
    """
    base = meta.get('base')
    if isinstance(base, str) and '@' in base:
        name, _, channel = base.partition('@')
        return name, channel
    bases = meta.get('bases')
    if isinstance(bases, list) and bases:
        first = _as_dict(cast('list[Any]', bases)[0])
        run_on = first.get('run-on')
        if isinstance(run_on, list) and run_on:
            first = _as_dict(cast('list[Any]', run_on)[0])
        name, channel = first.get('name'), first.get('channel')
        if name and channel:
            return str(name), str(channel)
    platforms = meta.get('platforms')
    if isinstance(platforms, dict):
        for platform in cast('dict[str, Any]', platforms):
            if '@' in platform:
                name, _, rest = platform.partition('@')
                return name, rest.split(':', 1)[0]
    return DEFAULT_BASE


def os_release(name: str, channel: str) -> str:
    """The ``/etc/os-release`` of a base."""
    if name != 'ubuntu':
        return f'NAME="{name}"\nID={name}\nVERSION_ID="{channel}"\n'
    codename, words = _UBUNTU_CODENAMES.get(channel, ('', ''))
    lts = ' LTS' if channel.endswith('.04') and int(channel.split('.')[0]) % 2 == 0 else ''
    version = f'{channel}{lts} ({words})' if words else f'{channel}{lts}'
    lines = [
        f'PRETTY_NAME="Ubuntu {channel}{lts}"',
        'NAME="Ubuntu"',
        f'VERSION_ID="{channel}"',
        f'VERSION="{version}"',
    ]
    if codename:
        lines.append(f'VERSION_CODENAME={codename}')
    lines += [
        'ID=ubuntu',
        'ID_LIKE=debian',
        'HOME_URL="https://www.ubuntu.com/"',
        'SUPPORT_URL="https://help.ubuntu.com/"',
        'BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"',
        'PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"',
    ]
    if codename:
        lines.append(f'UBUNTU_CODENAME={codename}')
    lines.append('LOGO=ubuntu-logo')
    return '\n'.join(lines) + '\n'


#: Left out of each unit's copy of the charm: version control, caches and
#: build output, none of which a charm ships.
_NOT_COPIED = frozenset({
    '.git',
    '.tox',
    '.venv',
    'venv',
    '.nox',
    'build',
    'node_modules',
    '__pycache__',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    # Context writes these into the charm directory from the metadata it is
    # given, so a copy would only make it warn that it is overwriting them.
    'metadata.yaml',
    'config.yaml',
    'actions.yaml',
})


def _ignored(directory: str, names: list[str]) -> set[str]:
    del directory
    return {n for n in names if n in _NOT_COPIED or n.endswith('.charm')}


def make_unit_root(
    parent: pathlib.Path,
    app_name: str,
    unit_id: int,
    *,
    meta: Mapping[str, Any],
    charm_source: pathlib.Path | None,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Create a unit's root, and its own copy of the charm if it has one on disk.

    The copy goes where Juju puts a unit's charm, under
    ``var/lib/juju/agents/unit-<app>-<id>/charm`` in the root, so it is
    inside the root without being translated.

    Returns:
        The root, and the unit's charm directory (``None`` for a charm with no
        source on disk).
    """
    root = parent / f'{app_name}-{unit_id}'
    (root / 'tmp').mkdir(parents=True)
    etc = root / 'etc'
    etc.mkdir()
    if charm_source is not None and not any(k in meta for k in ('base', 'bases', 'platforms')):
        # A charm with a metadata.yaml has its base in charmcraft.yaml all the same.
        charmcraft = charm_source / 'charmcraft.yaml'
        if charmcraft.exists():
            meta = yaml.safe_load(charmcraft.read_text()) or {}
    (etc / 'os-release').write_text(os_release(*charm_base(meta)))
    if charm_source is None:
        return root, None
    charm_dir = root / 'var' / 'lib' / 'juju' / 'agents' / f'unit-{app_name}-{unit_id}' / 'charm'
    shutil.copytree(charm_source.resolve(), charm_dir, symlinks=True, ignore=_ignored)
    return root, charm_dir
