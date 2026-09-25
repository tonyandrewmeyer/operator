# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""The mocking each charm provides for itself, and the defaults around it.

A charm configures its mocking in its ``pyproject.toml``::

    [tool.ops.testing.mocking]
    path = "tests/unit/mocking.py"  # The default.
    function = "mocked"             # The default.
    dependency-groups = ["unit"]    # Default: none.
    disable = ["network", "clock"]  # Default: none.

``function`` names a callable that takes keyword arguments and returns a
context manager. :class:`~ops.testing.Juju` calls it with the ``mocking=``
dict given to ``deploy()``, and opens the result around each of that charm's
dispatches, inside a set of defaults that keep a charm from reaching the host.

The same code runs in the test process, for a charm that runs there, and in
the isolated worker, for a charm that runs in its own interpreter. Only the
``mocking=`` dict crosses the process boundary.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Any, cast
from unittest import mock

from .errors import JujuError

#: The default mocks, by the name a charm uses to switch one off.
_DEFAULT_NAMES = (
    'charmlibs',
    'lightkube',
    'k8s-patch-libs',
    'subprocess',
    'snap',
    'clock',
    'hostname',
    'env',
    'network',
    'filesystem',
)

_CONFIG_KEYS = frozenset({'path', 'module', 'function', 'dependency-groups', 'disable'})
_DEFAULT_PATH = 'tests/unit/mocking.py'
_DEFAULT_FUNCTION = 'mocked'

#: The address every unit's name resolves to, matching the default
#: ``ingress-address`` in a :class:`~ops.testing.Network`.
_DEFAULT_ADDRESS = '192.0.2.0'

#: ``PATH`` as Juju sets it for a hook.
_JUJU_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


@dataclasses.dataclass(frozen=True)
class _MockingConfig:
    """A charm's ``[tool.ops.testing.mocking]`` table, with defaults filled in."""

    path: str | None = _DEFAULT_PATH
    module: str | None = None
    function: str = _DEFAULT_FUNCTION
    dependency_groups: tuple[str, ...] = ()
    disable: frozenset[str] = frozenset()
    #: Whether ``path`` was given explicitly, in which case a missing file is
    #: an error rather than a charm that has no mocking of its own.
    explicit_path: bool = False


def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    try:
        toml = importlib.import_module('tomllib')
    except ImportError:  # Python 3.10.
        try:
            toml = importlib.import_module('tomli')
        except ImportError:
            raise JujuError(
                f'Reading the mocking configuration in {path} on Python 3.10 needs '
                'the tomli package installed.'
            ) from None
    with path.open('rb') as f:
        return cast('dict[str, Any]', toml.load(f))


def _read_pyproject(charm_root: pathlib.Path) -> dict[str, Any] | None:
    path = charm_root / 'pyproject.toml'
    if not path.exists():
        return None
    if sys.version_info < (3, 11) and 'tool.ops' not in path.read_text():
        # Nothing to configure, so there is no need to require a TOML parser.
        return None
    return _load_toml(path)


def _read_mocking_config(charm_root: pathlib.Path | None, charm_name: str) -> _MockingConfig:
    """Read a charm's mocking configuration from its ``pyproject.toml``.

    Raises:
        JujuError: if the table has an unknown key, or ``disable`` names a
            default that doesn't exist.
    """
    pyproject = _read_pyproject(charm_root) if charm_root is not None else None
    table: dict[str, Any] = (
        (pyproject or {}).get('tool', {}).get('ops', {}).get('testing', {}).get('mocking')
    ) or {}
    unknown = sorted(set(table) - _CONFIG_KEYS)
    if unknown:
        raise JujuError(
            f'{charm_name}: unknown key {unknown[0]!r} in [tool.ops.testing.mocking]. '
            f'The keys are {", ".join(sorted(_CONFIG_KEYS))}.'
        )
    if 'path' in table and 'module' in table:
        raise JujuError(
            f'{charm_name}: [tool.ops.testing.mocking] sets both path and module; set one.'
        )
    disable = table.get('disable', [])
    if disable == 'all':
        disabled = frozenset(_DEFAULT_NAMES)
    else:
        if isinstance(disable, str):
            disable = [disable]
        for name in disable:
            if name not in _DEFAULT_NAMES:
                raise JujuError(
                    f'{charm_name}: [tool.ops.testing.mocking] disables {name!r}, which '
                    f'is not one of the default mocks: {", ".join(_DEFAULT_NAMES)}, '
                    'or "all".'
                )
        disabled = frozenset(disable)
    module = table.get('module')
    return _MockingConfig(
        path=None if module is not None else table.get('path', _DEFAULT_PATH),
        module=module,
        function=table.get('function', _DEFAULT_FUNCTION),
        dependency_groups=tuple(table.get('dependency-groups', ())),
        disable=disabled,
        explicit_path='path' in table,
    )


def check_mocking_json(mocking: Mapping[str, Any]) -> None:
    """Check that the ``mocking=`` dict can cross a process boundary as JSON.

    Raises:
        JujuError: naming the first key whose value can't be encoded.
    """
    for key, value in mocking.items():
        if not isinstance(key, str):
            raise JujuError(f'mocking= keys are keyword argument names, so strings: {key!r}.')
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            raise JujuError(
                f'mocking={{{key!r}: ...}} is not JSON-serialisable '
                f'({type(value).__name__}). The mocking dict is sent to the charm as '
                'data, so it can hold only JSON values.'
            ) from None


_REQUIREMENT_NAME = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)')


def _check_dependency_groups(
    charm_root: pathlib.Path, groups: Sequence[str], charm_name: str
) -> None:
    """Check that every package in the charm's mocking dependency groups is installed.

    Raises:
        JujuError: naming the group and the first missing package.
    """
    if not groups:
        return
    pyproject = _read_pyproject(charm_root) or {}
    declared: dict[str, list[Any]] = pyproject.get('dependency-groups', {})

    def requirements(group: str, seen: frozenset[str]) -> Iterator[str]:
        if group not in declared:
            raise JujuError(
                f'{charm_name}: [tool.ops.testing.mocking] names the dependency group '
                f'{group!r}, which its pyproject.toml does not declare.'
            )
        for entry in declared[group]:
            if isinstance(entry, dict):
                included = cast('dict[str, str]', entry).get('include-group')
                if included is not None and included not in seen:
                    yield from requirements(included, seen | {included})
                continue
            yield str(entry)

    for group in groups:
        for requirement in requirements(group, frozenset({group})):
            match = _REQUIREMENT_NAME.match(requirement)
            if match is None:
                continue
            name = match.group(1)
            try:
                importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                raise JujuError(
                    f'{charm_name}: the mocking needs {name} (from the {group!r} '
                    'dependency group), which is not installed in the environment the '
                    'charm runs in. Install it there.'
                ) from None


def _add_charm_paths(charm_root: pathlib.Path) -> None:
    """Make the charm's own source and bundled libraries importable."""
    for entry in (charm_root / 'src', charm_root / 'lib'):
        if entry.exists() and str(entry) not in sys.path:
            sys.path.insert(0, str(entry))


class CharmMocking:
    """One application's mocking: the charm's own, inside the defaults.

    Each application gets its own instance, and so its own copy of the
    charm's mocking module, imported under ``module_name``. Fake state that
    the module keeps at module level persists across that application's
    dispatches, but two applications of the same charm don't share it.
    """

    def __init__(
        self,
        charm_root: pathlib.Path | None,
        *,
        app_name: str,
        mocking: Mapping[str, Any],
        module_name: str,
    ):
        self._charm_root = charm_root
        self._app_name = app_name
        self._mocking = dict(mocking)
        self._module_name = module_name
        self._config = _read_mocking_config(charm_root, app_name)
        self._function: Callable[..., contextlib.AbstractContextManager[Any]] | None = None
        self._loaded = False
        self._clock = _FakeClock()

    def check(self) -> None:
        """The checks that need no import: an explicitly configured file exists.

        Raises:
            JujuError: naming the charm and the file.
        """
        config = self._config
        if config.explicit_path and self._charm_root is not None and config.path is not None:
            path = self._charm_root / config.path
            if not path.exists():
                raise JujuError(
                    f'{self._app_name}: the mocking file {path} configured in '
                    'pyproject.toml does not exist.'
                )

    def load(self) -> None:
        """Import the charm's mocking module, before the charm itself is imported.

        Raises:
            JujuError: if a configured module can't be found or imported, a
                dependency group isn't installed, the function is missing, or
                ``mocking=`` has a keyword the function doesn't take.
        """
        if self._loaded:
            return
        self._loaded = True
        self.check()
        config = self._config
        root = self._charm_root
        if root is None:
            return
        _check_dependency_groups(root, config.dependency_groups, self._app_name)
        _add_charm_paths(root)
        with self._defaults(unit_name=None, model_name=None):
            module = self._import_module(root)
        if module is None:
            if self._mocking:
                raise JujuError(
                    f'{self._app_name} was deployed with mocking={self._mocking!r}, '
                    'but the charm has no mocking of its own to pass it to.'
                )
            return
        function = getattr(module, config.function, None)
        if function is None or not callable(function):
            raise JujuError(
                f'{self._app_name}: the mocking module {module.__name__} has no '
                f'function {config.function!r}.'
            )
        try:
            inspect.signature(function).bind(**self._mocking)
        except TypeError as e:
            raise JujuError(
                f'{self._app_name}: mocking= does not match the arguments of the '
                f"charm's {config.function}(): {e}."
            ) from None
        self._function = cast('Callable[..., contextlib.AbstractContextManager[Any]]', function)

    def _import_module(self, root: pathlib.Path) -> Any:
        config = self._config
        if config.module is not None:
            try:
                return importlib.import_module(config.module)
            except ImportError as e:
                raise JujuError(
                    f'{self._app_name}: cannot import the mocking module {config.module}: {e}'
                ) from e
        assert config.path is not None
        path = root / config.path
        if not path.exists():
            return None  # The default path, and the charm has no mocking.
        if self._module_name in sys.modules:
            return sys.modules[self._module_name]
        spec = importlib.util.spec_from_file_location(self._module_name, path)
        if spec is None or spec.loader is None:
            raise JujuError(f'{self._app_name}: cannot load the mocking file {path}.')
        module = importlib.util.module_from_spec(spec)
        sys.modules[self._module_name] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            del sys.modules[self._module_name]
            raise JujuError(
                f'{self._app_name}: the mocking file {path} failed to import: {e}. It is '
                "loaded by path, so it can import from the charm's src/ and lib/ and from "
                "installed packages, but not from elsewhere in the charm's tests."
            ) from e
        except Exception as e:
            del sys.modules[self._module_name]
            raise JujuError(
                f'{self._app_name}: the mocking file {path} failed to import: {e!r}'
            ) from e
        return module

    def importing(self) -> contextlib.ExitStack[bool | None]:
        """The scope to import the charm in: the defaults, without the charm's own mocking."""
        return self._defaults(unit_name=None, model_name=None)

    @contextlib.contextmanager
    def dispatching(self, unit_name: str, model_name: str) -> Generator[None]:
        """The scope to run one dispatch in: the defaults, then the charm's own mocking."""
        self.load()
        with self._defaults(unit_name=unit_name, model_name=model_name):
            if self._function is None:
                yield
                return
            with self._function(**self._mocking):
                yield

    def _defaults(
        self, *, unit_name: str | None, model_name: str | None
    ) -> contextlib.ExitStack[bool | None]:
        stack = contextlib.ExitStack()
        disabled = self._config.disable
        if 'env' not in disabled:
            stack.enter_context(_juju_environment())
        if 'subprocess' not in disabled:
            stack.enter_context(_succeeding_subprocesses())
        if 'clock' not in disabled:
            stack.enter_context(self._clock.patched())
        if 'hostname' not in disabled:
            name = unit_name or f'{self._app_name}/0'
            stack.enter_context(_unit_hostname(name, model_name))
        if 'network' not in disabled:
            stack.enter_context(_no_outbound_connections())
        return stack


# The defaults


@contextlib.contextmanager
def _juju_environment() -> Generator[None]:
    """Run with only what Juju would set in the environment.

    ``Context`` adds the ``JUJU_*`` variables for the hook itself. The
    ``SCENARIO_*`` settings are the test harness's own, so they stay.
    """
    saved = dict(os.environ)
    os.environ.clear()
    os.environ['PATH'] = _JUJU_PATH
    for key, value in saved.items():
        if key.startswith('SCENARIO_'):
            os.environ[key] = value
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


class _SucceededProcess:
    """Stands in for a :class:`subprocess.Popen`: the command succeeded, silently."""

    def __init__(self, args: Any, *_: Any, **kwargs: Any):
        self.args = args
        self.pid = 0
        self.returncode = 0
        text = bool(
            kwargs.get('text')
            or kwargs.get('universal_newlines')
            or kwargs.get('encoding')
            or kwargs.get('errors')
        )
        empty: Any = '' if text else b''
        self._empty = empty
        self.stdin = None
        self.stdout = _empty_stream(text) if kwargs.get('stdout') == subprocess.PIPE else None
        self.stderr = _empty_stream(text) if kwargs.get('stderr') == subprocess.PIPE else None

    def communicate(self, input: Any = None, timeout: float | None = None) -> tuple[Any, Any]:
        del input, timeout
        stdout = self._empty if self.stdout is not None else None
        stderr = self._empty if self.stderr is not None else None
        return stdout, stderr

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def poll(self) -> int:
        return 0

    def kill(self) -> None:
        pass

    def terminate(self) -> None:
        pass

    def send_signal(self, sig: int) -> None:
        pass

    def __enter__(self) -> _SucceededProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _empty_stream(text: bool) -> Any:
    import io

    return io.StringIO('') if text else io.BytesIO(b'')


@contextlib.contextmanager
def _succeeding_subprocesses() -> Generator[None]:
    """Every command succeeds, with return code 0 and empty output.

    ``run``, ``check_output`` and the rest create a ``Popen``, and
    ``os.popen`` does too, so replacing ``Popen`` covers them.
    """

    def system(command: str) -> int:
        del command
        return 0

    with (
        mock.patch.object(subprocess, 'Popen', _SucceededProcess),
        mock.patch.object(os, 'system', system),
    ):
        yield


class _FakeClock:
    """``time.sleep`` advances ``time.time`` and ``time.monotonic`` instead of blocking.

    The offset accumulates over the application's dispatches, so a retry
    loop that sleeps, then reads the clock, sees the time it slept.
    """

    def __init__(self) -> None:
        self._offset = 0.0
        self._real_time = time.time
        self._real_monotonic = time.monotonic

    def _sleep(self, seconds: float) -> None:
        self._offset += max(0.0, seconds)

    def _time(self) -> float:
        return self._real_time() + self._offset

    def _monotonic(self) -> float:
        return self._real_monotonic() + self._offset

    @contextlib.contextmanager
    def patched(self) -> Generator[None]:
        with (
            mock.patch.object(time, 'sleep', self._sleep),
            mock.patch.object(time, 'time', self._time),
            mock.patch.object(time, 'monotonic', self._monotonic),
        ):
            yield


@contextlib.contextmanager
def _unit_hostname(unit_name: str, model_name: str | None) -> Generator[None]:
    """The unit's host names derive from its unit and model names."""
    hostname = unit_name.replace('/', '-')
    fqdn = f'{hostname}.{model_name}' if model_name else hostname

    def getfqdn(name: str = '') -> str:
        return fqdn if name in ('', hostname, fqdn) else name

    def gethostname() -> str:
        return hostname

    def gethostbyname(name: str) -> str:
        del name
        return _DEFAULT_ADDRESS

    with (
        mock.patch.object(socket, 'getfqdn', getfqdn),
        mock.patch.object(socket, 'gethostname', gethostname),
        mock.patch.object(socket, 'gethostbyname', gethostbyname),
    ):
        yield


def _outbound_error(address: Any) -> OSError:
    if isinstance(address, tuple):
        parts = cast('tuple[Any, ...]', address)
        target = ':'.join(str(part) for part in parts[:2])
    else:
        target = str(address)
    return ConnectionRefusedError(
        f'Outbound connection to {target} blocked: charms under ops.testing.Juju '
        "can't reach the network. Mock the service in the charm's own mocking, or "
        'disable the network default.'
    )


@contextlib.contextmanager
def _no_outbound_connections() -> Generator[None]:
    """TCP/IP connections raise, naming the host and port. Unix sockets are left alone."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise _outbound_error(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise _outbound_error(address)
        return real_connect_ex(self, address)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        del args, kwargs
        raise _outbound_error(address)

    with (
        mock.patch.object(socket.socket, 'connect', connect),
        mock.patch.object(socket.socket, 'connect_ex', connect_ex),
        mock.patch.object(socket, 'create_connection', create_connection),
    ):
        yield
