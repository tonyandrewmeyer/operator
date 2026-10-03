# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Default fakes for the libraries charms use to reach Kubernetes and snapd.

Each fake replaces attributes of the library's own classes and modules, where
they are defined, so it doesn't matter how the charm imported them. Nothing
is imported here: a library the charm hasn't imported by the time a dispatch
starts isn't in use, so there's nothing to fake.
"""

from __future__ import annotations

import contextlib
import re
import sys
import types
import urllib.parse
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from typing import Any, cast

#: What :func:`patched` sets: an object, an attribute name, and the new value.
Patch = tuple[Any, str, Any]


@contextlib.contextmanager
def patched(patches: list[Patch]) -> Generator[None]:
    """Set each attribute for the duration of the block, then put the originals back."""
    saved = [(owner, name, owner.__dict__.get(name, _MISSING)) for owner, name, _ in patches]
    for owner, name, value in patches:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, original in reversed(saved):
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)


_MISSING = object()


def _named(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Give a replacement method the name of the one it replaces.

    ``framework.observe`` stores an observer's method by name and looks it up
    again when the event is emitted, so the names have to match. One fake can
    stand in for several methods, so each gets its own wrapper.
    """

    def method(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = name
    return method


def _replace_methods(cls: type, replacements: dict[str, Any]) -> list[Patch]:
    """Patches for whichever of ``replacements`` the class actually has."""
    patches: list[Patch] = []
    for name, value in replacements.items():
        if not hasattr(cls, name):
            continue
        if callable(value) and not isinstance(value, (type, property)):
            value = _named(name, value)
        patches.append((cls, name, value))
    return patches


# lightkube


def _not_found_error(exceptions: types.ModuleType, res: Any, name: Any) -> BaseException:
    """The ``ApiError`` lightkube raises for a 404."""
    kind = getattr(res, '__name__', 'resource')
    message = f'{kind.lower()} "{name}" not found'
    body = {
        'kind': 'Status',
        'apiVersion': 'v1',
        'metadata': {},
        'status': 'Failure',
        'message': message,
        'reason': 'NotFound',
        'details': {'name': name, 'kind': kind.lower()},
        'code': 404,
    }
    api_error: Any = exceptions.ApiError
    httpx: Any = getattr(exceptions, 'httpx', None)
    if httpx is not None:
        request = httpx.Request('GET', f'https://kubernetes.invalid/{kind.lower()}/{name}')
        response = httpx.Response(404, json=body, request=request)
        return api_error(request=request, response=response)
    # A lightkube whose ApiError can't be built from a response: give it the
    # same status attributes.
    error = api_error.__new__(api_error)
    Exception.__init__(error, message)
    error.status = types.SimpleNamespace(
        code=404, reason='NotFound', message=message, status='Failure', details=body['details']
    )
    return error


async def _no_items() -> AsyncIterator[Any]:  # ruff: ignore[unused-async]
    return
    yield


def _written(obj: Any) -> Any:
    """What a write returns: the object written, when it's a resource."""
    return obj if hasattr(obj, 'metadata') else None


def _lightkube_patches() -> list[Patch]:
    """Lightkube's clients work without a cluster: nothing is there, and writes succeed."""
    exceptions = sys.modules.get('lightkube.core.exceptions')
    if exceptions is None:
        return []

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        namespace = kwargs.get('namespace', args[1] if len(args) > 1 else None)
        self._ops_testing_namespace = namespace or 'default'

    def namespace(self: Any) -> str:
        return getattr(self, '_ops_testing_namespace', 'default')

    def get(self: Any, res: Any, name: Any = None, *args: Any, **kwargs: Any) -> Any:
        raise _not_found_error(exceptions, res, name)

    def nothing(self: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        return iter(())

    def write(self: Any, obj: Any = None, *args: Any, **kwargs: Any) -> Any:
        return _written(obj)

    def patch(self: Any, res: Any, name: Any = None, obj: Any = None, *args: Any, **kwargs: Any):
        return _written(kwargs.get('obj', obj))

    def done(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    # These replace coroutine methods, so they have to be coroutines too.
    async def async_get(  # ruff: ignore[unused-async]
        self: Any, res: Any, name: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        raise _not_found_error(exceptions, res, name)

    def async_nothing(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        return _no_items()

    async def async_write(  # ruff: ignore[unused-async]
        self: Any, obj: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        return _written(obj)

    async def async_patch(  # ruff: ignore[unused-async]
        self: Any, res: Any, name: Any = None, obj: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        return _written(kwargs.get('obj', obj))

    async def async_done(  # ruff: ignore[unused-async]
        self: Any, *args: Any, **kwargs: Any
    ) -> None:
        return None

    patches: list[Patch] = []
    client_module = sys.modules.get('lightkube.core.client')
    client = getattr(client_module, 'Client', None)
    if isinstance(client, type):
        patches += _replace_methods(
            client,
            {
                '__init__': init,
                'namespace': property(namespace),
                'close': done,
                'get': get,
                'wait': get,
                'list': nothing,
                'watch': nothing,
                'log': nothing,
                'create': write,
                'apply': write,
                'replace': write,
                'patch': patch,
                'set': patch,
                'delete': done,
                'deletecollection': done,
            },
        )
    async_module = sys.modules.get('lightkube.core.async_client')
    async_client = getattr(async_module, 'AsyncClient', None)
    if isinstance(async_client, type):
        patches += _replace_methods(
            async_client,
            {
                '__init__': init,
                'namespace': property(namespace),
                'close': async_done,
                'aclose': async_done,
                'get': async_get,
                'wait': async_get,
                'list': async_nothing,
                'watch': async_nothing,
                'log': async_nothing,
                'create': async_write,
                'apply': async_write,
                'replace': async_write,
                'patch': async_patch,
                'set': async_patch,
                'delete': async_done,
                'deletecollection': async_done,
            },
        )
    return patches


# The Kubernetes patch libraries

_K8S_PATCH_MODULE = re.compile(
    r'^charms\.\w+\.v[01]\.'
    r'(kubernetes_service_patch|kubernetes_compute_resources_patch|kubernetes_statefulset_patch)$'
)


def _k8s_patch_lib_patches() -> list[Patch]:
    """The vendored libraries that patch what Juju created in Kubernetes do nothing."""

    def nothing(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    def ready(self: Any, *args: Any, **kwargs: Any) -> bool:
        return True

    def not_failed(self: Any, *args: Any, **kwargs: Any) -> tuple[bool, str]:
        return False, ''

    def not_in_progress(self: Any, *args: Any, **kwargs: Any) -> bool:
        return False

    def active(self: Any) -> Any:
        import ops

        return ops.ActiveStatus()

    def patcher_init(
        self: Any, namespace: str, statefulset_name: str, container_name: str, *args: Any
    ) -> None:
        self.namespace = namespace
        self.statefulset_name = statefulset_name
        self.container_name = container_name
        self.client = None

    # The libraries read the namespace from the service account's token
    # directory, which only exists in a pod. In Juju it is the model name.
    model_namespace = property(lambda self: self.model.name)

    replacements: dict[str, dict[str, Any]] = {
        'KubernetesServicePatch': {
            '_patch': nothing,
            '_on_upgrade_charm': nothing,
            '_remove_service': nothing,
            'is_patched': ready,
            '_is_patched': ready,
            '_namespace': model_namespace,
        },
        'KubernetesComputeResourcesPatch': {
            '_patch': nothing,
            '_on_config_changed': nothing,
            'is_ready': ready,
            'get_status': active,
            '_namespace': model_namespace,
        },
        'ResourcePatcher': {
            '__init__': patcher_init,
            'apply': nothing,
            'is_patched': ready,
            'is_ready': ready,
            'is_failed': not_failed,
            'is_in_progress': not_in_progress,
            'get_templated': nothing,
            'get_actual': nothing,
        },
        'KubernetesStatefulsetPatch': {
            '_patch_statefulset': nothing,
            '_is_patched': ready,
            '_namespace': model_namespace,
        },
    }
    patches: list[Patch] = []
    names = sorted(n for n in list(sys.modules) if n.startswith('charms.'))
    for module_name in names:
        if not _K8S_PATCH_MODULE.match(module_name):
            continue
        module = sys.modules[module_name]
        for class_name, methods in replacements.items():
            cls = getattr(module, class_name, None)
            if isinstance(cls, type) and cls.__module__ == module_name:
                patches += _replace_methods(cls, methods)
    return patches


# snap

_SNAPCACHE_MODULE = re.compile(
    r'^(charms\.operator_libs_linux\.v\d+\.snap|charmlibs\.snap\._snap)$'
)
_SNAP_PATH = re.compile(r'^/v2/snaps/([^/]+)$')


class _NoSnapd:
    """Stands in for ``SnapClient``: snapd knows nothing, and accepts everything."""

    def get_installed_snaps(self) -> list[Any]:
        return []

    def get_installed_snap_apps(self, name: str) -> list[Any]:
        del name
        return []

    def _put_snap_conf(self, name: str, conf: Any) -> None:
        del name, conf


def _snap_cache_patches(
    module: types.ModuleType, installed: dict[str, dict[str, str]]
) -> list[Patch]:
    """``SnapCache`` starts empty, and ``ensure`` installs into ``installed``."""
    snap_type: Any = getattr(module, 'Snap', None)
    cache_type: Any = getattr(module, 'SnapCache', None)
    state: Any = getattr(module, 'SnapState', None)
    if not isinstance(snap_type, type) or not isinstance(cache_type, type) or state is None:
        return []
    present = (state.Present, state.Latest)

    def make(name: str) -> Any:
        info = installed.get(name)
        snap = cast('Any', object.__new__(snap_type))
        snap._name = name
        snap._state = state.Latest if info else state.Available
        snap._channel = info['channel'] if info else ''
        snap._revision = info['revision'] if info else ''
        snap._confinement = info['confinement'] if info else ''
        snap._cohort = ''
        snap._apps = []
        snap._version = None
        snap._snap_client = _NoSnapd()
        return snap

    def cache_init(self: Any) -> None:
        self._snap_client = _NoSnapd()
        self._snap_map = {name: make(name) for name in sorted(installed)}

    def getitem(self: Any, name: str) -> Any:
        snap = self._snap_map.get(name)
        if snap is None:
            snap = self._snap_map[name] = make(name)
        return snap

    def ensure(
        self: Any,
        state: Any,
        classic: bool = False,
        devmode: bool = False,
        channel: str | None = None,
        cohort: str | None = None,
        revision: str | int | None = None,
        **kwargs: Any,
    ) -> None:
        del cohort, kwargs
        if classic and devmode:
            raise ValueError('Cannot set both classic and devmode confinement')
        if classic:
            self._confinement = 'classic'
        elif devmode:
            self._confinement = 'devmode'
        if state in present:
            self._channel = channel or self._channel or 'latest/stable'
            if revision:  # The library passes '' for no revision.
                self._revision = str(revision)
            elif not self._revision:
                self._revision = '1'
            installed[self._name] = {
                'channel': self._channel,
                'revision': self._revision,
                'confinement': self._confinement,
            }
        else:
            installed.pop(self._name, None)
        self._state = state

    def no_apps(self: Any) -> None:
        self._apps = []

    patches = _replace_methods(cache_type, {'__init__': cache_init, '__getitem__': getitem})
    patches += _replace_methods(snap_type, {'ensure': ensure, '_update_snap_apps': no_apps})
    # The module-level add(), remove() and ensure() keep one SnapCache for the
    # process; each dispatch starts without it, so it is built from this unit's snaps.
    holder: Any = getattr(module, '_Cache', None)
    if holder is not None and '_cache' in getattr(holder, '__dict__', {}):
        patches.append((holder, '_cache', None))
    client: Any = getattr(module, 'SnapClient', None)
    if isinstance(client, type):

        def request(self: Any, method: str, *args: Any, **kwargs: Any) -> Any:
            return [] if method == 'GET' else None

        patches += _replace_methods(client, {'_request': request})
    return patches


def _snapd_client_patches(
    module: types.ModuleType, installed: dict[str, dict[str, str]]
) -> list[Patch]:
    """charmlibs.snap's snapd client talks to an empty snapd that accepts every change."""
    make_error: Any = getattr(module, '_make_error', None)
    utils = sys.modules.get('charmlibs.snap._utils')
    resolve_channel: Any = getattr(utils, 'resolve_channel', None)

    def not_installed(name: str) -> BaseException:
        response: dict[str, Any] = {
            'type': 'error',
            'status-code': 404,
            'status': 'Not Found',
            'result': {
                'kind': 'snap-not-found',
                'message': f'snap "{name}" is not installed',
                'value': name,
            },
        }
        if make_error is None:
            return RuntimeError(response['result']['message'])
        return make_error(response)

    def info(name: str) -> dict[str, Any]:
        record = installed[name]
        return {
            'name': name,
            'tracking-channel': record['channel'],
            'channel': record['channel'],
            'revision': record['revision'],
            'version': '',
            'confinement': record['confinement'],
            'status': 'active',
            'apps': [],
        }

    def snap_name(path: str) -> str | None:
        match = _SNAP_PATH.match(path)
        return urllib.parse.unquote(match.group(1)) if match else None

    def get(path: str, query: Any = None) -> Any:
        del query
        if path == '/v2/snaps':
            return [info(name) for name in sorted(installed)]
        name = snap_name(path)
        if name is None:
            return {}
        if name not in installed:
            raise not_installed(name)
        return info(name)

    def post(path: str, body: Any = None) -> Any:
        body = body or {}
        name = snap_name(path)
        action = body.get('action')
        if name is not None and action in ('install', 'refresh'):
            record = installed.get(name, {})
            current = record.get('channel', 'latest/stable')
            channel = body.get('channel') or ''
            if resolve_channel is not None:
                tracking = resolve_channel(channel, current)
            else:
                tracking = channel or current
            installed[name] = {
                'channel': tracking,
                'revision': str(body.get('revision') or record.get('revision') or '1'),
                'confinement': 'classic'
                if body.get('classic')
                else record.get('confinement', 'strict'),
            }
        elif name is not None and action == 'remove':
            installed.pop(name, None)
        return None

    def put(path: str, body: Any = None) -> None:
        del path, body

    def get_logs(query: Any = None) -> list[Any]:
        del query
        return []

    replacements = {'get': get, 'post': post, 'put': put, 'get_logs': get_logs}
    return [(module, name, f) for name, f in replacements.items() if hasattr(module, name)]


def _snap_patches(installed: dict[str, dict[str, str]]) -> list[Patch]:
    patches: list[Patch] = []
    names = sorted(n for n in list(sys.modules) if n.startswith(('charms.', 'charmlibs.')))
    for module_name in names:
        if _SNAPCACHE_MODULE.match(module_name):
            patches += _snap_cache_patches(sys.modules[module_name], installed)
    client = sys.modules.get('charmlibs.snap._client')
    if client is not None:
        patches += _snapd_client_patches(client, installed)
    return patches


@contextlib.contextmanager
def lightkube() -> Generator[None]:
    """The ``lightkube`` default."""
    with patched(_lightkube_patches()):
        yield


@contextlib.contextmanager
def k8s_patch_libs() -> Generator[None]:
    """The ``k8s-patch-libs`` default."""
    with patched(_k8s_patch_lib_patches()):
        yield


@contextlib.contextmanager
def snap(installed: dict[str, dict[str, str]]) -> Generator[None]:
    """The ``snap`` default: ``installed`` is the unit's snaps, kept between dispatches."""
    with patched(_snap_patches(installed)):
        yield
