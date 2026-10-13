# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""What running a charm needs, both in the test process and in a worker."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import pathlib
import sys
import traceback
import unittest.mock
from collections.abc import Generator, Mapping

import ops

from . import _runtime
from . import state as _state

#: The alphabet of a Juju secret ID: an xid, 20 characters of base32hex.
_XID_ALPHABET = '0123456789abcdefghijklmnopqrstuv'


@contextlib.contextmanager
def secret_ids(seed: str | None) -> Generator[None]:
    """Make the IDs of secrets created inside the block depend only on ``seed``.

    Scenario gives a new secret a random ID. Under ``Juju`` that would make
    the final ``State`` differ from run to run, so each dispatch passes a
    seed, and the n-th secret it creates gets an ID derived from the seed and
    n. ``None`` leaves the IDs random.
    """
    if seed is None:
        yield
        return
    count = 0

    def generate() -> str:
        nonlocal count
        digest = hashlib.sha256(f'{seed}/{count}'.encode()).digest()
        count += 1
        value = int.from_bytes(digest[:13], 'big')
        chars: list[str] = []
        for _ in range(20):
            value, index = divmod(value, len(_XID_ALPHABET))
            chars.append(_XID_ALPHABET[index])
        return f'secret:{"".join(chars)}'

    with unittest.mock.patch.object(_state, '_generate_secret_id', generate):
        yield


def load_charm_type(charm_source: pathlib.Path, module_name: str = 'charm') -> type[ops.CharmBase]:
    """Import the charm module and return its CharmBase subclass.

    Adds ``charm_source/src`` and ``charm_source/lib`` to ``sys.path`` so that
    the charm's own source files and bundled charm libraries are importable.

    Args:
        charm_source: Path to the charm repository root.
        module_name: The name to import ``src/charm.py`` under. The worker
            runs one charm, so it uses ``charm``. Several charms loaded into
            one process each need their own name, and are loaded from the
            file rather than found on ``sys.path``.

    Returns:
        The charm class (a :class:`ops.CharmBase` subclass).

    Raises:
        RuntimeError: if zero or more than one charm class is found.
    """
    sources = [str(charm_source / 'src'), str(charm_source / 'lib')]
    for entry in sources:
        if pathlib.Path(entry).exists() and entry not in sys.path:
            sys.path.insert(0, entry)

    if module_name == 'charm':
        module = importlib.import_module('charm')
    elif module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(
            module_name, charm_source / 'src' / 'charm.py'
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Cannot load {charm_source}/src/charm.py.')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    charm_types = [
        t
        for t in module.__dict__.values()
        if isinstance(t, type) and issubclass(t, ops.CharmBase) and t is not ops.CharmBase
    ]
    if not charm_types:
        raise RuntimeError(f'No CharmBase subclass found in {charm_source}/src/charm.py.')
    if len(charm_types) > 1:
        raise RuntimeError(
            f'Multiple CharmBase subclasses found in {charm_source}/src/charm.py: '
            f'{[t.__name__ for t in charm_types]}. '
            'Exactly one is required.'
        )
    return charm_types[0]


def charm_exception(error: BaseException) -> BaseException:
    """The exception to report for a charm that raised.

    That's the charm's own exception if ``SCENARIO_BARE_CHARM_ERRORS`` is set,
    and otherwise the ``UncaughtCharmError`` that wraps it, as ``Context.run``
    would raise.
    """
    return (error.__cause__ or error) if _runtime._bare_charm_errors() else error


def charm_traceback(error: BaseException) -> str:
    """The traceback of :func:`charm_exception`, as text, to cross a process boundary."""
    return ''.join(traceback.format_exception(charm_exception(error)))


def machine_id(machines: Mapping[str, int], app_name: str, unit_id: int) -> str | None:
    """The unit's machine, as Juju gives it in ``JUJU_MACHINE_ID``; none on Kubernetes."""
    machine = machines.get(f'{app_name}/{unit_id}')
    return None if machine is None else str(machine)
