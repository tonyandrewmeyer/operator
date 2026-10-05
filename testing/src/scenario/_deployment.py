# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Model-level testing: drive several charms with Juju-shaped operations.

:class:`~ops.testing.Context` runs *one* event against *one* charm. This
module adds a layer above it: a :class:`Juju` owns a set of applications
(:class:`App`), each with its own units and its own :class:`State` per unit,
and exposes Juju-shaped operations (``deploy``, ``add_unit``, ``remove_unit``,
``config``) that describe **intents rather than events**. The ``Juju`` works
out the Juju-faithful event sequence each intent produces, and
:meth:`Juju.settle` drains it in a convergence loop.

A test therefore reads as a sequence of operations, a ``settle()``, and then
assertions on the resulting state, rather than as a hand-written event
sequence::

    from ops import testing

    juju = testing.Juju()
    web = juju.deploy('./charms/myapp', num_units=2)
    juju.config(web, {'log_level': 'debug'})
    juju.settle()
    assert web.leader.state.unit_status == testing.ActiveStatus('ready')

A charm comes from a path to its source, or, for a charm that runs in the test
process, from a :class:`CharmSpec` holding its class and metadata.

:class:`Juju` is its own class, not a subclass of :class:`~ops.testing.Model`:
``Model`` is a frozen dataclass held as ``State.model`` in every unit's
:class:`State`, so ``Juju`` produces the ``Model`` values that go into each
unit's state rather than being one. The identity it carries (``name``,
``uuid``, ``type``, ``cloud_spec``) is stamped into every unit's
:class:`State`, which stops two applications under the same ``Juju`` from
disagreeing about which model they are in.

Applications are related with ``integrate`` and unrelated with
``remove_relation``. Whatever a unit writes to a relation databag, and the
secrets it grants over a relation, reach the units on the other side, along
with the events Juju would fire for them there.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import pathlib
import shutil
import tempfile
import types
import weakref
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeAlias, cast
from uuid import uuid4

from . import _charm_mocking, _environment, _isolated_serde, _unit_filesystem
from . import state as _state_module
from ._isolated_worker import (
    _charm_exception,
    _charm_traceback,
    _load_charm_type,
    _secret_ids,
)
from ._isolation import IsolatedContext, _HookFailedError, _load_charm_spec
from .context import _DEFAULT_JUJU_VERSION, Context
from .errors import IsolationError, JujuError, MetadataNotFoundError, UncaughtCharmError
from .state import (
    CharmType,
    CloudSpec,
    Container,
    ErrorStatus,
    Model,
    PeerRelation,
    RawDataBagContents,
    Relation,
    RelationBase,
    Secret,
    State,
    _CharmSpec,
    _Event,
    _random_model_name,
)

if TYPE_CHECKING:
    from ops.charm import CharmBase


@dataclasses.dataclass(frozen=True)
class CharmSpec(Generic[CharmType]):
    """A charm that runs in the test process: its class, metadata, and mocking.

    Deploy one with :meth:`Juju.deploy` to run a charm class defined in the
    test, or a charm library's stand-in for the charm on the other end of a
    relation, which the library's testing package provides::

        spec = testing.CharmSpec(MyCharm, meta={'name': 'myapp'})
        app = juju.deploy(spec)

    A ``CharmSpec`` is frozen, so one can be deployed any number of times.
    """

    charm_type: type[CharmType]
    """The charm class."""

    meta: Mapping[str, Any]
    """The charm's metadata, shaped like ``charmcraft.yaml``.

    Config options go under ``config`` and actions under ``actions``, as they
    do in ``charmcraft.yaml``, rather than in separate mappings.
    """

    mocking: Callable[..., contextlib.AbstractContextManager[Any]] | None = None
    """A function returning a context manager that mocks what the charm needs.

    :class:`Juju` calls it with the ``mocked=`` given to :meth:`Juju.deploy`
    as keyword arguments, and opens the result around each of the
    application's dispatches, inside the default mocks. With no ``mocking``,
    the charm still gets the default mocks.
    """


#: What ``charm=`` accepts: a path to charm source on disk, a charm class, or a
#: :class:`CharmSpec`.
CharmSource: TypeAlias = 'str | pathlib.Path | type[CharmBase] | CharmSpec[Any]'

#: How many dispatches per unit :meth:`Juju.settle` allows before it decides
#: the model will not converge.
_SETTLE_DISPATCHES_PER_UNIT = 100

#: How many dispatches from the end of the trace a non-convergence error shows.
_TRACE_TAIL = 10

#: The model UUID when the test doesn't give one. It's fixed rather than random
#: so that two runs of the same test give the same final ``State``.
_DEFAULT_MODEL_UUID = '6f2b0c5e-3d4a-4b8e-9c1f-0a7d5e2b8c43'

#: Events that only say "look again". One already waiting for a unit, about
#: the same thing, covers a second.
_COALESCED_SUFFIXES = ('_relation_changed', 'config_changed', 'secret_changed')

#: ``State`` fields that describe a unit's place in the model. :class:`Juju`
#: sets these itself, so a ``state_template`` may not.
_JUJU_OWNED_FIELDS = ('leader', 'planned_units', 'model', 'config', 'relations', 'secrets')


@dataclasses.dataclass(frozen=True)
class Dispatch:
    """One event dispatched to one unit, as recorded in a settle trace.

    It's a record of the event as it happened, with no live handles: it
    describes the unit as it was at that point, however the model has moved
    on since, and two ``Dispatch`` objects compare by value. The application
    itself is ``juju.apps[dispatch.app]``, and the model is
    ``dispatch.state_in.model``.
    """

    event: _Event
    """The event that was dispatched."""

    app: str
    """The name of the application the unit belongs to."""

    unit_id: int
    """The unit's ID, as in ``myapp/2``."""

    state_in: State
    """The :class:`State` the unit was given.

    That's the state after :class:`Juju` wrote in the shared changes from other
    units, so it's what the charm actually saw, rather than the previous
    dispatch's :attr:`state_out`.
    """

    state_out: State
    """The :class:`State` the charm produced."""

    error: str | None = None
    """The charm's traceback, if it raised while handling the event.

    The charm's changes are discarded, as Juju discards a failed hook's, so
    :attr:`state_out` is :attr:`state_in` in error status.
    """

    _charm: _CharmForContext | None = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def unit_name(self) -> str:
        """The Juju unit name, for example ``myapp/2``."""
        return f'{self.app}/{self.unit_id}'

    def to_context(self) -> Context[CharmBase]:
        """A new :class:`~ops.testing.Context` for the unit as it was at this dispatch.

        The ``Context`` has the charm and metadata the test gave
        :meth:`Juju.deploy`, the application's trust, and the unit's ID and
        charm directory. Config and leadership come from :attr:`state_in`, so
        running :attr:`event` against :attr:`state_in` runs the dispatch again
        from the same starting point, without :class:`Juju`'s mocking::

            ctx = dispatch.to_context()
            state_out = ctx.run(dispatch.event, dispatch.state_in)

        Like :meth:`Unit.to_context`, it's new on every call, and a charm
        deployed in a worker process is imported into the test process.

        Raises:
            IsolationError: if the charm can't be imported into the test
                process.
            JujuError: if this ``Dispatch`` wasn't recorded by :meth:`Juju.settle`.
        """
        if self._charm is None:
            raise JujuError('Only a Dispatch from Juju.settle() knows its charm.')
        return self._charm.context(self.unit_id)


class _CharmForContext:
    """What a :class:`Context` for one of an application's units needs.

    Kept apart from the :class:`App` so that a :class:`Dispatch` can build a
    ``Context`` without holding the live application. The charm source and
    metadata never change after ``deploy()``.
    """

    def __init__(self, app: App):
        self._app_name = app.name
        self._charm_source = app._charm_source
        self._charm_type = app._charm_type
        self._metadata = dict(app._metadata)
        self._config_schema = dict(app._config_schema) if app._config_schema is not None else None
        self._actions = dict(app._actions) if app._actions is not None else None
        self._juju_version = app._juju_version
        self._trust = app._trust
        # Shared with the App, which adds each unit's directory as it's made.
        self._charm_roots = app._charm_roots

    def context(self, unit_id: int) -> Context[CharmBase]:
        if self._charm_type is None:
            assert self._charm_source is not None
            try:
                self._charm_type = _load_charm_type(
                    self._charm_source,
                    module_name=f'_ops_testing_charm_{uuid4().hex}',
                )
            except Exception as e:
                raise IsolationError(
                    f'Cannot import the charm for {self._app_name} into the test process, '
                    f'which a Context needs: {e!r}'
                ) from e
        return Context(
            self._charm_type,
            meta=dict(self._metadata),
            config=dict(self._config_schema) if self._config_schema is not None else None,
            actions=dict(self._actions) if self._actions is not None else None,
            app_name=self._app_name,
            unit_id=unit_id,
            juju_version=self._juju_version,
            app_trusted=self._trust,
            charm_root=self._charm_roots.get(unit_id),
        )


# Event sequences
#
# Each operation below maps to the events Juju emits for it. Keeping them in
# one place, named after the operation rather than the event, is what lets the
# convergence loop stay ignorant of Juju's hook semantics.


def _startup_events(app: App, unit_id: int) -> list[tuple[_Event, _Rebind | None]]:
    """The events a newly-added unit sees, in Juju's order.

    ``install`` first, then the leadership event (which one depends on whether
    this unit won the election), then ``config-changed``, then ``start``.
    Workload containers become ready after the unit has started.
    """
    events: list[tuple[_Event, _Rebind | None]] = [(_Event('install'), None)]
    if unit_id == app._leader_id:
        events.append((_Event('leader_elected'), None))
    else:
        events.append((_Event('leader_settings_changed'), None))
    events.append((_Event('config_changed'), None))
    events.append((_Event('start'), None))
    for name in app._container_names:
        events.append((_Event(f'{name}_pebble_ready'), _Rebind('container', name)))
    return events


def _teardown_events(app: App, unit_id: int) -> list[tuple[_Event, _Rebind | None]]:
    """The events a departing unit sees, in Juju's order."""
    del app, unit_id  # Same for every unit today; kept for symmetry with startup.
    return [(_Event('stop'), None), (_Event('remove'), None)]


# Runners
#
# Two ways to execute an event: in this process (a charm class, or a charm
# loaded from a path, exactly what Context does) or in a subprocess running
# the charm's own interpreter. The convergence loop only knows this interface.


class _Runner:
    """Executes a single event for one unit of one application.

    ``secret_seed`` decides the IDs of any secrets the charm creates in this
    dispatch, so that they are the same on every run.
    """

    def run(self, unit_id: int, event: _Event, state: State, secret_seed: str) -> State:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _NotStarted(_Runner):
    """The runner of an application that :class:`Juju` hasn't started yet."""

    def run(self, unit_id: int, event: _Event, state: State, secret_seed: str) -> State:
        raise JujuError('This application has not been started by Juju.deploy().')

    def close(self) -> None:
        pass


class _InProcessRunner(_Runner):
    """Runs a charm class in the test process, via :class:`Context`.

    No isolation: the charm shares the test process's interpreter and
    installed packages, which is the same trade-off a plain ``Context`` test
    makes. A separate ``Context`` is held per unit because the unit ID is
    fixed at construction time.
    """

    def __init__(
        self,
        charm_type: type[CharmBase],
        *,
        meta: Mapping[str, Any],
        config: Mapping[str, Any] | None,
        actions: Mapping[str, Any] | None,
        app_name: str,
        juju_version: str,
        app_trusted: bool,
        charm_roots: Mapping[int, pathlib.Path],
        unit_roots: Mapping[int, pathlib.Path],
        mocking: _charm_mocking.CharmMocking,
    ):
        self._charm_type = charm_type
        self._meta = meta
        self._config = config
        self._actions = actions
        self._app_name = app_name
        self._juju_version = juju_version
        self._app_trusted = app_trusted
        self._charm_roots = charm_roots
        self._unit_roots = unit_roots
        self._mocking = mocking
        self._contexts: dict[int, Context[CharmBase]] = {}

    def _context(self, unit_id: int) -> Context[CharmBase]:
        if unit_id not in self._contexts:
            # Context's own signature is still dict-typed, so copy at the
            # boundary rather than widening it as a drive-by.
            self._contexts[unit_id] = Context(
                self._charm_type,
                meta=dict(self._meta),
                config=dict(self._config) if self._config is not None else None,
                actions=dict(self._actions) if self._actions is not None else None,
                app_name=self._app_name,
                unit_id=unit_id,
                juju_version=self._juju_version,
                app_trusted=self._app_trusted,
                charm_root=self._charm_roots.get(unit_id),
            )
            self._contexts[unit_id]._wrap_charm_errors = True
        return self._contexts[unit_id]

    def run(self, unit_id: int, event: _Event, state: State, secret_seed: str) -> State:
        ctx = self._context(unit_id)
        with (
            self._mocking.dispatching(
                f'{self._app_name}/{unit_id}',
                state.model.name,
                filesystem_root=self._unit_roots.get(unit_id),
                allow=_unit_filesystem.framework_paths(ctx, state),
            ),
            _secret_ids(secret_seed),
        ):
            try:
                return ctx.run(event, state)
            except UncaughtCharmError as e:
                cause = _charm_exception(e)
                raise _HookFailedError(_charm_traceback(e), cause) from cause

    def close(self) -> None:
        self._contexts.clear()


class _IsolatedRunner(_Runner):
    """Runs an on-disk charm in a subprocess, via :class:`IsolatedContext`.

    One :class:`IsolatedContext`, and therefore one persistent worker process,
    serves every unit of the application; the unit ID travels with each
    request rather than being baked into the worker. The charm directory is
    per unit, so it travels with each request too.
    """

    def __init__(
        self,
        ctx: IsolatedContext,
        charm_roots: Mapping[int, pathlib.Path],
        unit_roots: Mapping[int, pathlib.Path],
    ):
        self._ctx = ctx
        self._charm_roots = charm_roots
        self._unit_roots = unit_roots

    def run(self, unit_id: int, event: _Event, state: State, secret_seed: str) -> State:
        self._ctx.charm_root = self._charm_roots.get(unit_id)
        self._ctx.filesystem_root = self._unit_roots.get(unit_id)
        return self._ctx._run_as(unit_id, event, state, secret_seed=secret_seed)

    def close(self) -> None:
        self._ctx.close()


# Public handles


class Unit:
    """One unit of an :class:`App`.

    Units are created by :meth:`Juju.deploy` and :meth:`Juju.add_unit`; there
    is no reason to construct one directly.
    """

    def __init__(self, app: App, unit_id: int, state: State):
        self._app = app
        self._id = unit_id
        self._state = state

    @property
    def app(self) -> App:
        """The application this unit belongs to."""
        return self._app

    @property
    def id(self) -> int:
        """The unit number, as in ``myapp/2``."""
        return self._id

    @property
    def name(self) -> str:
        """The Juju unit name, for example ``myapp/2``."""
        return f'{self._app.name}/{self._id}'

    @property
    def is_leader(self) -> bool:
        """Whether this unit currently holds leadership."""
        return self._id == self._app._leader_id

    @property
    def state(self) -> State:
        """This unit's :class:`State` as of the last event dispatched to it.

        Reading this never runs charm code. Call :meth:`Juju.settle` first to
        dispatch whatever the operations so far have queued, then assert.
        """
        return self._state

    @property
    def filesystem(self) -> pathlib.Path:
        """The root of this unit's own filesystem.

        What the charm writes outside its charm directory lands under this
        root, rather than on the test machine: ``/etc/nginx/nginx.conf``
        is at ``unit.filesystem / 'etc/nginx/nginx.conf'``. The unit's copy of
        the charm is in here too, where Juju would put it. Like
        ``Container.get_filesystem()``, this is for asserting on what the
        charm wrote. It isn't part of :attr:`state`, and it persists until the
        ``Juju`` is closed, even after the unit has been removed.
        """
        return self._app._unit_roots[self._id]

    def to_context(self) -> Context[CharmBase]:
        """A new :class:`~ops.testing.Context` for this unit's charm.

        Use it to run an action against the unit at this point in the test, or
        to carry on with a single-charm test from here::

            ctx = web.leader.to_context()
            ctx.run(ctx.on.action('backup'), web.leader.state)

        The ``Context`` has the application's metadata, config options,
        actions, name and trust, and this unit's ID and charm directory. It is
        new on every call, so its collections, such as the Juju log, start
        empty. Running it doesn't change this unit or anything else under the
        :class:`Juju`, and it has none of the mocking that :class:`Juju`
        applies around the application's dispatches.

        The charm runs in the test process, so a charm deployed in a worker
        process is imported into the test process here.

        Raises:
            IsolationError: if the charm can't be imported into the test
                process.
        """
        return self._app._for_context().context(self._id)

    def __repr__(self) -> str:
        return f'<Unit {self.name}>'


class App:
    """An application under a :class:`Juju`.

    Owns the charm reference, the metadata resolved from it, the environment
    it runs in, and the :class:`State` of each of its units. Applications are
    created by :meth:`Juju.deploy`, which resolves the metadata from the charm
    source first; there is no reason to construct one directly.
    """

    def __init__(
        self,
        juju: Juju,
        name: str,
        charm: CharmSource,
        *,
        meta: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        state_template: State | None = None,
        trust: bool = False,
        mocked: Mapping[str, Any] | None = None,
        juju_version: str = _DEFAULT_JUJU_VERSION,
    ):
        if state_template is not None:
            _check_state_template(state_template)
        self._juju = juju
        self._name = name
        self._charm = charm
        self._charm_source = (
            pathlib.Path(charm) if isinstance(charm, (str, pathlib.Path)) else None
        )
        # Each unit gets its own filesystem root, and its own copy of the
        # charm inside it (see _make_unit_root). The runners read these
        # mappings when they dispatch, so they are shared.
        self._unit_roots: dict[int, pathlib.Path] = {}
        self._charm_roots: dict[int, pathlib.Path] = {}
        self._meta = dict(meta)
        self._metadata, self._config_schema, self._actions = _split_meta(meta)
        self._config = _merged_config(self._config_schema, config)
        self._state_template = state_template if state_template is not None else State()
        self._trust = trust
        self._mocked = dict(mocked) if mocked is not None else {}
        _charm_mocking.check_mocking_json(self._mocked)
        self._juju_version = juju_version
        # Set by Juju once it knows where the charm runs; see _run_in_process
        # and _run_in_worker.
        self._runner: _Runner = _NotStarted()
        self._charm_type: type[CharmBase] | None = None
        self._context_source: _CharmForContext | None = None
        self._leader_id = 0
        self._units: dict[int, Unit] = {}
        # Units that remove_unit has started taking down: they leave their
        # relations, so nothing new is sent to them.
        self._dying: set[int] = set()
        self._next_unit_id = 0
        # One relation ID per peer endpoint: a peer relation is a single
        # relation that every unit is a member of, so the ID must agree across
        # units even though each unit holds its own view of the databags.
        self._peer_ids: dict[str, int] = {
            endpoint: juju._new_relation_id() for endpoint in self._peer_endpoints
        }

    def _for_context(self) -> _CharmForContext:
        if self._context_source is None:
            self._context_source = _CharmForContext(self)
        return self._context_source

    def _run_in_process(self) -> None:
        """Run the charm in the test process, loading it from its path if it has one."""
        if isinstance(self._charm, CharmSpec):
            charm_type = cast('type[CharmBase]', self._charm.charm_type)
            mocking = self._charm_mocking(
                None, function=self._charm.mocking, charm_sources=_class_sources(charm_type)
            )
        elif self._charm_source is None:
            charm_type = cast('type[CharmBase]', self._charm)
            mocking = self._charm_mocking(
                _charm_class_root(charm_type), charm_sources=_class_sources(charm_type)
            )
        else:
            charm_type: type[CharmBase] | None = None
            mocking = self._charm_mocking(self._charm_source)
        # The charm's mocking module loads before the charm itself, so its
        # import-time patches are in place when the charm is imported.
        mocking.load()
        if charm_type is None:
            assert self._charm_source is not None
            # The charm import itself runs inside the defaults.
            with mocking.importing():
                charm_type = _load_charm_type(
                    self._charm_source,
                    module_name=f'_ops_testing_charm_{uuid4().hex}',
                )
        self._charm_type = charm_type
        self._runner = _InProcessRunner(
            charm_type,
            meta=self._metadata,
            config=self._config_schema,
            actions=self._actions,
            app_name=self._name,
            juju_version=self._juju_version,
            app_trusted=self._trust,
            charm_roots=self._charm_roots,
            unit_roots=self._unit_roots,
            mocking=mocking,
        )

    def _run_in_worker(
        self,
        python_executable: str,
        extra_sys_path: Sequence[str],
        python_path: Sequence[str] = (),
    ) -> None:
        """Run the charm in a worker process, with the given interpreter.

        The interpreter has to be able to import the same ``ops`` as the test,
        either installed or from ``python_path`` (the front of the worker's
        ``PYTHONPATH``), and has to have everything else the charm needs.
        ``extra_sys_path`` is prepended to the worker's ``sys.path``.
        """
        assert self._charm_source is not None, 'only a charm on disk can run in a worker'
        self._runner = _IsolatedRunner(
            IsolatedContext(
                charm_source=self._charm_source,
                python_executable=python_executable,
                extra_sys_path=tuple(extra_sys_path),
                python_path=tuple(python_path),
                meta=self._metadata,
                config=self._config_schema,
                actions=self._actions,
                app_name=self._name,
                juju_version=self._juju_version,
                app_trusted=self._trust,
                mocking=self._mocked,
            ),
            self._charm_roots,
            self._unit_roots,
        )
        # The mocking itself loads in the worker; the parent reports the
        # configuration problems it can see without importing anything.
        self._charm_mocking(self._charm_source).check()

    def _charm_mocking(
        self,
        root: pathlib.Path | None,
        *,
        function: Callable[..., contextlib.AbstractContextManager[Any]] | None = None,
        charm_sources: Sequence[pathlib.Path] | None = None,
    ) -> _charm_mocking.CharmMocking:
        return _charm_mocking.CharmMocking(
            root,
            app_name=self._name,
            mocking=self._mocked,
            module_name=f'_ops_testing_mocking_{uuid4().hex}',
            function=function,
            charm_sources=charm_sources,
        )

    @property
    def name(self) -> str:
        """The application name."""
        return self._name

    @property
    def meta(self) -> Mapping[str, Any]:
        """The charm metadata, shaped like ``charmcraft.yaml``.

        Read from the charm source, or taken from the :class:`CharmSpec`. Config
        options are under ``config`` and actions under ``actions``, whichever
        files they were read from.
        """
        return self._meta

    @property
    def config(self) -> Mapping[str, Any]:
        """The application's current configuration."""
        return dict(self._config)

    @property
    def units(self) -> Sequence[Unit]:
        """This application's units, ordered by unit number."""
        return tuple(self._units[uid] for uid in sorted(self._units))

    @property
    def leader(self) -> Unit:
        """The unit that currently holds leadership."""
        try:
            return self._units[self._leader_id]
        except KeyError:
            raise JujuError(f'{self._name} has no units.') from None

    @property
    def _substrate(self) -> Literal['kubernetes', 'lxd']:
        """The substrate this application is deployed on.

        Every application under one :class:`Juju` shares its model's
        substrate; there is no per-application override.
        """
        return self._juju.type

    @property
    def _peer_endpoints(self) -> tuple[str, ...]:
        peers: dict[str, Any] = self._metadata.get('peers') or {}
        return tuple(peers)

    @property
    def _relation_endpoints(self) -> dict[str, tuple[str, Mapping[str, Any]]]:
        """Each ``provides`` and ``requires`` endpoint, with its role and metadata."""
        endpoints: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for role in ('provides', 'requires'):
            declared: dict[str, Any] = self._metadata.get(role) or {}
            for endpoint, spec in declared.items():
                endpoints[endpoint] = (role, spec or {})
        return endpoints

    @property
    def _live_units(self) -> tuple[Unit, ...]:
        """This application's units, leaving out those being removed."""
        return tuple(u for u in self.units if u.id not in self._dying)

    @property
    def _container_names(self) -> tuple[str, ...]:
        containers: dict[str, Any] = self._metadata.get('containers') or {}
        return tuple(containers)

    def _containers(self) -> list[Container]:
        """Each unit's starting containers: the template's, or a connectable default."""
        from_template = {c.name: c for c in self._state_template.containers}
        containers: list[Container] = []
        for name in self._container_names:
            containers.append(from_template.pop(name, Container(name=name, can_connect=True)))
        # Anything left names a container the metadata doesn't declare; keep
        # it, so that Context's consistency check reports it to the test.
        containers.extend(from_template.values())
        return containers

    def _make_unit_root(self, unit_id: int) -> None:
        """Give a unit its own filesystem root, and its own copy of the charm.

        Juju gives each unit its own copy of the charm, so a charm on disk is
        copied into the unit's root, where Juju would put it. That's the
        unit's charm directory, so the metadata files that ``Context`` writes
        there land in the copy rather than in the source tree, and so does
        anything the charm writes into its own directory.
        """
        root, charm_dir = _unit_filesystem.make_unit_root(
            self._juju._filesystems_parent(),
            self._name,
            unit_id,
            meta=self._meta,
            charm_source=self._charm_source,
        )
        self._unit_roots[unit_id] = root
        if charm_dir is not None:
            self._charm_roots[unit_id] = charm_dir

    def __repr__(self) -> str:
        return f'<App {self._name} ({len(self._units)} units)>'


class _RemoveUnit:
    """Queue entry marking where a removed unit's records should be dropped.

    Bookkeeping rather than a Juju event. It sits in the queue *after* the
    unit's teardown events so that those events still find the unit in place,
    and so the drop happens in queue order rather than eagerly at
    :meth:`Juju.remove_unit` time.
    """


class _Rebind(NamedTuple):
    """Which object in the unit's state an event should be re-bound to.

    A relation or workload event carries the relation or container *object*,
    and the consistency checker requires it to equal the one in the state it is
    dispatched against, not merely to share its ID or name. Events are queued
    before they are dispatched and the state moves in between, so an event
    queued with the object of the moment goes stale. Such events record what
    to look up here, and the object is bound at dispatch time instead.
    """

    kind: Literal['relation', 'container', 'secret']
    key: int | str
    """The relation ID, the container name, or the secret ID."""


class _Queued(NamedTuple):
    """An event waiting to be dispatched to a unit."""

    app: App
    unit_id: int
    event: _Event | _RemoveUnit
    rebind: _Rebind | None = None


class _End(NamedTuple):
    """One side of a relation between two applications."""

    app: App
    endpoint: str


@dataclasses.dataclass(frozen=True)
class _Integration:
    """A relation between two applications, as :meth:`Juju.integrate` made it.

    Each unit on either side holds its own view of the relation, as a
    :class:`Relation` in its :class:`State` with this ID. This records which
    applications and endpoints the relation joins, so that what one side
    writes can be carried to the other.
    """

    id: int
    ends: tuple[_End, _End]
    interface: str

    def local(self, app: App) -> _End:
        return self.ends[0] if self.ends[0].app is app else self.ends[1]

    def remote(self, app: App) -> _End:
        return self.ends[1] if self.ends[0].app is app else self.ends[0]

    def __str__(self) -> str:
        (a, ea), (b, eb) = self.ends
        return f'{a.name}:{ea} {b.name}:{eb}'


@dataclasses.dataclass
class _UserSecret:
    """A secret the test added with :meth:`Juju.add_secret`."""

    id: str
    name: str
    info: str | None
    # Each revision's content, oldest first.
    revisions: list[dict[str, str]]
    # The names of the applications it's granted to.
    grants: set[str] = dataclasses.field(default_factory=set[str])

    def as_source(self) -> Secret:
        """The secret as its owner would hold it, for :func:`_reader_copy`."""
        secret = Secret(dict(self.revisions[-1]), id=self.id, description=self.info)
        object.__setattr__(secret, '_tracked_revision', len(self.revisions))
        object.__setattr__(secret, '_latest_revision', len(self.revisions))
        return secret


class _JujuState:
    """The mutable half of a :class:`Juju`.

    Held in one object rather than as plain attributes so that the identity
    fields (``name``, ``uuid``, ``type``, ``cloud_spec``) stay the only things
    set directly on ``Juju``; everything mutable during a test run lives
    here instead.
    """

    def __init__(self) -> None:
        self.apps: dict[str, App] = {}
        # One queue per application, in the order the applications were
        # deployed; settle() takes from them in turn (see Juju._next_queued).
        self.queues: dict[str, deque[_Queued]] = {}
        self.turn = 0
        self.trace: list[Dispatch] = []
        self.closed = False
        # Relation IDs are per model in Juju, so they are per Juju here,
        # rather than drawn from the process-wide counter Relation uses: that
        # keeps them the same however many other tests ran first.
        self.next_relation_id = 0
        self.integrations: dict[int, _Integration] = {}
        # Every dispatch under this Juju, counted from 0; it seeds the IDs of
        # the secrets a dispatch creates.
        self.dispatches = 0
        # The secret IDs that each unit, keyed by application name and unit
        # ID, can read because another application granted them.
        self.granted: dict[tuple[str, int], set[str]] = {}
        # For each secret ID, the highest revision that secret-remove has
        # been queued for.
        self.removable_revisions: dict[str, int] = {}
        # Secrets the test added, as a user would with juju add-secret, by ID.
        self.user_secrets: dict[str, _UserSecret] = {}
        # Units whose charm raised, keyed by application name and unit ID,
        # with what it raised. Like a unit in a failed hook in Juju, each
        # gets no more events: those queued for it wait in ``held``.
        self.failed: dict[tuple[str, int], _HookFailedError] = {}
        self.held: dict[tuple[str, int], list[_Queued]] = {}

    def pending(self) -> int:
        """How many entries are queued, across every application."""
        return sum(len(queue) for queue in self.queues.values())


class Juju:
    """A set of applications, driven with Juju-shaped operations.

    Construct it much as you would a :class:`~ops.testing.Model`::

        juju = testing.Juju(model_name='my-model', type='lxd')

    Applications are added with :meth:`deploy`, which returns an :class:`App`
    handle::

        web = juju.deploy('./charms/myapp', num_units=2)
        db = juju.deploy(testing.CharmSpec(MyDatabaseCharm, meta=DB_META))
        juju.integrate(web, db)

    Each operation queues the events Juju would emit for it. Nothing runs until
    :meth:`settle` drains the queue, along with whatever the charms' own
    changes cause on their peers and on related applications, so most tests
    are a sequence of operations, a ``settle()``, and then assertions::

        juju.config(web, {'log_level': 'debug'})
        juju.settle()
        assert web.leader.state.unit_status == testing.ActiveStatus('ready')

    Charms run in the test process unless deployed with ``isolated=True``,
    in which case the charm runs in a worker process of its own. Call
    :meth:`close`, or use ``Juju`` as a context manager, to tear down any
    worker processes.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        uuid: str | None = None,
        type: Literal['kubernetes', 'lxd'] = 'kubernetes',
        cloud_spec: CloudSpec | None = None,
    ) -> None:
        """Create a simulated Juju model, as ``juju add-model`` would.

        Args:
            model_name: The model name. Defaults to a random name, matching
                :class:`~ops.testing.Model`.
            uuid: A unique identifier for the model. Defaults to a fixed
                UUID, the same on every run, so that anything derived from it
                (such as secret IDs) is too.
            type: The type of Juju model: ``'kubernetes'`` or ``'lxd'``
                (machine). Every application deployed under this ``Juju``
                shares this substrate, which decides which form of
                :meth:`remove_unit` it accepts.
            cloud_spec: Cloud specification information, as in
                :class:`~ops.testing.Model`.
        """
        self.name = model_name if model_name is not None else _random_model_name()
        self.uuid = uuid if uuid is not None else _DEFAULT_MODEL_UUID
        self.type: Literal['kubernetes', 'lxd'] = type
        self.cloud_spec = cloud_spec
        self._state = _JujuState()
        # Every unit's filesystem root, created with the first unit.
        self._filesystems: pathlib.Path | None = None
        self._filesystems_finalizer: weakref.finalize[Any, Any] | None = None

    @property
    def apps(self) -> Mapping[str, App]:
        """Every application deployed under this ``Juju``, by name.

        A test or fixture can reach any application through this without
        keeping the handle :meth:`deploy` returned, and a :class:`Dispatch`
        leads back to its application with ``juju.apps[dispatch.app]``. It's
        read-only: applications are added with :meth:`deploy`.
        """
        return types.MappingProxyType(self._state.apps)

    # Internals
    def _as_model(self) -> Model:
        """This ``Juju``'s identity as a plain :class:`Model`.

        Unit states get this rather than ``self``: a ``State`` is data that
        may be serialised out to a worker process, and it should not carry a
        handle to the ``Juju`` instance that is driving it.
        """
        return Model(
            name=self.name,
            uuid=self.uuid,
            type=self.type,
            cloud_spec=self.cloud_spec,
        )

    def _filesystems_parent(self) -> pathlib.Path:
        """The directory that holds each unit's filesystem root, until :meth:`close`."""
        if self._filesystems is None:
            self._filesystems = pathlib.Path(tempfile.mkdtemp(prefix='ops-testing-juju-'))
            self._filesystems_finalizer = weakref.finalize(
                self, shutil.rmtree, str(self._filesystems), True
            )
        return self._filesystems

    def _check_open(self) -> None:
        if self._state.closed:
            raise JujuError('This Juju has been closed.')

    def _new_relation_id(self) -> int:
        relation_id = self._state.next_relation_id
        self._state.next_relation_id += 1
        return relation_id

    # Operations
    def deploy(
        self,
        charm: CharmSource,
        app: str | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        state_template: State | None = None,
        trust: bool = False,
        num_units: int = 1,
        isolated: bool = False,
        requirements: str | pathlib.Path | None = None,
        mocked: Mapping[str, Any] | None = None,
        juju_version: str = _DEFAULT_JUJU_VERSION,
    ) -> App:
        """Deploy a charm, as ``juju deploy`` would.

        The units' startup events are queued; call :meth:`settle` to run them.

        By default, the charm runs in the test process, like a ``Context``
        test. Charms deployed that way share one set of imported modules, so
        two that need different versions of the same package need
        ``isolated=True`` for one of them.

        Args:
            charm: Charm source: a path to a charm's source directory, a
                :class:`CharmSpec`, or a :class:`ops.CharmBase` subclass. A
                path given as a ``str`` must be relative, as with
                ``juju deploy``; pass a :class:`pathlib.Path` for an absolute
                path. A charm class is shorthand for a ``CharmSpec`` with no
                mocking, with its metadata loaded from the files beside its
                source, as ``Context`` does.
            app: Application name. Defaults to the charm's name from its
                metadata.
            config: Application configuration. Merged over the defaults
                declared in the charm's config options.
            state_template: The starting :class:`State` for every unit of the
                application, including units added later with
                :meth:`add_unit`. Use it for what ``Juju`` can't know, such as
                ``Exec`` mocks and layers on containers, storage, resources,
                secrets and stored state. ``Juju`` sets each unit's
                ``leader``, ``planned_units``, ``model``, ``config`` and
                ``relations`` itself, so the template may not set those:
                relations come from peer endpoints, and config from
                ``config=``.
            trust: Whether the application has Juju trust, as with
                ``juju deploy --trust``.
            num_units: How many units to deploy. Unit ``0`` is the leader.
            isolated: Run the charm in its own worker process, in a virtual
                environment built from the charm's declared dependencies, so
                that it can need different versions of packages from the test
                and from other charms. Only for a charm deployed from a path.
                The dependencies are found from the build plugin in the
                charm's ``charmcraft.yaml`` (``charm``, ``python`` or
                ``uv``), along with the dependency groups its mocking
                configuration names, and installed with ``uv``, which has to
                be on ``PATH``. The environment is cached under
                ``$XDG_CACHE_HOME/ops-testing`` (``~/.cache/ops-testing`` by
                default), so only the first deploy of a charm builds it. The
                charm uses the test's own ``ops`` and ``ops.testing``.

                * inline: error
                * isolated: ok
            requirements: A requirements file to build the isolated
                environment from instead of the charm's build plugin, for a
                charm whose dependencies can't be found that way. It is
                installed as it is, so it has to include whatever the charm's
                mocking needs too. A relative path is relative to the working
                directory, as for ``charm``. Only with ``isolated=True``.

                * inline: error
                * isolated: ok
            mocked: Keyword arguments for the charm's own mocking: the
                function the charm configures in its ``pyproject.toml`` under
                ``[tool.ops.testing.mocking]``, or a ``CharmSpec``'s
                ``mocking``. Use it to vary how the charm's mocks behave in
                this test. Only JSON values are accepted, because for an
                isolated charm this dict is all that is sent to it.
            juju_version: The Juju agent version to simulate.

        Returns:
            The :class:`App` handle for the new application.

        Raises:
            JujuError: if an application of this name already exists,
                ``num_units`` is not positive, the charm or the template is
                not accepted, ``isolated`` or ``requirements`` is given
                where it can't apply, the charm's mocking can't be set up
                with ``mocked``, or the isolated environment can't be built:
                ``uv`` isn't on ``PATH``, the build plugin isn't supported,
                the charm's ``uv.lock`` is out of date, or a requirement
                can't be resolved.
        """
        self._check_open()
        if isinstance(charm, (str, pathlib.Path)):
            if requirements is not None and not isolated:
                raise JujuError('requirements= is for an isolated charm; pass isolated=True too.')
        elif isolated or requirements is not None:
            raise JujuError(
                'isolated= and requirements= need a charm on disk: a charm class or '
                'CharmSpec runs in the test process. Deploy it from a path.'
            )
        return self._deploy(
            charm,
            app,
            config=config,
            state_template=state_template,
            trust=trust,
            num_units=num_units,
            mocked=mocked,
            juju_version=juju_version,
            isolated=isolated,
            requirements=requirements,
        )

    def _deploy(
        self,
        charm: CharmSource,
        app: str | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        state_template: State | None = None,
        trust: bool = False,
        num_units: int = 1,
        mocked: Mapping[str, Any] | None = None,
        juju_version: str = _DEFAULT_JUJU_VERSION,
        isolated: bool = False,
        requirements: str | pathlib.Path | None = None,
        python_executable: str | None = None,
        extra_sys_path: Sequence[str] = (),
    ) -> App:
        """Deploy a charm, running it in a worker process if asked to.

        :meth:`deploy` without the checks on its public arguments. With
        ``isolated``, the charm's environment is built (from ``requirements``
        if given) and the charm runs in a worker with its interpreter. With
        ``python_executable``, the charm runs in a worker process with that
        interpreter, and ``extra_sys_path`` prepended to the worker's
        ``sys.path``; the interpreter must have the same ``ops`` installed as
        the test.
        """
        self._check_open()
        if num_units < 1:
            raise JujuError(f'num_units must be at least 1, not {num_units}.')
        meta = _resolve_meta(charm, app)

        app_name = app or meta.get('name')
        if not app_name:
            raise JujuError('Could not determine the application name; pass app=.')
        if app_name in self._state.apps:
            raise JujuError(f'An application named {app_name!r} is already deployed.')

        new_app = App(
            self,
            app_name,
            charm,
            meta=meta,
            config=config,
            state_template=state_template,
            trust=trust,
            mocked=mocked,
            juju_version=juju_version,
        )
        if isolated:
            assert isinstance(charm, (str, pathlib.Path))
            environment = _environment.build(
                pathlib.Path(charm).absolute(),
                app_name,
                pathlib.Path(requirements).absolute() if requirements is not None else None,
            )
            new_app._run_in_worker(
                environment.python_executable, extra_sys_path, environment.python_path
            )
        elif python_executable is not None:
            new_app._run_in_worker(python_executable, extra_sys_path)
        else:
            new_app._run_in_process()
        self._state.apps[app_name] = new_app
        self._state.queues[app_name] = deque()

        for _ in range(num_units):
            self._add_unit(new_app)
        return new_app

    def add_unit(self, app: App) -> Unit:
        """Add a unit to an application, as ``juju add-unit`` would.

        The new unit's startup events are queued. If the application has a
        peer relation, the units that were already there see the new unit join
        it. If it is related to other applications, the new unit gets
        ``relation-created`` for each relation after ``install``, and once it
        has started, ``relation-joined`` and ``relation-changed`` for each unit
        on the other side. Those units see the new unit join too.

        Returns:
            The new :class:`Unit`.
        """
        self._check_open()
        return self._add_unit(app)

    def remove_unit(self, *app_or_unit: App | Unit, num_units: int = 0) -> None:
        """Remove units from an application, as ``juju remove-unit`` would.

        Removal differs by substrate, and this API matches Juju in exposing one
        method for both rather than splitting it into two:

        - **Kubernetes**: units are fungible, so scale down by count. Pass one
          or more :class:`App` objects and a positive ``num_units``; the
          highest-numbered units are removed from each::

              juju.remove_unit(web, num_units=2)

        - **Machine**: units are individually addressable, so name them. Pass
          one or more :class:`Unit` objects, and no ``num_units``::

              juju.remove_unit(web.units[1])
              juju.remove_unit(u2, u3)

        Which form applies is decided by the application's substrate
        (:attr:`Juju.type`), not by which arguments happen to be passed: the
        wrong form for the substrate is rejected with a message naming the
        right one.

        Every unit related to a removed unit, peers and units of related
        applications alike, sees ``relation-departed`` for it. The removed
        unit gets ``relation-departed`` for each unit it is related to and
        ``relation-broken`` for each of its relations, then ``stop`` and
        ``remove``.

        Args:
            *app_or_unit: For the Kubernetes form, one or more :class:`App`
                objects. For the machine form, one or more :class:`Unit`
                objects, which need not all belong to the same application.
            num_units: How many units to remove from each application.
                Kubernetes form only; leave unset for the machine form.

        Raises:
            JujuError: if the arguments don't match the substrate's form for
                the applications involved, if removing them would leave an
                application with no units, or if a named unit is the leader.
        """
        self._check_open()
        if not app_or_unit:
            raise JujuError('remove_unit() requires at least one App or Unit.')
        is_apps = [isinstance(item, App) for item in app_or_unit]
        if any(is_apps) and not all(is_apps):
            raise JujuError(
                'remove_unit() takes either App objects (with num_units=, for '
                'Kubernetes) or Unit objects (for machine), not a mix.'
            )

        if is_apps[0]:
            apps = cast('tuple[App, ...]', app_or_unit)
            if num_units < 1:
                raise JujuError(
                    'remove_unit() with App objects scales down by count; pass a '
                    'positive num_units=.'
                )
            for app in apps:
                if app._substrate != 'kubernetes':
                    raise JujuError(
                        f'{app.name} is on a {app._substrate} substrate, which '
                        f'addresses units by name, not count. Use '
                        f'remove_unit(unit) or remove_unit(u1, u2, ...) instead.'
                    )
                if num_units >= len(app._units):
                    raise JujuError(
                        f'Cannot remove {num_units} units from {app.name}; it has '
                        f'only {len(app._units)}. Remove the application instead '
                        'to take it down entirely.'
                    )
            for app in apps:
                doomed_ids = set(sorted(app._units, reverse=True)[:num_units])
                for unit_id in sorted(doomed_ids, reverse=True):
                    self._remove_one_unit(app, unit_id, doomed_ids)
            return

        if num_units:
            raise JujuError(
                'num_units= is only valid with App objects (the Kubernetes '
                'scale-down form); pass Unit objects to remove named units.'
            )
        units = cast('tuple[Unit, ...]', app_or_unit)
        by_app: dict[App, list[Unit]] = {}
        for unit in units:
            by_app.setdefault(unit.app, []).append(unit)
        for app, doomed in by_app.items():
            if app._substrate == 'kubernetes':
                raise JujuError(
                    f'{app.name} is on a kubernetes substrate, which is scaled '
                    f'down by count, not by naming units. Use '
                    f'remove_unit({app.name}, num_units=...) instead.'
                )
            for unit in doomed:
                if unit.id not in app._units:
                    raise JujuError(f'{unit.name} is not part of {app.name} (already removed?).')
            if len(app._units) - len(doomed) < 1:
                raise JujuError(
                    f'Cannot remove the last unit of {app.name}; remove the application instead.'
                )
            for unit in doomed:
                if unit.id == app._leader_id:
                    raise JujuError(
                        f'Cannot remove {unit.name}: it is the leader, and this layer '
                        'does not elect a new one yet. Remove a non-leader unit, or '
                        'remove the application entirely.'
                    )
        for app, doomed in by_app.items():
            doomed_ids = {u.id for u in doomed}
            for unit in doomed:
                self._remove_one_unit(app, unit.id, doomed_ids)

    def _remove_one_unit(self, app: App, unit_id: int, doomed_ids: Collection[int] = ()) -> None:
        """Enqueue the departure and teardown sequence for one unit.

        Every unit related to the departing one sees it depart, peers and
        units of integrated applications alike. The departing unit then
        leaves each of its relations (``relation-departed`` for each unit on
        the other side, then ``relation-broken``), and is stopped and removed.

        Args:
            app: The application the unit belongs to.
            unit_id: The unit being removed.
            doomed_ids: Other unit IDs being removed in the same batch (see
                :meth:`remove_unit`). A peer in this set doesn't get a
                ``relation_departed`` enqueued for *this* departure: it is
                leaving too, and gets its own teardown instead, which also
                avoids queueing an event for a unit that may already be gone
                by the time this one dispatches.
        """
        departing = app._units[unit_id]
        remaining = [u for u in app._live_units if u is not departing and u.id not in doomed_ids]
        app._dying.add(unit_id)

        # The peers see the unit leave before it is torn down.
        for endpoint in app._peer_endpoints:
            for peer in remaining:
                relation = _peer_relation(peer._state, endpoint)
                if relation is not None:
                    self._enqueue(
                        app,
                        peer.id,
                        _Event(
                            f'{endpoint}_relation_departed',
                            relation=relation,
                            relation_remote_unit_id=unit_id,
                            relation_departed_unit_id=unit_id,
                        ),
                        rebind=_Rebind('relation', relation.id),
                    )
        # So do the units on the other side of each integration that have
        # seen it join.
        for integration in self._integrations_of(app):
            remote = integration.remote(app).app
            for other in remote._live_units:
                view = _relation_by_id(other._state, integration.id)
                if isinstance(view, Relation) and unit_id in view.remote_units_data:
                    self._enqueue(
                        remote,
                        other.id,
                        _Event(
                            f'{view.endpoint}_relation_departed',
                            relation=view,
                            relation_remote_unit_id=unit_id,
                            relation_departed_unit_id=unit_id,
                        ),
                        rebind=_Rebind('relation', view.id),
                    )
        for relation in sorted(departing._state.relations, key=lambda r: r.id):
            self._enqueue_leaving(app, departing, relation, departing_unit_id=unit_id)
        for event, rebind in _teardown_events(app, unit_id):
            self._enqueue(app, unit_id, event, rebind)

        self._state.queues[app.name].append(_Queued(app, unit_id, _RemoveUnit()))

    def _enqueue_leaving(
        self,
        app: App,
        unit: Unit,
        relation: RelationBase,
        *,
        departing_unit_id: int | None = None,
    ) -> None:
        """Enqueue a unit's side of leaving a relation.

        ``relation-departed`` for each unit on the other side that it has seen
        join, then ``relation-broken``. ``departing_unit_id`` is the unit
        being removed, if a unit removal is why it is leaving; for a removed
        relation, each remote unit is the departing one.
        """
        for remote_id in sorted(_remote_ids(relation)):
            self._enqueue(
                app,
                unit.id,
                _Event(
                    f'{relation.endpoint}_relation_departed',
                    relation=relation,
                    relation_remote_unit_id=remote_id,
                    relation_departed_unit_id=(
                        remote_id if departing_unit_id is None else departing_unit_id
                    ),
                ),
                rebind=_Rebind('relation', relation.id),
            )
        self._enqueue(
            app,
            unit.id,
            _Event(f'{relation.endpoint}_relation_broken', relation=relation),
            rebind=_Rebind('relation', relation.id),
        )

    # User secrets
    def add_secret(self, name: str, content: Mapping[str, Any], *, info: str | None = None) -> str:
        """Add a secret as a user would, matching ``juju add-secret``.

        The secret isn't visible to any application until it's granted with
        :meth:`grant_secret`. It's usually passed to a charm through a config
        option of type ``secret``.

        Args:
            name: The secret's name, unique in the model.
            content: The secret's content.
            info: A description of the secret.

        Returns:
            The secret's URI. It's derived from the model UUID and how many
            secrets the test has added, so it's the same on every run.

        Raises:
            JujuError: if the name is already used, or the content is empty.
        """
        self._check_open()
        if any(s.name == name for s in self._state.user_secrets.values()):
            raise JujuError(f'A secret named {name!r} already exists.')
        if not content:
            raise JujuError('A secret needs some content.')
        with _secret_ids(f'{self.uuid}/user-secrets/{len(self._state.user_secrets)}'):
            secret_id = _state_module._generate_secret_id()
        self._state.user_secrets[secret_id] = _UserSecret(
            secret_id, name, info, [{k: str(v) for k, v in content.items()}]
        )
        return secret_id

    def grant_secret(self, identifier: str, app: App | Iterable[App]) -> None:
        """Let an application read a secret the test added, matching ``juju grant-secret``.

        Args:
            identifier: The secret's URI or name.
            app: The application, or applications, to grant it to.
        """
        self._check_open()
        secret = self._user_secret(identifier)
        apps = [app] if isinstance(app, App) else list(app)
        for each in apps:
            self._check_deployed(each)
        secret.grants.update(each.name for each in apps)
        self._sync_secrets()

    def update_secret(self, identifier: str, content: Mapping[str, str]) -> None:
        """Add a new revision of a secret the test added, matching ``juju update-secret``.

        Each unit tracking the secret gets ``secret-changed``.

        Args:
            identifier: The secret's URI or name.
            content: The new content.
        """
        self._check_open()
        secret = self._user_secret(identifier)
        if not content:
            raise JujuError('A secret needs some content.')
        secret.revisions.append({k: str(v) for k, v in content.items()})
        self._sync_secrets()

    def remove_secret(self, identifier: str) -> None:
        """Remove a secret the test added, matching ``juju remove-secret``.

        It's removed from every unit's :class:`State`.

        Args:
            identifier: The secret's URI or name.
        """
        self._check_open()
        secret = self._user_secret(identifier)
        del self._state.user_secrets[secret.id]
        self._sync_secrets()

    def _user_secret(self, identifier: str) -> _UserSecret:
        for secret in self._state.user_secrets.values():
            if identifier in (secret.id, secret.name):
                return secret
        raise JujuError(f'No secret {identifier!r} was added with add_secret().')

    def config(self, app: App, config: Mapping[str, Any]) -> None:
        """Change an application's configuration, as ``juju config`` would.

        The new values are merged over the existing ones, and
        ``config-changed`` is queued for every unit.
        """
        self._check_open()
        app._config.update(config)
        for unit in app.units:
            unit._state = dataclasses.replace(unit._state, config=dict(app._config))
            self._enqueue(app, unit.id, _Event('config_changed'))

    def integrate(self, app1: App | tuple[App, str], app2: App | tuple[App, str]) -> None:
        """Relate two applications, as ``juju integrate`` would.

        Where only one pair of endpoints can match, the applications are
        enough::

            juju.integrate(web, db)

        Otherwise, name the endpoint on either side with an ``(App, endpoint)``
        tuple, the object form of ``juju integrate web:db db:database``::

            juju.integrate((web, 'db'), db)

        Every unit on both sides gets the relation in its :class:`State`, and
        ``relation-created`` is queued on each unit, then ``relation-joined``
        and ``relation-changed`` on each unit for each unit on the other side.

        Either side can be a charm on disk or a :class:`CharmSpec`, running in
        the test process or in a worker process.

        Raises:
            JujuError: if no pair of endpoints matches, more than one does,
                or the two are already related through that pair. The
                message names the candidates.
        """
        self._check_open()
        end1, end2, interface = self._match_endpoints(app1, app2)
        integration = _Integration(self._new_relation_id(), (end1, end2), interface)
        self._state.integrations[integration.id] = integration

        for end in integration.ends:
            for unit in end.app._live_units:
                view = self._relation_view(integration, end.app)
                unit._state = _with_relation(unit._state, view)
        for end in integration.ends:
            for unit in end.app._live_units:
                view = _relation_by_id(unit._state, integration.id)
                self._enqueue(
                    end.app,
                    unit.id,
                    _Event(f'{end.endpoint}_relation_created', relation=view),
                    rebind=_Rebind('relation', integration.id),
                )
        for end in integration.ends:
            remote = integration.remote(end.app).app
            for unit in end.app._live_units:
                for other in remote._live_units:
                    self._enqueue_joined(end.app, unit, integration, other.id)

    def remove_relation(self, app1: App | tuple[App, str], app2: App | tuple[App, str]) -> None:
        """Remove a relation between two applications, as ``juju remove-relation`` would.

        Takes the applications, or ``(App, endpoint)`` tuples, the same way as
        :meth:`integrate`. Each unit sees ``relation-departed`` for each unit
        on the other side, then ``relation-broken``; the relation leaves a
        unit's :class:`State` once ``relation-broken`` has been dispatched
        there. Secrets granted over the relation stop being readable on the
        other side when it is removed.

        Raises:
            JujuError: if the applications aren't related through a matching
                pair of endpoints, or are related through more than one.
        """
        self._check_open()
        (a1, ep1), (a2, ep2) = _split_end(app1), _split_end(app2)
        self._check_deployed(a1)
        self._check_deployed(a2)
        candidates: list[_Integration] = []
        for integration in self._state.integrations.values():
            for (x, ex), (y, ey) in (integration.ends, integration.ends[::-1]):
                if (
                    x is a1
                    and y is a2
                    and ep1 in (None, ex)
                    and ep2 in (None, ey)
                    and integration not in candidates
                ):
                    candidates.append(integration)
        if len(candidates) != 1:
            related = [str(i) for i in self._state.integrations.values()]
            if not candidates:
                raise JujuError(
                    f'{_end_name(a1, ep1)} and {_end_name(a2, ep2)} are not related. '
                    f'Relations: {", ".join(related) or "none"}.'
                )
            raise JujuError(
                f'{a1.name} and {a2.name} are related more than once: '
                f'{", ".join(str(c) for c in candidates)}. Pass (App, endpoint) tuples '
                'to say which relation to remove.'
            )
        integration = candidates[0]
        # From here on, nothing new is carried across, and no unit joins.
        del self._state.integrations[integration.id]
        for end in integration.ends:
            for unit in end.app._live_units:
                view = _relation_by_id(unit._state, integration.id)
                if view is not None:
                    self._enqueue_leaving(end.app, unit, view)

    def _match_endpoints(
        self, app1: App | tuple[App, str], app2: App | tuple[App, str]
    ) -> tuple[_End, _End, str]:
        """Find the one pair of endpoints that ``integrate`` should relate."""
        (a1, ep1), (a2, ep2) = _split_end(app1), _split_end(app2)
        self._check_deployed(a1)
        self._check_deployed(a2)
        if a1 is a2:
            raise JujuError(f'Cannot relate {a1.name} to itself; use a peer endpoint for that.')
        endpoints1, endpoints2 = a1._relation_endpoints, a2._relation_endpoints
        for app, endpoint, endpoints in ((a1, ep1, endpoints1), (a2, ep2, endpoints2)):
            if endpoint is not None and endpoint not in endpoints:
                raise JujuError(
                    f'{app.name} has no provides or requires endpoint {endpoint!r}. '
                    f'It has: {_list_endpoints(app)}.'
                )
        pairs: list[tuple[str, str, str]] = []
        for e1, (role1, spec1) in sorted(endpoints1.items()):
            if ep1 is not None and e1 != ep1:
                continue
            for e2, (role2, spec2) in sorted(endpoints2.items()):
                if ep2 is not None and e2 != ep2:
                    continue
                if role1 != role2 and spec1.get('interface') == spec2.get('interface'):
                    pairs.append((e1, e2, cast('str', spec1.get('interface'))))
        if not pairs:
            raise JujuError(
                f'No endpoints of {_end_name(a1, ep1)} and {_end_name(a2, ep2)} match: '
                'a provides endpoint relates to a requires endpoint with the same '
                f'interface. {a1.name} has: {_list_endpoints(a1)}; '
                f'{a2.name} has: {_list_endpoints(a2)}.'
            )
        if len(pairs) > 1:
            candidates = ', '.join(f'{a1.name}:{e1} {a2.name}:{e2}' for e1, e2, _ in pairs)
            raise JujuError(
                f'{a1.name} and {a2.name} can be related more than one way: {candidates}. '
                'Pass (App, endpoint) tuples to choose.'
            )
        e1, e2, interface = pairs[0]
        for app, endpoint in ((a1, e1), (a2, e2)):
            if app._relation_endpoints[endpoint][1].get('scope') == 'container':
                raise JujuError(
                    f'{app.name}:{endpoint} is a subordinate (container-scoped) endpoint, '
                    'and Juju here does not deploy subordinates.'
                )
        for integration in self._state.integrations.values():
            if {(end.app.name, end.endpoint) for end in integration.ends} == {
                (a1.name, e1),
                (a2.name, e2),
            }:
                raise JujuError(f'{a1.name}:{e1} and {a2.name}:{e2} are already related.')
        return _End(a1, e1), _End(a2, e2), interface

    def _check_deployed(self, app: App) -> None:
        if self._state.apps.get(app.name) is not app:
            raise JujuError(f'{app.name} is not deployed in this Juju.')

    def _integrations_of(self, app: App) -> list[_Integration]:
        """The live relations between ``app`` and other applications, by ID."""
        return [
            integration
            for _, integration in sorted(self._state.integrations.items())
            if any(end.app is app for end in integration.ends)
        ]

    def _relation_view(self, integration: _Integration, app: App) -> Relation:
        """A new unit's view of a relation: no remote units joined yet.

        The application databags on both sides are already readable, so they
        start as each side's leader has them.
        """
        local = integration.local(app)
        remote = integration.remote(app)
        return Relation(
            endpoint=local.endpoint,
            interface=integration.interface,
            id=integration.id,
            local_app_data=_app_databag(app, integration.id),
            remote_app_name=remote.app.name,
            remote_app_data=_app_databag(remote.app, integration.id),
            remote_units_data={},
        )

    def _enqueue_joined(
        self, app: App, unit: Unit, integration: _Integration, remote_id: int
    ) -> None:
        """Queue ``relation-joined`` then ``relation-changed`` for one remote unit.

        The remote unit's databag is added to this unit's view when the
        ``relation-joined`` is dispatched, not before.
        """
        endpoint = integration.local(app).endpoint
        view = _relation_by_id(unit._state, integration.id)
        for suffix in ('relation_joined', 'relation_changed'):
            self._enqueue(
                app,
                unit.id,
                _Event(
                    f'{endpoint}_{suffix}',
                    relation=view,
                    relation_remote_unit_id=remote_id,
                ),
                rebind=_Rebind('relation', integration.id),
            )

    # Convergence
    def settle(self) -> list[Dispatch]:
        """Dispatch queued events until the model converges.

        Convergence is reached when the queue is empty: every event produced
        by an operation has been dispatched, along with every follow-on event
        those dispatches produced.

        The order events are dispatched in is fixed, so the same operations
        on the same starting states settle the same way on every run. That
        relies on the charms being deterministic too: a charm that, for
        example, writes the current time to a databag ends up in a different
        :class:`State` each run, though the order of dispatch is the same.

        Returns:
            The events dispatched, in order, each with the unit it went to and
            that unit's :class:`State` afterwards.

        If a charm raises, the unit goes into error status, as it would in
        Juju after a failed hook: the charm's changes from that dispatch are
        discarded, and the unit gets no more events. There's no automatic
        retry. The rest of the model carries on settling, and once it has,
        this raises :class:`JujuError`. The model is left as it is, so a test
        that expects the failure can catch the error and assert on the unit's
        status.

        Raises:
            JujuError: if the model does not converge. The same event reaching
                the same unit with the same :class:`State` twice is a loop,
                and is reported as soon as it happens; otherwise, the limit
                grows with the number of units in the model. The message ends
                with the last events dispatched. Also raised, after settling,
                if a charm raised: the message names each unit put in error,
                with the charm's traceback, and the error is chained to the
                first exception.
        """
        self._check_open()
        self._state.trace = []
        already_failed = set(self._state.failed)
        seen: set[tuple[str, int, str, str]] = set()
        dispatched = 0
        while (queued := self._next_queued()) is not None:
            limit = _SETTLE_DISPATCHES_PER_UNIT * max(1, self._unit_count())
            if dispatched >= limit:
                raise JujuError(
                    f'Did not converge after {dispatched} events. Check for charms '
                    'that write a new value to a databag on every event.'
                    f'{self._trace_tail()}'
                )
            entry = self._next_dispatch(queued)
            if entry is None:
                continue
            app, unit, event = entry
            key = _dispatch_key(app, unit, event)
            if key is not None:
                if key in seen:
                    raise JujuError(
                        f'{unit.name} was about to handle {event.name} with the same '
                        'State it had the last time it handled it, so settling would '
                        f'never end.{self._trace_tail()}'
                    )
                seen.add(key)
            self._dispatch(app, unit, event)
            dispatched += 1
        failed = [(k, e) for k, e in self._state.failed.items() if k not in already_failed]
        if failed:
            details = '\n'.join(f'{app}/{unit_id}:\n{e.traceback}' for (app, unit_id), e in failed)
            raise JujuError(
                f'{len(failed)} unit(s) went into error status while settling.\n{details}'
            ) from (failed[0][1].cause or failed[0][1])
        return list(self._state.trace)

    def _unit_count(self) -> int:
        count = 0
        for app in self._state.apps.values():
            count += len(app._units)
        return count

    def _trace_tail(self) -> str:
        tail = self._state.trace[-_TRACE_TAIL:]
        lines = [f'  {d.event.name} on {d.unit_name}' for d in tail]
        return '\nLast events dispatched:\n' + '\n'.join(lines) if lines else ''

    def _next_queued(self) -> _Queued | None:
        """Take the next entry from the queues, one application at a time.

        Each application has its own queue, which runs in order, and the
        applications take turns, in the order they were deployed, one entry
        each. Within one application that is the order the events were
        caused in; across applications it is the closest a single thread gets
        to Juju running every unit agent at once.
        """
        queues = list(self._state.queues.values())
        for offset in range(len(queues)):
            index = (self._state.turn + offset) % len(queues)
            if queues[index]:
                self._state.turn = index + 1
                return queues[index].popleft()
        return None

    def _next_dispatch(self, queued: _Queued) -> tuple[App, Unit, _Event] | None:
        """Prepare a queue entry for dispatch, returning the charm invocation it holds, if any.

        A ``_RemoveUnit`` marker carries no charm invocation of its own, and
        neither does an event that no longer applies by the time its turn
        comes (its relation or container went away, or the remote unit it is
        about left first). Those are consumed here, and ``None`` is returned.
        """
        app, unit_id, event, rebind = queued
        if (app.name, unit_id) in self._state.failed:
            self._state.held.setdefault((app.name, unit_id), []).append(queued)
            return None
        if isinstance(event, _RemoveUnit):
            del app._units[unit_id]
            app._dying.discard(unit_id)
            self._drop_peer(app, unit_id)
            self._state.granted.pop((app.name, unit_id), None)
            self._sync_secrets()
            return None
        unit = app._units.get(unit_id)
        if unit is None:
            return None
        if (
            rebind is not None
            and rebind.kind == 'relation'
            and not self._enter_relation_event(app, unit, event, cast('int', rebind.key))
        ):
            return None
        if rebind is not None and rebind.kind == 'secret' and event.secret_revision is not None:
            _stop_owner_tracking(unit, cast('str', rebind.key), event.secret_revision)
        if rebind is not None:
            rebound = _rebind(unit._state, rebind)
            if rebound is None:
                # Whatever the event was about went away between queueing and
                # dispatch; there is nothing left for the charm to observe.
                return None
            event = dataclasses.replace(event, **{rebind.kind: rebound})
        return app, unit, event

    def _enter_relation_event(self, app: App, unit: Unit, event: _Event, relation_id: int) -> bool:
        """Bring a unit's view of a relation up to the moment of a relation event.

        Returns whether the event should still be dispatched. Juju changes
        which remote units a unit sees as the relation events reach it: a
        remote unit is in the relation from its ``relation-joined`` on, and
        out of it from its ``relation-departed``, and by ``relation-broken``
        none are left.
        """
        view = _relation_by_id(unit._state, relation_id)
        if view is None:
            return False
        name = event.name
        remote_id = event.relation_remote_unit_id
        if name.endswith('_relation_departed'):
            if remote_id is not None:
                unit._state = _with_relation(unit._state, _without_remote(view, remote_id))
            return True
        if name.endswith('_relation_broken'):
            unit._state = _with_relation(unit._state, _without_remote(view, None))
            return True
        if isinstance(view, PeerRelation):
            return unit.id not in app._dying
        if not isinstance(view, Relation):
            return True
        integration = self._state.integrations.get(relation_id)
        if name.endswith('_relation_created'):
            return True
        # The relation is being removed, or this unit is leaving it.
        if integration is None or unit.id in app._dying:
            return False
        remote_app = integration.remote(app).app
        if name.endswith('_relation_joined'):
            assert remote_id is not None
            remote_unit = remote_app._units.get(remote_id)
            if remote_unit is None or remote_id in remote_app._dying:
                return False
            remote_view = _relation_by_id(remote_unit._state, relation_id)
            data = dict(remote_view.local_unit_data) if remote_view is not None else {}
            units_data = {**view.remote_units_data, remote_id: data}
            unit._state = _with_relation(
                unit._state,
                dataclasses.replace(view, remote_units_data=dict(sorted(units_data.items()))),
            )
            return True
        if name.endswith('_relation_changed'):
            return remote_id is None or remote_id in view.remote_units_data
        return True

    def _dispatch(self, app: App, unit: Unit, event: _Event) -> None:
        state_in = unit._state
        seed = f'{self.uuid}/{unit.name}/{self._state.dispatches}'
        self._state.dispatches += 1
        try:
            state_out = app._runner.run(unit.id, event, state_in, seed)
        except _HookFailedError as e:
            self._fail(app, unit, event, e, state_in)
            return
        if not unit.is_leader:
            _check_no_app_data_writes(unit, state_in, state_out)
        unit._state = state_out
        self._state.trace.append(
            Dispatch(event, app.name, unit.id, state_in, state_out, _charm=app._for_context())
        )
        if event.name.endswith('_relation_broken') and event.relation is not None:
            self._leave_relation(unit, event.relation.id)
        if unit.id in app._dying:
            # A unit that is leaving doesn't publish anything new.
            return
        self._propagate_peers(app, unit)
        self._propagate_relations(app, unit)
        self._propagate_app_secrets(app, unit)
        self._propagate_app_status(app, unit)
        self._sync_secrets()

    def _fail(
        self, app: App, unit: Unit, event: _Event, error: _HookFailedError, state_in: State
    ) -> None:
        """Put a unit in error after its charm raised, as Juju does for a failed hook.

        Nothing the charm did in the dispatch is kept, and nothing is
        propagated to other units.
        """
        hook = event.name.replace('_', '-')
        unit._state = dataclasses.replace(
            unit._state, unit_status=ErrorStatus(f'hook failed: "{hook}"')
        )
        self._state.failed[app.name, unit.id] = error
        self._state.trace.append(
            Dispatch(
                event,
                app.name,
                unit.id,
                state_in,
                unit._state,
                error.traceback,
                _charm=app._for_context(),
            )
        )

    def _leave_relation(self, unit: Unit, relation_id: int) -> None:
        """Take a relation out of a unit's state after its ``relation-broken``.

        Grants of the unit's own secrets over the relation go with it, as
        they do in Juju.
        """
        state = _without_relation(unit._state, relation_id)
        secrets: list[Secret] = []
        for secret in state.secrets:
            if relation_id in secret.remote_grants:
                grants = {k: v for k, v in secret.remote_grants.items() if k != relation_id}
                secret = _replace_secret(secret, remote_grants=grants)
            secrets.append(secret)
        unit._state = dataclasses.replace(state, secrets=frozenset(secrets))

    # Units and peer relations
    def _add_unit(self, app: App) -> Unit:
        unit_id = app._next_unit_id
        app._next_unit_id += 1
        existing = app._live_units

        app._make_unit_root(unit_id)
        unit = Unit(app, unit_id, self._initial_state(app, unit_id))
        app._units[unit_id] = unit

        # Everyone's planned-units count moves as soon as the unit exists.
        for other in app.units:
            other._state = dataclasses.replace(other._state, planned_units=len(app._units))

        # Existing units see the newcomer join the peer relation before the
        # newcomer itself starts up. Juju follows relation-joined with
        # relation-changed for that unit, because the databag Juju itself
        # populates (the unit's addresses) becomes visible at the same moment.
        for endpoint in app._peer_endpoints:
            joining = _peer_relation(unit._state, endpoint)
            for peer in existing:
                relation = _peer_relation(peer._state, endpoint)
                if relation is None:
                    continue
                peers_data = dict(relation.peers_data)
                peers_data[unit_id] = dict(joining.local_unit_data) if joining else {}
                peer._state = _with_relation(
                    peer._state,
                    dataclasses.replace(relation, peers_data=peers_data),
                )
                for suffix in ('relation_joined', 'relation_changed'):
                    self._enqueue(
                        app,
                        peer.id,
                        _Event(
                            f'{endpoint}_{suffix}',
                            relation=relation,
                            relation_remote_unit_id=unit_id,
                        ),
                        rebind=_Rebind('relation', relation.id),
                    )

        # The units of related applications see the newcomer join too.
        integrations = self._integrations_of(app)
        for integration in integrations:
            remote = integration.remote(app).app
            for other in remote._live_units:
                self._enqueue_joined(remote, other, integration, unit_id)

        # The new unit enters its relations to other applications right after
        # install, as Juju runs relation-created before the unit starts, and
        # sees the remote units join once it has started.
        startup = _startup_events(app, unit_id)
        created = [
            (
                _Event(
                    f'{integration.local(app).endpoint}_relation_created',
                    relation=_relation_by_id(unit._state, integration.id),
                ),
                _Rebind('relation', integration.id),
            )
            for integration in integrations
        ]
        startup[1:1] = created
        for event, rebind in startup:
            self._enqueue(app, unit_id, event, rebind)
        for integration in integrations:
            for other in integration.remote(app).app._live_units:
                self._enqueue_joined(app, unit, integration, other.id)
        return unit

    def _initial_state(self, app: App, unit_id: int) -> State:
        relations: list[RelationBase] = []
        for endpoint in app._peer_endpoints:
            # A unit joining an existing application can read what its peers
            # have already published, so seed its view from theirs rather than
            # starting it empty; otherwise the first peer to run afterwards
            # looks like it just wrote data it had published long before.
            peers_data: dict[int, RawDataBagContents] = {}
            app_data: RawDataBagContents = {}
            for peer_id, peer in app._units.items():
                if peer_id == unit_id or peer_id in app._dying:
                    continue
                existing = _peer_relation(peer._state, endpoint)
                if existing is None:
                    continue
                peers_data[peer_id] = dict(existing.local_unit_data)
                if peer_id == app._leader_id:
                    app_data = dict(existing.local_app_data)
            relations.append(
                PeerRelation(
                    endpoint=endpoint,
                    id=app._peer_ids[endpoint],
                    local_app_data=app_data,
                    peers_data=peers_data,
                )
            )
        for integration in self._integrations_of(app):
            relations.append(self._relation_view(integration, app))
        secrets = list(app._state_template.secrets)
        if app._units:
            # The application's own secrets are visible to every unit, so a
            # new unit sees the ones the application already has.
            template_ids = {s.id for s in secrets}
            leader_state = app._units[app._leader_id]._state
            for secret in leader_state.secrets:
                if secret.owner == 'app' and secret.id not in template_ids:
                    secrets.append(secret)
        return dataclasses.replace(
            app._state_template,
            config=dict(app._config),
            relations=frozenset(relations),
            containers=frozenset(app._containers()),
            secrets=frozenset(secrets),
            leader=unit_id == app._leader_id,
            model=self._as_model(),
            planned_units=len(app._units) + 1,
        )

    def _drop_peer(self, app: App, unit_id: int) -> None:
        """Remove a departed unit from its peers' views of the peer relation."""
        for endpoint in app._peer_endpoints:
            for peer in app.units:
                relation = _peer_relation(peer._state, endpoint)
                if relation is None or unit_id not in relation.peers_data:
                    continue
                peers_data = dict(relation.peers_data)
                del peers_data[unit_id]
                peer._state = _with_relation(
                    peer._state,
                    dataclasses.replace(relation, peers_data=peers_data),
                )
        for peer in app.units:
            peer._state = dataclasses.replace(peer._state, planned_units=len(app._units))

    def _propagate_peers(self, app: App, source: Unit) -> None:
        """Publish a unit's peer databag writes to the rest of the application.

        A peer relation is one relation with one set of databags, but each
        unit holds its own view of it. After a unit runs, whatever it wrote to
        its own unit databag (or, as leader, to the application databag) has to
        appear in every other unit's view, and any unit whose view actually
        changed sees ``relation-changed``, which is what makes a multi-unit
        application converge rather than just run its events once.
        """
        for endpoint in app._peer_endpoints:
            source_relation = _peer_relation(source._state, endpoint)
            if source_relation is None:
                continue
            for peer in app._live_units:
                if peer is source:
                    continue
                relation = _peer_relation(peer._state, endpoint)
                if relation is None:
                    continue
                peers_data = dict(relation.peers_data)
                changed = False
                if peers_data.get(source.id) != source_relation.local_unit_data:
                    peers_data[source.id] = dict(source_relation.local_unit_data)
                    changed = True
                app_data: RawDataBagContents = relation.local_app_data
                if source.is_leader and app_data != source_relation.local_app_data:
                    app_data = dict(source_relation.local_app_data)
                    changed = True
                if not changed:
                    continue
                peer._state = _with_relation(
                    peer._state,
                    dataclasses.replace(
                        relation,
                        peers_data=peers_data,
                        local_app_data=app_data,
                    ),
                )
                self._enqueue(
                    app,
                    peer.id,
                    _Event(
                        f'{endpoint}_relation_changed',
                        relation=relation,
                        relation_remote_unit_id=source.id,
                    ),
                    rebind=_Rebind('relation', relation.id),
                )

    def _propagate_relations(self, app: App, source: Unit) -> None:
        """Carry a unit's relation databag writes to the other side of each relation.

        What the unit wrote to its own databag reaches each remote unit that
        has seen it join, and what the leader wrote to the application
        databag reaches every remote unit, and the rest of its own
        application. A remote unit whose view changed sees
        ``relation-changed``; its own application's units see nothing, as in
        Juju.
        """
        for integration in self._integrations_of(app):
            relation = _relation_by_id(source._state, integration.id)
            if not isinstance(relation, Relation):
                continue
            if source.is_leader:
                for peer in app._live_units:
                    view = _relation_by_id(peer._state, integration.id)
                    if (
                        peer is not source
                        and isinstance(view, Relation)
                        and view.local_app_data != relation.local_app_data
                    ):
                        peer._state = _with_relation(
                            peer._state,
                            dataclasses.replace(
                                view, local_app_data=dict(relation.local_app_data)
                            ),
                        )
            remote = integration.remote(app).app
            for other in remote._live_units:
                view = _relation_by_id(other._state, integration.id)
                if not isinstance(view, Relation):
                    continue
                changes: dict[str, Any] = {}
                about: int | None = None
                units_data = view.remote_units_data
                if source.id in units_data and units_data[source.id] != relation.local_unit_data:
                    changes['remote_units_data'] = {
                        **units_data,
                        source.id: dict(relation.local_unit_data),
                    }
                    about = source.id
                if source.is_leader and view.remote_app_data != relation.local_app_data:
                    changes['remote_app_data'] = dict(relation.local_app_data)
                    if about is None and units_data:
                        # Juju fires relation-changed for an application
                        # databag change with no remote unit, which a
                        # Scenario event can't express. It names the writer
                        # if this unit has seen it join, or else the
                        # lowest-numbered unit it has seen.
                        about = source.id if source.id in units_data else min(units_data)
                if not changes:
                    continue
                other._state = _with_relation(other._state, dataclasses.replace(view, **changes))
                if about is not None:
                    self._enqueue(
                        remote,
                        other.id,
                        _Event(
                            f'{view.endpoint}_relation_changed',
                            relation=view,
                            relation_remote_unit_id=about,
                        ),
                        rebind=_Rebind('relation', integration.id),
                    )

    def _propagate_app_secrets(self, app: App, source: Unit) -> None:
        """Share the leader's view of the application's own secrets with the other units.

        Only the leader can create, change or remove a secret the application
        owns, and every unit of the application can read it, so after the
        leader runs, each other unit's application-owned secrets are made to
        match the leader's.
        """
        if not source.is_leader:
            return
        owned: list[Secret] = [s for s in source._state.secrets if s.owner == 'app']
        for peer in app._live_units:
            if peer is source:
                continue
            others = [s for s in peer._state.secrets if s.owner != 'app']
            secrets = frozenset(others + owned)
            if secrets != peer._state.secrets:
                peer._state = dataclasses.replace(peer._state, secrets=secrets)

    def _propagate_app_status(self, app: App, source: Unit) -> None:
        """Give every unit the application status the leader set.

        Juju fires nothing when a status changes, and only the leader can
        read the application status, so this is for the test's sake: each
        unit's :class:`State` agrees on the application's status.
        """
        if not source.is_leader:
            return
        for peer in app._live_units:
            if peer is not source and peer._state.app_status != source._state.app_status:
                peer._state = dataclasses.replace(peer._state, app_status=source._state.app_status)

    def _sync_secrets(self) -> None:
        """Bring each unit's view of other applications' secrets up to date.

        A secret granted over a relation, to the application on the other side
        or to one of its units, is readable there for as long as the grant
        and the relation last. A new revision with new content updates the
        readers' latest content and queues ``secret-changed`` on them. Once
        no reader still tracks an old revision, the owner gets
        ``secret-remove`` for it.
        """
        owned: list[tuple[App, Unit, Secret]] = []
        for app in self._state.apps.values():
            for unit in app._live_units:
                for secret in sorted(unit._state.secrets, key=lambda s: s.id):
                    if secret.owner == 'unit' or (secret.owner == 'app' and unit.is_leader):
                        owned.append((app, unit, secret))

        wanted: dict[tuple[str, int], dict[str, Secret]] = {}
        for app, _, secret in owned:
            for relation_id, grantees in sorted(secret.remote_grants.items()):
                integration = self._state.integrations.get(relation_id)
                if integration is None or all(end.app is not app for end in integration.ends):
                    continue
                remote = integration.remote(app).app
                for unit in remote._live_units:
                    if remote.name in grantees or unit.name in grantees:
                        wanted.setdefault((remote.name, unit.id), {})[secret.id] = secret
        for user_secret in self._state.user_secrets.values():
            source = user_secret.as_source()
            for app_name in sorted(user_secret.grants):
                for unit in self._state.apps[app_name]._live_units:
                    wanted.setdefault((app_name, unit.id), {})[source.id] = source

        readers: dict[str, list[Secret]] = {}
        for app in self._state.apps.values():
            for unit in app.units:
                key = (app.name, unit.id)
                granted = self._state.granted.get(key, set())
                want = wanted.get(key, {})
                secrets = {s.id: s for s in unit._state.secrets}
                changed: list[str] = []
                for secret_id in sorted(granted - want.keys()):
                    if secrets.pop(secret_id, None) is not None:
                        changed.append(secret_id)
                now_granted: set[str] = set()
                notify: list[str] = []
                for secret_id, source in sorted(want.items()):
                    current = secrets.get(secret_id)
                    if current is not None and secret_id not in granted:
                        # The test put a secret with this ID in the unit's
                        # State itself; leave it alone.
                        continue
                    now_granted.add(secret_id)
                    if current is None:
                        secrets[secret_id] = _reader_copy(source, None)
                        changed.append(secret_id)
                    elif current._latest_revision != source._latest_revision:
                        secrets[secret_id] = _reader_copy(source, current)
                        changed.append(secret_id)
                        if current.latest_content != source.latest_content:
                            notify.append(secret_id)
                    readers.setdefault(secret_id, []).append(secrets[secret_id])
                if now_granted:
                    self._state.granted[key] = now_granted
                else:
                    self._state.granted.pop(key, None)
                if changed:
                    unit._state = dataclasses.replace(
                        unit._state, secrets=frozenset(secrets.values())
                    )
                for secret_id in notify:
                    self._enqueue(
                        app,
                        unit.id,
                        _Event('secret_changed', secret=secrets[secret_id]),
                        rebind=_Rebind('secret', secret_id),
                    )

        for app, unit, secret in owned:
            latest = secret._latest_revision
            tracked = [s._tracked_revision for s in readers.get(secret.id, [])]
            done = self._state.removable_revisions.get(secret.id, 0)
            unused_below = min([latest, *tracked])
            for revision in range(done + 1, unused_below):
                self._enqueue(
                    app,
                    unit.id,
                    _Event('secret_remove', secret=secret, secret_revision=revision),
                    rebind=_Rebind('secret', secret.id),
                )
            self._state.removable_revisions[secret.id] = max(done, unused_below - 1)

    def _enqueue(
        self,
        app: App,
        unit_id: int,
        event: _Event,
        rebind: _Rebind | None = None,
    ) -> None:
        """Queue an event for a unit.

        A ``*-changed`` event that is already waiting for the same unit, about
        the same thing, isn't queued twice: the one waiting will see the
        latest data when it runs. Juju coalesces these the same way.
        """
        queue = self._state.queues[app.name]
        if event.name.endswith(_COALESCED_SUFFIXES):
            for waiting in queue:
                if (
                    waiting.unit_id == unit_id
                    and isinstance(waiting.event, _Event)
                    and waiting.event.name == event.name
                    and waiting.rebind == rebind
                    and waiting.event.relation_remote_unit_id == event.relation_remote_unit_id
                ):
                    return
        queue.append(_Queued(app, unit_id, event, rebind))

    # Teardown
    def close(self) -> None:
        """Tear down every application's worker process, and every unit's filesystem.

        Safe to call more than once. Assert on :attr:`Unit.filesystem` before
        closing.
        """
        for app in self._state.apps.values():
            app._runner.close()
        if self._filesystems_finalizer is not None:
            self._filesystems_finalizer()
        self._state.apps.clear()
        self._state.queues.clear()
        self._state.closed = True

    def __enter__(self) -> Juju:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _class_sources(charm_type: type[Any]) -> list[pathlib.Path]:
    """The file a charm class is defined in, to find the charmlibs libraries it uses."""
    try:
        source = inspect.getsourcefile(charm_type)
    except TypeError:
        return []
    return [pathlib.Path(source)] if source else []


def _charm_class_root(charm_type: type[CharmBase]) -> pathlib.Path | None:
    """The source tree a charm class was loaded from, if it has a ``pyproject.toml``.

    Found the same way as ``Context`` finds a charm class's metadata: the
    class is expected to be in ``src/charm.py``.
    """
    try:
        root = pathlib.Path(inspect.getfile(charm_type)).parent.parent
    except (OSError, TypeError):
        return None
    return root if (root / 'pyproject.toml').exists() else None


def _check_state_template(template: State) -> None:
    """Reject a ``state_template`` that sets a field :class:`Juju` owns.

    ``State()`` gives every ``Model`` a random name and UUID, so those two
    can't be told apart from a template's own; a template's model is rejected
    only when its type or cloud spec differs from the default.
    """
    default = State()
    for name in _JUJU_OWNED_FIELDS:
        value = getattr(template, name)
        if name == 'model':
            if value.type != default.model.type or value.cloud_spec is not None:
                raise JujuError(
                    "state_template may not set model: Juju sets each unit's model. "
                    'Pass type= and cloud_spec= to Juju instead.'
                )
            continue
        if value != getattr(default, name):
            raise JujuError(
                f'state_template may not set {name}: Juju sets it for each unit. '
                'Use config= for configuration; relations come from the '
                "charm's peer endpoints and integrate(), and secrets from the "
                'charms themselves and add_secret().'
            )


def _dispatch_key(app: App, unit: Unit, event: _Event) -> tuple[str, int, str, str] | None:
    """Identify a dispatch by its unit, event and input state, for loop detection.

    Returns ``None`` when the event or state holds something the JSON codec
    can't encode, in which case that dispatch isn't checked.
    """
    try:
        return (
            app.name,
            unit.id,
            _isolated_serde.encode_event(event),
            unit._state._to_json(),
        )
    except TypeError:
        return None


def _resolve_meta(charm: CharmSource, app: str | None) -> Mapping[str, Any]:
    """A charm's metadata, shaped like ``charmcraft.yaml``, from wherever it comes from.

    Raises:
        JujuError: if a path isn't a charm directory, or a charm class has no
            metadata beside its source.
    """
    if isinstance(charm, CharmSpec):
        return charm.meta
    if isinstance(charm, (str, pathlib.Path)):
        if isinstance(charm, str) and pathlib.PurePath(charm).is_absolute():
            raise JujuError(
                f'A charm path given as a str must be relative, as with juju deploy: '
                f'{charm!r}. Pass a pathlib.Path for an absolute path.'
            )
        if not pathlib.Path(charm).is_dir():
            raise JujuError(f'No charm source directory at {str(charm)!r}.')
        try:
            meta, config, actions = _load_charm_spec(pathlib.Path(charm))
        except MetadataNotFoundError as e:
            raise JujuError(str(e)) from None
        return _joined_meta(meta, config, actions)
    try:
        spec = _CharmSpec.autoload(charm)
    except (MetadataNotFoundError, OSError, TypeError):
        name = app or charm.__name__.lower()
        raise JujuError(
            f'{charm.__name__} has no metadata beside its source to load. Wrap it in a '
            f'CharmSpec with the metadata: '
            f"deploy(CharmSpec({charm.__name__}, meta={{'name': {name!r}, ...}}))."
        ) from None
    return _joined_meta(spec.meta, spec.config, spec.actions)


def _joined_meta(
    meta: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    actions: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Metadata, config options and actions, joined in ``charmcraft.yaml`` shape."""
    joined = dict(meta)
    if config is not None:
        joined['config'] = dict(config)
    if actions is not None:
        joined['actions'] = dict(actions)
    return joined


def _split_meta(
    meta: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """``charmcraft.yaml``-shaped metadata, split into the three that ``Context`` takes."""
    metadata = dict(meta)
    config = metadata.pop('config', None)
    actions = metadata.pop('actions', None)
    return (
        metadata,
        dict(config) if config is not None else None,
        dict(actions) if actions is not None else None,
    )


def _merged_config(
    config_schema: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Charm config defaults, with the caller's values on top.

    Juju applies a charm's declared defaults to any option the deployer did not
    set, so a charm reading ``self.config['foo']`` finds the default rather
    than a ``KeyError``.
    """
    merged: dict[str, Any] = {}
    for name, option in (config_schema or {}).get('options', {}).items():
        if isinstance(option, dict) and 'default' in option:
            merged[name] = option['default']
    if config:
        merged.update(config)
    return merged


def _rebind(state: State, rebind: _Rebind) -> RelationBase | Container | Secret | None:
    """Look up the object an event should carry, in the state it will run against."""
    if rebind.kind == 'relation':
        return _relation_by_id(state, cast('int', rebind.key))
    if rebind.kind == 'secret':
        for secret in state.secrets:
            if secret.id == rebind.key:
                return secret
        return None
    for container in state.containers:
        if container.name == rebind.key:
            return container
    return None


def _peer_relation(state: State, endpoint: str) -> PeerRelation | None:
    for relation in state.relations:
        if relation.endpoint == endpoint and isinstance(relation, PeerRelation):
            return relation
    return None


def _relation_by_id(state: State, relation_id: int) -> RelationBase | None:
    for relation in state.relations:
        if relation.id == relation_id:
            return relation
    return None


def _with_relation(state: State, relation: RelationBase) -> State:
    """A copy of ``state`` with ``relation`` replacing the one with its ID."""
    relations = [r for r in state.relations if r.id != relation.id]
    relations.append(relation)
    return dataclasses.replace(state, relations=frozenset(relations))


def _without_relation(state: State, relation_id: int) -> State:
    relations = [r for r in state.relations if r.id != relation_id]
    return dataclasses.replace(state, relations=frozenset(relations))


def _remote_ids(relation: RelationBase) -> list[int]:
    """The units on the other side of a relation that this view has seen join."""
    if isinstance(relation, PeerRelation):
        return list(relation.peers_data)
    if isinstance(relation, Relation):
        return list(relation.remote_units_data)
    return []


def _without_remote(relation: RelationBase, remote_id: int | None) -> RelationBase:
    """A view of the relation without one remote unit, or without any if ``None``."""
    if isinstance(relation, PeerRelation):
        peers = {k: v for k, v in relation.peers_data.items() if remote_id not in (None, k)}
        return dataclasses.replace(relation, peers_data=peers)
    if isinstance(relation, Relation):
        units = {k: v for k, v in relation.remote_units_data.items() if remote_id not in (None, k)}
        return dataclasses.replace(relation, remote_units_data=units)
    return relation


def _app_databag(app: App, relation_id: int) -> dict[str, str]:
    """An application's databag in a relation, as its leader has it."""
    leader = app._units.get(app._leader_id)
    view = _relation_by_id(leader._state, relation_id) if leader is not None else None
    return dict(view.local_app_data) if view is not None else {}


def _check_no_app_data_writes(unit: Unit, state_in: State, state_out: State) -> None:
    """Fail a dispatch where a unit that isn't the leader changed application data.

    The charm can't do this through ops, which raises first. Juju refuses it
    too, so a ``State`` that shows it means the charm went around ops.
    """
    for relation in state_out.relations:
        before = _relation_by_id(state_in, relation.id)
        if before is not None and before.local_app_data != relation.local_app_data:
            raise JujuError(
                f'{unit.name} is not the leader, but its application databag in '
                f'{relation.endpoint}:{relation.id} changed. Juju only lets the leader '
                'write application data.'
            )


def _stop_owner_tracking(unit: Unit, secret_id: str, revision: int) -> None:
    """Move an owner off a revision it is about to be told to remove.

    Juju sends ``secret-remove`` once no reader of the secret tracks the
    revision; what the owner itself last read doesn't count. Scenario refuses
    to remove the revision the owner tracks, so the owner's view moves to the
    latest revision first.
    """
    secrets: list[Secret] = []
    for secret in unit._state.secrets:
        if secret.id == secret_id and secret._tracked_revision <= revision:
            secret = _replace_secret(secret, tracked_content=dict(secret.latest_content))
            object.__setattr__(secret, '_tracked_revision', secret._latest_revision)
        secrets.append(secret)
    unit._state = dataclasses.replace(unit._state, secrets=frozenset(secrets))


def _replace_secret(secret: Secret, **changes: Any) -> Secret:
    """``dataclasses.replace`` for a secret, keeping the revisions it tracks."""
    new = dataclasses.replace(secret, **changes)
    object.__setattr__(new, '_tracked_revision', secret._tracked_revision)
    object.__setattr__(new, '_latest_revision', secret._latest_revision)
    return new


def _reader_copy(source: Secret, current: Secret | None) -> Secret:
    """What a unit that was granted ``source`` sees of it.

    A new reader starts out tracking the latest revision. An existing one
    keeps the revision (and content) it tracks, and its own label, and learns
    the latest.
    """
    copy = Secret(
        dict(source.latest_content if current is None else current.tracked_content),
        latest_content=dict(source.latest_content),
        id=source.id,
        label=None if current is None else current.label,
    )
    tracked = source._latest_revision if current is None else current._tracked_revision
    object.__setattr__(copy, '_tracked_revision', tracked)
    object.__setattr__(copy, '_latest_revision', source._latest_revision)
    return copy


def _split_end(end: App | tuple[App, str]) -> tuple[App, str | None]:
    if isinstance(end, App):
        return end, None
    app, endpoint = end
    return app, endpoint


def _end_name(app: App, endpoint: str | None) -> str:
    return app.name if endpoint is None else f'{app.name}:{endpoint}'


def _list_endpoints(app: App) -> str:
    endpoints = app._relation_endpoints
    listed = [
        f'{name} ({role} {spec.get("interface")})'
        for name, (role, spec) in sorted(endpoints.items())
    ]
    return ', '.join(listed) or 'no provides or requires endpoints'
