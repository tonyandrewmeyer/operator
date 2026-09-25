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

:class:`Juju` is its own class, not a subclass of :class:`~ops.testing.Model`:
``Model`` is a frozen dataclass held as ``State.model`` in every unit's
:class:`State`, so ``Juju`` produces the ``Model`` values that go into each
unit's state rather than being one. The identity it carries (``name``,
``uuid``, ``type``, ``cloud_spec``) is stamped into every unit's
:class:`State`, which stops two applications under the same ``Juju`` from
disagreeing about which model they are in.

.. note::
    Cross-application operations (``integrate`` and the event propagation
    between related applications) are not in this layer yet. What is here is
    the single-application half: everything an application does on its own,
    including its peer relation.
"""

from __future__ import annotations

import dataclasses
import inspect
import pathlib
import shutil
import tempfile
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias, cast
from uuid import uuid4

from . import _charm_mocking, _isolated_serde
from ._isolated_worker import _load_charm_type
from ._isolation import IsolatedContext, _load_charm_spec
from .context import _DEFAULT_JUJU_VERSION, Context
from .errors import JujuError, MetadataNotFoundError
from .state import (
    CloudSpec,
    Container,
    Model,
    PeerRelation,
    RawDataBagContents,
    Secret,
    State,
    _CharmSpec,
    _Event,
    _next_relation_id,
    _random_model_name,
)

if TYPE_CHECKING:
    from ops.charm import CharmBase

#: What ``charm=`` accepts: a path to charm source on disk, or a charm class.
CharmSource: TypeAlias = 'str | pathlib.Path | type[CharmBase]'

#: How many dispatches per unit :meth:`Juju.settle` allows before it decides
#: the model will not converge.
_SETTLE_DISPATCHES_PER_UNIT = 100

#: How many dispatches from the end of the trace a non-convergence error shows.
_TRACE_TAIL = 10

#: ``State`` fields that describe a unit's place in the model. :class:`Juju`
#: sets these itself, so a ``state_template`` may not.
_JUJU_OWNED_FIELDS = ('leader', 'planned_units', 'model', 'config', 'relations')


class Dispatch(NamedTuple):
    """One event dispatched to one unit, as recorded in a settle trace."""

    event: _Event
    """The event that was dispatched."""

    unit: Unit
    """The unit it was dispatched to."""

    state: State
    """The unit's :class:`State` *after* the charm handled the event."""


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
    """Executes a single event for one unit of one application."""

    def run(self, unit_id: int, event: _Event, state: State) -> State:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


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
        return self._contexts[unit_id]

    def run(self, unit_id: int, event: _Event, state: State) -> State:
        with self._mocking.dispatching(f'{self._app_name}/{unit_id}', state.model.name):
            return self._context(unit_id).run(event, state)

    def close(self) -> None:
        self._contexts.clear()


class _IsolatedRunner(_Runner):
    """Runs an on-disk charm in a subprocess, via :class:`IsolatedContext`.

    One :class:`IsolatedContext`, and therefore one persistent worker process,
    serves every unit of the application; the unit ID travels with each
    request rather than being baked into the worker. The charm directory is
    per unit, so it travels with each request too.
    """

    def __init__(self, ctx: IsolatedContext, charm_roots: Mapping[int, pathlib.Path]):
        self._ctx = ctx
        self._charm_roots = charm_roots

    def run(self, unit_id: int, event: _Event, state: State) -> State:
        self._ctx.charm_root = self._charm_roots.get(unit_id)
        return self._ctx._run_as(unit_id, event, state)

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
        config_schema: Mapping[str, Any] | None,
        state_template: State | None = None,
        trust: bool = False,
        actions: Mapping[str, Any] | None,
        config: Mapping[str, Any],
        python_executable: str | None = None,
        extra_sys_path: Sequence[str] = (),
        mocking: Mapping[str, Any] | None = None,
        juju_version: str = _DEFAULT_JUJU_VERSION,
    ):
        if state_template is not None:
            _check_state_template(state_template)
        self._juju = juju
        self._name = name
        self._charm_source = (
            pathlib.Path(charm) if isinstance(charm, (str, pathlib.Path)) else None
        )
        # Each unit gets its own charm directory (see _make_charm_root). The
        # runners read this mapping when they dispatch, so it is shared.
        self._charm_roots: dict[int, pathlib.Path] = {}
        mocking = dict(mocking) if mocking is not None else {}
        _charm_mocking.check_mocking_json(mocking)
        charm_mocking = _charm_mocking.CharmMocking(
            self._charm_source or _charm_class_root(cast('type[CharmBase]', charm)),
            app_name=name,
            mocking=mocking,
            module_name=f'_ops_testing_mocking_{uuid4().hex}',
        )
        # A path with an interpreter runs the charm in its own process; a path
        # without one, or a charm class, runs it in this process. Which one it
        # is decides how events are executed, so the application owns that
        # choice rather than being handed it.
        self._runner: _Runner
        if python_executable is not None:
            if self._charm_source is None:
                raise JujuError(
                    'python_executable= needs a charm on disk: a charm class '
                    'cannot be sent to another interpreter. Deploy it from a path.'
                )
            self._runner = _IsolatedRunner(
                IsolatedContext(
                    charm_source=self._charm_source,
                    python_executable=python_executable,
                    extra_sys_path=tuple(extra_sys_path),
                    meta=meta,
                    config=config_schema,
                    actions=actions,
                    app_name=name,
                    juju_version=juju_version,
                    app_trusted=trust,
                    mocking=mocking,
                ),
                self._charm_roots,
            )
            # The mocking itself loads in the worker; the parent reports the
            # configuration problems it can see without importing anything.
            charm_mocking.check()
        else:
            if extra_sys_path:
                raise JujuError(
                    "extra_sys_path= adds to an isolated worker's sys.path, so it "
                    'needs python_executable= as well.'
                )
            # The charm's mocking module loads before the charm itself, so its
            # import-time patches are in place when the charm is imported.
            charm_mocking.load()
            if self._charm_source is None:
                charm_type = cast('type[CharmBase]', charm)
            else:
                with charm_mocking.importing():
                    charm_type = _load_charm_type(
                        self._charm_source,
                        module_name=f'_ops_testing_charm_{uuid4().hex}',
                    )
            self._runner = _InProcessRunner(
                charm_type,
                meta=meta,
                config=config_schema,
                actions=actions,
                app_name=name,
                juju_version=juju_version,
                app_trusted=trust,
                charm_roots=self._charm_roots,
                mocking=charm_mocking,
            )
        self._meta = meta
        self._config_schema = config_schema
        self._actions = actions
        self._config = dict(config)
        self._state_template = state_template if state_template is not None else State()
        self._trust = trust
        self._leader_id = 0
        self._units: dict[int, Unit] = {}
        self._next_unit_id = 0
        # One relation ID per peer endpoint: a peer relation is a single
        # relation that every unit is a member of, so the ID must agree across
        # units even though each unit holds its own view of the databags.
        self._peer_ids: dict[str, int] = {
            endpoint: _next_relation_id() for endpoint in self._peer_endpoints
        }

    @property
    def name(self) -> str:
        """The application name."""
        return self._name

    @property
    def meta(self) -> Mapping[str, Any]:
        """The charm metadata, as resolved from the charm source."""
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
        peers: dict[str, Any] = self._meta.get('peers') or {}
        return tuple(peers)

    @property
    def _container_names(self) -> tuple[str, ...]:
        containers: dict[str, Any] = self._meta.get('containers') or {}
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

    def _make_charm_root(self, unit_id: int) -> None:
        """Give a unit of an on-disk charm its own charm directory.

        Juju gives each unit its own copy of the charm. Here the directory
        links to each top-level entry of the charm source, so the charm finds
        the files it ships, while the metadata files that ``Context`` writes
        into its charm directory land in the unit's directory rather than in
        the source tree.
        """
        if self._charm_source is None:
            return
        root = pathlib.Path(tempfile.mkdtemp(prefix=f'ops-testing-{self._name}-{unit_id}-'))
        for entry in self._charm_source.resolve().iterdir():
            if entry.name in {'metadata.yaml', 'config.yaml', 'actions.yaml'}:
                continue
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        self._charm_roots[unit_id] = root

    def _drop_charm_root(self, unit_id: int) -> None:
        root = self._charm_roots.pop(unit_id, None)
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)

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

    kind: Literal['relation', 'container']
    name: str


class _Queued(NamedTuple):
    """An event waiting to be dispatched to a unit."""

    app: App
    unit_id: int
    event: _Event | _RemoveUnit
    rebind: _Rebind | None = None


class _JujuState:
    """The mutable half of a :class:`Juju`.

    Held in one object rather than as plain attributes so that the identity
    fields (``name``, ``uuid``, ``type``, ``cloud_spec``) stay the only things
    set directly on ``Juju``; everything mutable during a test run lives
    here instead.
    """

    def __init__(self) -> None:
        self.apps: dict[str, App] = {}
        self.queue: deque[_Queued] = deque()
        self.trace: list[Dispatch] = []
        self.closed = False


class Juju:
    """A set of applications, driven with Juju-shaped operations.

    Construct it much as you would a :class:`~ops.testing.Model`::

        juju = testing.Juju(model_name='my-model', type='lxd')

    Applications are added with :meth:`deploy`, which returns an :class:`App`
    handle::

        web = juju.deploy('./charms/myapp', num_units=2)
        db = juju.deploy(MyDatabaseCharm, app='db')

    Each operation queues the events Juju would emit for it. Nothing runs until
    :meth:`settle` drains the queue, so most tests are a sequence of
    operations, a ``settle()``, and then assertions::

        juju.config(web, {'log_level': 'debug'})
        juju.settle()
        assert web.leader.state.unit_status == testing.ActiveStatus('ready')

    Charms run in the test process unless ``deploy()`` is given a
    ``python_executable``, in which case the charm runs in a worker process
    with that interpreter. Call :meth:`close`, or use ``Juju`` as a context
    manager, to tear down any worker processes.
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
            uuid: A unique identifier for the model. Defaults to a randomly
                generated UUID.
            type: The type of Juju model: ``'kubernetes'`` or ``'lxd'``
                (machine). Every application deployed under this ``Juju``
                shares this substrate, which decides which form of
                :meth:`remove_unit` it accepts.
            cloud_spec: Cloud specification information, as in
                :class:`~ops.testing.Model`.
        """
        self.name = model_name if model_name is not None else _random_model_name()
        self.uuid = uuid if uuid is not None else str(uuid4())
        self.type: Literal['kubernetes', 'lxd'] = type
        self.cloud_spec = cloud_spec
        self._state = _JujuState()

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

    def _check_open(self) -> None:
        if self._state.closed:
            raise JujuError('This Juju has been closed.')

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
        meta: Mapping[str, Any] | None = None,
        config_schema: Mapping[str, Any] | None = None,
        actions: Mapping[str, Any] | None = None,
        python_executable: str | None = None,
        extra_sys_path: Sequence[str] = (),
        mocking: Mapping[str, Any] | None = None,
        juju_version: str = _DEFAULT_JUJU_VERSION,
    ) -> App:
        """Deploy a charm, as ``juju deploy`` would.

        The units' startup events are queued; call :meth:`settle` to run them.

        Args:
            charm: Charm source: a path to a charm's source directory, or a
                :class:`ops.CharmBase` subclass. A path given as a ``str``
                must be relative, as with ``juju deploy``; pass a
                :class:`pathlib.Path` for an absolute path. A charm class
                always runs in the test process, like a ``Context`` test.
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
            meta: Charm metadata, in ``metadata.yaml`` form. Read from the
                charm source when deploying from a path, and from the charm
                class's source tree when deploying a class.
            config_schema: Charm config options, in ``config.yaml`` form. Read
                from the charm source in the same way as ``meta``.
            actions: Charm actions, in ``actions.yaml`` form. Read from the
                charm source in the same way as ``meta``.
            python_executable: Interpreter to run the charm with, in its own
                worker process. Point it at a virtual environment built for
                the charm, which must have the same ``ops`` version installed
                as the test. Only for a charm deployed from a path. Without
                it, the charm runs in the test process, and shares its
                imported modules with any other charm deployed that way.
            extra_sys_path: Directories prepended to the worker's
                ``sys.path``. Only with ``python_executable``.
            mocking: Keyword arguments for the charm's own mocking function,
                which the charm configures in its ``pyproject.toml`` under
                ``[tool.ops.testing.mocking]``. Use it to vary how the
                charm's mocks behave in this test. Only JSON values are
                accepted, because for a charm in its own interpreter this
                dict is all that is sent.
            juju_version: The Juju agent version to simulate.

        Returns:
            The :class:`App` handle for the new application.

        Raises:
            JujuError: if an application of this name already exists,
                ``num_units`` is not positive, the charm path or the
                template is not accepted, ``python_executable`` or
                ``extra_sys_path`` is given where it can't apply, or the
                charm's mocking can't be set up with ``mocking``.
        """
        self._check_open()
        if num_units < 1:
            raise JujuError(f'num_units must be at least 1, not {num_units}.')

        if isinstance(charm, str) and pathlib.PurePath(charm).is_absolute():
            raise JujuError(
                f'A charm path given as a str must be relative, as with juju deploy: '
                f'{charm!r}. Pass a pathlib.Path for an absolute path.'
            )
        if isinstance(charm, (str, pathlib.Path)):
            if not pathlib.Path(charm).is_dir():
                raise JujuError(f'No charm source directory at {str(charm)!r}.')
            disk_meta, disk_config, disk_actions = _load_charm_spec(pathlib.Path(charm))
        else:
            try:
                spec = _CharmSpec.autoload(charm)
            except (MetadataNotFoundError, OSError, TypeError):
                # A charm class with no metadata on disk (or no file at all,
                # such as one defined interactively) has a name and nothing
                # else, unless the test passes meta=.
                disk_meta = {'name': app or charm.__name__.lower()}
                disk_config = disk_actions = None
            else:
                disk_meta = dict(spec.meta)
                disk_config = dict(spec.config) if spec.config is not None else None
                disk_actions = dict(spec.actions) if spec.actions is not None else None
        if meta is None:
            meta = disk_meta
        if config_schema is None:
            config_schema = disk_config
        if actions is None:
            actions = disk_actions

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
            config_schema=config_schema,
            state_template=state_template,
            trust=trust,
            actions=actions,
            config=_merged_config(config_schema, config),
            python_executable=python_executable,
            extra_sys_path=extra_sys_path,
            mocking=mocking,
            juju_version=juju_version,
        )
        self._state.apps[app_name] = new_app

        for _ in range(num_units):
            self._add_unit(new_app)
        return new_app

    def add_unit(self, app: App) -> Unit:
        """Add a unit to an application, as ``juju add-unit`` would.

        The new unit's startup events are queued. If the application has a
        peer relation, the units that were already there see the new unit join
        it.

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
        remaining = [u for u in app.units if u is not departing and u.id not in doomed_ids]

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
                        rebind=_Rebind('relation', endpoint),
                    )
        for event, rebind in _teardown_events(app, unit_id):
            self._enqueue(app, unit_id, event, rebind)

        self._state.queue.append(_Queued(app, unit_id, _RemoveUnit()))

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

    # Convergence
    def settle(self) -> list[Dispatch]:
        """Dispatch queued events until the model converges.

        Convergence is reached when the queue is empty: every event produced
        by an operation has been dispatched, along with every follow-on event
        those dispatches produced.

        Returns:
            The events dispatched, in order, each with the unit it went to and
            that unit's :class:`State` afterwards.

        Raises:
            JujuError: if the model does not converge. The same event reaching
                the same unit with the same :class:`State` twice is a loop,
                and is reported as soon as it happens; otherwise, the limit
                grows with the number of units in the model. The message ends
                with the last events dispatched.
        """
        self._check_open()
        self._state.trace = []
        seen: set[tuple[str, int, str, str]] = set()
        dispatched = 0
        while self._state.queue:
            limit = _SETTLE_DISPATCHES_PER_UNIT * max(1, self._unit_count())
            if dispatched >= limit:
                raise JujuError(
                    f'Did not converge after {dispatched} events. Check for charms '
                    'that write a new value to a databag on every event.'
                    f'{self._trace_tail()}'
                )
            entry = self._next_dispatch()
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
        return list(self._state.trace)

    def _unit_count(self) -> int:
        count = 0
        for app in self._state.apps.values():
            count += len(app._units)
        return count

    def _trace_tail(self) -> str:
        tail = self._state.trace[-_TRACE_TAIL:]
        lines = [f'  {d.event.name} on {d.unit.name}' for d in tail]
        return '\nLast events dispatched:\n' + '\n'.join(lines) if lines else ''

    def _next_dispatch(self) -> tuple[App, Unit, _Event] | None:
        """Pop the next queue entry, returning the charm invocation it holds, if any.

        A ``_RemoveUnit`` marker, and an event whose rebind target vanished
        before dispatch, carry no charm invocation of their own; both are
        consumed here and ``None`` is returned.
        """
        app, unit_id, event, rebind = self._state.queue.popleft()
        if isinstance(event, _RemoveUnit):
            del app._units[unit_id]
            app._drop_charm_root(unit_id)
            self._drop_peer(app, unit_id)
            return None
        unit = app._units[unit_id]
        if rebind is not None:
            rebound = _rebind(unit._state, rebind)
            if rebound is None:
                # Whatever the event was about went away between queueing and
                # dispatch; there is nothing left for the charm to observe.
                return None
            event = dataclasses.replace(event, **{rebind.kind: rebound})
        return app, unit, event

    def _dispatch(self, app: App, unit: Unit, event: _Event) -> None:
        state_out = app._runner.run(unit.id, event, unit._state)
        unit._state = state_out
        self._state.trace.append(Dispatch(event, unit, state_out))
        self._propagate_peers(app, unit)
        self._propagate_app_secrets(app, unit)

    # Units and peer relations
    def _add_unit(self, app: App) -> Unit:
        unit_id = app._next_unit_id
        app._next_unit_id += 1
        existing = app.units

        app._make_charm_root(unit_id)
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
                peer._state = _with_peer_relation(
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
                        rebind=_Rebind('relation', endpoint),
                    )

        for event, rebind in _startup_events(app, unit_id):
            self._enqueue(app, unit_id, event, rebind)
        return unit

    def _initial_state(self, app: App, unit_id: int) -> State:
        relations: list[PeerRelation] = []
        for endpoint in app._peer_endpoints:
            # A unit joining an existing application can read what its peers
            # have already published, so seed its view from theirs rather than
            # starting it empty; otherwise the first peer to run afterwards
            # looks like it just wrote data it had published long before.
            peers_data: dict[int, RawDataBagContents] = {}
            app_data: RawDataBagContents = {}
            for peer_id, peer in app._units.items():
                if peer_id == unit_id:
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
                peer._state = _with_peer_relation(
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
            for peer in app.units:
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
                peer._state = _with_peer_relation(
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
                    rebind=_Rebind('relation', endpoint),
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
        for peer in app.units:
            if peer is source:
                continue
            others = [s for s in peer._state.secrets if s.owner != 'app']
            secrets = frozenset(others + owned)
            if secrets != peer._state.secrets:
                peer._state = dataclasses.replace(peer._state, secrets=secrets)

    def _enqueue(
        self,
        app: App,
        unit_id: int,
        event: _Event,
        rebind: _Rebind | None = None,
    ) -> None:
        self._state.queue.append(_Queued(app, unit_id, event, rebind))

    # Teardown
    def close(self) -> None:
        """Tear down every application's worker process.

        Safe to call more than once. In-process applications have nothing to
        tear down.
        """
        for app in self._state.apps.values():
            app._runner.close()
            for unit_id in list(app._charm_roots):
                app._drop_charm_root(unit_id)
        self._state.apps.clear()
        self._state.queue.clear()
        self._state.closed = True

    def __enter__(self) -> Juju:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


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
                "charm's peer endpoints."
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


def _rebind(state: State, rebind: _Rebind) -> PeerRelation | Container | None:
    """Look up the object an event should carry, in the state it will run against."""
    if rebind.kind == 'relation':
        return _peer_relation(state, rebind.name)
    for container in state.containers:
        if container.name == rebind.name:
            return container
    return None


def _peer_relation(state: State, endpoint: str) -> PeerRelation | None:
    for relation in state.relations:
        if relation.endpoint == endpoint and isinstance(relation, PeerRelation):
            return relation
    return None


def _with_peer_relation(state: State, relation: PeerRelation) -> State:
    """A copy of ``state`` with ``relation`` replacing the one with its ID."""
    relations = [r for r in state.relations if r.id != relation.id]
    relations.append(relation)
    return dataclasses.replace(state, relations=frozenset(relations))
