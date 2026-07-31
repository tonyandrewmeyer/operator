# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Model-level testing: drive several charms with Juju-shaped operations.

:class:`~ops.testing.Context` runs *one* event against *one* charm.  This
module adds a layer above it: a :class:`Deployment` owns a set of applications
(:class:`App`), each with its own units and its own :class:`State` per unit,
and exposes Juju-shaped operations (``deploy``, ``add_unit``, ``remove_unit``,
``update_config``, ``run_action``) that describe **intents rather than
events**.  The deployment works out the Juju-faithful event sequence each
intent produces and drains it in a convergence loop.

A test therefore reads as a sequence of operations followed by assertions on
the resulting state, rather than as a hand-written event sequence::

    from ops import testing

    deployment = testing.Deployment()
    web = deployment.deploy('./charms/myapp', num_units=2)
    deployment.update_config(web, {'log_level': 'debug'})
    assert web.leader.state.unit_status == testing.ActiveStatus('ready')

:class:`Deployment` is a subclass of :class:`~ops.testing.Model`, so it *is* a
model description as well as a handle to drive one: the identity it carries
(``name``, ``uuid``, ``type``, ``cloud_spec``) is stamped into every unit's
:class:`State`, which is what stops two applications in one deployment from
disagreeing about which model they are in.  The plain :class:`Model` stays a
passive value object with no operations on it.

.. note::
    Cross-application operations — ``integrate`` and the event propagation
    between related applications — are not in this layer yet.  What is here is
    the single-application half: everything an application does on its own,
    including its peer relation.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections import deque
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

from .context import _DEFAULT_JUJU_VERSION, Context
from .isolation import IsolatedContext, _read_charm_metadata, _read_yaml
from .state import (
    Container,
    Model,
    PeerRelation,
    RawDataBagContents,
    State,
    _Action,
    _Event,
    _next_relation_id,
)

if TYPE_CHECKING:
    from ops.charm import CharmBase

__all__ = [
    'App',
    'Deployment',
    'DeploymentError',
    'Dispatch',
    'Unit',
]

#: What ``charm=`` accepts: a path to charm source on disk (run isolated, in
#: the charm's own interpreter) or a charm class (run in-process, like
#: ``Context``).
CharmSource: TypeAlias = 'str | pathlib.Path | type[CharmBase]'


class DeploymentError(RuntimeError):
    """Raised when an operation cannot be performed on a :class:`Deployment`.

    For example: removing the last unit of an application, referring to an
    application that was never deployed, or a convergence loop that does not
    terminate.
    """


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
# Each operation below maps to the events Juju emits for it.  Keeping them in
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
    for container in app._containers:
        events.append((
            _Event(f'{container.name}_pebble_ready', container=container),
            _Rebind('container', container.name),
        ))
    return events


def _teardown_events(app: App, unit_id: int) -> list[tuple[_Event, _Rebind | None]]:
    """The events a departing unit sees, in Juju's order."""
    del app, unit_id  # Same for every unit today; kept for symmetry with startup.
    return [(_Event('stop'), None), (_Event('remove'), None)]


# Runners
#
# Two ways to actually execute an event: in this process (an inline charm
# class, exactly what Context does today) or in a subprocess running the
# charm's own interpreter.  The convergence loop only knows this interface.


class _Runner:
    """Executes a single event for one unit of one application."""

    def run(self, unit_id: int, event: _Event, state: State) -> State:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _InProcessRunner(_Runner):
    """Runs an inline charm class in the test process, via :class:`Context`.

    No isolation: the charm shares the test process's interpreter and
    installed packages, which is the same trade-off a plain ``Context`` test
    makes today.  A separate ``Context`` is held per unit because the unit ID
    is fixed at construction time.
    """

    def __init__(
        self,
        charm_type: type[CharmBase],
        *,
        meta: dict[str, Any],
        config: dict[str, Any] | None,
        actions: dict[str, Any] | None,
        app_name: str,
        juju_version: str,
    ):
        self._charm_type = charm_type
        self._meta = meta
        self._config = config
        self._actions = actions
        self._app_name = app_name
        self._juju_version = juju_version
        self._contexts: dict[int, Context[CharmBase]] = {}

    def _context(self, unit_id: int) -> Context[CharmBase]:
        if unit_id not in self._contexts:
            self._contexts[unit_id] = Context(
                self._charm_type,
                meta=self._meta,
                config=self._config,
                actions=self._actions,
                app_name=self._app_name,
                unit_id=unit_id,
                juju_version=self._juju_version,
            )
        return self._contexts[unit_id]

    def run(self, unit_id: int, event: _Event, state: State) -> State:
        return self._context(unit_id).run(event, state)

    def close(self) -> None:
        self._contexts.clear()


class _IsolatedRunner(_Runner):
    """Runs an on-disk charm in a subprocess, via :class:`IsolatedContext`.

    One :class:`IsolatedContext` — and therefore one persistent worker
    process — serves every unit of the application; the unit ID travels with
    each request rather than being baked into the worker.
    """

    def __init__(self, ctx: IsolatedContext):
        self._ctx = ctx

    def run(self, unit_id: int, event: _Event, state: State) -> State:
        return self._ctx._run_as(unit_id, event, state)

    def close(self) -> None:
        self._ctx.close()


# Public handles


class Unit:
    """One unit of an :class:`App`.

    Units are created by :meth:`Deployment.deploy` and
    :meth:`Deployment.add_unit`; there is no reason to construct one directly.
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
        """This unit's :class:`State`.

        Reading this settles the deployment first if there are events pending,
        so assertions see converged state without an explicit
        :meth:`Deployment.settle` call.  Inside a
        :meth:`Deployment.stepping` block the implicit settle is suspended, so
        mid-convergence assertions do not drain the queue.
        """
        self._app._deployment._settle_if_pending()
        return self._state

    def __repr__(self) -> str:
        return f'<Unit {self.name}>'


class App:
    """An application in a :class:`Deployment`.

    Owns the charm reference, the metadata resolved from it, the environment
    it runs in, and the :class:`State` of each of its units.  Returned by
    :meth:`Deployment.deploy`.
    """

    def __init__(
        self,
        deployment: Deployment,
        name: str,
        runner: _Runner,
        *,
        meta: dict[str, Any],
        config_schema: dict[str, Any] | None,
        actions: dict[str, Any] | None,
        config: dict[str, Any],
    ):
        self._deployment = deployment
        self._name = name
        self._runner = runner
        self._meta = meta
        self._config_schema = config_schema
        self._actions = actions
        self._config = config
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
    def meta(self) -> dict[str, Any]:
        """The charm metadata, as resolved from the charm source."""
        return self._meta

    @property
    def config(self) -> Mapping[str, Any]:
        """The application's current configuration."""
        return dict(self._config)

    @property
    def units(self) -> tuple[Unit, ...]:
        """This application's units, ordered by unit number."""
        return tuple(self._units[uid] for uid in sorted(self._units))

    @property
    def leader(self) -> Unit:
        """The unit that currently holds leadership."""
        try:
            return self._units[self._leader_id]
        except KeyError:
            raise DeploymentError(f'{self._name} has no units.') from None

    @property
    def _peer_endpoints(self) -> tuple[str, ...]:
        peers: dict[str, Any] = self._meta.get('peers') or {}
        return tuple(peers)

    @property
    def _containers(self) -> tuple[Container, ...]:
        containers: dict[str, Any] = self._meta.get('containers') or {}
        return tuple(Container(name=name, can_connect=True) for name in containers)

    def __repr__(self) -> str:
        return f'<App {self._name} ({len(self._units)} units)>'


class _RemoveUnit:
    """Queue entry marking where a removed unit's records should be dropped.

    Bookkeeping rather than a Juju event.  It sits in the queue *after* the
    unit's teardown events so that those events still find the unit in place,
    and so the drop happens in queue order rather than eagerly at
    :meth:`Deployment.remove_unit` time.
    """


class _Rebind(NamedTuple):
    """Which object in the unit's state an event should be re-bound to.

    A relation or workload event carries the relation or container *object*,
    and the consistency checker requires it to equal the one in the state it is
    dispatched against — not merely to share its ID or name.  Events are queued
    before they are dispatched and the state moves in between, so an event
    queued with the object of the moment goes stale.  Such events record what
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


class _Stepper:
    """Dispatches queued events one at a time. See :meth:`Deployment.stepping`."""

    def __init__(self, deployment: Deployment):
        self._deployment = deployment

    def step(self) -> tuple[_Event, Unit] | None:
        """Dispatch exactly one queued event.

        Returns:
            The ``(event, unit)`` that was dispatched, or ``None`` if the queue
            was already empty.
        """
        return self._deployment._step()


class _DeploymentState:
    """The mutable half of a :class:`Deployment`.

    Held in one object so that a single ``object.__setattr__`` gets past the
    frozen dataclass this class inherits from, rather than one per field.
    """

    def __init__(self) -> None:
        self.apps: dict[str, App] = {}
        self.queue: deque[_Queued] = deque()
        self.trace: list[Dispatch] = []
        self.stepping = False
        self.closed = False


class Deployment(Model):
    """A set of applications, driven with Juju-shaped operations.

    A :class:`Deployment` is a :class:`Model` — it carries the same model
    identity, and that identity is stamped into every unit's :class:`State` —
    with operations added.  Construct it exactly as you would a ``Model``::

        deployment = testing.Deployment(name='my-model', type='lxd')

    Applications are added with :meth:`deploy`, which returns an :class:`App`
    handle::

        web = deployment.deploy('./charms/myapp', num_units=2)
        db = deployment.deploy(MyDatabaseCharm, app='db', meta=DB_META)

    Each operation enqueues the events Juju would emit for it.  The queue is
    drained by :meth:`settle`, which is also called implicitly the first time
    you read a unit's :attr:`Unit.state`, so most tests are a sequence of
    operations followed by assertions::

        deployment.update_config(web, {'log_level': 'debug'})
        assert web.leader.state.unit_status == testing.ActiveStatus('ready')

    Applications deployed from a path run in a subprocess with their own
    interpreter (see :class:`IsolatedContext`); applications deployed from a
    charm class run in the test process, with no isolation.  Call
    :meth:`close` — or use the deployment as a context manager — to tear down
    any worker processes.

    inline: ok
    isolated: ok
    """

    # -- internals ---------------------------------------------------------
    #
    # Model is a frozen dataclass, so there is no __init__ to hook: the
    # mutable state is created on first use instead.  That keeps Model's
    # generated constructor (and its defaults) as the single definition of how
    # a model identity is built.

    @property
    def _d(self) -> _DeploymentState:
        try:
            return self.__dict__['_deployment_state']
        except KeyError:
            state = _DeploymentState()
            object.__setattr__(self, '_deployment_state', state)
            return state

    def _as_model(self) -> Model:
        """This deployment's identity as a plain :class:`Model`.

        Unit states get this rather than ``self``: a ``State`` is data that
        may be serialised out to a worker process, and it should not carry a
        handle to the deployment that is driving it.
        """
        return Model(
            name=self.name,
            uuid=self.uuid,
            type=self.type,
            cloud_spec=self.cloud_spec,
        )

    def _check_open(self) -> None:
        if self._d.closed:
            raise DeploymentError('This deployment has been closed.')

    # -- operations --------------------------------------------------------

    def deploy(
        self,
        charm: CharmSource,
        app: str | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        num_units: int = 1,
        meta: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        actions: dict[str, Any] | None = None,
        python_executable: str | None = None,
        extra_sys_path: tuple[str, ...] = (),
        juju_version: str = _DEFAULT_JUJU_VERSION,
    ) -> App:
        """Deploy a charm, as ``juju deploy`` would.

        Args:
            charm: Charm source.  A path (``str`` or ``pathlib.Path``) to a
                charm repository root runs the charm isolated, in its own
                interpreter.  A :class:`ops.CharmBase` subclass runs it
                in-process, with no isolation — the same trade-off a plain
                ``Context`` test makes.
            app: Application name.  Defaults to the charm's name from its
                metadata.
            config: Application configuration.  Merged over the defaults
                declared in the charm's ``config.yaml``.
            num_units: How many units to deploy.  Unit ``0`` is the leader.
            meta: Charm metadata, in ``metadata.yaml`` form.  Read from the
                charm source when deploying from a path; required for an
                inline charm class that declares peers, containers, or
                relation endpoints.
            config_schema: Charm config schema, in ``config.yaml`` form.  Read
                from the charm source when deploying from a path.
            actions: Charm actions, in ``actions.yaml`` form.  Read from the
                charm source when deploying from a path.
            python_executable: Interpreter for the isolated worker.  Point it
                at a per-charm venv's ``bin/python`` to give this application
                its own dependencies.  Ignored for an inline charm class.
            extra_sys_path: Directories prepended to the worker's
                ``sys.path``.  Ignored for an inline charm class.
            juju_version: The Juju agent version to simulate.

        Returns:
            The :class:`App` handle for the new application.

        Raises:
            DeploymentError: if an application of this name already exists, or
                ``num_units`` is not positive.

        inline: ok
        isolated: ok
        """
        self._check_open()
        if num_units < 1:
            raise DeploymentError(f'num_units must be at least 1, not {num_units}.')

        if isinstance(charm, (str, pathlib.Path)):
            charm_root = pathlib.Path(charm)
            if meta is None:
                meta = _read_charm_metadata(charm_root)
            if config_schema is None:
                config_schema = _read_yaml(charm_root / 'config.yaml')
            if actions is None:
                actions = _read_yaml(charm_root / 'actions.yaml')
        elif meta is None:
            meta = {'name': app or getattr(charm, '__name__', 'charm').lower()}

        app_name = app or meta.get('name')
        if not app_name:
            raise DeploymentError('Could not determine the application name; pass app=.')
        if app_name in self._d.apps:
            raise DeploymentError(f'An application named {app_name!r} is already deployed.')

        runner: _Runner
        if isinstance(charm, (str, pathlib.Path)):
            runner = _IsolatedRunner(
                IsolatedContext(
                    charm_source=pathlib.Path(charm),
                    python_executable=python_executable,
                    extra_sys_path=extra_sys_path,
                    meta=meta,
                    config=config_schema,
                    actions=actions,
                    app_name=app_name,
                    juju_version=juju_version,
                )
            )
        else:
            runner = _InProcessRunner(
                charm,
                meta=meta,
                config=config_schema,
                actions=actions,
                app_name=app_name,
                juju_version=juju_version,
            )

        new_app = App(
            self,
            app_name,
            runner,
            meta=meta,
            config_schema=config_schema,
            actions=actions,
            config=_merged_config(config_schema, config),
        )
        self._d.apps[app_name] = new_app

        for _ in range(num_units):
            self._add_unit(new_app, startup=True)
        return new_app

    def add_unit(self, app: App) -> Unit:
        """Add a unit to an application, as ``juju add-unit`` would.

        The new unit sees the usual startup sequence.  If the application has a
        peer relation, the units that were already there see the new unit join
        it.

        Returns:
            The new :class:`Unit`.

        inline: ok
        isolated: ok
        """
        self._check_open()
        return self._add_unit(app, startup=True)

    def remove_unit(self, app: App, unit: int | Unit | None = None) -> None:
        """Remove a unit from an application, as ``juju remove-unit`` would.

        Args:
            app: The application to remove a unit from.
            unit: Which unit, as a :class:`Unit` or a unit number.  Defaults to
                the highest-numbered unit, which is the one Juju removes.

        Raises:
            DeploymentError: when removing the last unit of an application, or
                when the unit does not belong to ``app``.

        inline: ok
        isolated: ok
        """
        self._check_open()
        if unit is None:
            unit_id = max(app._units)
        elif isinstance(unit, Unit):
            if unit.app is not app:
                raise DeploymentError(f'{unit.name} is not a unit of {app.name}.')
            unit_id = unit.id
        else:
            unit_id = unit
        if unit_id not in app._units:
            raise DeploymentError(f'{app.name} has no unit {unit_id}.')
        if len(app._units) == 1:
            raise DeploymentError(
                f'Cannot remove the last unit of {app.name}; remove the application instead.'
            )

        departing = app._units[unit_id]
        remaining = [u for u in app.units if u is not departing]

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

        self._d.queue.append(_Queued(app, unit_id, _RemoveUnit()))

    def update_config(self, app: App, config: Mapping[str, Any]) -> None:
        """Change an application's configuration, as ``juju config`` would.

        The new values are merged over the existing ones, and every unit sees
        ``config-changed``.

        inline: ok
        isolated: ok
        """
        self._check_open()
        app._config.update(config)
        for unit in app.units:
            unit._state = dataclasses.replace(unit._state, config=dict(app._config))
            self._enqueue(app, unit.id, _Event('config_changed'))

    def run_action(
        self,
        target: App | Unit,
        action: str,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Run an action, as ``juju run`` would.

        Args:
            target: The unit to run the action on.  Passing an :class:`App`
                runs it on that application's leader, matching
                ``juju run <app>/leader``.
            action: The action name, as declared in ``actions.yaml``.
            params: Parameters for the action.

        .. note::
            The action's results are not yet returned — this queues the action
            event and lets the charm handle it, so its effect on the unit's
            :class:`State` is observable, but ``ActionOutput`` is not plumbed
            through the isolation boundary yet.

        inline: ok
        isolated: ok
        """
        self._check_open()
        unit = target.leader if isinstance(target, App) else target
        self._enqueue(unit.app, unit.id, _action_event(action, params))

    # -- convergence -------------------------------------------------------

    def settle(self, max_events: int = 1000) -> list[Dispatch]:
        """Drain the event queue until the deployment converges.

        Convergence is reached when the queue is empty: every event produced
        by an operation has been dispatched, along with every follow-on event
        those dispatches produced.

        Args:
            max_events: Safety valve.  Raises :class:`DeploymentError` rather
                than looping forever if the queue does not drain — most likely
                a charm and its peers writing to each other's databags on
                every event.

        Returns:
            The events dispatched, in order, each with the unit it went to and
            that unit's :class:`State` afterwards.  Appending to this record
            does not change the default settle path in any other way.

        inline: ok
        isolated: ok
        """
        self._check_open()
        self._d.trace = []
        dispatched = 0
        while self._d.queue:
            dispatched += 1
            if dispatched > max_events:
                raise DeploymentError(
                    f'Deployment did not converge after {max_events} events; '
                    'the queue is still not empty. Raise max_events if this is '
                    'expected, or check for charms that write to a databag on '
                    'every event.'
                )
            self._step()
        return list(self._d.trace)

    @contextmanager
    def stepping(self) -> Generator[_Stepper, None, None]:
        """Dispatch queued events one at a time, for mid-convergence assertions.

        Inside the block, reading a unit's :attr:`Unit.state` does *not*
        implicitly settle, so an assertion between steps sees exactly the state
        the last step produced::

            with deployment.stepping() as stepper:
                stepper.step()
                assert web.leader.state.unit_status == testing.MaintenanceStatus('setting up')
                stepper.step()

        Leaving the block re-enables implicit settling; it does not drain
        whatever is left in the queue.

        inline: ok
        isolated: ok
        """
        self._check_open()
        was_stepping = self._d.stepping
        self._d.stepping = True
        try:
            yield _Stepper(self)
        finally:
            self._d.stepping = was_stepping

    def _settle_if_pending(self) -> None:
        if self._d.queue and not self._d.stepping and not self._d.closed:
            self.settle()

    def _step(self) -> tuple[_Event, Unit] | None:
        """Dispatch the single event at the head of the queue."""
        if not self._d.queue:
            return None
        app, unit_id, event, rebind = self._d.queue.popleft()

        if isinstance(event, _RemoveUnit):
            del app._units[unit_id]
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
        state_out = app._runner.run(unit_id, event, unit._state)
        unit._state = state_out
        self._d.trace.append(Dispatch(event, unit, state_out))
        self._propagate_peers(app, unit)
        return event, unit

    # -- units and peer relations -----------------------------------------

    def _add_unit(self, app: App, *, startup: bool) -> Unit:
        unit_id = app._next_unit_id
        app._next_unit_id += 1
        existing = app.units

        unit = Unit(app, unit_id, self._initial_state(app, unit_id))
        app._units[unit_id] = unit

        # Everyone's planned-units count moves as soon as the unit exists.
        for other in app.units:
            other._state = dataclasses.replace(other._state, planned_units=len(app._units))

        # Existing units see the newcomer join the peer relation before the
        # newcomer itself starts up.  Juju follows relation-joined with
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
                if not startup:
                    continue
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

        if startup:
            for event, rebind in _startup_events(app, unit_id):
                self._enqueue(app, unit_id, event, rebind)
        return unit

    def _initial_state(self, app: App, unit_id: int) -> State:
        relations: list[PeerRelation] = []
        for endpoint in app._peer_endpoints:
            # A unit joining an existing application can read what its peers
            # have already published, so seed its view from theirs rather than
            # starting it empty — otherwise the first peer to run afterwards
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
        return State(
            config=dict(app._config),
            relations=relations,
            containers=app._containers,
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
        unit holds its own view of it.  After a unit runs, whatever it wrote to
        its own unit databag (or, as leader, to the application databag) has to
        appear in every other unit's view — and any unit whose view actually
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

    def _enqueue(
        self,
        app: App,
        unit_id: int,
        event: _Event,
        rebind: _Rebind | None = None,
    ) -> None:
        self._d.queue.append(_Queued(app, unit_id, event, rebind))

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Tear down every application's worker process.

        Safe to call more than once.  In-process applications have nothing to
        tear down.
        """
        for app in self._d.apps.values():
            app._runner.close()
        self._d.apps.clear()
        self._d.queue.clear()
        self._d.closed = True

    def __enter__(self) -> Deployment:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _merged_config(
    config_schema: dict[str, Any] | None,
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


def _action_event(name: str, params: Mapping[str, Any] | None) -> _Event:
    kwargs: dict[str, Any] = {}
    if params:
        kwargs['params'] = dict(params)
    return _Event(f'{name}_action', action=_Action(name, **kwargs))
