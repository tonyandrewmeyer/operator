# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.

"""Juju and Pebble mocking.

This module contains mocks for the Juju and Pebble APIs that are used by ops
to interact with the Juju controller and the Pebble service manager.
"""

from __future__ import annotations

import copy
import datetime
import heapq
import io
import shutil
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    TextIO,
    cast,
    get_args,
)

from ops import (
    JujuContext,
    JujuVersion,
    ModelError,
    RelationNotFoundError,
    SecretInfo,
    SecretNotFoundError,
    SecretRotate,
    pebble,
)
from ops._private.harness import ExecArgs, _TestingPebbleClient
from ops.model import CloudSpec as CloudSpec_Ops
from ops.model import Port as Port_Ops
from ops.model import Secret as Secret_Ops  # lol
from ops.model import (
    _format_action_result_dict,
    _ModelBackend,
    _SettableStatusName,
)
from ops.pebble import Client, ExecError

from .errors import ActionMissingFromContextError
from .logger import logger as scenario_logger
from .state import (
    CharmType,
    CheckInfo,
    JujuLogLine,
    Mount,
    Network,
    Notice,
    PeerRelation,
    Relation,
    RelationBase,
    ServiceBehaviour,
    ServiceExitCode,
    ServiceFailureMode,
    ServiceStart,
    Storage,
    SubordinateRelation,
    _EntityStatus,
    _port_cls_by_protocol,
    _RawPortProtocolLiteral,
)

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
    from .state import Container as ContainerSpec
    from .state import Exec, Secret, State, _CharmSpec, _Event

logger = scenario_logger.getChild('mocking')


def _rfc3339_now() -> str:
    """An RFC3339 timestamp for the current time, for synthesised Pebble task logs."""
    return (
        datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    )


def _pebble_task(
    name: str,
    kind: str,
    status: str,
    spawn_time: datetime.datetime,
    ready_time: datetime.datetime,
    log: list[str] | None = None,
) -> pebble.Task:
    """A single-service task for a synthesised start/stop/restart change."""
    return pebble.Task(
        id=pebble.TaskID(str(uuid.uuid4())),
        kind=kind,
        summary=f'{kind.capitalize()} service "{name}"',
        status=status,
        log=log or [],
        progress=pebble.TaskProgress(label='', done=1, total=1),
        spawn_time=spawn_time,
        ready_time=ready_time,
    )


class _MockExecProcess:
    def __init__(
        self,
        change_id: int,
        args: ExecArgs,
        return_code: int,
        stdin: TextIO | io.BytesIO | None,
        stdout: TextIO | io.BytesIO | None,
        stderr: TextIO | io.BytesIO | None,
    ):
        self._change_id = change_id
        self._args = args
        self._return_code = return_code
        self._waited = False
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

    def __del__(self):
        if not self._waited:
            self._close_stdin()

    def _close_stdin(self):
        if self._args.stdin is None and self.stdin is not None:
            self.stdin.seek(0)
            self._args.stdin = self.stdin.read()

    def wait(self):
        """Wait for the (mock) process to finish."""
        self._close_stdin()
        self._waited = True
        if self._return_code != 0:
            raise ExecError(list(self._args.command), self._return_code, None, None)

    def wait_output(self):
        """Wait for the (mock) process to finish and return tuple of (stdout, stderr)."""
        self._close_stdin()
        self._waited = True
        stdout = self.stdout.read() if self.stdout is not None else None
        stderr = self.stderr.read() if self.stderr is not None else None
        if self._return_code != 0:
            raise ExecError(
                list(self._args.command),
                self._return_code,
                stdout,  # type: ignore
                stderr,  # type: ignore
            )
        return stdout, stderr

    def send_signal(self, sig: int | str) -> NoReturn:
        """Send the given signal to the (mock) process."""
        raise NotImplementedError()


_NOT_GIVEN = object()  # non-None default value sentinel


# pyright: reportIncompatibleMethodOverride=false
class _MockModelBackend(_ModelBackend):  # type: ignore
    def __init__(
        self,
        state: State,
        event: _Event,
        charm_spec: _CharmSpec[CharmType],
        context: Context[CharmType],
        juju_context: JujuContext,
    ):
        super().__init__(juju_context=juju_context)
        self._state = state
        self._event = event
        self._context = context
        self._charm_spec = charm_spec

    def opened_ports(self) -> set[Port_Ops]:
        return {
            Port_Ops(protocol=port.protocol, port=port.port) for port in self._state.opened_ports
        }

    def open_port(
        self,
        protocol: _RawPortProtocolLiteral,
        port: int | None = None,
    ):
        port_ = _port_cls_by_protocol[protocol](port=port)  # type: ignore
        ports = set(self._state.opened_ports)
        if port_ not in ports:
            ports.add(port_)
        if ports != self._state.opened_ports:
            self._state._update_opened_ports(frozenset(ports))

    def close_port(
        self,
        protocol: _RawPortProtocolLiteral,
        port: int | None = None,
    ):
        port_ = _port_cls_by_protocol[protocol](port=port)  # type: ignore
        ports = set(self._state.opened_ports)
        if port_ in ports:
            ports.remove(port_)
        if ports != self._state.opened_ports:
            self._state._update_opened_ports(frozenset(ports))

    def get_pebble(self, socket_path: str) -> Client:
        container_name = socket_path.split('/')[
            3
        ]  # /charm/containers/<container_name>/pebble.socket
        container_root = self._context._get_container_root(container_name)
        mounts: Mapping[str, Mount]
        try:
            mounts = self._state.get_container(container_name).mounts
        except KeyError:
            # container not defined in state.
            mounts = {}

        return cast(
            'Client',
            _MockPebbleClient(
                socket_path=socket_path,
                container_root=container_root,
                mounts=mounts,
                state=self._state,
                event=self._event,
                charm_spec=self._charm_spec,
                context=self._context,
                container_name=container_name,
            ),
        )

    def _get_relation_by_id(self, rel_id: int, relation_name: str | None = None) -> RelationBase:
        try:
            relation = self._state.get_relation(rel_id)
        except KeyError:
            raise RelationNotFoundError() from None
        if relation_name is not None and relation.endpoint != relation_name:
            # Mimic Juju's behaviour when given a mismatched ``endpoint:id``.
            raise RelationNotFoundError()
        return relation

    def _get_secret(self, id: str | None = None, label: str | None = None):
        if JujuVersion(self._context.juju_version) < '3.0.2':
            raise ModelError(
                'secrets are only available in juju >= 3.0.2.'
                'Set ``Context.juju_version`` to 3.0.2+ to use them.',
            )

        canonicalize_id = Secret_Ops._canonicalize_id

        if id:
            # in scenario, you can create Secret(id="foo"),
            # but ops.Secret will prepend a "secret:" prefix to that ID.
            # we allow getting secret by either version.
            secrets = [
                s
                for s in self._state.secrets
                if canonicalize_id(s.id, model_uuid=self._state.model.uuid)
                == canonicalize_id(id, model_uuid=self._state.model.uuid)
            ]
            if not secrets:
                raise SecretNotFoundError(id)
            return secrets[0]

        if label:
            try:
                return self._state.get_secret(label=label)
            except KeyError:
                raise SecretNotFoundError(label) from None

        # if all goes well, this should never be reached. ops.model.Secret will check upon
        # instantiation that either an id or a label are set, and raise a TypeError if not.
        raise RuntimeError('need id or label.')

    def _check_app_data_access(self, is_app: bool):
        if not isinstance(is_app, bool):
            raise TypeError('is_app parameter to relation_get must be a boolean')

        if not is_app:
            return

        version = JujuVersion(self._context.juju_version)
        if not version.has_app_data():
            raise RuntimeError(
                f'setting application data is not supported on Juju version {version}',
            )

    def relation_get(
        self,
        relation_id: int,
        member_name: str,
        is_app: bool,
        *,
        relation_name: str | None = None,
    ):
        self._check_app_data_access(is_app)
        data = self._relation_get(
            relation_id, member_name=member_name, is_app=is_app, relation_name=relation_name
        )
        return dict(data)

    def _relation_get(
        self,
        relation_id: int,
        member_name: str,
        is_app: bool,
        relation_name: str | None = None,
    ):
        relation = self._get_relation_by_id(relation_id, relation_name)
        if is_app and member_name == self.app_name:
            return relation.local_app_data
        if is_app:
            if isinstance(relation, PeerRelation):
                return relation.local_app_data
            if isinstance(relation, (Relation, SubordinateRelation)):
                return relation.remote_app_data
            raise TypeError('relation_get: unknown relation type')
        if member_name == self.unit_name:
            return relation.local_unit_data

        unit_id = int(member_name.split('/')[-1])
        return relation._get_databag_for_remote(unit_id)

    def relation_model_get(
        self, relation_id: int, *, relation_name: str | None = None
    ) -> dict[str, Any]:
        if JujuVersion(self._context.juju_version) < '3.6.2':
            raise ModelError('Relation.remote_model is only available on Juju >= 3.6.2')

        relation = self._get_relation_by_id(relation_id, relation_name)
        # Only Relation has remote_model_uuid, not the other subclasses of RelationBase.
        if isinstance(relation, Relation) and relation.remote_model_uuid is not None:
            uuid = relation.remote_model_uuid
        else:
            uuid = self._state.model.uuid
        return {'uuid': uuid}

    def is_leader(self):
        return self._state.leader

    def status_get(self, *, is_app: bool = False):
        status = self._state.app_status if is_app else self._state.unit_status
        return {'status': status.name, 'message': status.message}

    def relation_ids(self, relation_name: str):
        return [rel.id for rel in self._state.relations if rel.endpoint == relation_name]

    def relation_list(
        self, relation_id: int, *, relation_name: str | None = None
    ) -> tuple[str, ...]:
        relation = self._get_relation_by_id(relation_id, relation_name)

        if isinstance(relation, PeerRelation):
            # The current unit should never be in `peers_data`, and there is a
            # consistency check to enforce that, but in case something has gone
            # wrong, filter it out to match Juju's behaviour.
            this_unit = int(self.unit_name.split('/')[-1])
            return tuple(
                f'{self.app_name}/{unit_id}'
                for unit_id in relation.peers_data
                if unit_id != this_unit
            )
        remote_name = self.relation_remote_app_name(relation_id, relation_name=relation_name)
        return tuple(f'{remote_name}/{unit_id}' for unit_id in relation._remote_unit_ids)

    def config_get(self) -> dict[str, str | int | float | bool]:
        state_config = dict(self._state.config)  # dedup or we'll mutate the state!

        # add defaults
        charm_config = self._charm_spec.config
        if not charm_config:
            return state_config

        for key, value in charm_config['options'].items():
            # if it has a default, and it's not overwritten from State, use it:
            if key not in state_config:
                default_value = value.get('default', _NOT_GIVEN)
                if default_value is not _NOT_GIVEN:  # accept False as default value
                    state_config[key] = default_value

        return state_config  # full config

    def network_get(self, binding_name: str, relation_id: int | None = None):
        # validation:
        extra_bindings = self._charm_spec.meta.get('extra-bindings', ())
        all_endpoints = self._charm_spec.get_all_relations()
        non_sub_relations = {
            name for name, meta in all_endpoints if meta.get('scope') != 'container'
        }

        # - is binding_name a valid binding name?
        if binding_name in extra_bindings:
            logger.warning('extra-bindings is a deprecated feature')  # fyi

            # - verify that if the binding is an extra binding, we're not ignoring a relation_id
            if relation_id is not None:
                # this should not happen
                logger.error(
                    'cannot pass relation_id to network_get if the binding name is '
                    'that of an extra-binding. Extra-bindings are not mapped to relation IDs.',
                )
        elif binding_name == 'juju-info':
            # implicit relation that always exists
            pass
        # - verify that the binding is a relation endpoint name, but not a subordinate one
        elif binding_name not in non_sub_relations:
            logger.error(
                f'cannot get network binding for {binding_name}: is not a valid relation '
                f'endpoint name nor an extra-binding.',
            )
            raise RelationNotFoundError()

        # We look in State.networks for an override. If not given, we return a default network.
        try:
            network = self._state.get_network(binding_name)
        except KeyError:
            network = Network('default')  # The name is not used in the output.
        return network._hook_tool_output_fmt()

    # setter methods: these can mutate the state.
    def application_version_set(self, version: str):
        if workload_version := self._state.workload_version:
            # do not record if empty = unset
            self._context.workload_version_history.append(workload_version)

        self._state._update_workload_version(version)

    def status_set(
        self,
        status: _SettableStatusName,
        message: str = '',
        *,
        is_app: bool = False,
    ):
        valid_names = get_args(_SettableStatusName)
        if status not in valid_names:
            raise ModelError(
                f'ERROR invalid status "{status}", expected one of [{", ".join(valid_names)}]',
            )
        self._context._record_status(self._state, is_app)
        status_obj = _EntityStatus.from_status_name(status, message)
        self._state._update_status(status_obj, is_app)

    def juju_log(self, level: str, message: str):
        self._context.juju_log.append(JujuLogLine(level, message))

    def relation_set(
        self,
        relation_id: int,
        data: Mapping[str, str],
        is_app: bool,
        *,
        relation_name: str | None = None,
    ) -> None:
        self._check_app_data_access(is_app)
        # NOTE: The code below currently does not have any effect, because
        # the dictionary has already had the same set/delete operations
        # applied to it by RelationDataContent -- unlike in production,
        # where this method calls out to Juju's relation-set to operate on
        # the real databag, this method currently operates on the same
        # dictionary object that RelationDataContent does.
        relation = self._get_relation_by_id(relation_id, relation_name)
        if is_app:
            if not self._state.leader:
                # will in practice not be reached because RelationData will check leadership
                # and raise RelationDataAccessError upstream on this path
                raise RuntimeError('needs leadership to set app data')
            tgt = cast('dict[str, str]', relation.local_app_data)
        else:
            tgt = cast('dict[str, str]', relation.local_unit_data)
        for key, value in data.items():
            if value == '':
                # Match the behavior of Juju, which is that setting the value to an
                # empty string will remove the key entirely from the relation data.
                tgt.pop(key, None)
            else:
                tgt[key] = value

    def secret_add(
        self,
        content: dict[str, str],
        *,
        label: str | None = None,
        description: str | None = None,
        expire: datetime.datetime | None = None,
        rotate: SecretRotate | None = None,
        owner: Literal['unit', 'application'] | None = None,
    ) -> str:
        from .state import Secret

        secret = Secret(
            content,
            label=label,
            description=description,
            expire=expire,
            rotate=rotate,
            # It's called 'application' in Ops, but 'app' in Scenario.
            owner='app' if owner == 'application' else owner,
        )
        secrets = set(self._state.secrets)
        secrets.add(secret)
        self._state._update_secrets(frozenset(secrets))
        return secret.id

    def _check_can_manage_secret(
        self,
        secret: Secret,
    ):
        if secret.owner is None:
            # The secret is in State.secrets (otherwise _get_secret would have
            # raised already), so the charm has read access; it just doesn't
            # own the secret and so cannot manage it. Set `Secret.owner` to
            # 'unit' or 'app' in the test setup if the charm is meant to own it.
            raise SecretNotFoundError(
                f'You must own secret {secret.id!r} to perform this operation',
            )
        if secret.owner == 'app' and not self.is_leader():
            understandable_error = SecretNotFoundError(
                f'App-owned secret {secret.id!r} can only be managed by the leader.',
            )
            # charm-facing side: respect ops error
            raise ModelError('ERROR permission denied') from understandable_error

    def secret_get(
        self,
        *,
        id: str | None = None,
        label: str | None = None,
        refresh: bool = False,
        peek: bool = False,
    ) -> dict[str, str]:
        secret = self._get_secret(id, label)
        # If both the id and label are provided, then update the label.
        if id is not None and label is not None:
            secret._set_label(label)
        juju_version = JujuVersion(self._context.juju_version)
        # In this medieval Juju chapter,
        # secret owners always used to track the latest revision.
        # ref: https://bugs.launchpad.net/juju/+bug/2037120
        if not (juju_version == '3.1.7' or juju_version >= '3.3.1') and secret.owner is not None:
            refresh = True

        if peek or refresh:
            if refresh:
                secret._track_latest_revision()
            assert secret.latest_content is not None
            return dict(secret.latest_content)

        return dict(secret.tracked_content)

    def secret_info_get(
        self,
        *,
        id: str | None = None,
        label: str | None = None,
    ) -> SecretInfo:
        secret = self._get_secret(id, label)
        # If both the id and label are provided, then update the label.
        if id is not None and label is not None:
            secret._set_label(label)

        # only "manage"=write access level can read secret info
        self._check_can_manage_secret(secret)

        return SecretInfo(
            id=secret.id,
            label=secret.label,
            description=secret.description,
            revision=secret._latest_revision,
            expires=secret.expire,
            rotation=secret.rotate,
            rotates=None,  # not implemented yet.
            model_uuid=self._state.model.uuid,
        )

    def secret_set(
        self,
        id: str,
        *,
        content: dict[str, str] | None = None,
        label: str | None = None,
        description: str | None = None,
        expire: datetime.datetime | None = None,
        rotate: SecretRotate | None = None,
    ):
        secret = self._get_secret(id, label)
        self._check_can_manage_secret(secret)

        if content == secret.latest_content:
            # In Juju 3.6 and higher, this is a no-op, but it's good to warn
            # charmers if they are doing this, because it's not generally good
            # practice.
            # https://bugs.launchpad.net/juju/+bug/2069238
            logger.warning(
                f'secret {id} contents set to the existing value: new revision not needed',
            )

        secret._update_metadata(
            content=content,
            label=label,
            description=description,
            expire=expire,
            rotate=rotate,
        )

    def secret_grant(self, id: str, relation_id: int, *, unit: str | None = None):
        secret = self._get_secret(id)
        self._check_can_manage_secret(secret)

        grantee = unit or self.relation_remote_app_name(
            relation_id,
            _raise_on_error=True,
        )
        assert grantee is not None  # guaranteed by _raise_on_error=True
        grants = cast('dict[int, frozenset[str]]', secret.remote_grants)
        grants[relation_id] = grants.get(relation_id, frozenset()).union({grantee})

    def secret_revoke(self, id: str, relation_id: int, *, unit: str | None = None):
        secret = self._get_secret(id)
        self._check_can_manage_secret(secret)

        grantee = unit or self.relation_remote_app_name(
            relation_id,
            _raise_on_error=True,
        )
        assert grantee is not None  # guaranteed by _raise_on_error=True
        grants = cast('dict[int, frozenset[str]]', secret.remote_grants)
        grants[relation_id] = grants[relation_id].difference({grantee})
        if not grants[relation_id]:
            del grants[relation_id]

    def secret_remove(self, id: str, *, revision: int | None = None):
        secret = self._get_secret(id)
        self._check_can_manage_secret(secret)

        # Removing all revisions means that the secret is removed, so is no
        # longer in the state.
        if revision is None:
            secrets = set(self._state.secrets)
            secrets.remove(secret)
            self._state._update_secrets(frozenset(secrets))
            return

        # Juju does not prevent removing the tracked or latest revision, but it
        # is always a mistake to do this. Rather than having the state model a
        # secret where the tracked/latest revision cannot be retrieved but the
        # secret still exists, we raise instead, so that charms know that there
        # is a problem with their code.
        if revision in (secret._tracked_revision, secret._latest_revision):
            raise ValueError(
                'Charms should not remove the latest revision of a secret. '
                'Add a new revision with `set_content()` instead, and the previous '
                'revision will be cleaned up by the secret owner when no longer in use.',
            )

        # For all other revisions, the content is not visible to the charm
        # (this is as designed: the secret is being removed, so it should no
        # longer be in use). That means that the state does not need to be
        # modified - however, unit tests should be able to verify that the remove call was
        # executed, so we track that in a history list in the context.
        self._context.removed_secret_revisions.append(revision)

    def relation_remote_app_name(
        self,
        relation_id: int,
        *,
        relation_name: str | None = None,
        _raise_on_error: bool = False,
    ) -> str | None:
        # ops catches RelationNotFoundErrors and returns None:
        try:
            relation = self._get_relation_by_id(relation_id, relation_name)
        except RelationNotFoundError:
            if _raise_on_error:
                raise
            return None

        if isinstance(relation, PeerRelation):
            return self.app_name
        if isinstance(relation, (Relation, SubordinateRelation)):
            return relation.remote_app_name
        raise TypeError('relation_remote_app_name: unknown relation type')

    def action_set(self, results: dict[str, Any]):
        if not self._event.action:
            raise ActionMissingFromContextError(
                'not in the context of an action event: cannot action-set',
            )
        # let ops validate the results dict
        _format_action_result_dict(results)
        # but then we will store it in its unformatted,
        # original form for testing ease
        if self._context.action_results:
            self._context.action_results.update(results)
        else:
            self._context.action_results = results

    def action_fail(self, message: str = ''):
        if not self._event.action:
            raise ActionMissingFromContextError(
                'not in the context of an action event: cannot action-fail',
            )
        self._context._action_failure_message = message

    def action_log(self, message: str):
        if not self._event.action:
            raise ActionMissingFromContextError(
                'not in the context of an action event: cannot action-log',
            )
        self._context.action_logs.append(message)

    def action_get(self):
        action = self._event.action
        if not action:
            raise ActionMissingFromContextError(
                'not in the context of an action event: cannot action-get',
            )
        return copy.deepcopy(action.params)

    def storage_add(self, name: str, count: int = 1):
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(
                f'storage count must be integer, got: {count} ({type(count)})',
            )

        if '/' in name:
            # this error is raised by Harness but not by ops at runtime
            raise ModelError('storage name cannot contain "/"')

        self._context.requested_storages[name] = count

    def storage_list(self, name: str) -> list[int]:
        return [storage.index for storage in self._state.storages if storage.name == name]

    def _storage_event_details(self) -> tuple[int, str]:
        storage = self._event.storage
        if not storage:
            # only occurs if this method is called when outside the scope of a storage event
            raise RuntimeError('unable to find storage key in ""')
        fs_path = storage.get_filesystem(self._context)
        return storage.index, str(fs_path)

    def storage_get(self, storage_name_id: str, attribute: str) -> str:
        if not len(attribute) > 0:  # assume it's an empty string.
            raise RuntimeError(
                'calling storage_get with `attribute=""` will return a dict '
                'and not a string. This usage is not supported.',
            )

        if attribute != 'location':
            # this should not happen: in ops it's hardcoded to be "location"
            raise NotImplementedError(
                f'storage-get not implemented for attribute={attribute}',
            )

        name, index = storage_name_id.split('/')
        index = int(index)
        storages: list[Storage] = [
            s for s in self._state.storages if s.name == name and s.index == index
        ]

        # should not really happen: sanity checks. In practice, ops will guard against these paths.
        if not storages:
            raise RuntimeError(f'Storage with name={name} and index={index} not found.')
        if len(storages) > 1:
            raise RuntimeError(
                f'Multiple Storage instances with name={name} and index={index} found. '
                f'Inconsistent state.',
            )

        storage = storages[0]
        fs_path = storage.get_filesystem(self._context)
        return str(fs_path)

    def planned_units(self) -> int:
        return self._state.planned_units

    # legacy ops API that we don't intend to mock:
    def pod_spec_set(
        self,
        spec: Mapping[str, Any],
        k8s_resources: Mapping[str, Any] | None = None,
    ) -> NoReturn:
        raise NotImplementedError(
            'pod-spec-set is not implemented in Scenario (and probably never will be: '
            "it's deprecated API)",
        )

    def add_metrics(
        self,
        metrics: Mapping[str, int | float],
        labels: Mapping[str, str] | None = None,
    ) -> NoReturn:
        raise NotImplementedError(
            'add-metrics is not implemented in Scenario '
            '(and never will be: it was removed in Juju 3.6.11)'
        )

    def resource_get(self, resource_name: str) -> str:
        # We assume that there are few enough resources that a linear search
        # will perform well enough.
        for resource in self._state.resources:
            if resource.name == resource_name:
                return str(resource.path)
        # ops will not let us get there if the resource name is unknown from metadata.
        # but if the user forgot to add it in State, then we remind you of that.
        raise RuntimeError(
            f'Inconsistent state: resource {resource_name} not found in State. please pass it.',
        )

    def credential_get(self) -> CloudSpec_Ops:
        if not self._context.app_trusted:
            raise ModelError(
                'ERROR charm is not trusted, initialise Context with `app_trusted=True`',
            )
        if not self._state.model.cloud_spec:
            raise ModelError(
                'ERROR cloud spec is empty, initialise it with '
                '`State(model=Model(..., cloud_spec=ops.CloudSpec(...)))`',
            )
        return self._state.model.cloud_spec._to_ops()


class _MockPebbleClient(_TestingPebbleClient):
    def __init__(
        self,
        socket_path: str,
        container_root: Path,
        mounts: Mapping[str, Mount],
        *,
        state: State,
        event: _Event,
        charm_spec: _CharmSpec[CharmType],
        context: Context[CharmType],
        container_name: str,
    ):
        self._state = state
        self.socket_path = socket_path
        self._event = event
        self._charm_spec = charm_spec
        self._context = context
        self._container_name = container_name

        # wipe just in case
        if container_root.exists():
            if any(container_root.iterdir()):
                logger.warning(
                    'Container %r has a non-empty filesystem that will be wiped before this run.'
                    ' If you are trying to mock filesystem contents, use Mounts instead.'
                    ' If you are reusing a Context instance across multiple ctx.run() calls,'
                    ' note that the filesystem is cleared between runs.',
                    container_name,
                )
            # Path.rmdir will fail if root is nonempty
            shutil.rmtree(container_root)

        # initialize simulated filesystem
        container_root.mkdir(parents=True)
        for mount in mounts.values():
            path = Path(mount.location).parts
            mounting_dir = container_root.joinpath(*path[1:])
            mounting_dir.parent.mkdir(parents=True, exist_ok=True)
            mounting_dir.symlink_to(mount.source)

        self._root = container_root

        self._notices: dict[tuple[str, str], pebble.Notice] = {}
        self._last_notice_id = 0
        self._changes: dict[str, pebble.Change] = {}

        # load any existing notices and check information from the state
        self._notices: dict[tuple[str, str], pebble.Notice] = {}
        self._check_infos: dict[str, pebble.CheckInfo] = {}
        try:
            container = state.get_container(self._container_name)
        except KeyError:
            # The container is in the metadata but not in the state - perhaps
            # this is an install event, at which point the container doesn't
            # exist yet. This means there will be no notices or check infos.
            pass
        else:
            for notice in container.notices:
                if hasattr(notice.type, 'value'):
                    notice_type = cast('pebble.NoticeType', notice.type).value
                else:
                    notice_type = str(notice.type)
                self._notices[notice_type, notice.key] = notice._to_ops()
            now = datetime.datetime.now()
            for check in container.check_infos:
                self._check_infos[check.name] = check._to_ops()
                kind = (
                    pebble.ChangeKind.PERFORM_CHECK.value
                    if check.status == pebble.CheckStatus.UP
                    else pebble.ChangeKind.RECOVER_CHECK.value
                )
                change = pebble.Change(
                    pebble.ChangeID(str(uuid.uuid4())),
                    kind,
                    summary=check.name,
                    status=pebble.ChangeStatus.DOING.value,
                    tasks=[],
                    ready=False,
                    err=None,
                    spawn_time=now,
                    ready_time=now,
                )
                assert check.change_id is not None
                self._changes[check.change_id] = change

    def get_plan(self) -> pebble.Plan:
        return self._container.plan

    def _update_state_check_infos(self):
        """Copy any new or changed check infos into the state."""
        infos: set[CheckInfo] = set()
        for info in self._check_infos.values():
            level = pebble.CheckLevel(info.level) if isinstance(info.level, str) else info.level
            if isinstance(info.status, str):
                status = pebble.CheckStatus(info.status)
            else:
                status = info.status
            check_info = CheckInfo(
                name=info.name,
                level=level,
                startup=info.startup,
                status=status,
                successes=info.successes,
                failures=info.failures,
                threshold=info.threshold,
                change_id=info.change_id,
            )
            infos.add(check_info)
        object.__setattr__(self._container, 'check_infos', frozenset(infos))

    def _update_state_notices(self):
        """Copy any new or changed notices into the state.

        Pebble keeps a notice once it has been recorded, so a notice the
        workload or the charm produced during the run belongs in the output
        state -- that is what lets the next run handle it.
        """
        object.__setattr__(
            self._container,
            'notices',
            [Notice._from_ops(notice) for notice in self._notices.values()],
        )

    def notify(
        self,
        type: pebble.NoticeType,
        key: str,
        *,
        data: dict[str, str] | None = None,
        repeat_after: datetime.timedelta | None = None,
    ) -> str:
        notice_id = super().notify(type, key, data=data, repeat_after=repeat_after)
        self._update_state_notices()
        return notice_id

    def _emit_service_notices(self, services: Iterable[str]) -> None:
        """Record the notices declared for each service that just started."""
        emitted = False
        for name in services:
            behaviour = self._find_service_behaviour(name)
            if behaviour is None or behaviour.start is ServiceStart.FAILS:
                # A service that never started can't have notified anything.
                continue
            for notice in behaviour.emits:
                notice_type = (
                    notice.type
                    if isinstance(notice.type, pebble.NoticeType)
                    else pebble.NoticeType(notice.type)
                )
                self._notify(notice_type, notice.key, data=dict(notice.last_data))
                emitted = True
        if emitted:
            self._update_state_notices()

    def replan_services(self, timeout: float = 30.0, delay: float = 0.1):
        # Real Pebble fails a replan the same way it fails an autostart when
        # an enabled service won't start, with kind='replan'.
        enabled = self._enabled_service_names()
        known_services = self._render_services()
        if self._needs_behaviour_path(enabled, known_services):
            return self._start_with_behaviours(enabled, kind='replan')
        super().replan_services(timeout=timeout, delay=delay)
        self._apply_exit_behaviours(enabled)
        self._emit_service_notices(enabled)
        self._update_state_check_infos()

    def start_services(
        self,
        services: list[str],
        timeout: float = 30.0,
        delay: float = 0.1,
    ) -> pebble.ChangeID:
        if isinstance(services, str):
            raise TypeError(f'start_services should take a list of names, not just "{services}"')
        known_services = self._render_services()
        if self._needs_behaviour_path(services, known_services):
            return self._start_with_behaviours(services)
        change_id = super().start_services(services, timeout=timeout, delay=delay)
        self._apply_exit_behaviours(services)
        self._emit_service_notices(services)
        return change_id

    def stop_services(
        self,
        services: list[str],
        timeout: float = 30.0,
        delay: float = 0.1,
    ) -> pebble.ChangeID:
        """Stop the named services, in Pebble's ``StopOrder``.

        Unlike ``start``/``restart``/``autostart``/``replan``, a stop cannot
        fail via a declared :class:`ServiceBehaviour` -- there is no
        ``ServiceStop`` outcome to declare, so this always succeeds for
        every known service, matching real Pebble (Real-Pebble probe #3,
        WORKLOAD-MOCK-DESIGN.md §22.2).
        """
        if isinstance(services, str):
            raise TypeError(f'stop_services should take a list of names, not just "{services}"')
        if not services:
            raise self._api_error(400, 'must specify services for start action')
        self._check_connection()

        known_services = self._render_services()
        for name in services:
            if name not in known_services:
                raise self._api_error(400, f'cannot stop services: service {name} does not exist')

        ordered = self._service_dependency_order(services, known_services, reverse=True)

        spawn_time = datetime.datetime.now(tz=datetime.timezone.utc)
        ready_time = spawn_time + datetime.timedelta(milliseconds=10)
        tasks: list[pebble.Task] = []
        for name in ordered:
            self._service_status[name] = pebble.ServiceStatus.INACTIVE
            tasks.append(_pebble_task(name, 'stop', 'Done', spawn_time, ready_time))

        # Matches start/restart, not autostart/replan: the summary leads
        # with the caller's first-requested name, not the topologically
        # first task (§22.3) -- stop has no autostart/replan-style entry
        # point to lead alphabetically instead.
        summary = f'Stop service "{services[0]}"'
        if len(ordered) > 1:
            summary += f' and {len(ordered) - 1} more'
        change = pebble.Change(
            id=pebble.ChangeID(str(uuid.uuid4())),
            kind='stop',
            summary=summary,
            status='Done',
            tasks=tasks,
            ready=True,
            err=None,
            spawn_time=spawn_time,
            ready_time=ready_time,
        )
        self._changes[change.id] = change
        return change.id

    def restart_services(
        self,
        services: list[str],
        timeout: float = 30.0,
        delay: float = 0.1,
    ) -> pebble.ChangeID:
        if isinstance(services, str):
            raise TypeError(f'restart_services should take a list of names, not just "{services}"')
        known_services = self._render_services()
        if self._needs_behaviour_path(services, known_services):
            return self._start_with_behaviours(services, kind='restart')
        change_id = super().restart_services(services, timeout=timeout, delay=delay)
        self._apply_exit_behaviours(services)
        self._emit_service_notices(services)
        return change_id

    def autostart_services(self, timeout: float = 30.0, delay: float = 0.1):
        self._check_connection()
        enabled = self._enabled_service_names()
        known_services = self._render_services()
        if self._needs_behaviour_path(enabled, known_services):
            return self._start_with_behaviours(enabled, kind='autostart')
        change_id = super().autostart_services(timeout=timeout, delay=delay)
        self._apply_exit_behaviours(enabled)
        self._emit_service_notices(enabled)
        return change_id

    def _enabled_service_names(self) -> list[str]:
        # The services that ``autostart``/``replan`` will try to start.
        return [
            name
            for name, service in self._render_services().items()
            if service.startup == pebble.ServiceStartup.ENABLED.value
        ]

    def _find_service_behaviour(self, service_name: str) -> ServiceBehaviour | None:
        for behaviour in self._container.service_behaviours:
            if behaviour.service_name == service_name:
                return behaviour
        return None

    def _has_failing_behaviour(self, services: Iterable[str]) -> bool:
        for name in services:
            behaviour = self._find_service_behaviour(name)
            if behaviour is not None and behaviour.start is ServiceStart.FAILS:
                return True
        return False

    def _needs_behaviour_path(
        self,
        services: list[str],
        known_services: dict[str, pebble.Service],
    ) -> bool:
        """Whether this request needs the per-service task-building path.

        True when a requested service is declared ``ServiceStart.FAILS``
        (the pre-existing trigger), or when the ``requires`` closure pulls in
        a service beyond what was requested (Real-Pebble probe #4 §24.3):
        either way, the change needs a real task list rather than the
        no-task-list fast path ``super()`` gives, since a pulled-in member is
        a full member of the change (§25.1) even though nothing in
        ``services`` names it directly.
        """
        members = self._service_requires_closure(services, known_services)
        return members != set(services) or self._has_failing_behaviour(services)

    def _apply_exit_behaviours(self, services: list[str]) -> None:
        """Overwrite the status of any requested ``EXITS`` service.

        Only called after a start/restart/replan/autostart call has already
        succeeded and set every requested service ``ACTIVE`` -- an ``EXITS``
        service's call still succeeds (unlike ``FAILS``), but its resulting
        status is the declared post-exit value instead.
        """
        known_services = self._render_services()
        for name in services:
            behaviour = self._find_service_behaviour(name)
            if behaviour is None or behaviour.start is not ServiceStart.EXITS:
                continue
            self._service_status[name] = self._service_exit_detail(known_services[name], behaviour)

    def _service_requires_closure(
        self,
        names: Iterable[str],
        known_services: dict[str, pebble.Service],
    ) -> set[str]:
        """Expand ``names`` to the full ``requires`` transitive closure.

        ``requires`` is membership, not ordering (Real-Pebble probe #4,
        WORKLOAD-MOCK-DESIGN.md §24.3): a service named only in another
        service's ``requires`` list becomes a full member of the resulting
        change even though nothing in ``names`` names it directly. That
        pull-in is transitive the full length of the chain (Real-Pebble
        probe #5, §25.1) -- a service three ``requires`` hops away is pulled
        in exactly as much as an immediate one, even when nothing in between
        names it either. ``before``/``after`` play no part here; see
        ``_service_dependency_order`` for the ordering half.

        This computes membership only. It does not hold a task down because
        a `requires`-related dependency failed -- that is cascade, which is
        not implemented by this function or by anything that calls it (see
        WORKLOAD-MOCK-DESIGN.md §26).

        Traversal is iterative over a visited set (``closure`` itself),
        never recursive, so a ``requires`` cycle terminates instead of
        looping forever. Real Pebble's own behaviour on a ``requires`` cycle
        has not been measured by any probe -- this is the defensive choice,
        not a claimed match to real Pebble; see the probe #6 question list
        in WORKLOAD-MOCK-DESIGN.md §26.
        """
        closure = set(names)
        frontier = list(closure)
        while frontier:
            name = frontier.pop()
            service = known_services.get(name)
            if service is None:
                # A `requires` target outside the known plan -- nothing to
                # expand from; leave it as an (unorderable) member and move on.
                continue
            for other in service.requires:
                if other not in closure:
                    closure.add(other)
                    frontier.append(other)
        return closure

    def _service_dependency_order(
        self,
        names: Iterable[str],
        known_services: dict[str, pebble.Service],
        *,
        reverse: bool = False,
    ) -> list[str]:
        """Order ``names`` the way Pebble's ``StartOrder``/``StopOrder`` would.

        A Kahn's-algorithm topological sort of the declared ``before``/
        ``after`` dependencies among ``names``, with services that have no
        outstanding dependency picked alphabetically -- the same tie-break
        ``tarjanSort`` gets from ``sort.Strings`` on Pebble's side. Measured
        against a real daemon in Real-Pebble probe #3
        (WORKLOAD-MOCK-DESIGN.md §22.1); this is not a plain alphabetical
        sort, which is what this mock did before.

        ``reverse=True`` computes ``StopOrder`` (§22.2): every ``before``/
        ``after`` edge points the other way, so a service that must *start*
        after another *stops* before it. This is not the same as reversing
        the start-ordered list -- see §22.2's worked example.

        This function only orders ``names`` -- it does not decide which
        services belong in ``names`` to begin with. A ``before``/``after``
        edge naming a service outside ``names`` is ignored here; membership
        (which services a start/stop/restart/autostart/replan request pulls
        in via ``requires``) is a separate, prior computation, done by
        ``_service_requires_closure`` (Real-Pebble probe #4 §24.3: ``requires``
        is membership, ``before``/``after`` is ordering -- the two are kept
        apart deliberately, because Pebble does).
        """
        pending = set(names)
        successors: dict[str, set[str]] = {name: set() for name in pending}
        in_degree: dict[str, int] = dict.fromkeys(pending, 0)

        def add_edge(first: str, second: str) -> None:
            # `first` must be ordered ahead of `second`.
            if second not in successors[first]:
                successors[first].add(second)
                in_degree[second] += 1

        for name in pending:
            service = known_services[name]
            for other in service.before:
                if other not in pending:
                    continue
                add_edge(other, name) if reverse else add_edge(name, other)
            for other in service.after:
                if other not in pending:
                    continue
                add_edge(name, other) if reverse else add_edge(other, name)

        ready = [name for name, degree in in_degree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            name = heapq.heappop(ready)
            ordered.append(name)
            for successor in sorted(successors[name]):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    heapq.heappush(ready, successor)

        if len(ordered) < len(pending):
            # A dependency cycle among the requested services. Real Pebble
            # rejects a plan whose layers combine into one of these, so
            # nothing in this mock should be able to construct this shape
            # today -- fall back to alphabetical for whatever's left rather
            # than hang, in case that changes.
            ordered.extend(sorted(pending.difference(ordered)))
        return ordered

    def _start_with_behaviours(self, services: list[str], kind: str = 'start') -> pebble.ChangeID:
        """Apply declared ``ServiceBehaviour``s for a start/restart/autostart request.

        Called whenever ``services`` needs a real per-service task list
        rather than the no-task-list fast path ``super()`` gives -- either
        because a requested service is declared ``ServiceStart.FAILS``, or
        because ``requires`` pulls in a service beyond what was requested
        (Real-Pebble probe #4 §24.3, probe #5 §25.1; see
        ``_needs_behaviour_path``). Only the first case raises
        ``pebble.ChangeError``, matching what real Pebble raises when a
        start fails; the second, on its own, still succeeds -- Pebble ran
        the pulled-in service too, it just didn't fail (§24.3's `yankee`/
        `zulu`, both `Done`). Services with no declared FAILS behaviour
        still reach a status before any error is raised -- ACTIVE by
        default, or their own declared EXITS status -- since Pebble tracks
        each service's start attempt independently.

        This resolves each service's status from its own declared
        ``ServiceBehaviour`` only (§3's "declare, don't derive"). A
        pulled-in member whose *own* dependency (not itself) is declared
        FAILS is not held down here -- that is cascade, and it is not
        implemented by this method (WORKLOAD-MOCK-DESIGN.md §26): a service
        three ``requires`` hops from a failure reaches ACTIVE the same as
        one with no failing relative at all, exactly the gap
        ``test_service_dependency_start_order_matches_probe`` already
        recorded for ``before``/``after`` chains before this stage existed.

        ``kind`` is the change kind Pebble uses for whichever entry point got
        us here (``start``/``restart``/``autostart``/``replan``), and the
        change summary's verb follows it.
        """
        if not services:
            raise self._api_error(400, 'must specify services for start action')
        known_services = self._render_services()
        for name in services:
            if name not in known_services:
                raise self._api_error(400, f'cannot start services: service {name} does not exist')

        members = self._service_requires_closure(services, known_services)
        ordered = self._service_dependency_order(members, known_services)

        failing: list[tuple[str, ServiceBehaviour]] = []
        for name in ordered:
            behaviour = self._find_service_behaviour(name)
            if behaviour is not None and behaviour.start is ServiceStart.FAILS:
                failing.append((name, behaviour))
            elif behaviour is not None and behaviour.start is ServiceStart.EXITS:
                # An EXITS service mixed into the same request as a FAILS
                # one still resolves to its own declared post-exit status,
                # not ACTIVE -- Pebble tracks each service's start attempt
                # independently, and that independence applies
                # just as much between FAILS and EXITS as within FAILS.
                self._service_status[name] = self._service_exit_detail(
                    known_services[name], behaviour
                )
            else:
                self._service_status[name] = pebble.ServiceStatus.ACTIVE

        # The services that did start have notified, even though the call as a
        # whole is about to fail: Pebble tracks each service independently.
        self._emit_service_notices(name for name in ordered if name not in dict(failing))

        tasks: list[pebble.Task] = []
        bullets: list[str] = []
        spawn_time = datetime.datetime.now(tz=datetime.timezone.utc)
        ready_time = spawn_time + datetime.timedelta(milliseconds=10)

        failing_by_name = dict(failing)

        # A restart change carries every "stop" task before any "start" task
        # -- real Pebble builds the two task sets independently (a StopOrder
        # pass, then a StartOrder pass) and concatenates them, rather than
        # interleaving stop/start per service. The stop pass uses StopOrder,
        # not StartOrder reversed -- see _service_dependency_order.
        if kind == 'restart':
            stop_ordered = self._service_dependency_order(members, known_services, reverse=True)
            for name in stop_ordered:
                tasks.append(_pebble_task(name, 'stop', 'Done', spawn_time, ready_time))

        for name in ordered:
            behaviour = failing_by_name.get(name)
            if behaviour is None:
                # Pebble emits a Done task for services that did start, not
                # only for the failures.
                tasks.append(_pebble_task(name, 'start', 'Done', spawn_time, ready_time))
                continue
            reason, status, log = self._service_failure_detail(known_services[name], behaviour)
            self._service_status[name] = status
            bullets.append(f'- Start service "{name}" ({reason})')
            tasks.append(_pebble_task(name, 'start', 'Error', spawn_time, ready_time, log))

        # The summary counts every service in the change -- every requested
        # one, plus any pulled in only via `requires` (§24.3) -- not only the
        # failing ones, and quotes the leading name. Which name leads,
        # though, depends on the entry point: real Pebble's start/restart
        # handler uses the client's request order verbatim
        # (payload.Services[0] in api_services.go), while autostart/replan
        # reassign payload.Services to an alphabetically-sorted list before
        # building the summary. So start/restart lead with the caller's
        # first-requested name and autostart/replan lead with the
        # topologically first task (§24.1) -- unverified for
        # autostart/replan once `requires` pulls in a member that doesn't
        # sort first (§25.3; WORKLOAD-MOCK-DESIGN.md §26 records this as
        # still open).
        leading = services[0] if kind in ('start', 'restart') else ordered[0]
        summary = f'{kind.capitalize()} service "{leading}"'
        if len(ordered) > 1:
            summary += f' and {len(ordered) - 1} more'

        if not bullets:
            # No FAILS behaviour anywhere in the closure: every pulled-in
            # member started successfully too (§24.3's `yankee`/`zulu`, both
            # `Done`) -- membership without cascade, this stage's whole
            # scope.
            change = pebble.Change(
                id=pebble.ChangeID(str(uuid.uuid4())),
                kind=kind,
                summary=summary,
                status='Done',
                tasks=tasks,
                ready=True,
                err=None,
                spawn_time=spawn_time,
                ready_time=ready_time,
            )
            self._changes[change.id] = change
            return change.id

        err = 'cannot perform the following tasks:\n' + '\n'.join(bullets)
        change = pebble.Change(
            id=pebble.ChangeID(str(uuid.uuid4())),
            kind=kind,
            summary=summary,
            status='Error',
            tasks=tasks,
            ready=True,
            err=err,
            spawn_time=spawn_time,
            ready_time=ready_time,
        )
        raise pebble.ChangeError(err, change)

    def _service_failure_detail(
        self,
        service: pebble.Service,
        behaviour: ServiceBehaviour,
    ) -> tuple[str, pebble.ServiceStatus | str, list[str]]:
        """Return (reason, status, task.log) for a FAILS service."""
        if behaviour.failure_mode is ServiceFailureMode.EXEC_ERROR:
            command = service.command.split()[0] if service.command else service.name
            reason = f'cannot start service: fork/exec {command}: no such file or directory'
            log = [f'{_rfc3339_now()} ERROR {reason}']
            return reason, pebble.ServiceStatus.INACTIVE, log

        verb = self._on_failure_verb(service.on_failure)
        reason = f'service start attempt: exited quickly with code 1, will {verb}'
        status: pebble.ServiceStatus | str = (
            pebble.ServiceStatus.ERROR if verb == 'ignore' else 'backoff'
        )
        log = [
            f'{_rfc3339_now()} INFO Most recent service output:',
            f'{_rfc3339_now()} ERROR {reason}',
        ]
        return reason, status, log

    @staticmethod
    def _on_failure_verb(on_failure: str) -> str:
        if on_failure in ('', 'restart'):
            return 'restart'
        if on_failure == 'ignore':
            return 'ignore'
        raise NotImplementedError(
            f'ServiceStart.FAILS does not model on-failure: {on_failure!r} yet; only '
            "'' (Pebble's default, equivalent to 'restart') and 'ignore'."
        )

    def _service_exit_detail(
        self,
        service: pebble.Service,
        behaviour: ServiceBehaviour,
    ) -> pebble.ServiceStatus | str:
        """Return the resulting status for an ``EXITS`` service.

        Unlike ``FAILS``, the call that reaches this point has already
        succeeded -- real Pebble doesn't fail the start/restart/replan/
        autostart call for an exit that happens after the service has been
        up for more than a second so there's no reason/log/ChangeError
        to build here, only the resulting status.
        """
        if behaviour.exit_code is ServiceExitCode.SUCCESS:
            verb = self._on_success_verb(service.on_success)
            return pebble.ServiceStatus.INACTIVE if verb == 'ignore' else 'backoff'
        verb = self._on_failure_verb(service.on_failure)
        return pebble.ServiceStatus.ERROR if verb == 'ignore' else 'backoff'

    @staticmethod
    def _on_success_verb(on_success: str) -> str:
        if on_success in ('', 'restart'):
            return 'restart'
        if on_success == 'ignore':
            return 'ignore'
        raise NotImplementedError(
            f'ServiceStart.EXITS does not model on-success: {on_success!r} yet; only '
            "'' (Pebble's default, equivalent to 'restart') and 'ignore'."
        )

    def add_layer(
        self,
        label: str,
        layer: str | pebble.LayerDict | pebble.Layer,
        *,
        combine: bool = False,
    ):
        super().add_layer(label, layer, combine=combine)
        self._update_state_check_infos()

    def start_checks(self, names: list[str]) -> list[str]:
        started = super().start_checks(names)
        self._update_state_check_infos()
        return started

    def stop_checks(self, names: list[str]) -> list[str]:
        stopped = super().stop_checks(names)
        self._update_state_check_infos()
        return stopped

    @property
    def _container(self) -> ContainerSpec:
        container_name = self.socket_path.split('/')[-2]
        try:
            return next(
                filter(lambda x: x.name == container_name, self._state.containers),
            )
        except StopIteration:
            raise RuntimeError(
                f'container with name={container_name!r} not found. '
                f'Did you forget a Container, or is the socket path '
                f'{self.socket_path!r} wrong?',
            ) from None

    # These properties override plain-dict attributes on the Harness parent
    # (self._layers / self._service_status) because the scenario mock keeps
    # state on the Container dataclass rather than on the client. Pyright
    # rightly flags overriding a variable with a differently-typed property,
    # but resolving it properly would require reworking the parent class to
    # let subclasses plug in a state source. Don't remove the ignores without
    # that refactor.
    @property
    def _layers(self) -> dict[str, pebble.Layer]:  # pyright: ignore[reportIncompatibleVariableOverride]
        # The runtime value is a dict, but Container.layers is annotated
        # Mapping. Ignored (rather than cast) so pyright alerts us if the
        # source type is ever narrowed to dict and this override is no
        # longer needed.
        return self._container.layers  # pyright: ignore[reportReturnType]

    @property
    def _service_status(self) -> dict[str, pebble.ServiceStatus | str]:  # pyright: ignore[reportIncompatibleVariableOverride]
        # See _layers.
        return self._container.service_statuses  # pyright: ignore[reportReturnType]

    def get_services(self, names: list[str] | None = None) -> list[pebble.ServiceInfo]:
        # Overridden (rather than relying on the parent implementation)
        # because the parent unconditionally does
        # ``pebble.ServiceStatus(status)``, which raises ValueError for a
        # status like ``'backoff'`` that real Pebble reports but that has no
        # ServiceStatus member.
        if isinstance(names, str):
            raise TypeError(f'start_services should take a list of names, not just "{names}"')
        self._check_connection()
        services = self._render_services()
        query = sorted(names) if names is not None else sorted(services.keys())
        infos: list[pebble.ServiceInfo] = []
        for name in query:
            try:
                service = services[name]
            except KeyError:
                # In pebble, it just returns "nothing matched" if there are 0 matches,
                # but it ignores services it doesn't recognise.
                continue
            status = self._service_status.get(name, pebble.ServiceStatus.INACTIVE)
            if service.startup == '':
                startup = pebble.ServiceStartup.DISABLED
            else:
                startup = pebble.ServiceStartup(service.startup)
            if isinstance(status, pebble.ServiceStatus):
                current: pebble.ServiceStatus | str = status
            else:
                try:
                    current = pebble.ServiceStatus(status)
                except ValueError:
                    current = status
            infos.append(pebble.ServiceInfo(name, startup=startup, current=current))
        return infos

    # Based on a method of the same name from Harness.
    def _find_exec_handler(self, command: list[str]) -> Exec | None:
        handlers = {exe.command_prefix: exe for exe in self._container.execs}
        # Start with the full command and, each loop iteration, drop the last
        # element, until it matches one of the command prefixes in the execs.
        # This includes matching against the empty list, which will match any
        # command, if there is not a more specific match.
        for prefix_len in reversed(range(len(command) + 1)):
            command_prefix = tuple(command[:prefix_len])
            if command_prefix in handlers:
                return handlers[command_prefix]
        # None of the command prefixes in the execs matched the command, no
        # matter how much of it was used, so we have failed to find a handler.
        return None

    def exec(
        self,
        command: list[str],
        *,
        environment: dict[str, str] | None = None,
        working_dir: str | None = None,
        timeout: float | None = None,
        user_id: int | None = None,
        user: str | None = None,
        group_id: int | None = None,
        group: str | None = None,
        stdin: str | bytes | TextIO | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        encoding: str | None = 'utf-8',
        combine_stderr: bool = False,
        **kwargs: Any,
    ):
        handler = self._find_exec_handler(command)
        if not handler:
            raise ExecError(
                command,
                127,
                '',
                f'mock for cmd {command} not found. Please patch out whatever '
                f'leads to the call, or pass to the Container {self._container.name} '
                f'a scenario.Exec mock for the command your charm is attempting '
                f'to run, such as '
                f"'Container(..., execs={{scenario.Exec({list(command)}, ...)}})'",
            )

        if stdin is None:
            proc_stdin = self._transform_exec_handler_output('', encoding)
        else:
            proc_stdin = None
            stdin = stdin.read() if hasattr(stdin, 'read') else stdin  # type: ignore
        if stdout is None:
            proc_stdout = self._transform_exec_handler_output(handler.stdout, encoding)
        else:
            proc_stdout = None
            stdout.write(handler.stdout)
        if stderr is None:
            proc_stderr = self._transform_exec_handler_output(handler.stderr, encoding)
        else:
            proc_stderr = None
            stderr.write(handler.stderr)

        args = ExecArgs(
            command=command,
            environment=environment or {},
            working_dir=working_dir,
            timeout=timeout,
            user_id=user_id,
            user=user,
            group_id=group_id,
            group=group,
            stdin=stdin,  # type:ignore  # If None, will be replaced by proc_stdin.read() later.
            encoding=encoding,
            combine_stderr=combine_stderr,
        )
        try:
            self._context.exec_history[self._container_name].append(args)
        except KeyError:
            self._context.exec_history[self._container_name] = [args]

        change_id = handler._run()
        return cast(
            'pebble.ExecProcess[Any]',
            _MockExecProcess(
                change_id=change_id,
                args=args,
                return_code=handler.return_code,
                stdin=proc_stdin,
                stdout=proc_stdout,
                stderr=proc_stderr,
            ),
        )

    def _check_connection(self):
        if not self._container.can_connect:
            msg = (
                f'Cannot connect to Pebble; did you forget to set '
                f'can_connect=True for container {self._container.name}?'
            )
            raise pebble.ConnectionError(msg)
