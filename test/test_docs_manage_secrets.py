# Copyright 2026 Canonical Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the migration example from the 'manage secrets' how-to.

The charm is read out of the Markdown source and executed as printed, so that
the page can't drift away from what works.
"""

from __future__ import annotations

import pathlib
import re
import textwrap
from typing import Any

import ops
from ops import testing

PAGE = pathlib.Path(__file__).parent.parent / 'docs' / 'howto' / 'manage-secrets.md'

MIGRATION_BLOCK = 'An earlier version of this charm wrote the credentials in plain text.'

META: dict[str, Any] = {'name': 'my-database', 'provides': {'database': {'interface': 'db'}}}

PLAIN_TEXT = {'username': 'admin', 'password': 'admin'}


def block(contains: str) -> str:
    """Return the page's only Python block that contains the given text."""
    pattern = r'^```python\n(.*?)^```'
    matching = [b for b in re.findall(pattern, PAGE.read_text(), re.DOTALL | re.MULTILINE)]
    matching = [b for b in matching if contains in b]
    count = len(matching)
    assert count == 1, f'expected one block containing {contains!r}, found {count}'
    return matching[0]


def context() -> testing.Context[ops.CharmBase]:
    namespace: dict[str, Any] = {'ops': ops}
    exec(textwrap.dedent(block(MIGRATION_BLOCK)), namespace)  # ruff: ignore[exec-builtin]
    return testing.Context(namespace['MyDatabaseCharm'], meta=META)


def assert_moved_to_secret(state: testing.State, relation: testing.Relation):
    databag = state.get_relation(relation.id).local_app_data
    assert set(databag) == {'secret-id'}
    secret = state.get_secret(id=databag['secret-id'])
    assert secret.latest_content == PLAIN_TEXT


def test_upgrade_moves_an_existing_relation_to_a_secret():
    relation = testing.Relation('database', local_app_data=dict(PLAIN_TEXT))
    state_in = testing.State(relations={relation}, leader=True)

    ctx = context()
    state_out = ctx.run(ctx.on.upgrade_charm(), state_in)

    assert_moved_to_secret(state_out, relation)


def test_new_relation_gets_a_secret():
    relation = testing.Relation('database', remote_units_data={0: {}})
    state_in = testing.State(relations={relation}, leader=True)

    ctx = context()
    state_out = ctx.run(ctx.on.relation_joined(relation, remote_unit=0), state_in)

    assert_moved_to_secret(state_out, relation)


def test_second_upgrade_keeps_the_secret():
    ctx = context()
    relation = testing.Relation('database', local_app_data=dict(PLAIN_TEXT))
    state_in = testing.State(relations={relation}, leader=True)
    first = ctx.run(ctx.on.upgrade_charm(), state_in)
    secret_id = first.get_relation(relation.id).local_app_data['secret-id']

    second = ctx.run(ctx.on.upgrade_charm(), first)

    assert second.get_relation(relation.id).local_app_data == {'secret-id': secret_id}
    assert len(second.secrets) == 1


def test_non_leader_leaves_the_databag_alone():
    relation = testing.Relation('database', local_app_data=dict(PLAIN_TEXT))
    state_in = testing.State(relations={relation}, leader=False)

    ctx = context()
    state_out = ctx.run(ctx.on.upgrade_charm(), state_in)

    assert state_out.get_relation(relation.id).local_app_data == PLAIN_TEXT
    assert not state_out.secrets
