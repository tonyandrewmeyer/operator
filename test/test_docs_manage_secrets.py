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

The handler is read out of the Markdown source and executed as printed, so that
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

CHARM_TEMPLATE = """
class MyDatabaseCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(
            self.on.database_relation_joined, self._on_database_relation_joined
        )

{handler}
"""


def block(contains: str) -> str:
    """Return the page's only Python block that contains the given text."""
    pattern = r'^```python\n(.*?)^```'
    matching = [b for b in re.findall(pattern, PAGE.read_text(), re.DOTALL | re.MULTILINE)]
    matching = [b for b in matching if contains in b]
    count = len(matching)
    assert count == 1, f'expected one block containing {contains!r}, found {count}'
    return matching[0]


def test_migrate_an_existing_charm_to_secrets():
    namespace: dict[str, Any] = {'ops': ops}
    handler = textwrap.indent(textwrap.dedent(block(MIGRATION_BLOCK)), ' ' * 4)
    exec(CHARM_TEMPLATE.format(handler=handler), namespace)  # ruff: ignore[exec-builtin]
    ctx = testing.Context(namespace['MyDatabaseCharm'], meta=META)
    relation = testing.Relation(
        'database',
        remote_units_data={0: {}},
        local_app_data={'username': 'admin', 'password': 'admin'},
    )
    state_in = testing.State(relations={relation}, leader=True)

    state_out = ctx.run(ctx.on.relation_joined(relation, remote_unit=0), state_in)

    databag = state_out.get_relation(relation.id).local_app_data
    assert set(databag) == {'secret-id'}
    secret = state_out.get_secret(id=databag['secret-id'])
    assert secret.latest_content == {'username': 'admin', 'password': 'admin'}
