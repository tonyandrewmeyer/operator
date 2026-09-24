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

"""Run the examples from the 'manage Pebble health checks' how-to.

The Python blocks are read out of the Markdown source and executed as printed,
so that the page can't drift away from what works. The page is a set of
snippets rather than a complete charm, so the only things supplied here are the
ones a reader would obviously supply: a logger, and the charm metadata that
``testing.Context`` would otherwise read from ``charmcraft.yaml``.
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Any

import pytest

import ops
from ops import testing

PAGE = (
    pathlib.Path(__file__).parent.parent
    / 'docs'
    / 'howto'
    / 'manage-containers'
    / 'manage-pebble-health-checks.md'
)

META: dict[str, Any] = {'name': 'postgres', 'containers': {'db': {}}}

CHARM_BLOCK = 'The http-test has stopped failing!'
SHARED_HANDLER_BLOCK = 'activate_alternative_configuration'
UNIT_TEST_BLOCK = 'def test_http_check_failing'


def run_block(contains: str, namespace: dict[str, Any]):
    """Execute the page's only Python block that contains the given text."""
    blocks = re.findall(r'^```python\n(.*?)^```', PAGE.read_text(), re.DOTALL | re.MULTILINE)
    matching = [block for block in blocks if contains in block]
    count = len(matching)
    assert count == 1, f'expected one block containing {contains!r}, found {count}'
    exec(matching[0], namespace)  # ruff: ignore[exec-builtin]


@pytest.fixture
def namespace(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A namespace to execute the page's blocks in."""
    real_context = testing.Context

    def context(charm_type: type[ops.CharmBase], **kwargs: Any):
        kwargs.setdefault('meta', META)
        return real_context(charm_type, **kwargs)

    monkeypatch.setattr(testing, 'Context', context)
    return {'ops': ops, 'logger': logging.getLogger(__name__)}


def test_respond_to_a_check_failing_or_recovering(namespace: dict[str, Any]):
    run_block(CHARM_BLOCK, namespace)
    ctx = testing.Context(namespace['PostgresCharm'], meta=META)
    layer = ops.pebble.Layer({
        'checks': {
            'http-test': {
                'override': 'replace',
                'threshold': 3,
                'http': {'url': 'http://localhost:8080/test'},
            },
        },
    })
    failed = testing.CheckInfo('http-test', failures=3, status=ops.pebble.CheckStatus.DOWN)
    container = testing.Container('db', check_infos={failed}, layers={'layer1': layer})
    state_out = ctx.run(
        ctx.on.pebble_check_failed(container, info=failed),
        testing.State(containers={container}),
    )
    assert state_out.unit_status == testing.ActiveStatus('Degraded functionality ...')

    recovered = testing.CheckInfo('http-test', failures=0, status=ops.pebble.CheckStatus.UP)
    container = testing.Container('db', check_infos={recovered}, layers={'layer1': layer})
    state_out = ctx.run(
        ctx.on.pebble_check_recovered(container, info=recovered),
        testing.State(containers={container}),
    )
    assert state_out.unit_status == testing.ActiveStatus()


def test_shared_handler(namespace: dict[str, Any]):
    run_block(SHARED_HANDLER_BLOCK, namespace)
    assert issubclass(namespace['PostgresCharm'], ops.CharmBase)


def test_write_unit_tests(namespace: dict[str, Any]):
    run_block(CHARM_BLOCK, namespace)
    run_block(UNIT_TEST_BLOCK, namespace)
    namespace['test_http_check_failing']()
