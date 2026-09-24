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

"""Run the examples from the 'manage stored state' how-to and its explanation page.

The blocks are read out of the Markdown source and executed as printed, so that
the pages can't drift away from what works. The how-to gives the charms in
fragments, so the fragments are assembled here the way the page describes:
handlers go in the body of the charm class, and the charm observes the events
whose handlers it defines.

The explanation page's blocks are snippets rather than complete charms, so they
are only checked for being valid Python.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re
import textwrap
from typing import Any

import pytest
import yaml

import ops
from ops import testing

DOCS = pathlib.Path(__file__).parent.parent / 'docs'
HOWTO = DOCS / 'howto' / 'manage-stored-state.md'
EXPLANATION = DOCS / 'explanation' / 'storedstate-guidance.md'

STORED_STATE_CHARM_BLOCK = '_stored = ops.StoredState()'
STORED_STATE_HANDLERS_BLOCK = 'def _on_install'
STORED_STATE_TESTS_BLOCK = 'def test_charm_sets_stored_state'
PEERS_YAML_BLOCK = 'my_charm_peers'
PEER_HANDLERS_BLOCK = 'def _on_stop'
PEER_TEST_BLOCK = 'def test_charm_sets_peer_data'

# The peer-relation section reuses the charm class from the section above, but
# with its own handlers, so the wiring the page describes in prose is spelled
# out here.
PEER_CHARM_TEMPLATE = """
class PeerCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)
        framework.observe(self.on.stop, self._on_stop)

    def _calculate_expensive_value(self) -> int:
        return 42

{handlers}
"""


def blocks(page: pathlib.Path, language: str = 'python') -> list[str]:
    """Return every block in that language on the page."""
    pattern = rf'^```{language}\n(.*?)^```'
    return re.findall(pattern, page.read_text(), re.DOTALL | re.MULTILINE)


def block(page: pathlib.Path, contains: str, language: str = 'python') -> str:
    """Return the page's only block in that language that contains the given text."""
    matching = [b for b in blocks(page, language) if contains in b]
    count = len(matching)
    assert count == 1, f'expected one block containing {contains!r}, found {count}'
    return matching[0]


def run(source: str, namespace: dict[str, Any]):
    """Execute the given source in the namespace."""
    exec(source, namespace)  # ruff: ignore[exec-builtin]


@pytest.fixture
def namespace(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A namespace with the how-to's two charms defined in it."""
    peers = yaml.safe_load(block(HOWTO, PEERS_YAML_BLOCK, 'yaml'))['peers']
    real_context = testing.Context

    def context(charm_type: type[ops.CharmBase], **kwargs: Any):
        meta: dict[str, Any] = {'name': 'my-charm'}
        if charm_type.__name__ == 'PeerCharm':
            meta['peers'] = peers
        kwargs.setdefault('meta', meta)
        return real_context(charm_type, **kwargs)

    monkeypatch.setattr(testing, 'Context', context)
    ns: dict[str, Any] = {'ops': ops, 'testing': testing, 'logger': logging.getLogger(__name__)}
    charm = block(HOWTO, STORED_STATE_CHARM_BLOCK)
    handlers = textwrap.indent(block(HOWTO, STORED_STATE_HANDLERS_BLOCK), ' ' * 4)
    run(f'{charm}\n{handlers}', ns)
    run(
        PEER_CHARM_TEMPLATE.format(
            handlers=textwrap.indent(block(HOWTO, PEER_HANDLERS_BLOCK), ' ' * 4)
        ),
        ns,
    )
    return ns


def test_stored_state_unit_tests(namespace: dict[str, Any]):
    run(block(HOWTO, STORED_STATE_TESTS_BLOCK), namespace)
    namespace['test_charm_sets_stored_state']()
    namespace['test_charm_logs_stored_state']()


def test_peer_relation_unit_test(namespace: dict[str, Any]):
    # The page's test says `MyCharm`, meaning the charm with the peer handlers.
    namespace['MyCharm'] = namespace['PeerCharm']
    run(block(HOWTO, PEER_TEST_BLOCK), namespace)
    namespace['test_charm_sets_peer_data']()


def test_peer_relation_non_leader(namespace: dict[str, Any]):
    ctx = testing.Context(namespace['PeerCharm'])
    peer = testing.PeerRelation('charm-peer')

    state_out = ctx.run(ctx.on.start(), testing.State(relations={peer}))

    assert state_out.get_relation(peer.id).local_app_data == {}


@pytest.mark.parametrize(
    'source',
    blocks(EXPLANATION),
    ids=[f'block{i}' for i, _ in enumerate(blocks(EXPLANATION), start=1)],
)
def test_explanation_blocks_parse(source: str):
    ast.parse(textwrap.dedent(source))
