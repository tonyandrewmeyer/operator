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

"""Run the examples from the 'manage configuration' how-to.

The blocks are read out of the Markdown source and executed as printed, so that
the page can't drift away from what works. The page gives the charm in
fragments, so the fragments are assembled here the way the page says to
assemble them: the `observe` call goes in `__init__`, and the handler goes in
the body of the charm class.
"""

from __future__ import annotations

import logging
import pathlib
import re
import textwrap
from typing import Any

import pytest
import yaml

import ops
from ops import testing

PAGE = pathlib.Path(__file__).parent.parent / 'docs' / 'howto' / 'manage-configuration.md'

CONFIG_CLASS_BLOCK = 'class WikiConfig'
OBSERVE_BLOCK = 'self.framework.observe(self.on.config_changed'
HANDLER_BLOCK = 'def _on_config_changed'
UNIT_TEST_BLOCK = 'def test_short_wiki_name'
CONFIG_YAML_BLOCK = 'skin for the Wiki'

CHARM_TEMPLATE = """
class MyCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
{observe}

{handler}

    def get_wiki_name(self) -> str:
        return 'Wiki'

    def set_wiki_name(self, name: str) -> None:
        pass
"""


def block(contains: str, language: str = 'python') -> str:
    """Return the page's only block in that language that contains the given text."""
    pattern = rf'^```{language}\n(.*?)^```'
    blocks = re.findall(pattern, PAGE.read_text(), re.DOTALL | re.MULTILINE)
    matching = [b for b in blocks if contains in b]
    count = len(matching)
    assert count == 1, f'expected one block containing {contains!r}, found {count}'
    return matching[0]


def run(source: str, namespace: dict[str, Any]):
    """Execute the given source in the namespace."""
    exec(source, namespace)  # ruff: ignore[exec-builtin]


@pytest.fixture
def namespace(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A namespace with the page's config class and charm defined in it."""
    config: dict[str, Any] = yaml.safe_load(block(CONFIG_YAML_BLOCK, 'yaml'))['config']
    real_context = testing.Context

    def context(charm_type: type[ops.CharmBase], **kwargs: Any):
        kwargs.setdefault('meta', {'name': 'wiki'})
        kwargs.setdefault('config', config)
        return real_context(charm_type, **kwargs)

    monkeypatch.setattr(testing, 'Context', context)
    ns: dict[str, Any] = {'ops': ops, 'logger': logging.getLogger(__name__)}
    run('import pydantic', ns)
    run(block(CONFIG_CLASS_BLOCK), ns)
    run(
        CHARM_TEMPLATE.format(
            observe=textwrap.indent(block(OBSERVE_BLOCK), ' ' * 8),
            handler=textwrap.indent(block(HANDLER_BLOCK), ' ' * 4),
        ),
        ns,
    )
    return ns


def test_valid_config(namespace: dict[str, Any]):
    ctx = testing.Context(namespace['MyCharm'])

    state_out = ctx.run(ctx.on.config_changed(), testing.State(config={'name': 'Charming'}))

    assert state_out.unit_status == testing.UnknownStatus()


def test_short_wiki_name(namespace: dict[str, Any]):
    run(block(UNIT_TEST_BLOCK), namespace)
    namespace['test_short_wiki_name']()


def test_name_with_spaces(namespace: dict[str, Any]):
    ctx = testing.Context(namespace['MyCharm'])

    state_out = ctx.run(ctx.on.config_changed(), testing.State(config={'name': 'has spaces'}))

    assert isinstance(state_out.unit_status, testing.BlockedStatus)
