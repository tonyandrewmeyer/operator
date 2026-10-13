# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Flapping: databag values whose order depends on the string-hash seed.

In Juju, each hook is a new process with its own string-hash seed, so a charm
that writes a set of strings without sorting it writes it in a different order
in each hook. Two charms that react to each other's writes then never settle.
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
from typing import Any

import pytest
from scenario import _deployment

import ops
from ops import testing

PROVIDER = """
    import json

    import ops

    MEMBERS = ['alpha', 'bravo', 'charlie', 'delta', 'echo']


    class ProviderCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            for event in ('relation_created', 'relation_joined', 'relation_changed'):
                framework.observe(getattr(self.on['shared'], event), self._publish)

        def _publish(self, event):
            if self.unit.is_leader():
                members = {order}
                event.relation.data[self.app]['members'] = json.dumps(members)
"""

#: The bug: iterating a set of strings, whose order depends on the hash seed.
UNSORTED = 'list(set(MEMBERS))'
#: The fix.
SORTED = 'sorted(set(MEMBERS))'

REQUIRER = """
    import ops


    class RequirerCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on['shared'].relation_changed, self._on_changed)

        def _on_changed(self, event):
            members = event.relation.data[event.app].get('members')
            if members:
                event.relation.data[self.unit]['seen'] = members
"""


def write_charm(root: pathlib.Path, name: str, source: str, role: str) -> pathlib.Path:
    (root / 'src').mkdir(parents=True)
    (root / 'src' / 'charm.py').write_text(textwrap.dedent(source))
    (root / 'charmcraft.yaml').write_text(
        textwrap.dedent(f"""\
            name: {name}
            type: charm
            summary: A charm.
            description: A charm.
            {role}:
              shared:
                interface: shared
        """)
    )
    return root


def deploy_pair(
    juju: testing.Juju, tmp_path: pathlib.Path, *, order: str, isolated: bool
) -> tuple[testing.App, testing.App]:
    provider = write_charm(
        tmp_path / 'provider', 'provider', PROVIDER.format(order=order), 'provides'
    )
    requirer = write_charm(tmp_path / 'requirer', 'requirer', REQUIRER, 'requires')
    kwargs: dict[str, Any] = {'python_executable': sys.executable} if isolated else {}
    p = juju._deploy(provider, **kwargs)
    r = juju._deploy(requirer, **kwargs)
    juju.integrate(p, r)
    return p, r


@pytest.fixture(params=[False, True], ids=['in-process', 'worker'])
def isolated(request: pytest.FixtureRequest) -> bool:
    return request.param


def test_settle_does_not_see_hash_order_flapping(tmp_path: pathlib.Path, isolated: bool):
    """A limit, not a feature: by default, the pair settles.

    The test process, and each charm's worker, keeps one hash seed for every
    dispatch, so the provider writes the members in the same order every time.
    In Juju, this pair would never settle.
    """
    with testing.Juju() as juju:
        provider, requirer = deploy_pair(juju, tmp_path, order=UNSORTED, isolated=isolated)
        juju.settle()
        members = provider.leader.state.get_relations('shared')[0].local_app_data['members']
        seen = requirer.leader.state.get_relations('shared')[0].local_unit_data['seen']
        assert seen == members


def test_check_flapping_reports_hash_order_flapping(tmp_path: pathlib.Path, isolated: bool):
    with testing.Juju() as juju:
        deploy_pair(juju, tmp_path, order=UNSORTED, isolated=isolated)
        with pytest.raises(testing.errors.JujuError, match='looks like flapping') as caught:
            juju.settle(check_flapping=True)
    message = str(caught.value)
    assert "'members' in provider's application data on shared:" in message
    assert "'seen' in requirer/0's unit data on shared:" in message
    assert 'Sort the value' in message


def test_check_flapping_passes_a_sorted_value(tmp_path: pathlib.Path, isolated: bool):
    with testing.Juju() as checked:
        provider, _ = deploy_pair(checked, tmp_path / 'checked', order=SORTED, isolated=isolated)
        checked_trace = checked.settle(check_flapping=True)
        checked_state = provider.leader.state
    with testing.Juju() as plain:
        provider, _ = deploy_pair(plain, tmp_path / 'plain', order=SORTED, isolated=isolated)
        plain_trace = plain.settle()
        plain_state = provider.leader.state
    assert [(d.event.name, d.unit_name) for d in checked_trace] == [
        (d.event.name, d.unit_name) for d in plain_trace
    ]
    checked_bag = checked_state.get_relations('shared')[0].local_app_data
    plain_bag = plain_state.get_relations('shared')[0].local_app_data
    assert checked_bag == plain_bag


class Requirer(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on['shared'].relation_changed, self._on_changed)

    def _on_changed(self, event: ops.RelationChangedEvent):
        assert event.app is not None
        members = event.relation.data[event.app].get('members')
        if members:
            event.relation.data[self.unit]['seen'] = members


def test_check_flapping_runs_a_charm_spec_in_process(tmp_path: pathlib.Path):
    provider = write_charm(
        tmp_path / 'provider', 'provider', PROVIDER.format(order=UNSORTED), 'provides'
    )
    spec = testing.CharmSpec(
        Requirer, meta={'name': 'requirer', 'requires': {'shared': {'interface': 'shared'}}}
    )
    with testing.Juju() as juju:
        p = juju._deploy(provider)
        r = juju.deploy(spec)
        juju.integrate(p, r)
        # The provider is the one that flaps, and it runs in a new process for
        # each dispatch, so the check still sees it.
        with pytest.raises(testing.errors.JujuError, match='looks like flapping'):
            juju.settle(check_flapping=True)
        assert r._runner is r._fresh_process_runner()


def test_a_real_loop_is_still_a_loop_under_check_flapping(tmp_path: pathlib.Path):
    looping = """
        import ops


        class LoopCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on['shared'].relation_changed, self._on_changed)

            def _on_changed(self, event):
                bag = event.relation.data[self.unit]
                bag['flip'] = 'b' if bag.get('flip') == 'a' else 'a'
    """
    with testing.Juju() as juju:
        a = juju._deploy(write_charm(tmp_path / 'a', 'a', looping, 'provides'))
        b = juju._deploy(write_charm(tmp_path / 'b', 'b', looping, 'requires'))
        juju.integrate(a, b)
        with pytest.raises(testing.errors.JujuError, match='never end') as caught:
            juju.settle(check_flapping=True)
    assert 'flapping' not in str(caught.value)


@pytest.mark.parametrize(
    ('first', 'second', 'same'),
    [
        ('["a", "b", "c"]', '["c", "a", "b"]', True),
        ('["a", "b"]', '["a", "b", "c"]', False),
        ('{"x": ["b", "a"], "y": 1}', '{"y": 1, "x": ["a", "b"]}', True),
        ('a,b,c', 'c,b,a', True),
        ('a b c', 'b c a', True),
        ('abc', 'cba', False),
        ('1', '2', False),
    ],
)
def test_values_that_differ_only_in_order(first: str, second: str, same: bool):
    assert (_deployment._unordered(first) == _deployment._unordered(second)) is same
