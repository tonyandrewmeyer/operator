"""A dict for `testing.State(relations=...)` (`spike-step-5/static-retry/
RESULT.md` §25, §26).

Live on `#2709` (§25, run 5-1), an answer passed `relations={'db': {1: {...}}}`.
`relations` takes `testing.Relation` objects: running a `State` built from a
dict fails with `AttributeError: 'str' object has no attribute 'endpoint'`, and
a dict inside the list fails `State(...)` itself with `TypeError: unhashable
type: 'dict'`, on 3.8.3 and `main`. Nine of the earlier saved `#2709` tests
(§10, §11, §13) are the same shape. These tests check both by running, and
what the rule lets through.
"""

from __future__ import annotations

import textwrap

import pytest

from static_test_check import check

_ATTRIBUTE_ERROR = "`AttributeError: 'str' object has no attribute 'endpoint'`"
_TYPE_ERROR = "`TypeError: unhashable type: 'dict'`"

_SOURCE = '''\
import ops
from ops import testing
{imports}

class MyCharm(ops.CharmBase):
    pass


def test_relations():
    ctx = testing.Context(MyCharm, meta={{"name": "myapp", "requires": {{"db": {{"interface": "db"}}}}, "peers": {{"cluster": {{"interface": "c"}}}}}})
{body}
'''


def _source(body: str, imports: str = "") -> str:
    return _SOURCE.format(body=textwrap.indent(textwrap.dedent(body), "    "), imports=imports)


def _run(source: str) -> None:
    namespace: dict = {}
    exec(compile(source, "test_relations.py", "exec"), namespace)
    namespace["test_relations"]()


def _dict_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if "`testing.State`" in r and "`testing.Relation`" in r]


_RUN_DICTS = [
    "ctx.run(ctx.on.update_status(), testing.State(relations={'db': {1: {'app': 'provider'}}}))",
    "ctx.run(ctx.on.update_status(), testing.State(relations={'db': {}}))",
    "state = testing.State(relations={'db': {'provider': {}}})\nctx.run(ctx.on.update_status(), state)",
    "ctx.run(ctx.on.update_status(), state=testing.State(leader=True, relations={'cluster': {}}))",
]

_ELEMENT_DICTS = [
    "testing.State(relations=[{'db': {'active': True}}])",
    "testing.State(relations=[{'name': 'db', 'relation_id': 2, 'app': 'provider'}])",
    "testing.State(relations=({'endpoint': 'db', 'id': 3},))",
    "testing.State(relations=[testing.Relation('db'), {'endpoint': 'cluster'}])",
]


@pytest.mark.parametrize("body", _RUN_DICTS)
def test_running_a_state_built_from_a_dict_is_an_attribute_error(body):
    with pytest.raises(AttributeError, match="'str' object has no attribute 'endpoint'"):
        _run(_source(body))


@pytest.mark.parametrize("body", _ELEMENT_DICTS)
def test_a_dict_in_the_list_is_a_type_error(body):
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        _run(_source(body))


@pytest.mark.parametrize("body", _RUN_DICTS)
def test_a_dict_for_a_state_that_is_run_is_rejected(body):
    (reason,) = _dict_reasons(_source(body))
    assert _ATTRIBUTE_ERROR in reason
    assert "build `testing.Relation(...)` objects instead" in reason


@pytest.mark.parametrize("body", _ELEMENT_DICTS)
def test_a_dict_in_the_list_is_rejected_wherever_it_is(body):
    (reason,) = _dict_reasons(_source(body))
    assert _TYPE_ERROR in reason


@pytest.mark.parametrize(
    ("body", "example"),
    [
        (_RUN_DICTS[0], "testing.Relation('db', id=1, remote_app_data={...})"),
        (_RUN_DICTS[3], "testing.Relation('cluster', id=1, remote_app_data={...})"),
        (_ELEMENT_DICTS[0], "testing.Relation('db', id=1, remote_app_data={...})"),
        (_ELEMENT_DICTS[1], "testing.Relation('db', id=2, remote_app_data={...})"),
        (_ELEMENT_DICTS[2], "testing.Relation('db', id=3, remote_app_data={...})"),
    ],
)
def test_the_reason_names_the_endpoint_and_id_the_dict_does(body, example):
    (reason,) = _dict_reasons(_source(body))
    assert f"`relations=[{example}]`" in reason
    assert reason.endswith("with the databag contents in its `local_app_data`, `remote_app_data` and `remote_units_data`")


def test_from_import_and_full_name_count_too():
    imported = _source("ctx.run(ctx.on.update_status(), State(relations={'db': {}}))", imports="from ops.testing import State")
    full = _source("ctx.run(ctx.on.update_status(), ops.testing.State(relations=[{'db': {}}]))", imports="import ops.testing")
    assert _dict_reasons(imported) and _dict_reasons(full)


@pytest.mark.parametrize(
    "body",
    [
        # No relations.
        "ctx.run(ctx.on.update_status(), testing.State(relations={}))",
        # Built and never run.
        "testing.State(relations={'db': {}})",
        "state = testing.State(relations={'db': {}})",
        # What the reason says to write.
        "ctx.run(ctx.on.update_status(), testing.State(relations=[testing.Relation('db', id=1)]))",
        # Not a literal it can read.
        "ctx.run(ctx.on.update_status(), testing.State(relations={**{}}))",
    ],
)
def test_what_runs_or_it_cannot_be_sure_of_passes(body):
    source = _source(body)
    _run(source)
    assert not _dict_reasons(source)


def test_a_state_that_is_not_ops_testings_passes():
    source = _source("ctx.run(ctx.on.update_status(), State(relations={'db': {}}))", imports="from mylib import State")
    assert not _dict_reasons(source)


def test_a_dict_through_a_name_is_not_followed():
    """It fails just the same, but only literals are read, as for §11's
    keywords."""
    assert not _dict_reasons(_source("rels = {'db': {}}\nctx.run(ctx.on.update_status(), testing.State(relations=rels))"))
