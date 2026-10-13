"""A relation lookup expected to raise, and a name only a handler binds
(`spike-step-5/static-retry/RESULT.md` §25, §26).

Live on `#2709` (§25, run 5-7), a test wrapped `model.get_relation('db', 1)` in
`pytest.raises(ops.RelationNotFoundError)`. The lookup runs `relation-ids` and
`relation-list`, never `relation-get`, so it failed "DID NOT RAISE" on ops 3.8.3
and `main`, and the classifier called that `reproduced_weaker`: a false comment.
§23's 4-4 and §25's 5-4, repaired, are the `try` form of the same misreading:
`except ops.ModelError as e: result = e`, then `assert isinstance(result, ...)`,
an `UnboundLocalError` on both versions. These tests check by running that the
blocks the rule rejects fail on the installed ops whatever the bug, cover what
it lets through, and do the same for `_unbound_after_try()`, which takes the
`try` form when the body is an unread databag lookup.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from static_test_check import check, retry_hints

from test_hook_tool_rules import _OUTPUT, _run, _shape

FIXTURES = Path(__file__).parent.parent / "fixtures"
_DENIED = "echo 'ERROR permission denied' >&2; exit 1"
_NEVER_READS = "never reads the databag"
_DID_NOT_RAISE = '"DID NOT RAISE" on every version of ops'
_UNBOUND = "fails with `UnboundLocalError` on every version of ops"
_LAZY = "only looks the databag up and never reads it"

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")


def _fakes(relation_get: str = _DENIED, ids: str = _OUTPUT["relation-ids"], units: str = _OUTPUT["relation-list"]):
    return {"relation-ids": ids, "relation-list": units, "relation-get": relation_get, "is-leader": "echo false"}


def _source(access: str, fakes: dict[str, str] | None = None, meta: str | None = None) -> str:
    kwargs = {} if meta is None else {"meta": meta}
    return "import pytest\n" + _shape(textwrap.dedent(access), fakes or _fakes(), **kwargs)


def _reasons(source: str, phrase: str) -> list[str]:
    return [r for r in check(source).reasons if phrase in r]


# Lookups whose `relation-ids` / `relation-list` fakes print what ops expects,
# so they raise nothing at all.
_LOOKUPS = [
    "model.get_relation('db', 2)",
    "model.get_relation('db', 7)",
    "model.get_relation('db')",
    "model.relations['db']",
    "model.relations['db'][0]",
    "model.get_relation('cluster', 2)",
    "model.get_relation(relation_name='db', relation_id=2)",
]

_RAISES = ["ops.RelationNotFoundError", "ops.ModelError", "Exception", "(KeyError, ops.ModelError)"]


def _raises_block(lookup: str, raised: str, assign: bool = False) -> str:
    body = f"relation = {lookup}" if assign else lookup
    return f"with pytest.raises({raised}):\n    {body}"


def _try_else(lookup: str) -> str:
    return f"try:\n    {lookup}\nexcept ops.RelationNotFoundError:\n    pass\nelse:\n    pytest.fail('expected RelationNotFoundError')"


def _try_unbound(lookup: str, handler: str = "ops.ModelError") -> str:
    return f"try:\n    {lookup}\nexcept {handler} as e:\n    result = e\nassert isinstance(result, ops.RelationNotFoundError)"


# -- by running ----------------------------------------------------------------


@pytest.mark.parametrize("lookup", _LOOKUPS)
@pytest.mark.parametrize("raised", _RAISES)
@pytest.mark.parametrize("relation_get", [_DENIED, "echo '{}'"])
def test_a_lookup_in_pytest_raises_did_not_raise(lookup, raised, relation_get, tmp_path, monkeypatch):
    """Whether or not `relation-get` is denied (the bug), and whatever is
    expected."""
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        _run(_source(_raises_block(lookup, raised), _fakes(relation_get)), tmp_path, monkeypatch)


@pytest.mark.parametrize("lookup", _LOOKUPS)
def test_a_lookup_in_a_try_with_a_failing_else_fails(lookup, tmp_path, monkeypatch):
    with pytest.raises(pytest.fail.Exception, match="expected RelationNotFoundError"):
        _run(_source(_try_else(lookup)), tmp_path, monkeypatch)


@pytest.mark.parametrize("lookup", _LOOKUPS)
def test_a_name_only_the_handler_binds_is_unbound(lookup, tmp_path, monkeypatch):
    with pytest.raises(UnboundLocalError):
        _run(_source(_try_unbound(lookup)), tmp_path, monkeypatch)


def test_a_denied_relation_list_is_not_a_lookup_that_raises_nothing(tmp_path, monkeypatch):
    """The lookup's own hook tools are where a version of ops could raise:
    3.8.3 raises `ModelError` here, and `main` treats the relation as gone
    and raises nothing. Either way the test differs by version, so the rule
    leaves it alone (below)."""
    source = _source(_raises_block("model.get_relation('db', 2)", "ops.ModelError"), _fakes(units=_DENIED))
    try:
        _run(source, tmp_path, monkeypatch)
    except pytest.fail.Exception:
        pass  # a version of ops that treats it as gone


def test_too_many_related_apps_does_raise(tmp_path, monkeypatch):
    """`get_relation(name)` with two IDs raises, so it is not a lookup that
    raises nothing (below)."""
    source = _source(
        _raises_block("model.get_relation('db')", "ops.TooManyRelatedAppsError"),
        _fakes(ids="""echo '["db:1", "db:2"]'"""),
    )
    _run(source, tmp_path, monkeypatch)


# -- the rule ------------------------------------------------------------------


@pytest.mark.parametrize("lookup", _LOOKUPS)
@pytest.mark.parametrize("raised", _RAISES)
def test_a_lookup_in_pytest_raises_is_rejected(lookup, raised):
    (reason,) = _reasons(_source(_raises_block(lookup, raised)), _NEVER_READS)
    assert _DID_NOT_RAISE in reason
    assert f"pytest.raises({raised})" in reason
    assert "the bug can only show when `relation.data[...]` is read" in reason
    assert "get the relation outside the `pytest.raises`" in reason


@pytest.mark.parametrize("lookup", _LOOKUPS)
def test_an_assigned_lookup_in_pytest_raises_is_rejected(lookup):
    assert _reasons(_source(_raises_block(lookup, "ops.RelationNotFoundError", assign=True)), _NEVER_READS)


@pytest.mark.parametrize("lookup", _LOOKUPS)
def test_a_lookup_in_a_try_with_a_failing_else_is_rejected(lookup):
    (reason,) = _reasons(_source(_try_else(lookup)), _NEVER_READS)
    assert "the `except ops.RelationNotFoundError` after it never runs" in reason


@pytest.mark.parametrize("lookup", _LOOKUPS)
@pytest.mark.parametrize("handler", ["ops.ModelError", "ops.RelationNotFoundError", "Exception"])
def test_a_lookup_whose_handler_alone_binds_the_result_is_rejected(lookup, handler):
    (reason,) = _reasons(_source(_try_unbound(lookup, handler)), _NEVER_READS)
    assert "so `result` is never bound" in reason
    assert _UNBOUND in reason


def test_the_reason_says_to_catch_and_assert_when_relation_get_fails():
    (reason,) = _reasons(_source(_raises_block("model.get_relation('db', 2)", "ops.RelationNotFoundError")), _NEVER_READS)
    assert reason.endswith(
        "`relation = model.get_relation('db', 2)`, then read the databag of the gone relation and assert what it "
        "returns (a relation that is gone reads as `{}`): the test's `relation-get` fake exits 1, so on a version "
        "of ops with the bug the read raises `ops.ModelError`, so catch that and assert on the result: "
        "`try: result = dict(relation.data[relation.app])` / `except ops.ModelError as e: result = e` / "
        "`assert result == {}`"
    )


def test_the_reason_reads_without_a_catch_when_relation_get_succeeds():
    (reason,) = _reasons(_source(_raises_block("model.relations['db']", "Exception"), _fakes("echo '{}'")), _NEVER_READS)
    assert "`relation = model.relations['db'][0]`" in reason
    assert reason.endswith("`assert dict(relation.data[relation.app]) == {}`")


@pytest.mark.parametrize(
    ("access", "fakes"),
    [
        # Its hook tools could raise, by version (above).
        (_raises_block("model.get_relation('db', 2)", "ops.ModelError"), _fakes(units=_DENIED)),
        (_raises_block("model.get_relation('db', 2)", "ops.ModelError"), _fakes(ids=_DENIED)),
        # `TooManyRelatedAppsError` (above), and an `IndexError`.
        (_raises_block("model.get_relation('db')", "ops.TooManyRelatedAppsError"), _fakes(ids="""echo '["db:1", "db:2"]'""")),
        (_raises_block("model.relations['db'][1]", "IndexError"), None),
        # Fakes it cannot be sure of.
        (_raises_block("model.get_relation('db', 2)", "ops.ModelError"), _fakes(units="echo $UNITS")),
        (_raises_block("model.get_relation('db', 2)", "ops.ModelError"), _fakes(ids="cat ids.json")),
        # Something in the block reads the databag.
        (
            "with pytest.raises(ops.ModelError):\n    relation = model.get_relation('db', 2)\n    dict(relation.data[relation.app])",
            None,
        ),
        # Something else in the block can raise.
        ("with pytest.raises(ValueError):\n    model.get_relation('db', 2)\n    int('x')", None),
        # A `try` that may complete without failing, and one whose name is
        # bound before it.
        ("try:\n    model.get_relation('db', 2)\nexcept ops.ModelError:\n    pass", None),
        ("result = None\n" + _try_unbound("model.get_relation('db', 2)"), None),
        # Conditional.
        ("if model.name:\n" + textwrap.indent(_raises_block("model.get_relation('db', 2)", "ops.ModelError"), "    "), None),
    ],
)
def test_what_it_cannot_be_sure_of_passes(access, fakes):
    assert not _reasons(_source(access, fakes), _NEVER_READS)


def test_an_undeclared_endpoint_passes():
    """`model.get_relation('nope', 2)` raises `KeyError`, which
    `pytest.raises(Exception)` catches."""
    assert not _reasons(_source(_raises_block("model.get_relation('nope', 2)", "Exception")), _NEVER_READS)


def test_two_fakes_of_relation_ids_are_not_certain():
    """Fakes are still file-wide (§24): a second test faking `relation-ids`
    makes neither certain. That is why §23's 4-4 repair still passes."""
    first = _source(_raises_block("model.get_relation('db', 2)", "ops.ModelError"))
    second = first.split("def test_access", 1)[1].replace("(tmp_path, monkeypatch):", "", 1)
    assert not _reasons(first + "\n\ndef test_again(tmp_path, monkeypatch):" + second, _NEVER_READS)


# -- `testing.Context` --------------------------------------------------------

_CONTEXT = '''\
import ops
import pytest
from ops import testing


class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.update_status, self._on_update_status)

    def _on_update_status(self, event):
{body}


def test_access():
    ctx = testing.Context(MyCharm, meta={{"name": "myapp", "requires": {{"db": {{"interface": "db"}}}}}})
    ctx.run(ctx.on.update_status(), testing.State({state}))
'''


def _context(body: str, state: str = "") -> str:
    return _CONTEXT.format(body=textwrap.indent(textwrap.dedent(body), "        "), state=state)


_CHARM_LOOKUPS = ["self.model.get_relation('db', 1)", "self.model.relations['db']"]


@pytest.mark.parametrize("lookup", _CHARM_LOOKUPS)
@pytest.mark.parametrize("state", ["", "relations=[testing.Relation('db', id=1)]"])
def test_a_charm_lookup_in_pytest_raises_did_not_raise(lookup, state):
    namespace: dict = {}
    exec(compile(_context(_raises_block(lookup, "ops.RelationNotFoundError"), state), "test_access.py", "exec"), namespace)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        namespace["test_access"]()


@pytest.mark.parametrize("lookup", _CHARM_LOOKUPS)
@pytest.mark.parametrize("raised", ["ops.RelationNotFoundError", "ops.ModelError"])
def test_a_charm_lookup_in_pytest_raises_is_rejected(lookup, raised):
    (reason,) = _reasons(_context(_raises_block(lookup, raised)), _NEVER_READS)
    assert "`testing.Context` answers them from the `State`" in reason
    assert _DID_NOT_RAISE in reason
    assert "assert on that after `ctx.run(...)` returns" in reason


@pytest.mark.parametrize(
    "body",
    [
        # An undeclared endpoint raises `KeyError`, which these catch.
        _raises_block("self.model.get_relation('db', 1)", "Exception"),
        _raises_block("self.model.get_relation('db', 1)", "KeyError"),
        # `TooManyRelatedAppsError` with two relations in the `State`.
        _raises_block("self.model.get_relation('db')", "ops.ModelError"),
        # Not `self.model`.
        _raises_block("other.get_relation('db', 1)", "ops.ModelError"),
    ],
)
def test_charm_lookups_it_cannot_be_sure_of_pass(body):
    assert not _reasons(_context(body), _NEVER_READS)


def test_a_harness_file_is_left_alone():
    source = _context(_raises_block("self.model.get_relation('db', 1)", "ops.ModelError")) + "\nfrom ops.testing import Harness\n"
    assert not _reasons(source, _NEVER_READS)


# -- the `try` form with an unread databag lookup (`_unbound_after_try()`) ----

_LAZY_TRY = """\
relation = model.get_relation('db', 2)
try:
    relation.data[{key}]
except ops.ModelError as e:
    result = e
assert result == {{}}
"""


@pytest.mark.parametrize("key", ["relation.app", "model.app", "model.unit"])
@pytest.mark.parametrize("relation_get", [_DENIED, "echo '{}'"])
def test_an_unread_lookup_whose_handler_alone_binds_the_result_is_unbound(key, relation_get, tmp_path, monkeypatch):
    with pytest.raises(UnboundLocalError):
        _run(_source(_LAZY_TRY.format(key=key), _fakes(relation_get)), tmp_path, monkeypatch)


@pytest.mark.parametrize("key", ["relation.app", "model.app", "model.unit"])
def test_an_unread_lookup_whose_handler_alone_binds_the_result_is_rejected(key):
    (reason,) = _reasons(_source(_LAZY_TRY.format(key=key)), _LAZY)
    assert "so `result` is never bound, and line" in reason
    assert _UNBOUND in reason
    assert "read the databag inside the `try` and bind the result there too" in reason
    assert reason.endswith(
        f"`try: result = dict(relation.data[{key}])` / `except ops.ModelError as e: result = e` / `assert result == {{}}`"
    )


@pytest.mark.parametrize(
    "access",
    [
        # The read binds it in the `try`.
        "relation = model.get_relation('db', 2)\ntry:\n    result = dict(relation.data[relation.app])\nexcept ops.ModelError as e:\n    result = e\nassert result == {}",
        # A key that is not certainly in the databag (`KeyError`).
        "relation = model.get_relation('db', 2)\nother = 'provider'\ntry:\n    relation.data[other]\nexcept Exception as e:\n    result = e\nassert result",
        # Nothing reads the name.
        "relation = model.get_relation('db', 2)\ntry:\n    relation.data[relation.app]\nexcept ops.ModelError as e:\n    result = e",
        # Bound in an `else` too.
        "relation = model.get_relation('db', 2)\ntry:\n    relation.data[relation.app]\nexcept ops.ModelError as e:\n    result = e\nelse:\n    result = {}\nassert result == {}",
        # The test stops first.
        "relation = model.get_relation('db', 2)\ntry:\n    relation.data[relation.app]\nexcept ops.ModelError as e:\n    result = e\npytest.skip('later')\nassert result == {}",
    ],
)
def test_unbound_shapes_it_cannot_be_sure_of_pass(access):
    """(§20's rule still rejects the one where nothing can fail on an
    assertion, for its own reason.)"""
    assert not _reasons(_source(access), _UNBOUND)


# -- the saved tests -----------------------------------------------------------


def _corpus(name: str):
    for record in json.loads((FIXTURES / "static_test_check" / name).read_text()):
        yield pytest.param(record, id=f"{record['source']}-{record['run']}")


# §25's five test files: run -> phrases each reason contains.
EXPECTED = {
    1: [["`relations=` is a dict", "`AttributeError: 'str' object has no attribute 'endpoint'`"]],
    2: [["needs `relation-ids`", "which is not JSON"], ["needs `relation-list`", "which is not JSON"]],
    3: [["its fake prints `[]` whatever its arguments are"]],
    4: [["the test does not fake `relation-list`"]],
    7: [[_NEVER_READS, "`pytest.raises(ops.RelationNotFoundError)`", _DID_NOT_RAISE]],
}


def test_the_corpus_is_the_5_tests():
    assert {p.values[0]["run"] for p in _corpus("2709-hooktool5.json")} == set(EXPECTED)


@pytest.mark.parametrize("record", list(_corpus("2709-hooktool5.json")))
def test_calibration_verdict(record):
    expected = EXPECTED[record["run"]]
    result = check(record["test_file"]["body"])
    assert not result.passed
    assert len(result.reasons) == len(expected), result.reasons
    for phrases in expected:
        assert any(all(p in reason for p in phrases) for reason in result.reasons), (phrases, result.reasons)


@pytest.mark.parametrize("record", list(_corpus("2709-hooktool5-repaired.json")))
def test_the_hand_repairs_pass_the_check(record):
    result = check(record["test_file"]["body"])
    assert result.passed, result.reasons
    assert retry_hints(result) == []


def test_5_4_with_its_relation_list_fake_is_rejected():
    """§25's probe: 5-4 with the `relation-list` fake it lacked. It passed the
    check before and dies with `UnboundLocalError` on 3.8.3 and `main`."""
    (record,) = [p.values[0] for p in _corpus("2709-hooktool5.json") if p.values[0]["run"] == 4]
    body = record["test_file"]["body"].replace(
        "    fake_hook_tool(bin_dir, 'is-leader', 'echo true')",
        "    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'[\"provider/0\"]\\'')\n"
        "    fake_hook_tool(bin_dir, 'is-leader', 'echo true')",
    )
    (reason,) = check(body).reasons
    assert reason.startswith("line 24: `model.get_relation('db', 1)` never reads the databag")
    assert "so `got` is never bound, and line 27 fails with `UnboundLocalError`" in reason


def test_4_4s_repair_still_passes():
    """Its two tests each fake every tool, so no fake is certain (§24)."""
    (record,) = [p.values[0] for p in _corpus("2709-hooktool4-repaired.json") if p.values[0]["run"] == 4]
    assert check(record["test_file"]["body"]).passed
