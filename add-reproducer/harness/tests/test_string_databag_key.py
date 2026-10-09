"""A relation databag indexed with a string (`spike-step-5/static-retry/RESULT.md` §21).

`RelationData` is keyed by `ops.Unit` and `ops.Application` objects, so
`relation.data['provider']` raises `KeyError` at the subscript, on ops 3.8.3
and on `main`. Two of §19's `#2709` tests did that (3-7 and 3b-2) and §20's
rules let both through. These tests cover the rule, and check by running on
the installed ops that a string key raises `KeyError` and that the keys the
reason suggests do not.

The hand repair of 3b-2 from the new reason is under
`fixtures/static_test_check/2709-hooktool3-key-repaired.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from static_test_check import check, retry_hints

from test_hook_tool_rules import _OUTPUT, _run, _shape

FIXTURES = Path(__file__).parent.parent / "fixtures"

_STRING_KEY = "indexes the databag with the string"
_DENIED = "echo 'ERROR permission denied' >&2; exit 1"
_RELATION = "relation = model.get_relation('db', 2)\n"
_TOOLS = {t: _OUTPUT[t] for t in ("relation-ids", "relation-list", "is-leader")}

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")


def _rule_test(access: str, fakes: dict | None = None, relation: str = _RELATION) -> str:
    fakes = dict(_TOOLS, **{"relation-get": "echo '{}'"}) if fakes is None else fakes
    return "import pytest\n" + _shape(relation + access, fakes) + "\n\ndef other():\n    return 'provider'\n"


def _string_key_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if _STRING_KEY in r]


# -- by running ------------------------------------------------------------------


@pytest.mark.parametrize("key", ["provider", "provider/0", "myapp", "myapp/0", "db"])
@pytest.mark.parametrize("use", ["relation.data[{key!r}]", "dict(relation.data[{key!r}])"])
def test_a_string_key_raises_key_error_at_the_subscript(key, use, tmp_path, monkeypatch):
    """With `relation-get` unfaked, a lookup that reached a read would raise
    `FileNotFoundError` instead."""
    source = _shape(_RELATION + use.format(key=key), dict(_TOOLS))
    with pytest.raises(KeyError) as raised:
        _run(source, tmp_path, monkeypatch)
    assert raised.value.args == (key,)


@pytest.mark.parametrize(
    "key",
    ["relation.app", "model.get_unit('provider/0')", "model.app", "model.unit", "next(iter(relation.units))"],
)
def test_the_keys_the_reason_suggests_work(key, tmp_path, monkeypatch):
    source = _shape(_RELATION + f"assert dict(relation.data[{key}]) == {{}}", dict(_TOOLS, **{"relation-get": "echo '{}'"}))
    _run(source, tmp_path, monkeypatch)


# -- the rule ----------------------------------------------------------------------

_REJECTED = [
    "relation.data['provider']",
    "dict(relation.data['provider'])",
    "assert relation.data['provider/0'] == {}",
    "x = relation.data['myapp']['k']",
    "assert dict(relation.data['provider']) == {}",
    "with pytest.raises(ops.ModelError):\n    relation.data['provider']",
    "with pytest.raises(ops.RelationNotFoundError):\n    dict(relation.data['provider'])",
    "try:\n    result = dict(relation.data['provider'])\nexcept ops.ModelError as e:\n    result = e\nassert result == {}",
    "try:\n    relation.data['provider']\nexcept ops.RelationNotFoundError:\n    pass\nexcept Exception as e:\n    pytest.fail(str(e))",
    "if True:\n    relation.data['provider']",
    "for _ in range(1):\n    relation.data['provider']",
    "x = 1\nrelation.data['provider']",
    # An inline relation access whose fakes print what ops expects.
    "model.get_relation('db', 2).data['provider']",
    "try:\n    dict(model.get_relation('db', 2).data['provider'])\nexcept ops.ModelError:\n    pass",
]


@pytest.mark.parametrize("access", _REJECTED)
def test_a_string_key_is_rejected(access):
    reasons = _string_key_reasons(_rule_test(access))
    assert len(reasons) == 1, check(_rule_test(access)).reasons
    assert "raises `KeyError:" in reasons[0]


@pytest.mark.parametrize("access", _REJECTED)
@pytest.mark.parametrize("relation_get", ["echo '{}'", _DENIED])
def test_a_rejected_string_key_fails_whether_or_not_the_bug_is_there(access, relation_get, tmp_path, monkeypatch):
    """With the bug's denial or the fix's `{}`, the test fails, on the
    `KeyError` or on the second handler that turns it into `pytest.fail`."""
    source = _rule_test(access, dict(_TOOLS, **{"relation-get": relation_get}))
    with pytest.raises((KeyError, pytest.fail.Exception)):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("relation", "ids"),
    [
        ("relation = model.relations['db'][0]\n", _OUTPUT["relation-ids"]),
        ("relation = model.get_relation('db')\n", _OUTPUT["relation-ids"]),
        ("relation = model.get_relation(relation_name='db', relation_id=2)\n", _OUTPUT["relation-ids"]),
    ],
)
def test_other_assigned_relations_are_followed(relation, ids):
    fakes = dict(_TOOLS, **{"relation-ids": ids, "relation-get": "echo '{}'"})
    assert _string_key_reasons(_rule_test("relation.data['provider']", fakes, relation))


def test_the_reason_reads_whole():
    (reason,) = check(_rule_test("assert dict(relation.data['provider']) == {}")).reasons
    assert reason == (
        "line 28: `relation.data['provider']` indexes the databag with the string `'provider'`, but "
        "`Relation.data` is keyed by `ops.Application` and `ops.Unit` objects, not by name, so this raises "
        "`KeyError: 'provider'` at the subscript, before anything is read, with or without the bug; index it "
        "with the object instead: `relation.data[relation.app]` for the remote application's databag "
        "(`relation.data[model.get_unit('<unit name>')]` for a remote unit's; `relation.data[model.app]` or "
        "`relation.data[model.unit]` for this charm's own)"
    )


@pytest.mark.parametrize(
    ("key", "use"),
    [
        ("provider", "`relation.data[relation.app]` for the remote application's databag"),
        ("provider/0", "`relation.data[model.get_unit('provider/0')]` for that unit's databag"),
        ("myapp", "`relation.data[model.app]` for this application's databag"),
        ("myapp/0", "`relation.data[model.unit]` for this unit's databag"),
    ],
)
def test_the_reason_suggests_the_object_the_string_names(key, use):
    (reason,) = _string_key_reasons(_rule_test(f"relation.data[{key!r}]\nassert relation"))
    assert f"index it with the object instead: {use}" in reason


def test_the_reason_names_the_inline_access():
    (reason,) = _string_key_reasons(_rule_test("model.get_relation('db', 2).data['provider']"))
    assert "`model.get_relation('db', 2).data[model.get_relation('db', 2).app]`" in reason


def test_the_reason_brings_no_hint():
    """It says what to write, and not "hook tool", so the `is-leader` hint
    stays out of it."""
    result = check(_rule_test("assert dict(relation.data['provider']) == {}"))
    assert not result.passed
    assert retry_hints(result) == []


@pytest.mark.parametrize(
    "access",
    [
        # Not a string literal.
        "relation.data[relation.app]",
        "key = 'provider'\nrelation.data[key]",
        "relation.data[other()]",
        "relation.data[f'{relation.app.name}']",
        "relation.data[relation.app.name]",
        "relation.data[b'provider']",
        # Not a relation it knows is one.
        "other().data['provider']",
        "model.get_relation('db', other()).data['provider']",
        "model.data['provider']",
        "relation.data2['provider']",
        # Might not run, or its `KeyError` might be caught.
        "if os.environ.get('X'):\n    relation.data['provider']",
        "x = os.environ.get('X') and relation.data['provider']",
        "assert relation, relation.data['provider']",
        "f = lambda: relation.data['provider']",
        "[relation.data['provider'] for _ in os.environ]",
        "try:\n    relation.data['provider']\nexcept KeyError:\n    pass",
        "try:\n    relation.data['provider']\nexcept Exception:\n    pass",
        "try:\n    relation.data['provider']\nexcept (ops.ModelError, KeyError):\n    pass",
        "with pytest.raises(KeyError):\n    relation.data['provider']",
        "with pytest.raises(Exception):\n    relation.data['provider']",
        "def inner():\n    relation.data['provider']",
        # Below a handler that swallows a `ModelError`, after something that
        # might raise one.
        "try:\n    os.getcwd()\n    relation.data['provider']\nexcept ops.ModelError:\n    pass",
        "try:\n    x = [os.getcwd(), relation.data['provider']]\nexcept ops.ModelError:\n    pass",
        # After the test stops.
        "pytest.skip('x')\nrelation.data['provider']",
        # Something might make it a mapping keyed by strings.
        "relation.data = {'provider': {}}\nrelation.data['provider']",
        "monkeypatch.setattr(relation, 'data', {'provider': {}})\nrelation.data['provider']",
    ],
)
def test_a_key_it_cannot_be_sure_of_passes(access):
    source = _rule_test(access + "\nassert relation")
    assert not _string_key_reasons(source), check(source).reasons


@pytest.mark.parametrize(
    ("relation", "ids"),
    [
        # `None` when `relation-ids` prints no ID.
        ("relation = model.get_relation('db')\n", "echo '[]'"),
        # Not known to print exactly one.
        ("relation = model.get_relation('db')\n", None),
        # A list, not a relation.
        ("relation = model.relations['db']\n", _OUTPUT["relation-ids"]),
    ],
)
def test_a_name_that_might_not_be_a_relation_passes(relation, ids):
    fakes = dict(_TOOLS, **{"relation-get": "echo '{}'"})
    if ids is None:
        del fakes["relation-ids"]
    else:
        fakes["relation-ids"] = ids
    source = _rule_test("relation.data['provider']\nassert relation", fakes, relation)
    assert not _string_key_reasons(source), check(source).reasons


def test_an_inline_access_that_might_raise_a_swallowed_model_error_passes():
    """A denied `relation-list` raises a `ModelError` the handler swallows, and
    the subscript never runs."""
    fakes = dict(_TOOLS, **{"relation-list": _DENIED, "relation-get": "echo '{}'"})
    access = "try:\n    model.get_relation('db', 2).data['provider']\nexcept ops.ModelError:\n    pass"
    assert not _string_key_reasons(_rule_test(access + "\nassert model", fakes, ""))


def test_a_test_that_fakes_no_hook_tools_is_left_alone():
    source = "import ops\n\n\ndef test_x(relation):\n    assert dict(relation.data['provider']) == {}\n"
    assert check(source).passed


# -- §19's run 3b-2, repaired from the reason ----------------------------------------


def _repaired():
    (record,) = json.loads((FIXTURES / "static_test_check" / "2709-hooktool3-key-repaired.json").read_text())
    return record


def test_the_hand_repair_passes_the_check():
    record = _repaired()
    assert (record["source"], record["run"]) == ("retry-hooktool3b-2709.json", 2)
    assert check(record["test_file"]["body"]).passed


def test_the_hand_repair_passes_whether_or_not_the_bug_is_there(tmp_path, monkeypatch):
    """The lookup reads nothing, so nothing raises and nothing fails: on ops
    3.8.3 and on `main` it is `did_not_reproduce`."""
    record = _repaired()
    namespace: dict = {}
    exec(compile(record["test_file"]["body"], record["test_file"]["path"], "exec"), namespace)
    (test,) = [v for k, v in namespace.items() if k.startswith("test")]
    test(tmp_path, monkeypatch)
