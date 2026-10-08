"""Unread databags, `RelationNotFoundError` around a read, `get_relation(name)`,
and accesses inside other statements (`spike-step-5/static-retry/RESULT.md` §20).

Live, §18's prompt gave 7 `#2709` tests, and 2 would have been posted as
reproductions that fail the same way with the fix (§19). Both end with
`relation.data[relation.app]` as a statement of its own, inside
`pytest.raises(RelationNotFoundError)` or a `try` expecting it: the databag
loads lazily, so nothing runs, and a read could not raise
`RelationNotFoundError` anyway, because `RelationDataContent._load()` turns it
into `{}`. Two more passed the check because its walk did not follow
`get_relation(name)` without an ID, or an access under a second `except`.
These tests cover the rules for those, and check what each shape does by
running it on the installed ops.

The calibration corpus is the 5 §19 tests, under
`fixtures/static_test_check/2709-hooktool3.json`, and their hand repairs from
the new reasons, under `2709-hooktool3-repaired.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import static_test_check
from static_test_check import check, retry_hints

from test_hook_tool_rules import _OUTPUT, _run, _shape

FIXTURES = Path(__file__).parent.parent / "fixtures"

_LOOKED_UP = "only looks the databag up and never reads it"
_CANNOT_RAISE = "cannot raise `RelationNotFoundError`: `RelationDataContent._load()` catches it"
_NOT_JSON = "which is not JSON"
_NOT_FAKED = "and the test does not fake"
_UNDECLARED = "which the `CharmMeta` on line"
_DENIED = "echo 'ERROR permission denied' >&2; exit 1"
_IS_LEADER_RUNS = static_test_check._security_event_runs_is_leader()
_ON = static_test_check._hook_tool_failure_raises_model_error()

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")


def _ops():
    return static_test_check._ops()


# -- the calibration corpus ---------------------------------------------------

# (source, run) -> phrases each expected reason contains, one list per reason.
EXPECTED = {
    # §19's two false comments: a lookup that is never read, expected to
    # raise `RelationNotFoundError`.
    ("retry-hooktool3-2709.json", 0): [[_LOOKED_UP, "the `except ops.RelationNotFoundError` after it never runs"]],
    ("retry-hooktool3-2709.json", 6): [[_LOOKED_UP, 'fails with "DID NOT RAISE"']],
    # A fake's quoting breaks the file.
    ("retry-hooktool3-2709.json", 7): [["does not parse"]],
    # `get_relation('db')`, followed now to `relation-list`, whose fake
    # prints `provider`; and the lookup inside `pytest.raises`.
    ("retry-hooktool3b-2709.json", 1): [
        ["`model.get_relation('db')` needs `relation-list`", _NOT_JSON],
        [_LOOKED_UP, 'fails with "DID NOT RAISE"'],
    ],
    # Under a second handler that fails the test.
    ("retry-hooktool3b-2709.json", 2): [[_UNDECLARED], [_NOT_FAKED + " `relation-ids` and `relation-list`"]],
}


def _corpus(name: str):
    for record in json.loads((FIXTURES / "static_test_check" / name).read_text()):
        yield pytest.param(record, id=f"{record['source']}-{record['run']}")


def test_the_corpus_is_the_5_tests():
    records = [p.values[0] for p in _corpus("2709-hooktool3.json")]
    assert {(r["source"], r["run"]) for r in records} == set(EXPECTED)


@pytest.mark.parametrize("record", list(_corpus("2709-hooktool3.json")))
def test_calibration_verdict(record):
    expected = EXPECTED[(record["source"], record["run"])]
    result = check(record["test_file"]["body"])
    assert not result.passed
    assert len(result.reasons) == len(expected), result.reasons
    for phrases in expected:
        assert any(all(p in reason for p in phrases) for reason in result.reasons), (phrases, result.reasons)


def test_the_false_comments_are_told_to_read_and_assert():
    records = {(r["source"], r["run"]): r for r in (p.values[0] for p in _corpus("2709-hooktool3.json"))}
    for key in [("retry-hooktool3-2709.json", 0), ("retry-hooktool3-2709.json", 6)]:
        (reason,) = check(records[key]["test_file"]["body"]).reasons
        assert "a relation that is gone reads as `{}`" in reason
        assert (
            "`try: result = dict(relation.data[relation.app])` / `except ops.ModelError as e: result = e` / "
            "`assert result == {}`"
        ) in reason
        assert "hook tool" not in reason  # so the `is-leader` hint stays out of it
        assert retry_hints(check(records[key]["test_file"]["body"])) == []


# The hand repairs: exactly what the reasons (and the hint) say.
REPAIRED = {
    ("retry-hooktool3-2709.json", 0): AssertionError,
    ("retry-hooktool3-2709.json", 6): AssertionError,
    ("retry-hooktool3b-2709.json", 1): AssertionError,
    # Indexes the databag with the string `'provider'`, which no rule here
    # catches: `KeyError`, turned into `pytest.fail` by its second handler.
    ("retry-hooktool3b-2709.json", 2): pytest.fail.Exception,
}


@pytest.mark.parametrize("record", list(_corpus("2709-hooktool3-repaired.json")))
def test_the_hand_repairs_pass_the_check(record):
    assert check(record["test_file"]["body"]).passed, check(record["test_file"]["body"]).reasons


@pytest.mark.skipif(not _IS_LEADER_RUNS, reason="the installed ops has the #2709 fix")
@pytest.mark.parametrize("record", list(_corpus("2709-hooktool3-repaired.json")))
def test_the_hand_repairs_ran_as_recorded(record, tmp_path, monkeypatch):
    """On ops 3.8.3, three fail on `assert ModelError('ERROR permission
    denied\\n') == {}`, the bug, and run 3b-2 still dies on its string key."""
    namespace: dict = {}
    exec(compile(record["test_file"]["body"], record["test_file"]["path"], "exec"), namespace)
    (test,) = [v for k, v in namespace.items() if k.startswith("test")]
    expected = REPAIRED[(record["source"], record["run"])]
    with pytest.raises(expected) as raised:
        test(tmp_path, monkeypatch)
    assert type(raised.value) is expected
    if expected is AssertionError:
        # The `assert result == {}` failed on what the read raised.
        (result,) = [t.locals["result"] for t in raised.traceback if "result" in t.locals]
        assert type(result) is _ops().ModelError
        assert str(result) == "ERROR permission denied\n"


# -- by running: a lookup runs nothing, a read cannot raise RelationNotFoundError

_RELATION = "relation = model.get_relation('db', 2)\n"
_TOOLS = {t: _OUTPUT[t] for t in ("relation-ids", "relation-list", "is-leader")}

# What `relation-get` does: the bug (denied, `ModelError` on 3.8.3), the fix
# (the same denial read as `{}`, which is what `main` gives), an empty
# databag, and a relation Juju says is gone.
_RELATION_GET = {
    "denied": _DENIED,
    "fixed": "echo '{}'",
    "empty": "echo '{}'",
    "gone": "echo 'ERROR relation not found' >&2; exit 1",
}

_LOOKUPS = [
    "relation.data[relation.app]",
    "relation.data[model.app]",
    "relation.data[model.unit]",
    "relation.data[app]",
]


@pytest.mark.parametrize("lookup", _LOOKUPS)
def test_a_lookup_runs_no_hook_tool(lookup, tmp_path, monkeypatch):
    """With `relation-get` unfaked, a lookup that loaded the databag would
    raise `FileNotFoundError`."""
    source = _shape(_RELATION + "app = relation.app\n" + lookup, dict(_TOOLS))
    _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "read",
    [
        "dict(relation.data[relation.app])",
        "relation.data[relation.app] == {}",
        "relation.data[relation.app]['k']",
        "relation.data[relation.app].get('k')",
        "[k for k in relation.data[relation.app]]",
        "len(relation.data[relation.app])",
        "bag = relation.data[relation.app]\ndict(bag)",
    ],
)
def test_a_read_runs_relation_get_and_is_not_a_lookup(read, tmp_path, monkeypatch):
    source = _shape(_RELATION + read, dict(_TOOLS))
    assert not [r for r in check(source).reasons if _LOOKED_UP in r]
    with pytest.raises(FileNotFoundError) as raised:
        _run(source, tmp_path, monkeypatch)
    assert raised.value.filename == "relation-get"


@pytest.mark.parametrize("bag", ["relation-get " + k for k in _RELATION_GET])
def test_a_read_never_raises_relation_not_found(bag, tmp_path, monkeypatch):
    """`_load()` turns `RelationNotFoundError` into `{}`."""
    fakes = dict(_TOOLS, **{"relation-get": _RELATION_GET[bag.split()[1]]})
    source = _shape(_RELATION + "dict(relation.data[relation.app])", fakes)
    try:
        _run(source, tmp_path, monkeypatch)
    except Exception as e:
        assert not isinstance(e, _ops().RelationNotFoundError)


# -- rule 1: a databag looked up and never read --------------------------------

_LOOKUP_REJECTED = [
    "with pytest.raises(ops.RelationNotFoundError):\n    relation.data[relation.app]",
    "with pytest.raises(ops.ModelError):\n    relation.data[relation.app]",
    "with pytest.raises((ops.RelationNotFoundError, ops.ModelError)):\n    relation.data[model.app]",
    "with pytest.raises(ops.model.RelationNotFoundError):\n    relation.data[relation.app]\n    pytest.fail('no')",
    (
        "try:\n    relation.data[relation.app]\nexcept ops.RelationNotFoundError:\n    pass\n"
        "else:\n    raise AssertionError('expected RelationNotFoundError')"
    ),
    "try:\n    relation.data[relation.app]\n    pytest.fail('no')\nexcept ops.RelationNotFoundError:\n    pass",
    (
        "try:\n    relation.data[relation.app]\nexcept ops.RelationNotFoundError:\n    pass\n"
        "except Exception as e:\n    pytest.fail(str(e))\nelse:\n    assert False"
    ),
    "if True:\n    with pytest.raises(ops.RelationNotFoundError):\n        relation.data[relation.app]",
    # A relation access whose fakes print what ops expects raises no `ModelError`.
    "with pytest.raises(ops.ModelError):\n    model.get_relation('db', 2).data[model.app]",
    # Nothing in the test can fail on an assertion.
    "relation.data[relation.app]",
    "try:\n    relation.data[relation.app]\nexcept ops.RelationNotFoundError:\n    pass",
]


def _rule_test(access: str, relation_get: str = _DENIED, extra: dict | None = None) -> str:
    fakes = dict(_TOOLS, **{"relation-get": relation_get}, **(extra or {}))
    return "import pytest\n" + _shape(_RELATION + access, fakes)


@pytest.mark.parametrize("access", _LOOKUP_REJECTED)
def test_a_lookup_that_certainly_fails_is_rejected(access):
    reasons = [r for r in check(_rule_test(access)).reasons if _LOOKED_UP in r]
    assert len(reasons) == 1, check(_rule_test(access)).reasons
    assert "runs no `relation-get` and cannot raise an `ops.ModelError`, with or without the bug" in reasons[0]


@pytest.mark.parametrize("access", _LOOKUP_REJECTED[:-2])
@pytest.mark.parametrize("relation_get", list(_RELATION_GET))
def test_a_rejected_lookup_fails_whether_or_not_the_bug_is_there(access, relation_get, tmp_path, monkeypatch):
    """With the bug or the fix's `{}`, the test fails (and not on the
    lookup's own `KeyError`)."""
    source = _rule_test(access, _RELATION_GET[relation_get])
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        _run(source, tmp_path, monkeypatch)


def test_the_reason_for_a_lookup_reads_whole():
    access = "with pytest.raises(ops.RelationNotFoundError):\n    relation.data[relation.app]"
    (reason,) = check(_rule_test(access)).reasons
    assert reason == (
        "line 29: `relation.data[relation.app]` only looks the databag up and never reads it: "
        "`RelationDataContent` loads lazily, so this runs no `relation-get` and cannot raise an "
        "`ops.ModelError`, with or without the bug, and the `pytest.raises(ops.RelationNotFoundError)` "
        'around it fails with "DID NOT RAISE" whether or not the bug is there; reading it would not '
        "raise `RelationNotFoundError` either: `RelationDataContent._load()` catches that and returns "
        "`{}`, in every version of ops, so a relation that is gone reads as `{}`; read the databag "
        "instead, outside the `pytest.raises`, and compare it with what the issue says it should be: "
        "the test's `relation-get` fake exits 1, so on a version of ops with the bug the read raises "
        "`ops.ModelError`, so catch that and assert on the result: `try: result = "
        "dict(relation.data[relation.app])` / `except ops.ModelError as e: result = e` / "
        "`assert result == {}`"
    )


def test_without_a_failing_relation_get_the_reason_says_to_assert_the_read():
    access = "with pytest.raises(ops.ModelError):\n    relation.data[relation.app]"
    (reason,) = check(_rule_test(access, "echo '{}'")).reasons
    assert reason.endswith(
        "compare it with what the issue says it should be (a relation that is gone reads as `{}`): "
        "`assert dict(relation.data[relation.app]) == {}`"
    )
    assert "reading it would not raise" not in reason  # not a `RelationNotFoundError` guard


@pytest.mark.parametrize(
    "access",
    [
        # reads
        "with pytest.raises(ops.ModelError):\n    dict(relation.data[relation.app])",
        "relation.data[relation.app] == {}",
        "assert dict(relation.data[relation.app]) == {}",
        "relation.data[relation.app]['k']",
        "relation.data[relation.app].get('k')",
        "len(relation.data[relation.app])",
        "for k in relation.data[relation.app]:\n    pass",
        "bag = relation.data[relation.app]\nassert dict(bag) == {}",
        "relation.data[relation.app].keys()",
        # a lookup, but the test reads afterwards, or the block need not fail
        "relation.data[relation.app]\nassert dict(relation.data[relation.app]) == {}",
        "try:\n    relation.data[relation.app]\nexcept ops.RelationNotFoundError:\n    pass",
        "try:\n    relation.data[relation.app]\nexcept ops.ModelError:\n    pass\nelse:\n    x = 1",
        "try:\n    relation.data[relation.app]\nexcept Exception:\n    pass\nelse:\n    assert False",
        "try:\n    relation.data[relation.app]\nexcept KeyError:\n    pass\nelse:\n    assert False",
        "with pytest.raises(KeyError):\n    relation.data[relation.app]",
        "with pytest.raises(Exception):\n    relation.data[relation.app]",
        "with pytest.raises(ops.ModelError):\n    relation.data[relation.app]\n    other()",
        "with pytest.raises(ops.ModelError), open(os.devnull):\n    relation.data[relation.app]",
        "if os.environ.get('X'):\n    with pytest.raises(ops.ModelError):\n        relation.data[relation.app]",
        # a lookup it does not know runs nothing
        "with pytest.raises(ops.ModelError):\n    relation.data['provider']",
        "with pytest.raises(ops.ModelError):\n    relation.data[next(iter(relation.units))]",
        "with pytest.raises(ops.ModelError):\n    relation.data[relation.app.status]",
        "with pytest.raises(ops.ModelError):\n    other().data[relation.app]",
        "with pytest.raises(ops.ModelError):\n    model.get_relation('db', 2, other()).data[model.app]",
    ],
)
def test_a_read_or_a_lookup_it_cannot_be_sure_of_passes(access):
    # The `assert` at the end keeps the no-assertion case out of it.
    source = _rule_test(access + "\nassert relation", "echo '{}'") + "\n\ndef other():\n    return None\n"
    assert not [r for r in check(source).reasons if _LOOKED_UP in r], check(source).reasons


def test_a_relation_access_whose_fakes_succeed_counts_as_a_relation(tmp_path, monkeypatch):
    """`model.get_relation('db').data[model.get_relation('db').app]`, as in
    §19's run 3b-1: with `relation-ids` printing one ID and `relation-list`
    unit names, the two calls run them once (the second is cached) and
    nothing else."""
    access = "with pytest.raises(ops.RelationNotFoundError):\n    model.get_relation('db').data[model.get_relation('db').app]"
    fakes = dict(_TOOLS, **{"relation-ids": """echo '["db:1"]'""", "relation-get": _DENIED})
    source = "import pytest\n" + _shape(access, fakes)
    assert [r for r in check(source).reasons if _LOOKED_UP in r]
    with pytest.raises(pytest.fail.Exception):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("ids", "flagged"),
    [
        ("""echo '["db:1"]'""", True),
        ("""echo '["db:1", "db:2"]'""", False),  # `TooManyRelatedAppsError`, a `ModelError`
        ("echo '[]'", True),  # no relation: `AttributeError`
        ("exit 1", False),
        (None, True),  # unfaked: `FileNotFoundError`
    ],
)
def test_a_relation_access_is_a_relation_only_when_it_cannot_raise_model_error(ids, flagged):
    access = "with pytest.raises(ops.ModelError):\n    model.get_relation('db').data[model.app]"
    fakes = {"relation-list": _OUTPUT["relation-list"], "relation-get": "echo '{}'"}
    if ids is not None:
        fakes["relation-ids"] = ids
    reasons = check("import pytest\n" + _shape(access, fakes)).reasons
    assert any(_LOOKED_UP in r for r in reasons) is flagged, reasons


def test_the_rule_is_off_when_the_test_sets_hook_is_running():
    access = "backend._hook_is_running = 'x'\nwith pytest.raises(ops.ModelError):\n    relation.data[relation.app]"
    assert not [r for r in check(_rule_test(access)).reasons if _LOOKED_UP in r]


def test_a_test_that_fakes_no_hook_tools_is_left_alone():
    source = (
        "import pytest\nimport ops\n\n\ndef test_x(relation):\n"
        "    with pytest.raises(ops.RelationNotFoundError):\n        relation.data[relation.app]\n"
    )
    assert check(source).passed


# -- rule 2: RelationNotFoundError around a databag read -----------------------

_READ_REJECTED = [
    "with pytest.raises(ops.RelationNotFoundError):\n    dict(relation.data[relation.app])",
    "with pytest.raises(ops.model.RelationNotFoundError):\n    relation.data[relation.app] == {}",
    "with pytest.raises(ops.RelationNotFoundError):\n    len(relation.data[relation.app])",
    "with pytest.raises(ops.RelationNotFoundError):\n    relation.data[relation.app].get('k')",
    "with pytest.raises(ops.RelationNotFoundError):\n    x = dict(relation.data[relation.app])\n    assert x == {}",
    (
        "try:\n    dict(relation.data[relation.app])\nexcept ops.RelationNotFoundError:\n    pass\n"
        "else:\n    pytest.fail('expected RelationNotFoundError')"
    ),
    "try:\n    x = dict(relation.data[relation.app])\n    assert False\nexcept ops.RelationNotFoundError:\n    pass",
    (
        "bag = relation.data[relation.app]\ntry:\n    dict(bag)\nexcept ops.RelationNotFoundError:\n    pass\n"
        "except Exception:\n    raise\nelse:\n    raise AssertionError('no')"
    ),
]


@pytest.mark.parametrize("access", _READ_REJECTED)
def test_relation_not_found_around_a_read_is_rejected(access):
    reasons = [r for r in check(_rule_test(access, "echo '{}'")).reasons if _CANNOT_RAISE in r]
    assert len(reasons) == 1, check(_rule_test(access, "echo '{}'")).reasons
    assert "so a relation that is gone reads as `{}`" in reasons[0]
    assert "assert that the databag reads as `{}` instead" in reasons[0]


@pytest.mark.parametrize("access", _READ_REJECTED)
@pytest.mark.parametrize("relation_get", list(_RELATION_GET))
def test_relation_not_found_around_a_read_fails_whether_or_not_the_bug_is_there(
    access, relation_get, tmp_path, monkeypatch
):
    source = _rule_test(access, _RELATION_GET[relation_get])
    with pytest.raises((AssertionError, pytest.fail.Exception, _ops().ModelError)) as raised:
        _run(source, tmp_path, monkeypatch)
    assert not isinstance(raised.value, _ops().RelationNotFoundError)


def test_the_reason_for_a_read_reads_whole():
    access = "with pytest.raises(ops.RelationNotFoundError):\n    dict(relation.data[relation.app])"
    reasons = [r for r in check(_rule_test(access, "echo '{}'")).reasons if _CANNOT_RAISE in r]
    assert reasons == [
        "line 29: `dict(relation.data[relation.app])` cannot raise `RelationNotFoundError`: "
        "`RelationDataContent._load()` catches it and returns `{}`, in every version of ops, so a "
        "relation that is gone reads as `{}`, and the `pytest.raises(ops.RelationNotFoundError)` around "
        'it fails with "DID NOT RAISE" whether or not the bug is there; assert that the databag reads as '
        "`{}` instead, outside the `pytest.raises`: `assert dict(relation.data[relation.app]) == {}`"
    ]


def test_with_a_failing_relation_get_the_read_reason_says_to_catch_it():
    access = "with pytest.raises(ops.RelationNotFoundError):\n    dict(relation.data[relation.app])"
    reasons = check(_rule_test(access)).reasons
    (read,) = [r for r in reasons if _CANNOT_RAISE in r]
    assert read.endswith(
        "`try: result = dict(relation.data[relation.app])` / `except ops.ModelError as e: result = e` / "
        "`assert result == {}`"
    )
    if _ON:  # and §18's rule says the same `ModelError` gets out
        assert any("does not catch a plain `ModelError`" in r for r in reasons)


@pytest.mark.parametrize(
    "access",
    [
        # not a `RelationNotFoundError` alone: a read can raise `ModelError`
        "with pytest.raises(ops.ModelError):\n    dict(relation.data[relation.app])",
        "with pytest.raises((ops.RelationNotFoundError, ops.ModelError)):\n    dict(relation.data[relation.app])",
        # the prompt's pattern, with a dead `RelationNotFoundError` handler
        (
            "try:\n    result = dict(relation.data[relation.app])\nexcept ops.RelationNotFoundError:\n"
            "    result = {}\nexcept ops.ModelError as e:\n    result = e\nassert result == {}"
        ),
        # nothing certainly fails when the block completes
        "try:\n    dict(relation.data[relation.app])\nexcept ops.RelationNotFoundError:\n    pass",
        "try:\n    dict(relation.data[relation.app])\nexcept ops.RelationNotFoundError:\n    pass\nelse:\n    x = 1",
        # a relation access that is not a databag read can raise it on `main`
        "with pytest.raises(ops.RelationNotFoundError):\n    model.get_relation('db', 2)",
        "with pytest.raises(ops.RelationNotFoundError):\n    backend.relation_get(2, 'provider', True)",
        "with pytest.raises(ops.RelationNotFoundError):\n    dict(model.get_relation('db', 2).data[model.app])",
        # something else in the block could raise it
        "with pytest.raises(ops.RelationNotFoundError):\n    dict(relation.data[relation.app])\n    other()",
        "with pytest.raises(ops.RelationNotFoundError):\n    x = [dict(relation.data[relation.app]), other()]",
        "with pytest.raises(ops.RelationNotFoundError):\n    relation.data[relation.app.status].get('k')",
        "with pytest.raises(ops.RelationNotFoundError):\n    dict(relation.data[relation.app])['k'].upper()",
        # a handler that might swallow it and not fail
        (
            "try:\n    dict(relation.data[relation.app])\nexcept ops.RelationNotFoundError:\n    pass\n"
            "except Exception:\n    x = 1\nelse:\n    assert False"
        ),
    ],
)
def test_a_read_it_cannot_be_sure_of_passes(access):
    source = _rule_test(access, "echo '{}'") + "\n\ndef other():\n    return None\n"
    assert not [r for r in check(source).reasons if _CANNOT_RAISE in r], check(source).reasons


# -- rule 3: get_relation(name) without an ID ----------------------------------


@pytest.mark.parametrize("access", ["model.get_relation('db')", "model.relations['db']"])
@pytest.mark.parametrize(
    ("ids", "relation_list", "reason", "raises"),
    [
        ("""echo '["db:1"]'""", None, "does not fake `relation-list`", FileNotFoundError),
        ("""echo '["db:1"]'""", "echo provider", _NOT_JSON, json.JSONDecodeError),
        ("""echo '["db:1", "db:2"]'""", None, "does not fake `relation-list`", FileNotFoundError),
        ("echo '[]'", None, None, None),  # no IDs: no `relation-list`
        ("echo '[]'", "echo provider", None, None),
    ],
)
def test_an_access_by_name_runs_relation_list_once_relation_ids_prints_ids(
    access, ids, relation_list, reason, raises, tmp_path, monkeypatch
):
    fakes = {"relation-ids": ids}
    if relation_list is not None:
        fakes["relation-list"] = relation_list
    source = _shape(access, fakes)
    reasons = [r for r in check(source).reasons if _NOT_FAKED in r or _NOT_JSON in r]
    if reason is None:
        assert not reasons
        _run(source, tmp_path, monkeypatch)
        return
    (found,) = reasons
    assert reason in found
    with pytest.raises(raises):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "ids",
    ["echo $IDS", "cat ids.json", 'echo "[\\"db:1\\"]"', "echo '[\"db:x\"]'", "echo '{\"db\": 1}'", "echo '[1]'"],
)
def test_an_access_by_name_is_not_followed_when_the_ids_are_not_certain(ids):
    source = _shape("model.get_relation('db')", {"relation-ids": ids, "relation-list": "echo provider"})
    assert not [r for r in check(source).reasons if "needs `relation-list`" in r], check(source).reasons


@pytest.mark.parametrize(
    "access",
    [
        "model.get_relation('db')",
        "with pytest.raises(ops.RelationNotFoundError):\n    model.get_relation('db')",
    ],
)
def test_the_section_19_probe_is_rejected(access):
    """§19: with `relation-list` faked to print `provider`, `get_relation('db',
    1)` was rejected and `get_relation('db')` passed, bare or inside
    `pytest.raises`."""
    fakes = {"relation-ids": """echo '["db:1"]'""", "relation-list": "echo provider"}
    source = "import pytest\n" + _shape(access, fakes)
    assert any("`model.get_relation('db')` needs `relation-list`" in r and _NOT_JSON in r for r in check(source).reasons)


def test_two_accesses_in_one_expression_report_the_first():
    """The subscript's value is evaluated before its key, so the first
    `get_relation('db')` runs first; the second is cached."""
    access = "with pytest.raises(ops.RelationNotFoundError):\n    model.get_relation('db').data[model.get_relation('db').app]"
    fakes = {"relation-ids": """echo '["db:1"]'""", "relation-list": "echo provider", "relation-get": "echo '{}'"}
    reasons = [r for r in check("import pytest\n" + _shape(access, fakes)).reasons if _NOT_JSON in r]
    assert len(reasons) == 1 and reasons[0].startswith("line 27: `model.get_relation('db')` needs `relation-list`")


def test_a_read_after_an_access_by_name_needs_only_relation_get():
    """`relation-list` ran at the assignment, and is reported there."""
    access = "relation = model.get_relation('db')\ndict(relation.data[relation.app])"
    reasons = [r for r in check(_shape(access, {"relation-ids": """echo '["db:1"]'"""})).reasons if _NOT_FAKED in r]
    assert len(reasons) == 2
    assert "line 23: `model.get_relation('db')` needs the `relation-ids` and `relation-list`" in reasons[0]
    assert "line 24: `dict(relation.data[relation.app])` needs the `relation-get` hook tool" in reasons[1]


@pytest.mark.parametrize(
    ("statement", "first"),
    [
        ("x = f(a(), b())", "a()"),
        ("x = f(a(), b())", "b()"),
        ("x = f(g(a()))", "a()"),
        ("x[a()] = b()", "b()"),
        ("x[a()] = b()", "a()"),
        ("a().data[b().app]", "a()"),
        ("a().data[b().app]", "b()"),
        ("assert a() == b()", "b()"),
    ],
)
def test_called_first(statement, first):
    tree = static_test_check.ast.parse(statement)
    (node,) = [
        n for n in static_test_check.ast.walk(tree) if static_test_check.ast.unparse(n) == first
    ]
    expected = {
        ("x = f(a(), b())", "a()"): True,
        ("x = f(a(), b())", "b()"): False,
        ("x = f(g(a()))", "a()"): True,
        ("x[a()] = b()", "b()"): True,
        ("x[a()] = b()", "a()"): False,
        ("a().data[b().app]", "a()"): True,
        ("a().data[b().app]", "b()"): False,
        ("assert a() == b()", "b()"): False,
    }[(statement, first)]
    assert static_test_check._called_first(tree.body[0], node) is expected


# -- rule 4: accesses inside other statements -----------------------------------

_DBC = "model.get_relation('dbc', 1)"

_ALSO_CERTAIN = [
    "try:\n    {access}\nexcept ops.RelationNotFoundError:\n    pass\nexcept Exception as e:\n    pytest.fail(f'got {{e!r}}')",
    "try:\n    {access}\nexcept Exception as e:\n    raise AssertionError(e)",
    "try:\n    {access}\nexcept Exception:\n    raise",
    "try:\n    {access}\nexcept ops.ModelError:\n    pass\nexcept Exception:\n    assert False",
    "if True:\n    {access}",
    "if 0:\n    pass\nelse:\n    {access}",
    "with contextlib.nullcontext():\n    {access}",
    "with open(os.devnull):\n    {access}",
    "for _ in range(1):\n    {access}",
    "for _ in [1, 2]:\n    {access}",
    "while True:\n    {access}",
    "if True:\n    x = 1\n    with contextlib.nullcontext():\n        {access}",
]


def _wrapped(wrap: str, access: str, fakes: dict, meta: str | None = None) -> str:
    kwargs = {} if meta is None else {"meta": meta}
    return "import contextlib\nimport pytest\n" + _shape(wrap.format(access=access), fakes, **kwargs)


@pytest.mark.parametrize("wrap", _ALSO_CERTAIN)
def test_an_undeclared_endpoint_inside_a_statement_that_certainly_runs_is_rejected(wrap, tmp_path, monkeypatch):
    """§19's run 3b-2: `get_relation('dbc', 1)` with no relations declared."""
    source = _wrapped(wrap, _DBC, dict(_OUTPUT), meta="name: myapp\n")
    assert any("reads the `dbc` endpoint" in r for r in check(source).reasons), check(source).reasons
    with pytest.raises((KeyError, AssertionError, pytest.fail.Exception)):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize("wrap", _ALSO_CERTAIN)
def test_an_unfaked_tool_inside_a_statement_that_certainly_runs_is_rejected(wrap, tmp_path, monkeypatch):
    fakes = {"relation-ids": _OUTPUT["relation-ids"]}
    source = _wrapped(wrap, "model.get_relation('db', 2)", fakes)
    assert any("does not fake `relation-list`" in r for r in check(source).reasons), check(source).reasons
    with pytest.raises((FileNotFoundError, AssertionError, pytest.fail.Exception)):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize("wrap", _ALSO_CERTAIN)
def test_a_non_json_fake_inside_a_statement_that_certainly_runs_is_rejected(wrap, tmp_path, monkeypatch):
    fakes = {"relation-ids": _OUTPUT["relation-ids"], "relation-list": "echo provider"}
    source = _wrapped(wrap, "model.get_relation('db', 2)", fakes)
    assert any(_NOT_JSON in r for r in check(source).reasons), check(source).reasons
    with pytest.raises((json.JSONDecodeError, AssertionError, pytest.fail.Exception)):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.skipif(not _ON, reason="the installed ops does not raise ModelError for every failed hook tool")
@pytest.mark.parametrize("wrap", _ALSO_CERTAIN[4:])
def test_an_uncaught_error_inside_a_statement_that_certainly_runs_is_rejected(wrap, tmp_path, monkeypatch):
    fakes = dict(_OUTPUT, **{"relation-get": _DENIED})
    source = _wrapped(wrap, "dict(relation.data[relation.app])", fakes).replace(
        "    model = ops.Model(meta, backend)\n", "    model = ops.Model(meta, backend)\n" + "    " + _RELATION
    )
    assert any("and nothing around it catches that" in r for r in check(source).reasons), check(source).reasons
    with pytest.raises(_ops().ModelError):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.skipif(not _ON, reason="the installed ops does not raise ModelError for every failed hook tool")
@pytest.mark.parametrize("wrap", _ALSO_CERTAIN[:4])
def test_a_handler_that_catches_model_error_still_passes_the_uncaught_rule(wrap):
    """`except Exception` catches `ModelError`: the test fails in the handler,
    not on an uncaught exception, so §18's rule has nothing to say."""
    fakes = dict(_OUTPUT, **{"relation-get": _DENIED})
    source = _wrapped(wrap, "dict(relation.data[relation.app])", fakes).replace(
        "    model = ops.Model(meta, backend)\n", "    model = ops.Model(meta, backend)\n" + "    " + _RELATION
    )
    assert not [r for r in check(source).reasons if "so the installed ops raises `ops.ModelError` there" in r]


_NOT_CERTAIN = [
    "if os.environ.get('X'):\n    {access}",
    "if False:\n    {access}",
    "if True:\n    pass\nelse:\n    {access}",
    "for _ in range(0):\n    {access}",
    "for _ in os.environ:\n    {access}",
    "for _ in [1]:\n    break\n    {access}",
    "for _ in [1]:\n    if os.environ.get('X'):\n        continue\n    {access}",
    "while os.environ.get('X'):\n    {access}",
    "while True:\n    {access}\n    break",
    "with contextlib.suppress(Exception):\n    {access}",
    "with something():\n    {access}",
    "try:\n    {access}\nexcept Exception:\n    pass",
    "try:\n    {access}\nexcept Exception as e:\n    print(e)\n    pytest.fail('x')",
    "try:\n    {access}\nexcept Exception:\n    raise ops.ModelError('x')",
    "try:\n    {access}\nexcept Exception:\n    raise make_error()",
    "try:\n    pass\nexcept Exception:\n    {access}",
    "try:\n    pass\nexcept Exception:\n    pass\nelse:\n    {access}",
    "f = lambda: {access}",
    "match 1:\n    case 1:\n        {access}",
]
_HELPERS = "\n\ndef something():\n    return contextlib.nullcontext()\n\n\ndef make_error():\n    return ValueError()\n"


@pytest.mark.parametrize("wrap", _NOT_CERTAIN)
def test_an_undeclared_endpoint_that_might_not_be_read_or_whose_error_might_be_swallowed_is_left_alone(wrap):
    source = _wrapped(wrap, _DBC, dict(_OUTPUT), meta="name: myapp\n") + _HELPERS
    assert not [r for r in check(source).reasons if "reads the `dbc` endpoint" in r]


@pytest.mark.parametrize(
    "wrap",
    [
        *_NOT_CERTAIN,
        "def inner():\n    {access}",
        "try:\n    pass\nfinally:\n    {access}",
        "try:\n    os.getcwd()\n    {access}\nexcept ops.ModelError:\n    pass",
        "with pytest.raises(ops.ModelError):\n    if True:\n        os.getcwd()\n        {access}",
    ],
)
def test_an_access_that_might_not_run_or_whose_error_might_be_swallowed_is_left_alone(wrap):
    source = _wrapped(wrap, "model.get_relation('db', 2)", {"relation-list": _OUTPUT["relation-list"]}) + _HELPERS
    assert not [r for r in check(source).reasons if _NOT_FAKED in r]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* is Python 3.11 and later")
def test_an_access_in_an_except_star_try_is_left_alone():
    source = _wrapped("try:\n    {access}\nexcept* Exception:\n    pytest.fail('x')", _DBC, dict(_OUTPUT), meta="name: myapp\n")
    assert not [r for r in check(source).reasons if "reads the `dbc` endpoint" in r]


# -- the re-ask -----------------------------------------------------------------


def test_the_new_reasons_bring_no_hint_of_their_own():
    """Each says what to write; and none says "hook tool", so the `is-leader`
    hint only comes with the reasons it came with before."""
    for access in [_LOOKUP_REJECTED[0], _READ_REJECTED[0]]:
        result = check(_rule_test(access, "echo '{}'"))
        assert not result.passed
        assert retry_hints(result) == []
