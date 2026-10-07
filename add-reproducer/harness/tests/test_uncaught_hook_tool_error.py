"""An uncaught hook-tool error (`spike-step-5/static-retry/RESULT.md` §18).

Repaired by hand from §17's reasons, 4 of `#2709`'s 8 fake-hook-tool tests
reached the bug and passed with the fix, and none was valid: two read the
databag with nothing around the read, and one wrapped it in
`pytest.raises(RelationNotFoundError)`, which the plain `ModelError` ops 3.8.3
raises gets out of. Each died on the exception, which rung 1c keeps silent.
These tests cover the rule that rejects that shape, check what ops raises for
each access by running it on the installed ops, and cover the prompt's
instruction to catch the error and assert.

The calibration corpus is §17's hand repairs, rebuilt from what §17 says it
changed, under `fixtures/static_test_check/2709-hooktool-repaired.json`.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import extraction
import static_test_check
from static_test_check import check, retry_hints

from test_hook_tool_rules import ACCESSES, _fakes, _ids, _run, _shape

FIXTURES = Path(__file__).parent.parent / "fixtures"

_UNCAUGHT = "so the installed ops raises `ops.ModelError` there"
_NOTHING_CATCHES = "and nothing around it catches that"
_DENIED = "echo 'ERROR permission denied' >&2; exit 1"
_ON = static_test_check._hook_tool_failure_raises_model_error()
_IS_LEADER_RUNS = static_test_check._security_event_runs_is_leader()

pytestmark = [
    pytest.mark.skipif(not _ON, reason="the installed ops does not raise ModelError for every failed hook tool"),
    pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up"),
]


def _model_error():
    return static_test_check._ops().ModelError


def _uncaught(source: str) -> list[str]:
    return [r for r in check(source).reasons if _UNCAUGHT in r]


# -- the calibration corpus ---------------------------------------------------

# (run, repair) -> phrases the one §18 reason contains, or [] when the file
# passes the check.
EXPECTED = {
    (0, "§17 reasons"): [_NOTHING_CATCHES, "`try: result = dict(rel.data[rel.app])`"],
    (1, "§17 reasons"): [
        "not `RelationNotFoundError`, and the `pytest.raises(RelationNotFoundError)` around it",
        "read that through `ops.Model`",
        "`pytest.raises` is only right when the issue says an exception should be raised",
    ],
    # Two `assert`s come before the read: if one failed on the buggy version
    # the test would be valid, so the rule lets it through (it dies on
    # `is-leader`, or with the hint on the uncaught `ModelError`).
    (2, "§17 reasons"): [],
    (2, "§17 reasons and the is-leader hint"): [],
    (3, "§17 reasons"): [_NOTHING_CATCHES, "`try: result = dict(relation.data[relation.app])`"],
    # Run 4 catches `Exception`, run 5 never reads a databag, run 6 catches
    # `ModelError`, and run 7 fakes `relation-get` twice.
    (4, "§17 reasons"): [],
    (5, "§17 reasons"): [],
    (6, "§17 reasons"): [],
    (6, "§17 reasons and the is-leader hint"): [],
    (7, "§17 reasons"): [],
    # Each repaired again from the §18 reason: all three are valid.
    (0, "§17 reasons, then the §18 reason"): [],
    (1, "§17 reasons, then the §18 reason (read through ops.Model) and the check's reasons on that"): [],
    (3, "§17 reasons, then the §18 reason"): [],
}


def _corpus():
    for record in json.loads((FIXTURES / "static_test_check" / "2709-hooktool-repaired.json").read_text()):
        yield pytest.param(record, id=f"run-{record['run']}-{record['repair']}")


def test_the_corpus_is_the_13_repairs():
    records = [p.values[0] for p in _corpus()]
    assert len(records) == 13 == len(EXPECTED)
    assert {(r["run"], r["repair"]) for r in records} == set(EXPECTED)


@pytest.mark.parametrize("record", list(_corpus()))
def test_calibration_verdict(record):
    phrases = EXPECTED[(record["run"], record["repair"])]
    result = check(record["test_file"]["body"])
    if not phrases:
        assert result.passed, result.reasons
        return
    (reason,) = result.reasons
    assert _UNCAUGHT in reason
    for phrase in phrases:
        assert phrase in reason, (phrase, reason)


@pytest.mark.parametrize("record", list(_corpus()))
def test_the_repaired_tests_ran_as_recorded(record, tmp_path, monkeypatch):
    """Runs 0, 1 and 3 die on the uncaught `ModelError` on the installed ops
    (3.8.3), and once repaired again from the §18 reason fail on their
    assertion instead (`assert ModelError('ERROR permission denied\\n') == {}`
    under pytest)."""
    if not _IS_LEADER_RUNS:
        pytest.skip("the installed ops has the #2709 fix")
    if record["run"] not in (0, 1, 3):
        return
    namespace: dict = {}
    exec(compile(record["test_file"]["body"], record["test_file"]["path"], "exec"), namespace)
    (test,) = [v for k, v in namespace.items() if k.startswith("test")]
    expected = AssertionError if "§18" in record["repair"] else _model_error()
    with pytest.raises(expected) as raised:
        test(tmp_path, monkeypatch)
    assert type(raised.value) is expected


# -- what ops raises, by running it ------------------------------------------

# Scripts certain to fail, and what the rule reads from them.
_FAILING = {
    "denied": _DENIED,
    "boom": "echo 'ERROR boom' >&2; exit 1",
    "exit": "exit 3",
    "stdout": "echo 'ERROR boom'; exit 2",
    "two lines": "echo ERROR >&2\necho boom >&2\nexit 255",
}


def _failing_first(access: str, tools: list[str], first: str, script: str) -> dict[str, str]:
    fakes = _fakes(tools)
    fakes[first] = script
    if first != "is-leader":
        fakes["is-leader"] = "echo false"
    return fakes


_CASES = [
    (access, tools, first, name)
    for access, tools, first in ACCESSES
    for name in _FAILING
    if not (first == "is-leader" and name == "denied")
]


@pytest.mark.parametrize(
    ("access", "tools", "first", "failing"), _CASES, ids=[f"{_ids([c])[0]}-{c[3]}" for c in _CASES]
)
def test_an_access_whose_first_tool_fails_raises_model_error_and_is_rejected(
    access, tools, first, failing, tmp_path, monkeypatch
):
    source = _shape(access, _failing_first(access, tools, first, _FAILING[failing]))
    (reason,) = check(source).reasons
    assert f"runs `{first}` first, and the test's `{first}` fake exits" in reason
    assert _UNCAUGHT in reason and _NOTHING_CATCHES in reason
    assert "except ops.ModelError as e: result = e" in reason
    with pytest.raises(_model_error()) as raised:
        _run(source, tmp_path, monkeypatch)
    assert type(raised.value) is _model_error()


def _caught(access: str) -> str:
    """`access` with its last line wrapped the way the reason says."""
    *setup, last = textwrap.dedent(access).strip().splitlines()
    return "\n".join([*setup, "try:", f"    result = {last}", "except ops.ModelError as e:", "    result = e"])


@pytest.mark.parametrize(
    ("access", "tools", "first", "failing"), _CASES, ids=[f"{_ids([c])[0]}-{c[3]}" for c in _CASES]
)
def test_the_same_access_caught_as_the_reason_says_passes_and_runs(
    access, tools, first, failing, tmp_path, monkeypatch
):
    source = _shape(_caught(access), _failing_first(access, tools, first, _FAILING[failing]))
    assert check(source).passed, check(source).reasons
    _run(source, tmp_path, monkeypatch)


@pytest.mark.skipif(not _IS_LEADER_RUNS, reason="the installed ops does not check leadership there")
def test_a_denied_is_leader_recurses_and_is_left_alone(tmp_path, monkeypatch):
    """On 3.8.3 a denied `is-leader` logs a security event, which checks
    leadership with `is-leader` again: `RecursionError`, not `ModelError`."""
    source = _shape("backend.is_leader()", {"is-leader": _DENIED})
    assert not _uncaught(source)
    with pytest.raises(RecursionError):
        _run(source, tmp_path, monkeypatch)


_DATABAG = "relation = model.get_relation('db', 2)\ndict(relation.data[relation.app])"
_TOOLS = ["relation-ids", "relation-list", "relation-get"]


def test_printing_error_and_exiting_0_is_not_a_model_error(tmp_path, monkeypatch):
    """ops only looks at the exit status; the output goes to `json.loads()`."""
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", "echo 'ERROR boom' >&2"))
    assert not _uncaught(source)
    with pytest.raises(json.JSONDecodeError):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("access", "tools", "first"),
    [a for a in ACCESSES if a[2].startswith("relation-")],
    ids=_ids([a for a in ACCESSES if a[2].startswith("relation-")]),
)
def test_relation_not_found_is_left_alone(access, tools, first, tmp_path, monkeypatch):
    """ops raises `RelationNotFoundError` for it, which a databag read turns
    into `{}` (and then `['k']` is a `KeyError`)."""
    source = _shape(access, _failing_first(access, tools, first, "echo 'ERROR relation not found' >&2; exit 1"))
    assert not _uncaught(source)
    try:
        _run(source, tmp_path, monkeypatch)
    except (static_test_check._ops().RelationNotFoundError, KeyError):
        pass


# -- what catches it ----------------------------------------------------------


def _databag_test(access: str, **extra: str) -> str:
    return "import pytest\n" + _shape(access, dict(_failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED), **extra))


@pytest.mark.parametrize(
    "wrap",
    [
        "try:\n    {read}\nexcept ops.ModelError:\n    pass",
        "try:\n    {read}\nexcept ops.model.ModelError as e:\n    x = e",
        "try:\n    {read}\nexcept (ops.RelationNotFoundError, ops.ModelError):\n    pass",
        "try:\n    {read}\nexcept Exception:\n    pass",
        "try:\n    {read}\nexcept:\n    pass",
        "with pytest.raises(ops.ModelError):\n    {read}",
        "with pytest.raises(Exception):\n    {read}",
    ],
)
def test_an_access_inside_something_that_catches_model_error_passes(wrap):
    access = "relation = model.get_relation('db', 2)\n" + wrap.format(read="dict(relation.data[relation.app])")
    assert not _uncaught(_databag_test(access)), check(_databag_test(access)).reasons


def test_pytest_raises_with_a_subclass_is_rejected_and_lets_model_error_out(tmp_path, monkeypatch):
    access = (
        "relation = model.get_relation('db', 2)\n"
        "with pytest.raises(ops.RelationNotFoundError):\n"
        "    dict(relation.data[relation.app])"
    )
    source = _databag_test(access)
    (reason,) = _uncaught(source)
    assert "not `ops.RelationNotFoundError`, and the `pytest.raises(ops.RelationNotFoundError)` around it" in reason
    assert "`pytest.raises` is only right when the issue says an exception should be raised" in reason
    with pytest.raises(_model_error()) as raised:
        _run(source, tmp_path, monkeypatch)
    assert type(raised.value) is _model_error()


def test_except_with_a_subclass_is_rejected():
    access = (
        "relation = model.get_relation('db', 2)\n"
        "try:\n"
        "    dict(relation.data[relation.app])\n"
        "except ops.RelationDataError:\n"
        "    pass"
    )
    (reason,) = _uncaught(_databag_test(access))
    assert "the `except` for `ops.RelationDataError` around it does not catch a plain `ModelError`" in reason
    assert "`pytest.raises` is only right" not in reason


# -- what it cannot be sure of ------------------------------------------------


@pytest.mark.parametrize(
    "access",
    [
        # something before it could fail on an assertion
        "relation = model.get_relation('db', 2)\nassert relation.app.name == 'provider'\ndict(relation.data[relation.app])",
        "relation = model.get_relation('db', 2)\ncheck_it(relation)\ndict(relation.data[relation.app])",
        "relation = model.get_relation('db', 2)\ncheck_deeper(relation)\ndict(relation.data[relation.app])",
        # another call in the statement, which could run first
        "relation = model.get_relation('db', 2)\nprint(dict(relation.data[relation.app]))",
        "relation = model.get_relation('db', 2)\nassert other() == dict(relation.data[relation.app])",
        # conditional, or after the test stops
        "relation = model.get_relation('db', 2)\nif os.environ.get('X'):\n    dict(relation.data[relation.app])",
        "relation = model.get_relation('db', 2)\npytest.skip('no')\ndict(relation.data[relation.app])",
        # a databag the rule does not follow
        "relation = model.get_relation('db', 2)\nbag = get_bag(relation)\ndict(bag)",
        "relation = model.get_relation('db', 2)\nrelation.data[relation.app].keys()",
    ],
)
def test_an_access_it_cannot_be_sure_of_is_let_through(access):
    source = _databag_test(access) + "\n\ndef check_it(r):\n    assert r\n\n\ndef check_deeper(r):\n    check_it(r)\n\n\ndef other():\n    return {}\n\n\ndef get_bag(r):\n    return r.data[r.app]\n"
    assert not _uncaught(source), check(source).reasons


@pytest.mark.parametrize(
    "script",
    [
        "echo 'ERROR permission denied' >&2",  # exits 0
        "exit 0",
        "exit 256",  # wraps to 0
        "echo \"ERROR $X\" >&2; exit 1",  # not a literal message
        "echo ERROR >&2 | cat; exit 1",
        "false; exit 1",  # not only `echo`s before the exit
        "if true; then exit 1; fi",
        "echo -n ERROR >&2; exit 1",
        "echo ERROR 2>&1; exit 1",
        "echo 'ERROR relation not found' >&2; exit 1",  # ops raises RelationNotFoundError
        "echo 'ERROR secret not found' >&2; exit 1",
    ],
)
def test_a_fake_that_is_not_certain_to_fail_with_model_error_is_let_through(script):
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", script))
    assert not _uncaught(source), check(source).reasons


def test_a_fake_that_is_not_a_literal_is_let_through():
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED))
    source = source.replace(repr(_DENIED), "DENIED")
    source = f"DENIED = {_DENIED!r}\n" + source
    assert not _uncaught(source)


def test_a_tool_faked_twice_is_let_through():
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED))
    source += "\n\ndef helper(bin_dir):\n    fake_hook_tool(bin_dir, 'relation-get', 'exit 0')\n"
    assert not _uncaught(source)


def test_a_fake_written_after_the_access_is_let_through():
    fakes = _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED)
    del fakes["relation-get"]
    source = _shape(_DATABAG + f"\nfake_hook_tool(bin_dir, 'relation-get', {_DENIED!r})", fakes)
    assert not _uncaught(source)


def test_a_fake_written_by_a_fixture_or_helper_is_let_through():
    fakes = _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED)
    del fakes["relation-get"]
    source = _shape("deny(bin_dir)\n" + _DATABAG, fakes)
    source += f"\n\ndef deny(bin_dir):\n    fake_hook_tool(bin_dir, 'relation-get', {_DENIED!r})\n"
    assert not _uncaught(source)


@pytest.mark.skipif(not _IS_LEADER_RUNS, reason="the installed ops does not check leadership there")
@pytest.mark.parametrize("is_leader", [None, "echo yes", "echo true; exit 1", "echo $LEADER"])
def test_an_authorisation_error_needs_a_certain_is_leader_fake(is_leader):
    """Without one, ops dies on `is-leader` first (`FileNotFoundError`,
    `JSONDecodeError` or another `ModelError`), which other rules or the hint
    cover."""
    fakes = _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED)
    if is_leader is None:
        del fakes["is-leader"]
    else:
        fakes["is-leader"] = is_leader
    assert not _uncaught(_shape(_DATABAG, fakes))


def test_a_failure_that_is_not_an_authorisation_error_needs_no_is_leader():
    fakes = _failing_first(_DATABAG, _TOOLS, "relation-get", "echo 'ERROR boom' >&2; exit 1")
    del fakes["is-leader"]
    assert _uncaught(_shape(_DATABAG, fakes))


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace('    monkeypatch.setenv("JUJU_VERSION", "3.6.0")\n', ""),
        lambda s: s.replace('"JUJU_VERSION", "3.6.0"', '"JUJU_VERSION", "2.6.0"'),
        lambda s: s.replace('"JUJU_VERSION", "3.6.0"', '"JUJU_VERSION", VERSION').replace("import os\n", "import os\nVERSION = '3.6.0'\n", 1),
        lambda s: s.replace('"JUJU_VERSION", "3.6.0"', '"JUJU_VERSION", "not a version"'),
        lambda s: s + '\n\ndef test_other(monkeypatch):\n    monkeypatch.delenv("JUJU_VERSION")\n',
        # set after the backend is built, which is when ops reads it
        lambda s: s.replace('    monkeypatch.setenv("JUJU_VERSION", "3.6.0")\n', "").replace(
            "    model = ops.Model(meta, backend)\n",
            '    model = ops.Model(meta, backend)\n    monkeypatch.setenv("JUJU_VERSION", "3.6.0")\n',
        ),
        # a backend built somewhere else in the file
        lambda s: s + '\n\nOTHER = _ModelBackend("other/0")\n',
    ],
)
def test_an_app_databag_read_needs_juju_version_set_first(change):
    """Without application data in `JUJU_VERSION` when the backend is built,
    ops raises `RuntimeError` before it runs `relation-get`."""
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED))
    assert _uncaught(source)
    assert not _uncaught(change(source)), check(change(source)).reasons


def test_juju_version_set_through_os_environ_counts():
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED))
    source = source.replace('    monkeypatch.setenv("JUJU_VERSION", "3.6.0")\n', '    os.environ["JUJU_VERSION"] = "3.6.0"\n')
    assert _uncaught(source)


@pytest.mark.parametrize(
    ("call", "rejected"),
    [
        ("backend.relation_get(2, 'provider/0', False)", True),
        ("backend.relation_get(relation_id=2, member_name='provider/0', is_app=False)", True),
        ("backend.relation_get(2, 'provider', True)", True),  # JUJU_VERSION is set first
        ("backend.relation_get(2, 'provider/0', IS_APP)", False),
        ("backend.relation_get(2, 'provider/0', 'no')", False),  # TypeError
        ("backend.relation_get(2, 'provider/0')", False),  # TypeError
        ("backend.relation_get(2, 'provider/0', False, unit='x')", False),  # TypeError
    ],
)
def test_a_direct_relation_get_must_bind(call, rejected):
    source = "IS_APP = False\n" + _shape(call, {"relation-get": _DENIED, "is-leader": "echo false"})
    assert bool(_uncaught(source)) is rejected, check(source).reasons


def test_a_direct_backend_call_is_told_to_read_through_the_model():
    (reason,) = _uncaught(_shape("backend.relation_ids('db')", {"relation-ids": "exit 1"}))
    assert "`backend.relation_ids()` is `_ModelBackend`, a layer below what a charm sees" in reason
    assert "read that through `ops.Model`" in reason


def test_a_decorated_test_and_pytestmark_are_let_through():
    source = _databag_test(_DATABAG)
    assert _uncaught(source)
    assert not _uncaught(source.replace("def test_access(", "@pytest.mark.xfail\ndef test_access("))
    assert not _uncaught(source + "\npytestmark = []\n")


def test_the_rule_is_off_when_ops_raises_something_else(monkeypatch):
    """With the `#2709` fix, ops raises `RelationNotFoundError` for some
    denied relation tools, so the rule turns itself off rather than say
    `ModelError`."""
    source = _databag_test(_DATABAG)
    assert _uncaught(source)
    monkeypatch.setattr(static_test_check, "_hook_tool_failure_raises_model_error", lambda: False)
    assert not _uncaught(source)


# -- the reason, the hints and the prompt -------------------------------------


def test_the_reason_reads_whole():
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED))
    assert check(source).reasons == [
        "line 27: `dict(relation.data[relation.app])` runs `relation-get` first, and the test's "
        '`relation-get` fake exits 1, printing "ERROR permission denied" to stderr, so the installed '
        "ops raises `ops.ModelError` there, and nothing around it catches that: the test dies on that "
        "exception instead of failing an assertion, so its run cannot tell the bug from a broken test; "
        "catch `ops.ModelError` around the access and assert on what the issue says should happen "
        "instead: `try: result = dict(relation.data[relation.app])` / `except ops.ModelError as e: "
        "result = e` / `assert result == <what the issue says it should be>`, so the buggy version "
        "fails that assertion and the fixed version passes"
    ]


@pytest.mark.parametrize(
    ("read", "suggested"),
    [
        ("relation.data[relation.app] == {}", "dict(relation.data[relation.app])"),
        ("{} != relation.data[relation.app]", "dict(relation.data[relation.app])"),
        ("'k' in relation.data[relation.app]", "dict(relation.data[relation.app])"),
        ("relation.data[relation.app]['k']", "relation.data[relation.app]['k']"),
        ("len(relation.data[relation.app])", "len(relation.data[relation.app])"),
    ],
)
def test_the_reason_suggests_reading_the_databag(read, suggested):
    access = f"relation = model.get_relation('db', 2)\n{read}"
    (reason,) = _uncaught(_shape(access, _failing_first(access, _TOOLS, "relation-get", _DENIED)))
    assert f"`try: result = {suggested}`" in reason


def test_a_stdout_only_failure_prints_nothing_to_stderr():
    source = _shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", "echo 'ERROR boom'; exit 2"))
    (reason,) = _uncaught(source)
    assert "fake exits 2, so the installed ops raises" in reason


def test_the_reason_alone_brings_no_hint():
    """It says what to do itself, and `is-leader` is already faked or not
    needed, so neither existing hint applies."""
    result = check(_shape(_DATABAG, _failing_first(_DATABAG, _TOOLS, "relation-get", _DENIED)))
    assert retry_hints(result) == []


def test_the_prompt_says_to_catch_the_error_and_assert():
    text = " ".join(extraction._SCHEMA_INSTRUCTIONS.split())
    assert "wrap the access that raises in `try:` / `except ops.ModelError as e: result = e`" in text
    assert "`assert result == <what the issue says it should be>`" in text
    assert "on the buggy version the test fails with an `AssertionError`, and on the fixed version it passes" in text
    assert (
        "Use `pytest.raises` only when the issue says an exception should be raised, and then only with "
        "the exact class ops raises" in text
    )


def test_the_prompt_example_still_catches_and_asserts():
    text = extraction._SCHEMA_INSTRUCTIONS
    assert "      # Catch the error the bug raises, so the bug fails the assert below.\n      try:\n" in text
    # Not `#2709`'s relation-data shape: that is the issue it is measured on.
    example = text[text.index("  import os\n") : text.index("      assert config == {}\n")]
    assert "relation" not in example
