"""Names scoped to each test function, for the relation rules, and the §23
corpus (`spike-step-5/static-retry/RESULT.md` §24).

The check counts bindings across the whole file and treats a name bound twice
as unknown (§2). Live on `#2709` (§23), run 4-4 had two similar tests that each
assigned `model`, so neither `model` was tracked, and both tests passed though
each alone is rejected. The relation rules (`_scan_tests()` and the
undeclared-endpoint rule) now see a name local to a test function as that
function sees it. Everything else keeps file-wide counting.

The calibration corpus is §23's 6 tests, under
`fixtures/static_test_check/2709-hooktool4.json`, and the hand repairs of the
three whose verdict changed, under `2709-hooktool4-repaired.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import static_test_check
from static_test_check import check, retry_hints

from test_hook_tool_rules import _OUTPUT, _shape

FIXTURES = Path(__file__).parent.parent / "fixtures"

_UNDECLARED = "which the `CharmMeta` on line"
_NOT_FAKED = "and the test does not fake"
_NOT_JSON = "which is not JSON"
_UNCAUGHT = "so the installed ops raises `ops.ModelError` there"
_IS_LEADER_RUNS = static_test_check._security_event_runs_is_leader()

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")


def _two_tests(first: str, second: str) -> str:
    """`first`, then the test function of `second` (a `_shape()` file) as
    `test_access_two`."""
    return first + "\n\ndef test_access_two(" + second.split("\n\n\ndef test_access(", 1)[1]


_UNDECLARED_ACCESS = _shape("model.get_relation('other', 2)", {"relation-get": _OUTPUT["relation-get"]})


def test_the_helper_gives_two_tests():
    source = _two_tests(_UNDECLARED_ACCESS, _UNDECLARED_ACCESS)
    assert "def test_access(" in source and "def test_access_two(" in source
    assert source.count("model = ops.Model(meta, backend)") == 2


# -- 4-4's shape -----------------------------------------------------------------


def test_one_test_alone_is_rejected():
    reasons = check(_UNDECLARED_ACCESS).reasons
    assert any(_UNDECLARED in r for r in reasons)
    assert any(_NOT_FAKED in r for r in reasons)


def test_two_similar_tests_are_each_rejected():
    reasons = check(_two_tests(_UNDECLARED_ACCESS, _UNDECLARED_ACCESS)).reasons
    assert len([r for r in reasons if _UNDECLARED in r]) == 2, reasons
    assert len([r for r in reasons if _NOT_FAKED in r]) == 2, reasons


def test_a_second_test_with_a_valid_access_does_not_hide_the_first():
    valid = _shape("model.get_relation('db', 2)", dict(_OUTPUT))
    reasons = check(_two_tests(_UNDECLARED_ACCESS, valid)).reasons
    assert any(_UNDECLARED in r for r in reasons)


@pytest.mark.parametrize(
    "access",
    [
        "relation = model.get_relation('db', 2)\nrelation.data['provider']",  # §21
        "model.get_relation('db', 2)",  # §17, non-JSON `relation-list`
    ],
)
def test_every_relation_rule_sees_each_tests_names(access):
    fakes = dict(_OUTPUT, **{"relation-get": "echo 'ERROR permission denied' >&2; exit 1", "relation-list": "echo provider"})
    one = check(_shape(access, fakes)).reasons
    assert one
    # The fakes are file-wide (two of each are not certain), so the second
    # test fakes nothing; its names are what would otherwise hide the first.
    second = _shape("relation = model.get_relation('db', 2)\nassert relation", {"status-get": "exit 1"})
    second = second.replace('    fake_hook_tool(bin_dir, "status-get", "exit 1")\n', "", 1)
    two = check(_two_tests(_shape(access, fakes), second)).reasons
    assert set(one) <= set(two), (one, two)


# -- what stays file-wide ---------------------------------------------------------


def test_the_uncaught_error_rule_still_needs_juju_version_set_before_every_backend():
    """§18's rule needs `JUJU_VERSION` set once in the file, before every
    `_ModelBackend(...)` in it. That is not a name, so it stays file-wide, and
    a second test that builds a backend still turns the rule off (§24)."""
    access = "relation = model.get_relation('db', 2)\nassert dict(relation.data[relation.app]) == {}"
    fakes = dict(_OUTPUT, **{"relation-get": "echo 'ERROR permission denied' >&2; exit 1"})
    assert any(_UNCAUGHT in r for r in check(_shape(access, fakes)).reasons)
    second = _shape("assert model", {"status-get": "exit 1"})
    assert not any(_UNCAUGHT in r for r in check(_two_tests(_shape(access, fakes), second)).reasons)



def test_a_name_bound_twice_in_one_test_is_still_unknown():
    source = _shape("model = ops.Model(meta, backend)\nmodel.get_relation('other', 2)", {"relation-get": _OUTPUT["relation-get"]})
    assert not any(_UNDECLARED in r or _NOT_FAKED in r for r in check(source).reasons)


def test_a_global_name_stays_file_wide():
    source = _UNDECLARED_ACCESS.replace("    bin_dir = tmp_path", "    global model\n    bin_dir = tmp_path", 1)
    reasons = check(_two_tests(source, _UNDECLARED_ACCESS)).reasons
    # Only the second test, whose `model` is its own.
    assert len([r for r in reasons if _NOT_FAKED in r]) == 1, reasons


def test_a_parameter_counts_as_a_binding():
    source = _UNDECLARED_ACCESS.replace("(tmp_path, monkeypatch)", "(tmp_path, monkeypatch, model=None)", 1)
    assert not any(_NOT_FAKED in r for r in check(source).reasons)


def test_the_undefined_name_rule_still_ignores_scope():
    """A name read in one test and bound only in another is not reported:
    that rule only fires when a name cannot resolve at all (§2)."""
    source = "def test_a():\n    x = 1\n    assert x\n\n\ndef test_b():\n    assert x\n"
    assert check(source).passed


# -- the §23 corpus ---------------------------------------------------------------

# run -> phrases each expected reason contains, one list per reason.
EXPECTED = {
    0: [["`ops.Relation` is not built by hand", "(it is missing `our_unit`, `backend`, and `cache`)"]],
    1: [["needs `relation-ids`", _NOT_JSON], ["needs `relation-list`", _NOT_JSON], [_UNCAUGHT]],
    2: [[_UNDECLARED], ["needs `relation-list`", _NOT_JSON], [_NOT_FAKED + " `relation-ids`"]],
    3: [[_UNCAUGHT]],
    4: [[_UNDECLARED, "line 21"], [_NOT_FAKED, "line 21"], [_UNDECLARED, "line 35"], [_NOT_FAKED, "line 35"]],
    6: [["its fake prints `[]` whatever its arguments are", "`TypeError: unhashable type: 'list'`"]],
}


def _corpus(name: str):
    for record in json.loads((FIXTURES / "static_test_check" / name).read_text()):
        yield pytest.param(record, id=f"{record['source']}-{record['run']}")


def test_the_corpus_is_the_6_tests():
    records = [p.values[0] for p in _corpus("2709-hooktool4.json")]
    assert {r["run"] for r in records} == set(EXPECTED)


@pytest.mark.parametrize("record", list(_corpus("2709-hooktool4.json")))
def test_calibration_verdict(record):
    expected = EXPECTED[record["run"]]
    result = check(record["test_file"]["body"])
    assert not result.passed
    assert len(result.reasons) == len(expected), result.reasons
    for phrases in expected:
        assert any(all(p in reason for p in phrases) for reason in result.reasons), (phrases, result.reasons)


@pytest.mark.parametrize("record", list(_corpus("2709-hooktool4-repaired.json")))
def test_the_hand_repairs_pass_the_check(record):
    result = check(record["test_file"]["body"])
    assert result.passed, result.reasons
    assert retry_hints(result) == []


# What each repair raises on ops 3.8.3: 4-0 fails on `assert ModelError(...)
# == {}`, the bug; 4-4's `result` is never bound, on any version; 4-6's read
# lets the `ModelError` out, which the check misses behind the `assert` before
# it (§24 leaves that as it is).
REPAIRED = {0: AssertionError, 4: UnboundLocalError, 6: "ModelError"}


@pytest.mark.skipif(not _IS_LEADER_RUNS, reason="the installed ops has the #2709 fix")
@pytest.mark.parametrize("record", list(_corpus("2709-hooktool4-repaired.json")))
def test_the_hand_repairs_ran_as_recorded(record, tmp_path, monkeypatch):
    namespace: dict = {}
    exec(compile(record["test_file"]["body"], record["test_file"]["path"], "exec"), namespace)
    tests = [v for k, v in namespace.items() if k.startswith("test")]
    expected = REPAIRED[record["run"]]
    if expected == "ModelError":
        expected = static_test_check._ops().ModelError
    for i, test in enumerate(tests):
        work = tmp_path / str(i)
        work.mkdir()
        with monkeypatch.context() as patch, pytest.raises(expected) as raised:
            test(work, patch)
        assert type(raised.value) is expected
        if expected is AssertionError:
            (data,) = [t.locals["data"] for t in raised.traceback if "data" in t.locals]
            assert type(data) is static_test_check._ops().ModelError
            assert str(data) == "ERROR permission denied\n"
