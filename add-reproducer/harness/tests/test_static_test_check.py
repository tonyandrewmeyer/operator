"""The static check on an extraction's embedded test file, and the one re-ask
it triggers (`spike-step-5/static-retry/RESULT.md`).

The calibration corpus is the 33 saved `#2045` extractions under
`fixtures/static_test_check/`: 32 from `spike-step-5/assert-choice/` (8 per
prompt version) and the one live dispatch at `7888bb13`. The verdicts below
are that RESULT's table. `assert-choice/RESULT.md` read five of the 33 tests
as valid (after2 runs 1 and 3, after3 runs 1, 2 and 7); every one of them
must pass, because a false rejection costs a reproduction.

No network, no key: the extraction tests drive `Extractor` through a
scripted LLM seam.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import static_test_check
from extraction import Extractor
from extraction_record import build as build_record
from extraction_record import render as render_record
from inscope_second_pass import TwoPassExtractor
from models import Hypothesis, Issue, StaticCheckResult
from static_test_check import check, check_commands

FIXTURES = Path(__file__).parent.parent / "fixtures"
CORPUS = FIXTURES / "static_test_check"

# (file, run) -> (valid per assert-choice/RESULT.md, check passes, a phrase
# each expected reason contains). `None` for "passes" means there is no
# embedded test file to check.
EXPECTED = {
    ("before", 0): (False, False, ["`str` and a `pathlib.Path`"]),
    ("before", 1): (
        False,
        False,
        ["no module-level `def test_", "inside the charm", "`str` and a `pathlib.Path`", "at module level"],
    ),
    ("before", 2): (False, False, ["inside the charm", "`str` and a `pathlib.Path`"]),
    ("before", 3): (False, False, ["no module-level `def test_", "inside the charm", "`str` and a `pathlib.Path`"]),
    ("before", 4): (False, True, []),
    ("before", 5): (False, False, ["no module-level `def test_", "read-only `CharmBase` property", "at module level"]),
    ("before", 6): (False, False, ["read-only `CharmBase` property"]),
    ("before", 7): (
        False,
        False,
        ["no module-level `def test_", "inside the charm", "`str` and a `pathlib.Path`", "at module level"],
    ),
    ("after", 0): (False, False, ["`ctx.charm`", "`ctx.fs`"]),
    ("after", 1): (False, False, ["`ctx.charm`"]),
    ("after", 2): (False, False, ["`ctx.charm`", "`ctx.framework`", "`str` and a `pathlib.Path`"]),
    ("after", 3): (False, False, ["`ctx.charm`", "`str` and a `pathlib.Path`"]),
    ("after", 4): (False, False, ["`ctx.charm_dir`", "`str` and a `pathlib.Path`", "before the handlers run"]),
    ("after", 5): (False, False, ["`ctx.charm_dir`", "`str` and a `pathlib.Path`"]),
    ("after", 6): (False, False, ["`ctx.charm_dir`", "`str` and a `pathlib.Path`", "before the handlers run"]),
    ("after", 7): (
        False,
        False,
        ["`ops` is used but never", "`testing` is used but never", "`str` and a `pathlib.Path`"],
    ),
    ("after2", 0): (False, None, []),
    ("after2", 1): (True, True, []),
    ("after2", 2): (False, False, ["`ctx.charm_dir`", "before the handlers run"]),
    ("after2", 3): (True, True, []),
    ("after2", 4): (False, False, ["before the handlers run"]),
    ("after2", 5): (False, False, ["`ctx.charm_dir`", "before the handlers run"]),
    ("after2", 6): (False, False, ["before the handlers run"]),
    ("after2", 7): (False, False, ["before the handlers run"]),
    ("after3", 0): (False, False, ["`ctx.charm_dir`"]),
    ("after3", 1): (True, True, []),
    ("after3", 2): (True, True, []),
    ("after3", 3): (False, False, ["`ctx.charm`"]),
    ("after3", 4): (False, False, ["`ctx.charm_dir`", "`ctx.charm`"]),
    ("after3", 5): (False, False, ["`ctx.charm_dir`", "`ctx.charm`"]),
    ("after3", 6): (False, False, ["`ctx.charm_dir`"]),
    ("after3", 7): (True, True, []),
    ("dispatch", 0): (False, False, ["`ctx.charm_dir`", "`ctx.charm`"]),
}


def _corpus():
    for path in sorted(CORPUS.glob("2045-*.json")):
        version = path.stem.removeprefix("2045-")
        for record in json.loads(path.read_text()):
            yield pytest.param(version, record, id=f"{version}-{record['run']}")


def test_the_corpus_is_the_33_extractions():
    assert len(list(_corpus())) == 33 == len(EXPECTED)


@pytest.mark.parametrize(("version", "record"), list(_corpus()))
def test_calibration_verdict(version, record):
    valid, passes, phrases = EXPECTED[version, record["run"]]
    hypothesis = Hypothesis.from_dict(2045, record)
    result = check_commands(hypothesis.commands)
    if passes is None:
        assert result is None
        return
    assert result is not None
    assert result.passed is passes, result.reasons
    assert bool(result.reasons) is not passes
    # Each rule that should fire does; no rule fires that the table does not
    # name (a reason that matches none of the phrases is a new, unreviewed
    # rejection).
    for phrase in phrases:
        assert any(phrase in reason for reason in result.reasons), (phrase, result.reasons)
    for reason in result.reasons:
        assert any(phrase in reason for phrase in phrases), reason


@pytest.mark.parametrize(
    ("version", "record"),
    [param for param in _corpus() if EXPECTED[param.values[0], param.values[1]["run"]][0]],
)
def test_no_valid_test_is_rejected(version, record):
    """The non-negotiable half of the calibration, stated on its own."""
    result = check_commands(Hypothesis.from_dict(2045, record).commands)
    assert result is not None and result.passed, result.reasons


def test_there_are_five_valid_tests():
    assert sum(valid for valid, _, _ in EXPECTED.values()) == 5


# -- individual rules, on hand-written files ------------------------------

_HEADER = "import os\nimport ops\nfrom ops import testing\n\n"


def test_syntax_error_fails_with_the_line():
    result = check("def test_x(:\n    pass\n")
    assert not result.passed
    assert "does not parse" in result.reasons[0]
    assert "line 1" in result.reasons[0]


def test_a_test_class_counts_as_a_test_function():
    assert check("class TestX:\n    def test_y(self):\n        assert True\n").passed


def test_context_attributes_come_from_the_installed_ops():
    allowed = static_test_check.context_attributes()
    assert allowed is not None
    # Set in `Context.__init__`, so invisible to `dir()` on the class.
    assert {"on", "emitted_events", "juju_log", "charm_root", "app_name"} <= allowed
    assert {"run", "run_action"} <= allowed
    assert not {"charm", "charm_dir", "framework", "fs"} & allowed


def test_real_context_attributes_pass():
    body = _HEADER + (
        "def test_x():\n"
        "    ctx = testing.Context(ops.CharmBase, meta={'name': 'x'}, app_name='y')\n"
        "    out = ctx.run(ctx.on.start(), testing.State())\n"
        "    assert ctx.emitted_events == [] or ctx.unit_status_history is not None\n"
        "    assert ctx.app_name == 'y' and ctx.charm_root is None\n"
    )
    assert check(body).passed


@pytest.mark.parametrize(
    "construct",
    ["testing.Context(ops.CharmBase)", "ops.testing.Context(ops.CharmBase)", "Context(ops.CharmBase)"],
)
def test_every_spelling_of_context_is_tracked(construct):
    body = _HEADER + f"from ops.testing import Context\n\ndef test_x():\n    c = {construct}\n    c.charm\n"
    result = check(body)
    assert not result.passed
    assert "`c.charm`" in result.reasons[0]


def test_context_bound_by_a_with_statement_is_tracked():
    body = _HEADER + "def test_x():\n    with testing.Context(ops.CharmBase) as c:\n        c.charm_dir\n"
    assert "`c.charm_dir`" in check(body).reasons[0]


def test_attributes_on_anything_else_are_not_checked():
    body = _HEADER + (
        "def test_x():\n"
        "    ctx = testing.Context(ops.CharmBase, meta={'name': 'x'})\n"
        "    with ctx(ctx.on.start(), testing.State()) as mgr:\n"
        "        mgr.charm.charm_dir\n"
        "    state = testing.State()\n"
        "    state.anything\n"
    )
    assert check(body).passed


def test_a_name_rebound_to_something_else_is_not_tracked():
    body = _HEADER + (
        "def test_x():\n"
        "    ctx = testing.Context(ops.CharmBase)\n"
        "    ctx = object()\n"
        "    ctx.charm\n"
    )
    assert check(body).passed


def test_a_parameter_named_ctx_is_not_tracked():
    body = _HEADER + "def test_x(ctx):\n    ctx.charm\n"
    assert check(body).passed


def test_a_module_level_context_is_seen_inside_the_test():
    body = _HEADER + "ctx = testing.Context(ops.CharmBase)\n\ndef test_x():\n    ctx.charm_dir\n"
    assert "`ctx.charm_dir`" in check(body).reasons[0]


def test_str_on_both_sides_passes():
    body = "import pathlib\n" + _HEADER + (
        "captured = {}\n\n"
        "class C(ops.CharmBase):\n"
        "    def _on(self, e):\n"
        "        captured['cwd'] = os.getcwd()\n"
        "        captured['dir'] = self.charm_dir\n\n"
        "def test_x():\n"
        "    assert captured['cwd'] == str(captured['dir'])\n"
        "    assert pathlib.Path(captured['cwd']) == captured['dir']\n"
    )
    assert check(body).passed


def test_a_slot_with_two_kinds_of_value_is_not_typed():
    body = _HEADER + (
        "class C(ops.CharmBase):\n"
        "    def _on(self, e):\n"
        "        self.where = os.getcwd()\n"
        "        self.where = self.charm_dir\n\n"
        "def test_x():\n"
        "    c = None\n"
        "    assert c.where == c.charm_dir\n"
    )
    assert check(body).passed


def test_a_charm_dir_the_file_assigns_itself_is_not_typed_as_a_path():
    body = _HEADER + (
        "class Thing:\n"
        "    def __init__(self):\n"
        "        self.charm_dir = os.getcwd()\n\n"
        "def test_x():\n"
        "    assert Thing().charm_dir == os.getcwd()\n"
    )
    assert check(body).passed


def test_reading_a_capture_after_mgr_run_passes():
    body = _HEADER + (
        "class C(ops.CharmBase):\n"
        "    def __init__(self, framework):\n"
        "        super().__init__(framework)\n"
        "        framework.observe(self.on.start, self._on_start)\n\n"
        "    def _on_start(self, event):\n"
        "        self.cwd = os.getcwd()\n\n"
        "def test_x():\n"
        "    ctx = testing.Context(C, meta={'name': 'x'})\n"
        "    with ctx(ctx.on.start(), testing.State()) as mgr:\n"
        "        mgr.run()\n"
        "        assert mgr.charm.cwd == str(mgr.charm.charm_dir)\n"
    )
    assert check(body).passed


def test_something_set_in_init_can_be_read_inside_the_with_block():
    body = _HEADER + (
        "class C(ops.CharmBase):\n"
        "    def __init__(self, framework):\n"
        "        super().__init__(framework)\n"
        "        self.cwd = os.getcwd()\n\n"
        "def test_x():\n"
        "    ctx = testing.Context(C, meta={'name': 'x'})\n"
        "    with ctx(ctx.on.start(), testing.State()) as mgr:\n"
        "        assert mgr.charm.cwd == str(mgr.charm.charm_dir)\n"
    )
    assert check(body).passed


def test_star_import_switches_the_undefined_name_rule_off():
    assert check("from ops.testing import *\n\ndef test_x():\n    Context\n").passed


def test_the_context_rule_is_off_when_ops_cannot_be_introspected(monkeypatch):
    monkeypatch.setattr(static_test_check, "context_attributes", lambda: None)
    body = _HEADER + "def test_x():\n    ctx = testing.Context(ops.CharmBase)\n    ctx.charm_dir\n"
    assert check(body).passed


def test_no_embedded_test_file_is_not_checked():
    assert check_commands(["uv init --bare .", "uv run pytest test_x.py"]) is None


# -- the re-ask ------------------------------------------------------------

_VALID_BODY = (
    "import os\nimport ops\nfrom ops import testing\n\ncaptured = {}\n\n"
    "class MyCharm(ops.CharmBase):\n"
    "    def __init__(self, framework):\n"
    "        super().__init__(framework)\n"
    "        framework.observe(self.on.start, self._on_start)\n\n"
    "    def _on_start(self, event):\n"
    "        captured['cwd'] = os.getcwd()\n"
    "        captured['charm_dir'] = str(self.framework.charm_dir)\n\n"
    "def test_cwd_is_charm_dir():\n"
    "    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})\n"
    "    ctx.run(ctx.on.start(), testing.State())\n"
    "    assert captured['cwd'] == captured['charm_dir']"
)
_INVALID_BODY = _VALID_BODY.replace(
    "assert captured['cwd'] == captured['charm_dir']", "assert captured['cwd'] == str(ctx.charm_dir)"
)


def _extraction(body: str | None, *, in_scope: bool = True) -> dict:
    commands = ["uv init --bare .", "uv add 'ops[testing]'"]
    if body is not None:
        commands += [f"cat > test_cwd.py << 'PYEOF'\n{body}\nPYEOF", "uv run pytest test_cwd.py -v"]
    return {
        "in_scope": in_scope,
        "moving_parts": {"substrate": "none" if in_scope else None},
        "commands": commands,
        "expected": "cwd is the charm root",
        "observed": "it is not",
        "confidence": "high",
    }


class _ScriptedLLM:
    """Returns each queued response in turn, recording what it was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.contexts: list[dict] = []
        self.calls: list[dict] = []

    def complete_json(self, *, purpose, prompt, context):
        self.prompts.append(prompt)
        self.contexts.append(context)
        self.calls.append({"purpose": purpose, "model": None, "provider": None, "usage": None})
        return self.responses.pop(0)


def _issue() -> Issue:
    return Issue(
        number=2045,
        title="cwd is not the charm root",
        body="b",
        labels=[],
        state="OPEN",
        created_at="",
        author="a",
        repo="canonical/operator",
    )


def test_a_test_that_passes_first_time_is_not_retried():
    llm = _ScriptedLLM([_extraction(_VALID_BODY)])
    extractor = Extractor(llm)
    hypothesis = extractor.extract(_issue())
    assert len(llm.prompts) == 1
    assert hypothesis.in_scope
    assert extractor.last_test_retried is False
    assert [c.passed for c in extractor.last_test_checks] == [True]


def test_a_failed_check_is_retried_once_with_the_reasons():
    llm = _ScriptedLLM([_extraction(_INVALID_BODY), _extraction(_VALID_BODY)])
    extractor = Extractor(llm)
    hypothesis = extractor.extract(_issue())
    assert len(llm.prompts) == 2
    assert llm.prompts[1].startswith(llm.prompts[0])
    assert "`ctx.charm_dir`, which `testing.Context` does not have" in llm.prompts[1]
    assert "`ctx.charm_dir`" not in llm.prompts[0].split("Repo:")[-1]
    # The second answer is the one kept.
    assert "assert captured['cwd'] == captured['charm_dir']" in "\n".join(hypothesis.commands)
    assert extractor.last_test_retried is True
    assert [c.passed for c in extractor.last_test_checks] == [False, True]


def test_a_missing_context_attribute_retry_lists_the_ones_there_are():
    llm = _ScriptedLLM([_extraction(_INVALID_BODY), _extraction(_VALID_BODY)])
    Extractor(llm).extract(_issue())
    hint = "The public attributes of `testing.Context` are, in full:"
    assert hint not in llm.prompts[0]
    retry = llm.prompts[1]
    assert hint in retry
    assert "`charm_root`" in retry and "`run`" in retry
    # The reasons come first, then the hint, then the instruction.
    assert retry.index("does not have") < retry.index(hint) < retry.index("Return the whole")


def test_retry_hints_only_for_a_missing_context_attribute():
    other = StaticCheckResult(
        path="test_x.py", passed=False, reasons=["line 3: the test file has no test function"]
    )
    assert static_test_check.retry_hints(other) == []
    missing = StaticCheckResult(
        path="test_x.py",
        passed=False,
        reasons=["line 9: the test accesses `ctx.mgr`, which `testing.Context` does not have"],
    )
    (hint,) = static_test_check.retry_hints(missing)
    assert "`emitted_events`" in hint
    assert "`_" not in hint
    # Listed bare, `charm_root` invited a test comparing the cwd with `None`.
    assert "`None` when none was passed" in hint


def test_retry_hints_are_off_when_ops_cannot_be_introspected(monkeypatch):
    monkeypatch.setattr(static_test_check, "context_attributes", lambda: None)
    missing = StaticCheckResult(
        path="test_x.py",
        passed=False,
        reasons=["line 9: the test accesses `ctx.mgr`, which `testing.Context` does not have"],
    )
    assert static_test_check.retry_hints(missing) == []


def test_two_failures_keep_the_second_and_stop():
    second = _INVALID_BODY.replace("str(ctx.charm_dir)", "ctx.charm.cwd")
    llm = _ScriptedLLM([_extraction(_INVALID_BODY), _extraction(second)])
    extractor = Extractor(llm)
    hypothesis = extractor.extract(_issue())
    assert len(llm.prompts) == 2
    assert "ctx.charm.cwd" in "\n".join(hypothesis.commands)
    assert extractor.last_test_retried is True
    first, retried = extractor.last_test_checks
    assert not first.passed and not retried.passed
    assert any("`ctx.charm_dir`" in r for r in first.reasons)
    assert any("`ctx.charm`" in r for r in retried.reasons)


def test_an_out_of_scope_answer_is_not_checked():
    llm = _ScriptedLLM([_extraction(_INVALID_BODY, in_scope=False)])
    extractor = Extractor(llm)
    assert not extractor.extract(_issue()).in_scope
    assert len(llm.prompts) == 1
    assert extractor.last_test_checks == []


def test_no_test_file_is_not_retried():
    llm = _ScriptedLLM([_extraction(None)])
    extractor = Extractor(llm)
    extractor.extract(_issue())
    assert len(llm.prompts) == 1
    assert extractor.last_test_checks == []


def test_a_new_issue_does_not_inherit_the_previous_checks():
    llm = _ScriptedLLM([_extraction(_INVALID_BODY), _extraction(_INVALID_BODY), _extraction(None)])
    extractor = Extractor(llm)
    extractor.extract(_issue())
    extractor.extract(_issue())
    assert extractor.last_test_checks == []
    assert extractor.last_test_retried is False


def test_the_record_shows_the_retry_and_both_verdicts():
    llm = _ScriptedLLM([_extraction(_INVALID_BODY), _extraction(_INVALID_BODY)])
    extractor = TwoPassExtractor(llm)
    hypothesis = extractor.extract(_issue())
    record = build_record(2045, extractor=extractor, hypothesis=hypothesis, llm=llm)
    test_check = record["test_check"]
    assert record["schema_version"] == 2
    assert test_check["ran"] is True
    assert test_check["retried"] is True
    assert test_check["both_failed"] is True
    assert [c["passed"] for c in test_check["checks"]] == [False, False]
    assert all(c["path"] == "test_cwd.py" for c in test_check["checks"])
    assert len(record["llm_calls"]) == 2
    lines = render_record(record)
    assert "  test check 1: failed (test_cwd.py)" in lines
    assert "  test check 2: failed (test_cwd.py)" in lines
    assert "  test check: retried=True both_failed=True" in lines
    assert any("`ctx.charm_dir`" in line for line in lines)


def test_the_record_for_a_passing_first_answer():
    llm = _ScriptedLLM([_extraction(_VALID_BODY)])
    extractor = TwoPassExtractor(llm)
    hypothesis = extractor.extract(_issue())
    test_check = build_record(2045, extractor=extractor, hypothesis=hypothesis, llm=llm)["test_check"]
    assert test_check == {
        "ran": True,
        "checks": [{"path": "test_cwd.py", "passed": True, "reasons": []}],
        "retried": False,
        "both_failed": False,
    }


def test_the_record_with_nothing_to_check():
    llm = _ScriptedLLM([_extraction(None)])
    extractor = TwoPassExtractor(llm)
    hypothesis = extractor.extract(_issue())
    record = build_record(2045, extractor=extractor, hypothesis=hypothesis, llm=llm)
    assert record["test_check"] == {"ran": False, "checks": [], "retried": False, "both_failed": False}
    assert "  test check: nothing to check" in render_record(record)
