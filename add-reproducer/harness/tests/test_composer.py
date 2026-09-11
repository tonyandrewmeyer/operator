"""Comment composition tests (PLAN.md Approach §6).

`compose_template()` (the pre-existing deterministic renderer) already had
no test file of its own -- it was only exercised indirectly through
`test_pipeline_e2e.py`'s fixture-mode `Pipeline` runs. These tests cover it
directly, plus the new `Composer` LLM path and its fallback behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from composer import (
    AUTOMATION_PREFIX,
    Composer,
    ComposerInvalid,
    _resolved_test_file,
    compose_template,
)
from models import (
    COMMENT_OUTCOMES,
    CommandResult,
    Hypothesis,
    Issue,
    MovingParts,
    Outcome,
    RunResult,
    TestFile,
)
from seams.llm import FixtureLLM, LLMError


def after_prefix(text: str) -> str:
    """The comment with `AUTOMATION_PREFIX` stripped.

    Every composed comment opens with the prefix (criterion 2's accepted-risk
    mitigation), so assertions about what a *renderer* or the *model* wrote
    start one line further down.
    """
    assert text.startswith(AUTOMATION_PREFIX + "\n\n"), text[:200]
    return text[len(AUTOMATION_PREFIX) + 2 :]


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def _load_hypothesis(number: int) -> Hypothesis:
    raw = json.loads((FIXTURES / "extractions" / f"{number}.json").read_text())
    return Hypothesis.from_dict(number, raw)


def _load_run(number: int) -> RunResult:
    return RunResult.from_dict(json.loads((FIXTURES / "runs" / f"{number}.json").read_text()))


class _StaticLLM:
    """Returns a canned dict (or raises) regardless of prompt content."""

    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls = 0

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._response


# --- compose_template() -------------------------------------------------


def test_template_silent_outcome_returns_none():
    hyp = _load_hypothesis(2341)
    run = _load_run(2341)
    assert compose_template(hyp, _load_issue(2341), run, Outcome.DID_NOT_REPRODUCE, "n/a", run_id="r", timestamp="t") is None


def test_template_partial_wording():
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(hypothesis_number=3, branch="none", commands=[CommandResult(command="uv venv", exit_code=1)])
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.PARTIAL, "diverged early", run_id="r1", timestamp="2026-01-01T00:00:00Z")
    assert after_prefix(body).startswith("**Partial reproduction attempt**")
    assert "<!-- add-reproducer:issue=3:run=r1 -->" in body


def test_template_partial_says_attempted_not_reproduced():
    # A `partial` outcome did not reproduce the bug -- the trailing summary
    # line must not contradict that (spike-step-5/composer-live/RESULT.md).
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none", base="ubuntu@24.04"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(hypothesis_number=3, branch="none", commands=[CommandResult(command="uv venv", exit_code=1)])
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.PARTIAL, "diverged early", run_id="r1", timestamp="2026-01-01T00:00:00Z")
    assert "Attempted on ubuntu@24.04/none at 2026-01-01T00:00:00Z." in body
    assert "Reproduced on" not in body


def test_template_versions_line_includes_observed_juju_version_when_present():
    """`spike-step-5/wallclock-substrate/RESULT.md` §5: the pinned
    `moving_parts.juju_version` (often nothing) and the juju that actually
    produced the verdict can differ -- a reader trusting a "Reproduced"
    comment needs both, not just the pin."""
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s", base="ubuntu@24.04"),
        commands=["true"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=3,
        branch="k8s-scratch",
        commands=[CommandResult(command="true", exit_code=0)],
        observed_juju_version="3.6.27-ubuntu-amd64",
    )
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "Versions: repo=unpinned, juju=unpinned, base=ubuntu@24.04, substrate=k8s, observed_juju=3.6.27-ubuntu-amd64" in body


def test_template_versions_line_omits_observed_juju_version_when_absent():
    """The `none` branch never touches a juju substrate, so
    `observed_juju_version` stays `None` -- the versions line must not claim
    an "observed_juju=" value nobody measured."""
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none", base="ubuntu@24.04"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(hypothesis_number=3, branch="none", commands=[CommandResult(command="uv venv", exit_code=0)])
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "observed_juju=" not in body
    assert "Versions: repo=unpinned, juju=unpinned, base=ubuntu@24.04, substrate=none" in body


def test_template_2639_includes_control_run():
    # spike-step-5/composer-live/RESULT.md Finding 1: the control run is the
    # entire basis for a `reproduced_positive_signal_absent` verdict, and it
    # was silently dropped from both render paths before this fix.
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    assert run.control is not None
    assert run.control.command in body
    assert "Recorded notice 2" in body  # control's stdout
    assert "observed notice: canonical.com/repro/notice-root in workload" in body


def test_template_control_label_explains_what_it_rules_out():
    # spike-step-5/maintainer-review/RESULT.md §7: rated C1, this label was
    # flagged as possibly unclear cold -- it separates
    # reproduced_positive_signal_absent from a broken observer, and the
    # unclarified label didn't say so. One clause, not a new section.
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    assert "rules out a broken check" in body


def test_composer_llm_response_missing_control_falls_back_to_template():
    # A response that never mentions "control" at all is rejected even
    # though it's otherwise well-formed -- this outcome's claim depends on
    # showing that run, not just asserting it happened.
    llm = _StaticLLM(response={"comment_body": "Reproduced cleanly, no mention of the other run."})
    composer = Composer(llm)
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    want = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    assert got == want
    assert "Recorded notice 2" in got


def test_composer_control_gate_not_applied_when_no_control_exists():
    # The control-mention requirement only applies when run_result.control
    # is actually set -- a plain `reproduced` outcome with no control must
    # not be penalised for never mentioning one.
    llm = _StaticLLM(response={"comment_body": "Reproduced cleanly, nothing about any control."})
    composer = Composer(llm)
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest -v", exit_code=1, stderr='"boom"')],
    )
    hyp.observed = 'crashes with "boom"'
    # #2341's real extraction embeds a test file as a heredoc in commands[]
    # (spike-step-5/maintainer-review/FOLLOWUPS.md §1) -- irrelevant to what
    # this test checks, but would otherwise trip the test-file gate too.
    hyp.commands = ["uv run pytest -v"]
    issue = _load_issue(2341)
    body = composer.compose(hyp, issue, run, Outcome.REPRODUCED, "matched observed output", run_id="run-x", timestamp="t")
    assert after_prefix(body).startswith("Reproduced cleanly, nothing about any control.")


# --- Log-record threading (spike-step-5/composer-scale/RESULT.md Finding
# 1) -- same shape as Finding 1's control-run gap, for `reproduced_log_only`
# ------------------------------------------------------------------------


def _log_only_run():
    # spike-step-4/2341's shape: exit 0, no stdout/stderr worth reading, the
    # bug's only evidence is a captured log record.
    return RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[
            CommandResult(
                command="uv run pytest repro_test.py -v --log-cli-level=DEBUG",
                exit_code=0,
                stdout="1 passed in 0.08s",
                log_records=["WARNING consistency_checker: \"remote_unit=0 implicit\" (matches observed)"],
            )
        ],
    )


def test_template_log_only_includes_log_records():
    hyp = _load_hypothesis(2341)
    run = _log_only_run()
    issue = _load_issue(2341)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_LOG_ONLY, "log record matched", run_id="run-x", timestamp="t")
    assert "remote_unit=0 implicit" in body
    # And the clean-looking stdout is still shown too -- this is additive,
    # not a replacement for the (now potentially misleading-on-its-own)
    # stdout/stderr section.
    assert "1 passed in 0.08s" in body


def test_composer_llm_response_missing_log_records_falls_back_to_template():
    # A response that claims "reproduced" but never mentions the log record
    # at all would read as self-contradictory next to a clean "1 passed"
    # stdout block -- rejected even though it's otherwise well-formed.
    llm = _StaticLLM(response={"comment_body": "The bug reproduced. Observed output: 1 passed in 0.08s."})
    composer = Composer(llm)
    hyp = _load_hypothesis(2341)
    run = _log_only_run()
    issue = _load_issue(2341)
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED_LOG_ONLY, "log record matched", run_id="run-x", timestamp="t")
    want = compose_template(hyp, issue, run, Outcome.REPRODUCED_LOG_ONLY, "log record matched", run_id="run-x", timestamp="t")
    assert got == want
    assert "remote_unit=0 implicit" in got


def test_composer_log_gate_not_applied_when_no_log_records_exist():
    # The log-record-mention requirement only applies when a command
    # actually carries log_records -- a plain `reproduced` outcome with no
    # log records must not be penalised for never mentioning any.
    llm = _StaticLLM(response={"comment_body": "Reproduced cleanly, no log records involved."})
    composer = Composer(llm)
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest -v", exit_code=1, stderr='"boom"')],
    )
    hyp.observed = 'crashes with "boom"'
    # Same reason as test_composer_control_gate_not_applied_when_no_control_exists
    # above -- strip #2341's real heredoc-embedded test file, unrelated here.
    hyp.commands = ["uv run pytest -v"]
    issue = _load_issue(2341)
    body = composer.compose(hyp, issue, run, Outcome.REPRODUCED, "matched observed output", run_id="run-x", timestamp="t")
    assert after_prefix(body).startswith("Reproduced cleanly, no log records involved.")


def test_template_2639_contains_marker_and_versions():
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="2026-07-27T00:00:00Z")
    assert "substrate=k8s" in body
    assert "add-reproducer:issue=2639:run=run-x" in body
    for c in run.commands:
        assert c.command in body


# --- Composer: gating ----------------------------------------------------


def test_composer_never_calls_llm_for_silent_outcome():
    llm = _StaticLLM(response={"comment_body": "should never be seen"})
    composer = Composer(llm)
    hyp = _load_hypothesis(2341)
    run = _load_run(2341)
    result = composer.compose(hyp, _load_issue(2341), run, Outcome.DID_NOT_REPRODUCE, "n/a", run_id="r", timestamp="t")
    assert result is None
    assert llm.calls == 0


# --- Composer: LLM path ---------------------------------------------------


def test_composer_uses_llm_body_when_valid():
    llm = _StaticLLM(response={"comment_body": "Reproduced cleanly.\n\nVersions: repo=main\n\nControl run confirmed the observer works."})
    composer = Composer(llm)
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="2026-07-27T00:00:00Z")
    assert after_prefix(body).startswith("Reproduced cleanly.")
    assert llm.calls == 1
    # Marker is always code-appended, never trusted to the model.
    assert body.endswith("<!-- add-reproducer:issue=2639:run=run-x -->")
    assert body.count("add-reproducer:issue=2639:run=run-x") == 1


# --- Composer: fallback on failure ---------------------------------------


def test_composer_falls_back_to_template_on_llm_error():
    llm = _StaticLLM(error=LLMError("boom"))
    composer = Composer(llm)
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="2026-07-27T00:00:00Z")
    want = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="2026-07-27T00:00:00Z")
    assert got == want


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"comment_body": ""},
        {"comment_body": "   "},
        {"comment_body": 12345},
        {"wrong_key": "x"},
        "not even a dict",
        None,
    ],
)
def test_composer_falls_back_to_template_on_malformed_response(response):
    llm = _StaticLLM(response=response)
    composer = Composer(llm)
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="2026-07-27T00:00:00Z")
    want = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="2026-07-27T00:00:00Z")
    assert got == want


def test_composer_fixture_llm_no_composition_recording_falls_back():
    # No fixtures/compositions/2639.json exists -- FixtureLLM raises LLMError,
    # Composer catches it. This is the exact path fixture-mode Pipeline runs
    # take today (see test_pipeline_e2e.py::test_2639_calibration_mode_reproduces).
    llm = FixtureLLM(FIXTURES)
    composer = Composer(llm)
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="test-run", timestamp="2026-07-23T00:00:00Z")
    want = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="test-run", timestamp="2026-07-23T00:00:00Z")
    assert got == want
    assert "add-reproducer:issue=2639:run=test-run" in got


# --- Synthesized test-file threading (spike-step-5/composer-live/RESULT.md
# Finding 4) --------------------------------------------------------------


def _hypothesis_with_test_file(*, path="test_issue_9999_repro.py", body="def test_x():\n    assert False\n"):
    return Hypothesis(
        issue_number=9999,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv init --bare .", "uv add 'ops[testing]'", f"uv run pytest {path} -v"],
        expected="passes",
        observed="fails",
        confidence="medium",
        synthesized_test_file=TestFile(path=path, body=body),
    )


def _issue_9999():
    return Issue(number=9999, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")


def test_template_includes_synthesized_test_file_body():
    hyp = _hypothesis_with_test_file()
    run = RunResult(
        hypothesis_number=9999,
        branch="none",
        commands=[CommandResult(command=c, exit_code=0) for c in hyp.commands],
    )
    body = compose_template(hyp, _issue_9999(), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "test_issue_9999_repro.py" in body
    assert "def test_x():" in body
    # The file body must appear before the commands section, per
    # composer-live/RESULT.md Finding 4 ("the reader needs the file to exist
    # before the commands that reference it make sense").
    assert body.index("def test_x():") < body.index("Commands run:")
    # spike-step-5/maintainer-review/RESULT.md §7's cross-cutting note:
    # inline it collapsed, since these bodies are long.
    assert "<details>" in body
    assert "<summary>Synthesized test file: test_issue_9999_repro.py" in body
    assert body.index("<summary>") < body.index("def test_x():") < body.index("</details>")


# --- Heredoc-embedded test file (spike-step-5/maintainer-review/
# FOLLOWUPS.md §1: three of five passing maintainer ratings were
# conditional on seeing a test file that `synthesized_test_file` alone
# never covered -- #2341's real extraction writes its own test file via a
# `cat > ... << 'EOF'` heredoc in commands[], not LLM synthesis, and the
# executed run's own commands (e.g. a runner that replays only the final
# pytest invocation) can drop that heredoc before it ever reaches the
# composer.) -----------------------------------------------------------


def test_template_includes_heredoc_embedded_test_file():
    hyp = _load_hypothesis(2341)  # real extraction; commands[2] is the heredoc
    # Mirrors composer-scale's 2341-log-only-synthetic RunResult shape: the
    # executed run replays only the final pytest invocation, not the setup
    # or the heredoc that wrote the file.
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed in 0.08s")],
    )
    body = compose_template(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    # "Commands run" only ever renders run_result.commands (the executed
    # subset), never hypothesis.commands -- the heredoc line that actually
    # creates the file is not among them.
    commands_section = body[body.index("Commands run:") :]
    assert "cat >" not in commands_section
    assert "class DummyCharm(ops.CharmBase):" in body  # the heredoc's actual body, shown elsewhere
    assert "<details>" in body
    assert "<summary>Test file: repro_test.py" in body
    assert body.index("<summary>") < body.index("class DummyCharm") < body.index("</details>") < body.index("Commands run:")


def test_composer_llm_response_missing_heredoc_test_file_falls_back_to_template():
    llm = _StaticLLM(response={"comment_body": "Reproduced cleanly, no file shown here."})
    composer = Composer(llm)
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed in 0.08s")],
    )
    issue = _load_issue(2341)
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="run-x", timestamp="t")
    want = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="run-x", timestamp="t")
    assert got == want
    assert "repro_test.py" in got


def test_composer_llm_response_including_heredoc_test_file_is_accepted():
    llm = _StaticLLM(
        response={
            "comment_body": "Reproduced.\n\n<details>\n<summary>Test file: repro_test.py</summary>\n\n```python\n...\n```\n\n</details>\n"
        }
    )
    composer = Composer(llm)
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed in 0.08s")],
    )
    got = composer.compose(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="run-x", timestamp="t")
    assert after_prefix(got).startswith("Reproduced.")


def test_resolved_test_file_prefers_synthesized_over_embedded_heredoc():
    # If a hypothesis somehow carries both (shouldn't happen in practice --
    # the two sources are mutually exclusive in the real pipeline -- but the
    # priority should be principled, not accidental): synthesized_test_file
    # is a copy of the file actually written to disk for the run that
    # happened, so it wins.
    hyp = _load_hypothesis(2341)
    hyp.synthesized_test_file = TestFile(path="different.py", body="def test_y(): pass\n")
    resolved = _resolved_test_file(hyp)
    assert resolved.path == "different.py"


def test_template_omits_test_file_section_when_none():
    hyp = _load_hypothesis(2341)
    run = _load_run(2341)
    hyp.observed = 'crashes with "boom"'
    # #2341's real extraction embeds a test file as a heredoc in commands[]
    # (spike-step-5/maintainer-review/FOLLOWUPS.md §1) -- strip it so this
    # case actually has no test file, which is what the test name claims.
    hyp.commands = ["uv run pytest -v"]
    run.commands[-1] = CommandResult(command="uv run pytest -v", exit_code=1, stderr='"boom"')
    body = compose_template(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "Synthesized test file" not in body
    assert "<details>" not in body


def test_composer_llm_response_missing_test_file_falls_back_to_template():
    # Same shape as the control-drop test: a response that never includes
    # the synthesized file is rejected even though it's otherwise
    # well-formed, because the commands it shows reference a file the
    # reader was never given.
    llm = _StaticLLM(response={"comment_body": "Reproduced cleanly, no file shown here."})
    composer = Composer(llm)
    hyp = _hypothesis_with_test_file()
    run = RunResult(
        hypothesis_number=9999,
        branch="none",
        commands=[CommandResult(command=c, exit_code=0) for c in hyp.commands[:-1]]
        + [CommandResult(command=hyp.commands[-1], exit_code=1, stderr="AssertionError")],
    )
    issue = _issue_9999()
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="run-x", timestamp="t")
    want = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="run-x", timestamp="t")
    assert got == want
    assert "test_issue_9999_repro.py" in got


def test_composer_llm_response_including_test_file_is_accepted():
    llm = _StaticLLM(
        response={
            "comment_body": (
                "Reproduced.\n\ntest_issue_9999_repro.py:\n```python\ndef test_x():\n    assert False\n```\n"
            )
        }
    )
    composer = Composer(llm)
    hyp = _hypothesis_with_test_file()
    run = RunResult(hypothesis_number=9999, branch="none", commands=[CommandResult(command=c, exit_code=0) for c in hyp.commands])
    got = composer.compose(hyp, _issue_9999(), run, Outcome.REPRODUCED, "matched", run_id="run-x", timestamp="t")
    assert after_prefix(got).startswith("Reproduced.")
    assert "test_issue_9999_repro.py" in got


def test_composer_test_file_gate_not_applied_when_none_exists():
    # #2639 has no synthesized test file -- a comment that never mentions
    # one must not be penalised for it.
    llm = _StaticLLM(response={"comment_body": "Reproduced cleanly.\n\nControl run confirmed the observer works."})
    composer = Composer(llm)
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    assert after_prefix(body).startswith("Reproduced cleanly.")


def test_composer_fixture_llm_replays_a_composition_recording(tmp_path):
    # Symmetry check with Extractor/SurfaceInferrer: if a recording *is*
    # present, FixtureLLM serves it and Composer uses it directly, no
    # fallback.
    fixtures_dir = tmp_path
    (fixtures_dir / "compositions").mkdir()
    (fixtures_dir / "compositions" / "2639.json").write_text(
        json.dumps({"comment_body": "Recorded composition body.\n\nControl run confirmed the observer works."})
    )
    llm = FixtureLLM(fixtures_dir)
    composer = Composer(llm)
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    got = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-y", timestamp="t")
    assert after_prefix(got).startswith("Recorded composition body.")
    assert "add-reproducer:issue=2639:run=run-y" in got


# --- AUTOMATION_PREFIX (criterion 2's accepted-risk mitigation) ----------


def _prefix_case() -> tuple[Hypothesis, Issue, RunResult]:
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none", base="ubuntu@24.04"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=3,
        branch="none",
        commands=[CommandResult(command="uv venv", exit_code=1)],
    )
    issue = Issue(
        number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator"
    )
    return hyp, issue, run


@pytest.mark.parametrize("outcome", sorted(COMMENT_OUTCOMES, key=lambda o: o.value))
def test_template_prefixes_every_comment_worthy_outcome(outcome):
    """Criterion 2 rests on this line being present, so it is asserted on
    every rung that can reach a maintainer -- including `partial`, which is
    why the wording says "attempt" rather than "reproduced"."""
    hyp, issue, run = _prefix_case()
    body = compose_template(hyp, issue, run, outcome, "reason", run_id="r1", timestamp="t")
    assert body.startswith(AUTOMATION_PREFIX + "\n\n")


def test_llm_path_prefixes_the_model_body():
    hyp, issue, run = _prefix_case()
    llm = _StaticLLM({"comment_body": "Reproduced. Here is what happened."})
    got = Composer(llm).compose(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert got.startswith(AUTOMATION_PREFIX + "\n\n")
    assert after_prefix(got).startswith("Reproduced. Here is what happened.")


def test_llm_path_does_not_double_the_disclaimer():
    """The prompt tells the model not to open with one; this is what happens
    when it does anyway. Cosmetic, so repaired rather than rejected."""
    hyp, issue, run = _prefix_case()
    llm = _StaticLLM(
        {"comment_body": "> **Automated attempt** -- please verify.\n\nReproduced. Here is what happened."}
    )
    got = Composer(llm).compose(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert got.count("Automated") == 1
    assert after_prefix(got).startswith("Reproduced. Here is what happened.")


def test_fallback_to_template_prefixes_exactly_once():
    """`Composer.compose()` returns `compose_template()` directly on failure,
    which already prefixes -- the prefix must not be applied twice."""
    hyp, issue, run = _prefix_case()
    llm = _StaticLLM(error=LLMError("no key"))
    got = Composer(llm).compose(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert got.count(AUTOMATION_PREFIX) == 1
    assert got.startswith(AUTOMATION_PREFIX + "\n\n")
