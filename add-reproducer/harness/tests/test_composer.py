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
    _rendered_command_sequences,
    _resolved_test_file,
    _test_file_to_render,
    _validate,
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
        self.prompts: list[str] = []

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        self.calls += 1
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return self._response


def _commands_block(run: RunResult) -> str:
    """The commands block a well-behaved response has to carry.

    `spike-step-5/fifth-dispatch/RESULT.md` §2: `_validate()` now requires
    that the response render `run_result.commands` exactly, so a stub whose
    `comment_body` is a single sentence is no longer a *valid* response --
    it is one that would fall back to the template. The ten stubs below that
    predate the check are given a correct block rather than exempted from
    it: each of those tests is about some other gate (control, log records,
    test file, the prefix, the marker), and every one of them is a sharper
    test of that gate when the response around it is otherwise well-formed.
    """
    lines = "\n".join(f"$ {c.command}" for c in run.commands)
    return f"```shell\n{lines}\n```"


def _stub_body(text: str, run: RunResult) -> dict:
    return {"comment_body": f"{text}\n\n{_commands_block(run)}\n"}


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
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest -v", exit_code=1, stderr='"boom"')],
    )
    llm = _StaticLLM(response=_stub_body("Reproduced cleanly, nothing about any control.", run))
    composer = Composer(llm)
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
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest -v", exit_code=1, stderr='"boom"')],
    )
    llm = _StaticLLM(response=_stub_body("Reproduced cleanly, no log records involved.", run))
    composer = Composer(llm)
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
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    llm = _StaticLLM(
        response=_stub_body(
            "Reproduced cleanly.\n\nVersions: repo=main\n\nControl run confirmed the observer works.", run
        )
    )
    composer = Composer(llm)
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
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed in 0.08s")],
    )
    llm = _StaticLLM(
        response=_stub_body(
            "Reproduced.\n\n<details>\n<summary>Test file: repro_test.py</summary>\n\n```python\n...\n```\n\n</details>", run
        )
    )
    composer = Composer(llm)
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
    hyp = _hypothesis_with_test_file()
    run = RunResult(hypothesis_number=9999, branch="none", commands=[CommandResult(command=c, exit_code=0) for c in hyp.commands])
    llm = _StaticLLM(
        response=_stub_body(
            "Reproduced.\n\ntest_issue_9999_repro.py:\n```python\ndef test_x():\n    assert False\n```", run
        )
    )
    composer = Composer(llm)
    got = composer.compose(hyp, _issue_9999(), run, Outcome.REPRODUCED, "matched", run_id="run-x", timestamp="t")
    assert after_prefix(got).startswith("Reproduced.")
    assert "test_issue_9999_repro.py" in got


def test_composer_test_file_gate_not_applied_when_none_exists():
    # #2639 has no synthesized test file -- a comment that never mentions
    # one must not be penalised for it.
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    llm = _StaticLLM(response=_stub_body("Reproduced cleanly.\n\nControl run confirmed the observer works.", run))
    composer = Composer(llm)
    body = composer.compose(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    assert after_prefix(body).startswith("Reproduced cleanly.")


def test_composer_fixture_llm_replays_a_composition_recording(tmp_path):
    # Symmetry check with Extractor/SurfaceInferrer: if a recording *is*
    # present, FixtureLLM serves it and Composer uses it directly, no
    # fallback.
    fixtures_dir = tmp_path
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    (fixtures_dir / "compositions").mkdir()
    (fixtures_dir / "compositions" / "2639.json").write_text(
        json.dumps(
            _stub_body("Recorded composition body.\n\nControl run confirmed the observer works.", run)
        )
    )
    llm = FixtureLLM(fixtures_dir)
    composer = Composer(llm)
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
    llm = _StaticLLM(_stub_body("Reproduced. Here is what happened.", run))
    got = Composer(llm).compose(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert got.startswith(AUTOMATION_PREFIX + "\n\n")
    assert after_prefix(got).startswith("Reproduced. Here is what happened.")


def test_llm_path_does_not_double_the_disclaimer():
    """The prompt tells the model not to open with one; this is what happens
    when it does anyway. Cosmetic, so repaired rather than rejected."""
    hyp, issue, run = _prefix_case()
    llm = _StaticLLM(
        _stub_body("> **Automated attempt** -- please verify.\n\nReproduced. Here is what happened.", run)
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


def test_template_versions_line_includes_observed_ops_version_when_present():
    """`spike-step-5/first-dispatch/RESULT.md` §6.1: `charmcraft pack`
    resolves `ops~=3.8` from PyPI inside its own managed LXD instance, so the
    `ops` a verdict was produced on is not the checked-out tree and is not
    what `repo=` names either -- `repo=` is the extraction's pin, usually
    nothing. Disclosed for the same reason `observed_juju` is."""
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
        observed_juju_version="4.0.14-genericlinux-amd64",
        observed_ops_version="3.8.2",
    )
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert (
        "Versions: repo=unpinned, juju=unpinned, base=ubuntu@24.04, substrate=k8s, "
        "observed_juju=4.0.14-genericlinux-amd64, observed_ops=3.8.2" in body
    )


def test_template_versions_line_omits_observed_ops_version_when_absent():
    """A branch that packs no charm has no pack log to read, and an
    unparseable one gives `None` -- either way the line must not claim an
    "observed_ops=" value nobody measured."""
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
    assert "observed_ops=" not in body


# --- The `<summary>` contradiction (spike-step-5/second-dispatch/RESULT.md
# §5.2) ----------------------------------------------------------------
#
# The first comment this project ever composed on GitHub Actions
# (`tonyandrewmeyer/operator` run 35219949266, 2026-09-17) rendered `#2045`'s
# fourteen-line test file twice: once inside a `<details>` block whose
# `<summary>` said it was "not shown in the commands actually run below", and
# again four lines later as the `cat > test_cwd.py << 'PYEOF'` heredoc that
# wrote it. The commands and the file body below are transcribed from that
# comment, which `second-dispatch/RESULT.md` §5.1 quotes verbatim and
# complete; the run artefact itself was not committed, and the live
# extraction's `expected`/`observed` prose was never recorded anywhere, so
# those two fields are left empty here rather than guessed at. Nothing the
# assertions below check reads them.

_DISPATCH_TEST_FILE_BODY = """\
import os
import pytest
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, *args):
        super().__init__(*args)
        assert os.getcwd() == self.framework.charm_dir


def test_cwd_in_scenario():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.update_status(), testing.State()) as mgr:
        pass"""

_DISPATCH_HEREDOC = f"cat > test_cwd.py << 'PYEOF'\n{_DISPATCH_TEST_FILE_BODY}\nPYEOF"

_DISPATCH_COMMANDS = [
    "uv init --bare .",
    "uv add 'ops[testing]'",
    _DISPATCH_HEREDOC,
    "uv run pytest test_cwd.py -v",
]

_DISPATCH_PYTEST_OUTPUT = """\
test_cwd.py::test_cwd_in_scenario FAILED                                 [100%]

=================================== FAILURES ===================================
_____________________________ test_cwd_in_scenario _____________________________

    def __init__(self, *args):
        super().__init__(*args)
>       assert os.getcwd() == self.framework.charm_dir
E       AssertionError: assert '/home/runner/work/operator/operator' == PosixPath('/tmp/tmph5kel9et')
"""


def _dispatch_hypothesis() -> Hypothesis:
    return Hypothesis(
        issue_number=2045,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=list(_DISPATCH_COMMANDS),
        expected="",
        observed="",
        confidence="medium",
    )


def _dispatch_run() -> RunResult:
    """The `none` branch runs `commands[]` one `bash -c` at a time, so the
    heredoc that writes the file is itself one of the executed commands --
    which is exactly why the comment showed the body twice."""
    return RunResult(
        hypothesis_number=2045,
        branch="none",
        commands=[
            CommandResult(command=_DISPATCH_COMMANDS[0], exit_code=0),
            CommandResult(command=_DISPATCH_COMMANDS[1], exit_code=0),
            CommandResult(command=_DISPATCH_COMMANDS[2], exit_code=0),
            CommandResult(command=_DISPATCH_COMMANDS[3], exit_code=1, stdout=_DISPATCH_PYTEST_OUTPUT),
        ],
    )


def _dispatch_issue() -> Issue:
    return Issue(
        number=2045,
        title="os.getcwd() is not the charm root in ops[testing]",
        body="b",
        labels=[],
        state="OPEN",
        created_at="",
        author="a",
        repo="canonical/operator",
    )


def test_composed_comment_does_not_render_the_heredoc_body_twice():
    """The defect itself, against the real composed text: the body appeared
    once in the `<details>` block and once in the commands block."""
    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        _dispatch_run(),
        Outcome.REPRODUCED_WEAKER,
        "'uv run pytest test_cwd.py -v' (last command) exited non-zero, no substring match",
        run_id="35219949266",
        timestamp="2026-09-17T12:14:36.483167+00:00",
    )
    assert body.count("def test_cwd_in_scenario():") == 1


def test_composed_comment_does_not_deny_showing_a_file_it_shows():
    """The reader-visible half: the `<summary>` said the file was "not shown
    in the commands actually run below" directly above the commands that
    show it."""
    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        _dispatch_run(),
        Outcome.REPRODUCED_WEAKER,
        "reason",
        run_id="r1",
        timestamp="t",
    )
    assert "not shown in the commands actually run below" not in body
    assert "<details>" not in body


def test_the_heredoc_body_survives_in_the_commands_block():
    """Dropping the duplicate must not drop the file. The reader still sees
    every line of it, in the `cat` heredoc that writes it."""
    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        _dispatch_run(),
        Outcome.REPRODUCED_WEAKER,
        "reason",
        run_id="r1",
        timestamp="t",
    )
    commands_section = body[body.index("Commands run:") :]
    for line in _DISPATCH_TEST_FILE_BODY.splitlines():
        assert line in commands_section


def test_a_heredoc_the_executed_commands_drop_is_still_rendered():
    """The other side of the same question, and the reason it is asked about
    visibility rather than provenance: `#2341`'s extraction embeds its test
    file the same way, but its run replays only the final `pytest`
    invocation, so the body really is absent from the commands block and the
    `<details>` section -- and its "not shown" wording -- are correct."""
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed")],
    )
    body = compose_template(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "<summary>Test file: repro_test.py" in body
    assert "not shown in the commands actually run below" in body


def test_a_synthesized_file_is_always_rendered():
    """Synthesis writes the file straight to disk, so no command ever shows
    it -- this path is untouched by the visibility question."""
    hyp = _dispatch_hypothesis()
    hyp.commands = ["uv run pytest test_x.py -v"]
    hyp.synthesized_test_file = TestFile(path="test_x.py", body="def test_x():\n    assert False\n")
    run = RunResult(
        hypothesis_number=2045,
        branch="none",
        commands=[CommandResult(command="uv run pytest test_x.py -v", exit_code=1, stdout="E assert False")],
    )
    body = compose_template(hyp, _dispatch_issue(), run, Outcome.REPRODUCED_WEAKER, "r", run_id="r1", timestamp="t")
    assert "<summary>Synthesized test file: test_x.py" in body


def test_the_llm_path_is_not_asked_for_a_file_the_commands_already_show():
    """`_build_prompt()` and `_validate()` have to agree with the renderer:
    asking the model to include a `<details>` block whose heading says the
    commands do not show the file would reintroduce the contradiction one
    layer up, and demanding the path back would reject a correct comment."""
    run = _dispatch_run()
    llm = _StaticLLM(response=_stub_body("The bug reproduced in this automated attempt.", run))
    composer = Composer(llm)
    got = composer.compose(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        run,
        Outcome.REPRODUCED_WEAKER,
        "reason",
        run_id="r1",
        timestamp="t",
    )
    assert after_prefix(got).startswith("The bug reproduced in this automated attempt.")
    assert "<details>" not in got
    assert "not shown in the commands actually run below" not in llm.prompts[0]


def test_a_none_branch_comment_now_discloses_the_ops_it_resolved():
    """`spike-step-5/second-dispatch/RESULT.md` §6.2, closed in
    `fourth-dispatch/RESULT.md` §1.

    The second dispatch's comment rendered ` + ops==3.8.2` in its observed
    output and `Versions: ... substrate=none` with no `observed_ops=` four
    lines above it. Same commands, same branch, with the version the seam now
    captures: the versions line states what the observed output already
    showed, and the two halves of the comment stop disagreeing."""
    run = _dispatch_run()
    run.observed_ops_version = "3.8.2"

    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        run,
        Outcome.REPRODUCED_WEAKER,
        "'uv run pytest test_cwd.py -v' (last command) exited non-zero, no substring match",
        run_id="35219949266",
        timestamp="2026-09-17T12:14:36.483167+00:00",
    )

    assert "Versions: repo=unpinned, juju=unpinned, base=unpinned, substrate=none, observed_ops=3.8.2" in body
    # Still no juju: this branch provisions no substrate, and the asymmetry
    # between the two fields is the point, not an oversight.
    assert "observed_juju=" not in body


# --- The live composer's invented test-file section
# (spike-step-5/fourth-dispatch/RESULT.md §3, fixed in fifth-dispatch) ----
#
# On the path where the extraction's test file is a heredoc inside
# `commands[]`, `_test_file_to_render()` returns `None` and `_build_prompt()`
# emits no test-file section. `deepseek/deepseek-chat` rendered one anyway on
# both of the fourth round's live runs, with different damage each time --
# run 1 dropped the heredoc from its commands block (its three printed
# commands give "ERROR: file or directory not found: test_cwd.py", exit 4,
# when pasted), run 2 kept it and printed the body twice under a malformed
# `<Synthesized test file>` tag.
#
# The four files under `fixtures/composed/` are those two runs' own
# artefacts, byte-for-byte as GitHub Actions uploaded them: `<n>-run.json` is
# the `add-reproducer-run` artefact's `2045-run.json` and `<n>.md` is its
# `2045.md`, the composed-and-withheld comment. Nothing below is transcribed
# or reconstructed, so these are regressions against what the model really
# returned rather than against a description of it.

_LIVE_RUNS = {
    "run1": "35442613976",  # dropped the heredoc from its commands block
    "run2": "35442768800",  # kept it, and printed the body twice
}


def _live(which: str) -> tuple[Hypothesis, RunResult, str]:
    """The hypothesis, run result and model `comment_body` of one fourth-round
    live run.

    `2045.md` is the *composed* comment, so it carries `AUTOMATION_PREFIX`
    and the marker that `Composer.compose()` adds around the model's body.
    Both are stripped back off here, which reconstructs exactly the string
    `_validate()` was handed at the time.
    """
    run_id = _LIVE_RUNS[which]
    raw = json.loads((FIXTURES / "composed" / f"2045-{run_id}-run.json").read_text())
    run = RunResult.from_dict(raw)
    hyp = Hypothesis.from_dict(2045, raw["hypothesis"])
    composed = (FIXTURES / "composed" / f"2045-{run_id}.md").read_text()
    body = after_prefix(composed)
    marker = f"<!-- add-reproducer:issue=2045:run={run_id} -->"
    assert body.rstrip().endswith(marker), body[-200:]
    return hyp, run, body.rstrip()[: -len(marker)].rstrip()


def _validate_live(which: str) -> None:
    """`_validate()` with exactly the arguments `Composer.compose()` builds
    for this run."""
    hyp, run, body = _live(which)
    test_file = _test_file_to_render(hyp, run)
    assert test_file is None, "these runs are on the withheld path; that is the premise"
    _validate(
        {"comment_body": body},
        control_output=None,
        test_file_path=None,
        log_output=None,
        commands=[c.command for c in run.commands],
        withheld_test_file=_resolved_test_file(hyp),
    )


def test_the_fourth_rounds_live_runs_are_both_on_the_withheld_path():
    """The premise, asserted rather than assumed: no synthesised file, the
    test file a heredoc inside the extraction's own commands, the run
    executing that heredoc -- so the commands block shows the body and
    `_test_file_to_render()` correctly declines to show it again."""
    for which, run_id in _LIVE_RUNS.items():
        hyp, run, _ = _live(which)
        assert hyp.synthesized_test_file is None, which
        assert _resolved_test_file(hyp) is not None, which
        assert _test_file_to_render(hyp, run) is None, which
        assert run.branch == "none", which
        assert len(run.commands) == 4, (which, run_id)


def test_run_1s_real_output_is_rejected():
    """The commands check, against the artefact that motivated it. Run 1's
    comment prints three of the four commands that ran, dropping the
    `cat > test_cwd.py << 'PYEOF'` heredoc."""
    with pytest.raises(ComposerInvalid, match="commands block is not the 4 commands"):
        _validate_live("run1")


def test_run_2s_real_output_is_rejected():
    """The test-file-section check, against the artefact that motivated it.
    Run 2's commands block is faithful -- all four, verbatim -- so nothing
    about the commands could have caught it; what is wrong is the section
    above them."""
    with pytest.raises(ComposerInvalid, match="more than once"):
        _validate_live("run2")


def test_neither_live_run_could_have_been_caught_before():
    """The gap this round closes, stated as a test rather than as prose.

    `_validate()`'s test-file check keys on `test_file_path`, which is
    `None` on exactly this path, and none of the other three checks looks at
    the commands at all -- so the pre-existing gates accept both comments."""
    for which in _LIVE_RUNS:
        hyp, run, body = _live(which)
        _validate(
            {"comment_body": body},
            control_output=None,
            test_file_path=None,
            log_output=None,
            commands=[],
            withheld_test_file=None,
        )


def test_run_1s_commands_are_a_subsequence_of_what_ran():
    """Why "matches" is sequence equality and not subsequence, measured on
    the artefact rather than argued.

    Run 1 dropped a command without reordering or inventing one, so what it
    printed *is* an in-order subsequence of what ran -- a subsequence test
    would accept the one comment this check exists to reject."""
    _, run, body = _live("run1")
    ran = [c.command for c in run.commands]
    printed = _rendered_command_sequences(body)[0]
    assert printed != ran
    assert len(printed) == 3

    def is_subsequence(small, large):
        it = iter(large)
        return all(item in it for item in small)

    assert is_subsequence(printed, ran)


def test_a_reordered_commands_block_is_rejected():
    """Why "matches" is sequence equality and not set equality. Set equality
    rejects run 1 too, but accepts this -- and "write the file, then install,
    then run it" reordered into "run it, then write the file" fails for a
    reader exactly as an omission does."""
    _, run, _ = _live("run2")
    ran = [c.command for c in run.commands]
    shuffled = [ran[3], ran[0], ran[1], ran[2]]
    block = "\n".join(f"$ {c}" for c in shuffled)
    with pytest.raises(ComposerInvalid, match="commands block is not the 4 commands"):
        _validate(
            {"comment_body": f"Reproduced.\n\n```shell\n{block}\n```\n"},
            control_output=None,
            test_file_path=None,
            log_output=None,
            commands=ran,
            withheld_test_file=None,
        )
    assert sorted(shuffled) == sorted(ran)


def test_a_faithful_commands_block_is_accepted():
    """The check must not reject the comment this round is trying to get.
    Run 2's commands block is the faithful one, heredoc and all, so it
    passes the commands check on its own."""
    _, run, body = _live("run2")
    _validate(
        {"comment_body": body},
        control_output=None,
        test_file_path=None,
        log_output=None,
        commands=[c.command for c in run.commands],
        withheld_test_file=None,
    )


def test_a_heredoc_command_is_read_as_one_command():
    """The parse that makes the check usable at all: a heredoc that writes a
    twenty-line test file is one `CommandResult.command`, and it renders as a
    `$ ` line followed by twenty lines that are not. Splitting on lines would
    make every heredoc in the corpus look like a mismatch."""
    _, run, body = _live("run2")
    printed = _rendered_command_sequences(body)[0]
    assert printed == [c.command for c in run.commands]
    assert printed[2].startswith("cat > test_getcwd.py << 'PYEOF'")
    assert printed[2].endswith("PYEOF")
    assert "\n" in printed[2]


@pytest.mark.parametrize("fence", ["```shell", "```sh", "```bash", "```console", "```", "~~~shell"])
def test_the_commands_block_is_found_whatever_its_fence(fence):
    """The prompt asks for ```shell and run 1 used it, but nothing makes a
    model do so -- and a block tagged `sh` or tagged nothing is the same
    block to a reader. Rejecting over the info string would be a fallback
    bought for no reader-visible gain."""
    close = "~~~" if fence.startswith("~~~") else "```"
    _, run, _ = _live("run2")
    ran = [c.command for c in run.commands]
    block = "\n".join(f"$ {c}" for c in ran)
    _validate(
        {"comment_body": f"Reproduced.\n\n{fence}\n{block}\n{close}\n"},
        control_output=None,
        test_file_path=None,
        log_output=None,
        commands=ran,
        withheld_test_file=None,
    )


def test_a_control_blocks_dollar_line_is_not_mistaken_for_the_commands():
    """`_control_output()` renders its own `$ ` line, so "the block with
    dollar signs in it" is not a unique description. The check asks whether
    *any* block is the sequence that ran rather than guessing which one was
    meant -- and the control block, being one command, is not it."""
    _, run, _ = _live("run2")
    ran = [c.command for c in run.commands]
    block = "\n".join(f"$ {c}" for c in ran)
    body = (
        f"Reproduced.\n\n```shell\n{block}\n```\n\n"
        "Control (re-run to confirm the observer itself works):\n"
        "```\n$ juju debug-log --replay\nunit-my-charm-0: nothing\n```\n"
    )
    sequences = _rendered_command_sequences(body)
    assert len(sequences) == 2
    assert sequences[1] == ["juju debug-log --replay\nunit-my-charm-0: nothing"]
    _validate(
        {"comment_body": body},
        control_output="$ juju debug-log --replay",
        test_file_path=None,
        log_output=None,
        commands=ran,
        withheld_test_file=None,
    )


def test_a_collapsed_section_is_rejected_where_none_was_supplied():
    """The structural half of the test-file check, and the half that catches
    run 1: its `<details>` body appears only once in the comment, because it
    dropped the heredoc, so counting the body does not see it."""
    hyp, run, body = _live("run1")
    assert body.count(_resolved_test_file(hyp).body) == 1
    assert "<details>" in body
    with pytest.raises(ComposerInvalid, match="collapsed section"):
        _validate(
            {"comment_body": body},
            control_output=None,
            test_file_path=None,
            log_output=None,
            commands=[],  # commands check disabled, so only this one can fire
            withheld_test_file=_resolved_test_file(hyp),
        )


def test_run_2s_duplication_is_the_defect_the_third_dispatch_closed():
    """`third-dispatch/RESULT.md` §1 stopped the *template* rendering the
    body twice. Run 2 is the same fourteen-line duplication arriving through
    the model instead, and this is it measured on the real comment."""
    hyp, _, body = _live("run2")
    assert body.count(_resolved_test_file(hyp).body) == 2


def test_the_withheld_check_does_not_fire_when_the_section_was_supplied():
    """`#2341`'s shape: the executed run replays only the final pytest
    invocation, so the heredoc never reaches the commands block, the section
    is genuinely required, and a `<details>` block is correct. This must stay
    accepted."""
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed")],
    )
    assert _test_file_to_render(hyp, run) is not None
    llm = _StaticLLM(
        response=_stub_body(
            "Reproduced.\n\n<details>\n<summary>Test file: repro_test.py</summary>\n\n"
            "```python\n...\n```\n\n</details>",
            run,
        )
    )
    got = Composer(llm).compose(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert after_prefix(got).startswith("Reproduced.")
    assert "<details>" in got


def test_a_rejected_live_comment_falls_back_to_the_template():
    """What `_validate()` failing *does*, end to end and for both runs: the
    same thing every other check in it already does. Not a retry (a second
    live call, with no evidence it converges -- the defect was 2/2 and the
    damage differed), and not an annotation (which ships the wrong commands
    block with a note admitting it). The template is not an arbitrary
    fallback here: it renders `run_result.commands` verbatim and applies
    `_test_file_to_render()` itself, so it is right by construction about
    exactly the two things these checks test."""
    for which in _LIVE_RUNS:
        hyp, run, body = _live(which)
        issue = _dispatch_issue()
        llm = _StaticLLM(response={"comment_body": body})
        got = Composer(llm).compose(
            hyp, issue, run, Outcome.REPRODUCED_WEAKER, "reason", run_id=_LIVE_RUNS[which], timestamp="t"
        )
        want = compose_template(
            hyp, issue, run, Outcome.REPRODUCED_WEAKER, "reason", run_id=_LIVE_RUNS[which], timestamp="t"
        )
        assert got == want, which
        assert llm.calls == 1, which
        # And the fallback is a comment a reader can actually paste: every
        # command that ran, in order, the heredoc included.
        assert _rendered_command_sequences(got)[0] == [c.command for c in run.commands], which
        assert "<details>" not in got, which


# --- The prompt's conditional (fifth-dispatch item 1) --------------------


def test_the_withheld_path_forbids_the_section_rather_than_omitting_it():
    """The fix to the prompt. Before this round `_SCHEMA_INSTRUCTIONS` was one
    string carrying the "If a ... section is given below" paragraph whether or
    not one was, so the model's only cue was an absence it had to notice."""
    hyp, run, _ = _live("run1")
    prompt = Composer._build_prompt(
        hyp, _dispatch_issue(), run, Outcome.REPRODUCED_WEAKER, "reason", "t"
    )
    assert "So do NOT write one." in prompt
    assert "No `<details>` block and no `<summary>` anywhere in the" in prompt
    assert "A \"Synthesized test file\" or \"Test file\" section is given below" not in prompt
    # Still no section, which is the property the third dispatch added and
    # this round must not regress.
    assert "Test file: " not in prompt
    assert "Synthesized test file: " not in prompt


def test_the_supplied_path_still_asks_for_the_section():
    """The other side of the conditional: `#2341`'s shape must keep the
    rendering instructions it has always had, verbatim."""
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed")],
    )
    prompt = Composer._build_prompt(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", "t")
    assert 'A "Synthesized test file" or "Test file" section is given below' in prompt
    assert "So do NOT write one." not in prompt
    assert "<summary>` is the given heading text verbatim" in prompt
    assert "Test file: repro_test.py" in prompt


def test_both_branches_keep_the_rest_of_the_instructions():
    """Splitting one string into head/branch/tail is the kind of change that
    silently drops a paragraph. Both prompts must still carry the numbered
    list, the log-record paragraph, the control paragraph and the closing
    warning, in that order."""
    supplied_hyp = _load_hypothesis(2341)
    supplied_run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed")],
    )
    withheld_hyp, withheld_run, _ = _live("run1")
    prompts = [
        Composer._build_prompt(supplied_hyp, _load_issue(2341), supplied_run, Outcome.REPRODUCED, "m", "t"),
        Composer._build_prompt(withheld_hyp, _dispatch_issue(), withheld_run, Outcome.REPRODUCED_WEAKER, "r", "t"),
    ]
    markers = [
        '  "comment_body": str',
        "1. One line stating the outcome plainly",
        "5. One line: \"Reproduced on <base>/<substrate> at <timestamp>.\"",
        'If a "Captured log records" section is given below',
        'If a "Control run" section is given below',
        "A confidently wrong comment is worse than no",
    ]
    for prompt in prompts:
        positions = [prompt.index(m) for m in markers]
        assert positions == sorted(positions), prompt[:400]
