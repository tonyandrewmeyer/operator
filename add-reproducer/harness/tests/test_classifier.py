"""Classifier rung tests (PLAN.md Approach §5).

Four tests load the *real* captured output from spike-step-4 (the
`fixtures/runs/<n>.json` files, derived verbatim from
`spike-step-4/<n>/RESULT.md`) through `classifier.classify()` directly --
these are the task's stated acceptance cases: "the classifier must land
each on the correct rung."

Two of the four (#2341, #2327) are CLOSED issues with no `repo_version`,
so in the full pipeline they never reach the classifier at all -- Approach
§4's skip-when-stale gate intercepts them first (see test_runner_stage.py
and test_pipeline_e2e.py). Testing them here, bypassing the gate, is
deliberate: it's exactly what spike-step-4 itself did ("all 4 in_scope:
true extractions walked by hand", regardless of confidence or staleness)
to calibrate the classifier ladder against real captured output, and it
proves the log-only / did-not-reproduce distinction the ladder needs to
get right even when the gate isn't the one doing the work.

The remaining rungs the real corpus doesn't route through at the pipeline
level (log-only *with a match*, API-shape-mismatch) are exercised with
synthetic-but-realistic data lifted from the same RESULT.md files, since
no hypothesis in this small corpus reaches them while also clearing the
stale gate.
"""

import json
from pathlib import Path

from classifier import classify
from models import COMMENT_OUTCOMES, CommandResult, Hypothesis, MovingParts, Outcome, RunResult, SurfaceInference
from surface_inference import SYNTHESIS_INCOMPLETE_MARKER

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_hypothesis(number: int) -> Hypothesis:
    raw = json.loads((FIXTURES / "extractions" / f"{number}.json").read_text())
    return Hypothesis.from_dict(number, raw)


def _load_run(number: int) -> RunResult:
    return RunResult.from_dict(json.loads((FIXTURES / "runs" / f"{number}.json").read_text()))


def _load_surface(number: int) -> SurfaceInference:
    return SurfaceInference.from_dict(json.loads((FIXTURES / "surface" / f"{number}.json").read_text()))


def test_2639_reproduced_positive_signal_absent():
    outcome, reason = classify(_load_hypothesis(2639), _load_surface(2639), _load_run(2639))
    assert outcome == Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT
    assert "control" in reason


def test_2341_did_not_reproduce_on_current_ops():
    # Real captured result: already fixed, pytest passes cleanly, no
    # warning at any log level (spike-step-4/2341/RESULT.md).
    outcome, reason = classify(_load_hypothesis(2341), None, _load_run(2341))
    assert outcome == Outcome.DID_NOT_REPRODUCE


def test_2327_did_not_reproduce_on_current_ops():
    # Real captured result: partial fix landed, empty databag instead of
    # the reported leak (spike-step-4/2327/RESULT.md).
    outcome, reason = classify(_load_hypothesis(2327), None, _load_run(2327))
    assert outcome == Outcome.DID_NOT_REPRODUCE


def test_2484_unrunnable_test_selector_stale():
    outcome, reason = classify(_load_hypothesis(2484), None, _load_run(2484))
    assert outcome == Outcome.UNRUNNABLE_TEST_SELECTOR_STALE


def test_reproduced_log_only_synthetic():
    # If #2341's bug were still open: the exact warning text quoted in
    # the issue body, landing in captured log records, exit code zero.
    hypothesis = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[
            CommandResult(
                command="uv run pytest repro_test.py -v --log-cli-level=DEBUG",
                exit_code=0,
                stdout="1 passed in 0.08s",
                log_records=[
                    "WARNING ops-scenario.runtime.consistency_checker:_consistency_checker.py:122 "
                    "This scenario is probably inconsistent. Double check, and ignore this warning "
                    "if you're sure. The following warnings were found: "
                    "\"'rel_relation_changed' is implicitly using 0 as the remote unit. "
                    "Consider passing `remote_unit` explicitly.\""
                ],
            )
        ],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome == Outcome.REPRODUCED_LOG_ONLY


def test_unrunnable_api_shape_mismatch_synthetic():
    # spike-step-4/2327/RESULT.md's legacy-API attempt:
    # `TypeError: Relation.__init__() got an unexpected keyword argument 'id'`.
    hypothesis = _load_hypothesis(2327)
    run = RunResult(
        hypothesis_number=2327,
        branch="none",
        commands=[
            CommandResult(command="uv pip install ops==2.19.4 ops-scenario==6.1.6", exit_code=0),
            CommandResult(
                command="uv run pytest repro_test_legacy.py -v",
                exit_code=1,
                stderr="TypeError: Relation.__init__() got an unexpected keyword argument 'id'",
            ),
        ],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome == Outcome.UNRUNNABLE_API_SHAPE_MISMATCH
    assert "symbol_anchor" in reason


def test_reproduced_traceback_substring_match():
    hypothesis = Hypothesis(
        issue_number=1,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv run pytest repro_test.py -v"],
        expected="no crash",
        observed='Reporter says it crashes with "IndexError: list index out of range" on line 12.',
        confidence="high",
    )
    run = RunResult(
        hypothesis_number=1,
        branch="none",
        commands=[
            CommandResult(
                command="uv run pytest repro_test.py -v",
                exit_code=1,
                stderr="IndexError: list index out of range",
            )
        ],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome == Outcome.REPRODUCED


def test_reproduced_weaker_last_command_failed_no_match():
    hypothesis = Hypothesis(
        issue_number=2,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv venv", "uv run pytest repro_test.py -v"],
        expected="passes",
        observed="fails somehow, reporter wasn't specific",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=2,
        branch="none",
        commands=[
            CommandResult(command="uv venv", exit_code=0),
            CommandResult(command="uv run pytest repro_test.py -v", exit_code=1, stderr="AssertionError"),
        ],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome == Outcome.REPRODUCED_WEAKER


def test_partial_earlier_command_failed():
    hypothesis = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["git clone https://github.com/canonical/operator", "uv run pytest repro_test.py -v"],
        expected="passes",
        observed="reporter's exact traceback text, unquoted here",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=3,
        branch="none",
        commands=[
            # Deliberately NOT an environment-build command: a failed
            # `uv pip install` now lands on rung 0c (INFRASTRUCTURE_FAILED)
            # instead, because a run whose environment was never built is
            # not evidence about the bug -- and `partial` composes a comment.
            # `partial` keeps its meaning for a non-environment step that
            # failed part-way through the hypothesised sequence.
            CommandResult(command="git clone https://github.com/canonical/operator", exit_code=1, stderr="fatal"),
            CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="did not even run"),
        ],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome == Outcome.PARTIAL


def test_did_not_reproduce_all_commands_succeed():
    hypothesis = Hypothesis(
        issue_number=4,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv venv", "uv run pytest repro_test.py -v"],
        expected="passes",
        observed="crashes",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=4,
        branch="none",
        commands=[
            CommandResult(command="uv venv", exit_code=0),
            CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed"),
        ],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome == Outcome.DID_NOT_REPRODUCE


def test_2107_lxd_did_not_reproduce_no_quotable_observed_substring():
    # The one real substrate:lxd extraction in the whole corpus (see
    # fixtures/extractions/2107.json's _provenance). Its `observed` field
    # ("Traceback shows ValueError when parsing '0/lxd/4' as an integer")
    # is prose with no double-quoted substring for `_quoted_snippets()` to
    # match against -- PLAN.md's 2026-07-25 entry already documented this
    # as common for hand-written-style `observed` text, not something this
    # branch introduces. fixtures/runs/2107.json's (synthetic) captured
    # `juju debug-log` output contains the real crash verbatim, but every
    # command still exits 0 (juju CLI reads succeed regardless of what's in
    # the log), so the ladder correctly lands on `did_not_reproduce` rather
    # than fabricating a match. Pinned here so this stays a known, expected
    # limitation rather than a silent surprise.
    outcome, reason = classify(_load_hypothesis(2107), None, _load_run(2107))
    assert outcome == Outcome.DID_NOT_REPRODUCE


def test_lxd_scratch_reproduced_positive_signal_absent_synthetic():
    # Mirrors test_2639_reproduced_positive_signal_absent, but for the new
    # lxd-scratch branch -- no real hypothesis in the corpus reaches this
    # rung via lxd (see the #2107 test above), so this pins the classifier
    # decision with hand-built-but-realistic data, the same way
    # test_reproduced_log_only_synthetic etc. do for their rungs.
    hypothesis = Hypothesis(
        issue_number=9101,
        in_scope=True,
        moving_parts=MovingParts(substrate="lxd"),
        commands=[],
        expected="the charm reacts to a non-root Pebble notice on a machine unit",
        observed="no reaction to the notice",
        confidence="medium",
    )
    surface = SurfaceInference(
        charm_name="repro-i9101-x",
        pebble_service={"service": "workload", "command": "pebble notify canonical.com/repro/notice key=value", "user": "_daemon_"},
        ops_api_surface="pebble-custom-notice",
        expected_signal="observed notice:",
    )
    run = RunResult(
        hypothesis_number=9101,
        branch="lxd-scratch",
        commands=[CommandResult(command="juju status repro/0", exit_code=0, stdout="Workload: unknown")],
        control=CommandResult(
            command="juju ssh repro/0 -- ...; juju status repro/0",
            exit_code=0,
            stdout="observed notice: canonical.com/repro/notice in workload",
        ),
    )
    outcome, reason = classify(hypothesis, surface, run)
    assert outcome == Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT
    assert "control" in reason


def test_unrunnable_synthesis_incomplete_marker_beats_reproduced_weaker():
    # spike-step-5/composer-live/RESULT.md Finding 6: a test-file-synthesis
    # fallback stub (surface_inference._loud_failing_test_file) raises an
    # AssertionError carrying SYNTHESIS_INCOMPLETE_MARKER. Without this rung
    # checked ahead of rung 6, this exact shape (non-zero exit, no observed
    # substring match) would misread the fallback stub as `reproduced_weaker`
    # -- a comment-worthy outcome for a scaffold that never actually
    # exercised the reported bug.
    hypothesis = Hypothesis(
        issue_number=2045,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv add 'ops[testing]' pytest", "uv run pytest test_issue_2045_repro.py -v"],
        expected="cwd inside a hook is the charm root",
        observed="cwd is not the charm root",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=2045,
        branch="none",
        commands=[
            CommandResult(command="uv add 'ops[testing]' pytest", exit_code=0),
            CommandResult(
                command="uv run pytest test_issue_2045_repro.py -v",
                exit_code=1,
                stderr=(
                    f"AssertionError: {SYNTHESIS_INCOMPLETE_MARKER}: LLM-driven test-file "
                    "synthesis did not produce a usable assertion for this issue"
                ),
            ),
        ],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome == Outcome.UNRUNNABLE_SYNTHESIS_INCOMPLETE
    assert outcome not in COMMENT_OUTCOMES


def test_positive_signal_absent_no_control_stays_silent():
    hypothesis = _load_hypothesis(2639)
    surface = _load_surface(2639)
    run = RunResult(
        hypothesis_number=2639,
        branch="k8s-scratch",
        commands=[CommandResult(command="juju status repro/0", exit_code=0, stdout="Workload: unknown")],
        control=None,
    )
    outcome, reason = classify(hypothesis, surface, run)
    assert outcome == Outcome.DID_NOT_REPRODUCE
    assert "broken observer" in reason
