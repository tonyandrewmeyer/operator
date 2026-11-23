"""Classifier rung tests.

Four tests load *real* captured output (the `fixtures/runs/<n>.json`
files) through `classifier.classify()` directly -- the classifier must land
each on the correct rung.

Two of the four (#2341, #2327) are CLOSED issues with no `repo_version`,
so in the full pipeline they never reach the classifier at all -- the
skip-when-stale gate intercepts them first (see test_runner_stage.py
and test_pipeline_e2e.py). Testing them here, bypassing the gate, is
deliberate: it calibrates the classifier ladder against real captured
output regardless of confidence or staleness, and it proves the log-only /
did-not-reproduce distinction the ladder needs to get right even when the
gate isn't the one doing the work.

The remaining rungs the real corpus doesn't route through at the pipeline
level (log-only *with a match*, API-shape-mismatch) are exercised with
synthetic-but-realistic data based on the same issues, since no hypothesis
in this small corpus reaches them while also clearing the stale gate.
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
    # warning at any log level.
    outcome, reason = classify(_load_hypothesis(2341), None, _load_run(2341))
    assert outcome == Outcome.DID_NOT_REPRODUCE


def test_2327_did_not_reproduce_on_current_ops():
    # Real captured result: partial fix landed, empty databag instead of
    # the reported leak.
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
    # #2327's legacy-API attempt:
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
            # Not an environment-build command: a failed `uv pip install`
            # lands on rung 0c instead.
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
    # The `observed` prose has no double-quoted substring to match, and every
    # command exits 0 even though the debug-log holds the crash, so the ladder
    # lands on `did_not_reproduce`. Pinned as a known limitation.
    outcome, reason = classify(_load_hypothesis(2107), None, _load_run(2107))
    assert outcome == Outcome.DID_NOT_REPRODUCE


def test_lxd_scratch_reproduced_positive_signal_absent_synthetic():
    # Mirrors test_2639_reproduced_positive_signal_absent for the lxd-scratch
    # branch, with hand-built data.
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
    # The synthesis fallback stub raises an AssertionError carrying
    # SYNTHESIS_INCOMPLETE_MARKER, which must not be read as
    # `reproduced_weaker`.
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


# --- rung 5: a control that did not fire is not a control ---
#
# The control's `juju status` read polls for the signal, so a non-zero exit
# means the signal never appeared, and that must not count as confirming the
# observer works.


def _absent_signal_case(control: CommandResult | None) -> tuple[Hypothesis, SurfaceInference, RunResult]:
    """#2639's real hypothesis and surface, with a main run whose output does
    not carry `expected_signal` -- the only state in which rung 5 reads the
    control at all."""
    return (
        _load_hypothesis(2639),
        _load_surface(2639),
        RunResult(
            hypothesis_number=2639,
            branch="k8s-scratch",
            commands=[CommandResult(command="juju status repro-i2639/0", exit_code=0, stdout="Workload: unknown")],
            control=control,
        ),
    )


def test_positive_signal_absent_with_a_timed_out_control_stays_silent():
    hypothesis, surface, run = _absent_signal_case(
        CommandResult(
            command="juju ssh ...; for _ in $(seq 1 24); do ... done; ...",
            exit_code=1,
            stdout="Unit  Workload  Agent  Address  Ports  Message\nrepro-i2639/0*  active  idle  10.1.0.72",
            stderr="timed out after 120s waiting for the control signal 'observed notice:' to appear in the unit status",
        )
    )

    outcome, reason = classify(hypothesis, surface, run)

    assert outcome == Outcome.DID_NOT_REPRODUCE
    assert outcome not in COMMENT_OUTCOMES
    assert "exited 1" in reason
    assert "broken observer" in reason


def test_positive_signal_absent_with_a_control_killed_by_its_step_budget_stays_silent():
    """`_execute_sequence` records a subprocess timeout as exit 124 with no
    captured output at all, so the exit code is the only thing left to read.
    Same verdict, reached without any text to match against."""
    hypothesis, surface, run = _absent_signal_case(
        CommandResult(command="juju ssh ...; ...", exit_code=124, stderr="timed out after 300s")
    )

    outcome, reason = classify(hypothesis, surface, run)

    assert outcome == Outcome.DID_NOT_REPRODUCE
    assert "exited 124" in reason


def test_positive_signal_absent_with_a_silent_control_stays_silent():
    """A control that exited 0 and still shows no signal. Distinct from the
    timeout above only in how it is described -- the verdict is the same,
    because in both cases nothing demonstrated that the observer works."""
    hypothesis, surface, run = _absent_signal_case(
        CommandResult(command="juju ssh ...; ...", exit_code=0, stdout="Unit  Workload  Agent\nrepro-i2639/0*  active")
    )

    outcome, reason = classify(hypothesis, surface, run)

    assert outcome == Outcome.DID_NOT_REPRODUCE
    assert "did not produce the signal" in reason


def test_positive_signal_absent_with_a_control_that_fired_still_reproduces():
    """The rung still reaches its comment-worthy outcome when the control
    genuinely succeeded -- the exit-code check tightens what counts as a
    confirmation, it does not remove the rung."""
    hypothesis, surface, run = _absent_signal_case(
        CommandResult(
            command="juju ssh ...; ...",
            exit_code=0,
            stdout="Unit  Workload  Agent  Address  Ports  Message\n"
            "repro-i2639/0*  active  idle  10.1.0.72  observed notice: canonical.com/repro/notice-1 in workload",
        )
    )

    outcome, reason = classify(hypothesis, surface, run)

    assert outcome == Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT
    assert outcome in COMMENT_OUTCOMES
    assert "control" in reason
