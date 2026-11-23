"""Two small things that keep a real run's failure readable.

1. Rung 0's reason quotes the first line of a failed command's output that
   looks like an error. juju writes progress to stderr alongside errors, so
   quoting the *first* line of a collided `juju deploy` would report
   `'deploy' failed: Located local charm "i2639-pebble-notice", revision 0`
   -- which reads as though nothing went wrong -- while the actual
   `ERROR cannot add application ...` sits three lines below.
2. `CommandResult` carries a duration, so an end-to-end run time can be
   attributed across pack/deploy/wait.
"""

from __future__ import annotations

import subprocess

import classifier
from models import CommandResult, Hypothesis, MovingParts, Outcome, RunResult
from seams.runner import SubprocessRunnerSeam, build_plan


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        issue_number=2639,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s"),
        commands=[],
        expected="expected",
        observed="observed",
        confidence="medium",
    )


# The exact stderr of a collided deploy.
_COLLIDED_DEPLOY_STDERR = (
    'Located local charm "i2639-pebble-notice", revision 0\n'
    'Deploying "repro" from local charm "i2639-pebble-notice", revision 0 on ubuntu@24.04/stable\n'
    'ERROR cannot add application "repro": application already exists: \n'
    "deploy application using an alias name:\n"
    "    juju deploy <application> <alias>\n"
)


def test_rung_zero_quotes_the_error_line_not_the_progress_line():
    run_result = RunResult(
        hypothesis_number=2639,
        branch="k8s-scratch",
        commands=[
            CommandResult(command="sudo concierge prepare -p k8s", exit_code=0, step="prepare"),
            CommandResult(command="charmcraft pack", exit_code=0, step="pack"),
            CommandResult(command="juju deploy ...", exit_code=1, stderr=_COLLIDED_DEPLOY_STDERR, step="deploy"),
        ],
        aborted_at_step="deploy",
        skipped_steps=["wait", "stimulus", "status", "debug-log"],
    )

    outcome, reason = classifier.classify(_hypothesis(), None, run_result)

    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert "cannot add application" in reason
    assert "Located local charm" not in reason


def test_rung_zero_falls_back_to_the_first_line_when_nothing_looks_like_an_error():
    run_result = RunResult(
        hypothesis_number=2639,
        branch="k8s-scratch",
        commands=[CommandResult(command="charmcraft pack", exit_code=2, stderr="something went sideways", step="pack")],
        aborted_at_step="pack",
        skipped_steps=["cleanup", "deploy"],
    )

    _outcome, reason = classifier.classify(_hypothesis(), None, run_result)

    assert "something went sideways" in reason


def test_lowercase_and_prefixed_error_lines_are_both_recognised():
    """pebble writes `error: ...`; juju writes `ERROR ...`; charmcraft writes
    a message with `error:` partway through the line."""
    for stderr, expected in [
        ("progress\nerror: invalid argument for flag `--timeout'\n", "error: invalid argument"),
        ("progress\nERROR cannot add application\n", "ERROR cannot add application"),
        ("progress\ncraft-application error: bad platform\n", "craft-application error: bad platform"),
    ]:
        run_result = RunResult(
            hypothesis_number=2639,
            branch="k8s-scratch",
            commands=[CommandResult(command="x", exit_code=1, stderr=stderr, step="pack")],
            aborted_at_step="pack",
            skipped_steps=[],
        )
        _outcome, reason = classifier.classify(_hypothesis(), None, run_result)
        assert expected in reason, stderr


def test_commands_record_elapsed_time(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=""),
    )

    result = SubprocessRunnerSeam().run(
        branch="k8s-scratch", hypothesis=_hypothesis(), surface=None, context={"charm_dir": "/w/c"}
    )

    assert result.commands
    for command in result.commands:
        assert command.elapsed_s is not None
        assert command.elapsed_s >= 0


def test_elapsed_time_survives_a_round_trip():
    restored = CommandResult.from_dict(
        {"command": "charmcraft pack", "exit_code": 0, "step": "pack", "elapsed_s": 312.5}
    )

    assert restored.elapsed_s == 312.5


def test_replayed_fixtures_have_no_elapsed_time():
    """Fixture runs have no timing to report; `None` says so rather than
    implying a measured zero."""
    restored = CommandResult.from_dict({"command": "charmcraft pack", "exit_code": 0})

    assert restored.elapsed_s is None


def test_wait_gets_a_budget_longer_than_the_timeout_inside_its_own_command():
    """`wait` polls for up to 15 minutes, so it needs its own entry in
    `STEP_TIMEOUTS_S`: under the flat 5-minute default the subprocess is
    killed at 5 and juju's own timeout is unreachable, and a merely-slow unit
    comes back as `'wait' failed (timed out): timed out after 300s`. Any step
    whose command embeds a timeout needs an outer budget larger than the
    inner one, or the inner one is dead code."""
    from seams.runner import STEP_TIMEOUTS_S, UNIT_READY_TIMEOUT_S, build_plan

    plan = build_plan(branch="k8s-scratch", hypothesis=_hypothesis(), surface=None, context={"charm_dir": "/w/c"})
    wait = next(p.command for p in plan if p.step == "wait")

    assert "15m" in wait
    assert STEP_TIMEOUTS_S["wait"] > UNIT_READY_TIMEOUT_S
