"""`k8s-clone` stops at a failed prerequisite, like the scratch branches.

The scratch branches record `CommandResult.step` and
`RunResult.aborted_at_step`/`skipped_steps`, and stop at the first failed
prerequisite, with classifier rung 0 returning `INFRASTRUCTURE_FAILED` for the
result. If `_run_k8s_clone` kept its own loop without that, a failed
`git clone` would leave every later command running in an empty directory and
the classifier -- which reaches rung 0 only via `aborted_at_step` -- would
read their exit codes as evidence about the bug.

That matters because `k8s-clone` is the branch that any issue linking a CI
run, with no self-contained snippet, routes to.
"""

from __future__ import annotations

import subprocess

import classifier
from models import COMMENT_OUTCOMES, Hypothesis, MovingParts, Outcome
from seams.runner import SubprocessRunnerSeam, build_plan


def _clone_hypothesis() -> Hypothesis:
    return Hypothesis(
        issue_number=1329,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s", ci_run_url="https://github.com/canonical/operator/actions/runs/1"),
        commands=["git clone ignored-first-entry", "uv run pytest -k test_exec_timeout", "echo after"],
        expected="expected",
        observed="observed",
        confidence="medium",
    )


def _run_with(monkeypatch, *, clone_returncode: int):
    calls: list[str] = []

    def fake(args, **kwargs):
        command = args[2]
        calls.append(command)
        returncode = clone_returncode if command.startswith("git clone") else 0
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout="", stderr="fatal: repository not found" if returncode else ""
        )

    monkeypatch.setattr(subprocess, "run", fake)
    result = SubprocessRunnerSeam().run(
        branch="k8s-clone",
        hypothesis=_clone_hypothesis(),
        surface=None,
        context={"repo": "canonical/operator", "workdir": "/w"},
    )
    return calls, result


def test_failed_clone_stops_the_run(monkeypatch):
    calls, result = _run_with(monkeypatch, clone_returncode=128)

    assert calls == ["git clone https://github.com/canonical/operator"]
    assert result.aborted_at_step == "clone"
    assert result.skipped_steps == ["run[1]", "run[2]"]
    assert result.commands[0].step == "clone"


def test_failed_clone_reaches_rung_zero_not_a_verdict_about_the_bug(monkeypatch):
    _calls, result = _run_with(monkeypatch, clone_returncode=128)

    outcome, reason = classifier.classify(_clone_hypothesis(), None, result)

    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert outcome not in COMMENT_OUTCOMES
    assert "clone" in reason


def test_successful_clone_runs_the_remaining_commands(monkeypatch):
    calls, result = _run_with(monkeypatch, clone_returncode=0)

    assert calls == [
        "git clone https://github.com/canonical/operator",
        "uv run pytest -k test_exec_timeout",
        "echo after",
    ]
    assert result.aborted_at_step is None
    assert [c.step for c in result.commands] == ["clone", "run[1]", "run[2]"]


def test_plan_and_run_share_one_sequence(monkeypatch):
    """The reason `_clone_sequence()` exists: if `build_plan()` and the real
    runner each built this branch's commands separately, a reviewed plan and
    a real run could diverge."""
    calls, _result = _run_with(monkeypatch, clone_returncode=0)
    plan = build_plan(
        branch="k8s-clone",
        hypothesis=_clone_hypothesis(),
        surface=None,
        context={"repo": "canonical/operator", "workdir": "/w"},
    )

    assert [p.command for p in plan] == calls


def test_clone_url_is_the_issue_repo_not_the_ci_run_repo():
    """`choose_branch()` routes on *any* GitHub Actions URL in the body, and
    `#1329`'s belongs to `canonical/mysql-k8s-operator`. Routing is affected;
    what gets cloned must not be."""
    hypothesis = _clone_hypothesis()
    hypothesis.moving_parts.ci_run_url = "https://github.com/canonical/mysql-k8s-operator/actions/runs/10500119558"
    plan = build_plan(branch="k8s-clone", hypothesis=hypothesis, surface=None, context={"repo": "canonical/operator"})

    assert plan[0].command == "git clone https://github.com/canonical/operator"
