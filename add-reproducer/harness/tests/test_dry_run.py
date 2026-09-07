"""Tests for the dry-run plan mode (PLAN.md Approach §4 delta): emit the
exact command sequence a branch would execute, without executing it.

Two things get pinned here that the mocked-subprocess tests in
`test_seams_runner.py` can't: that `build_plan()` (pure, no subprocess) and
`SubprocessRunnerSeam._run_scratch_branch` (executes for real) agree on the
sequence -- and that the committed `#2639` artefact
(`../spike-step-5/2639-k8s-scratch-dry-run-plan.md`) is what the generator
actually produces today, not a stale hand-edited snapshot.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import dry_run
import pytest
import runnability
import runner_stage
from models import Hypothesis, MovingParts, SurfaceInference
from seams.runner import PlannedCommand, SubprocessRunnerSeam, build_plan

FIXTURES = Path(__file__).parent.parent / "fixtures"
ARTEFACT = Path(__file__).parent.parent.parent / "spike-step-5" / "2639-k8s-scratch-dry-run-plan.md"

# A whole bracketed token, e.g. `<unit>` / `<k8s-charm-with-workload>` --
# #2639's own hand extraction's unfilled-placeholder shape. Deliberately not
# a bare `<`/`>` check: those characters also appear legitimately in shell
# redirects (`2>&1`, `/dev/null`), which a real resolved command may contain.
_PLACEHOLDER_RE = re.compile(r"<[a-zA-Z][a-zA-Z0-9_-]*>")


def _hypothesis(substrate: str, commands: list[str], **mp_kwargs) -> Hypothesis:
    return Hypothesis(
        issue_number=9999,
        in_scope=True,
        moving_parts=MovingParts(substrate=substrate, **mp_kwargs),
        commands=commands,
        expected="expected",
        observed="observed",
        confidence="medium",
    )


# --- build_plan(): "none" and "k8s-clone" are a thin pass-through ---


def test_build_plan_none_is_hypothesis_commands_verbatim():
    hyp = _hypothesis("none", ["uv venv", "uv run pytest test_x.py -v"])
    plan = build_plan(branch="none", hypothesis=hyp, surface=None, context={})
    assert [p.command for p in plan] == ["uv venv", "uv run pytest test_x.py -v"]
    assert [p.step for p in plan] == ["run[0]", "run[1]"]


def test_build_plan_k8s_clone_replaces_first_command_with_git_clone():
    hyp = _hypothesis(
        "k8s",
        ["git clone https://github.com/canonical/operator", "cd operator", "tox -e integration -- -k test_x"],
        ci_run_url="https://github.com/canonical/operator/actions/runs/1",
    )
    plan = build_plan(branch="k8s-clone", hypothesis=hyp, surface=None, context={"repo": "canonical/operator"})
    assert plan[0] == PlannedCommand("clone", "git clone https://github.com/canonical/operator")
    assert [p.command for p in plan[1:]] == ["cd operator", "tox -e integration -- -k test_x"]


# --- build_plan() vs SubprocessRunnerSeam: same sequence, one pure/one executed ---


def test_build_plan_k8s_scratch_matches_executed_sequence(monkeypatch):
    hyp = _hypothesis("k8s", [])
    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={"container": "workload", "user": "_daemon_", "command": "pebble notify x key=value"},
        expected_signal="observed notice:",
    )
    context = {"charm_dir": "charm-9999"}

    plan = build_plan(branch="k8s-scratch", hypothesis=hyp, surface=surface, context=context)

    calls: list[str] = []

    def fake_run(args, **kwargs):
        if args[:2] != ["bash", "-c"]:
            # `_observed_juju_version()`'s own `["juju", "version"]` call --
            # not part of the planned sequence being pinned here.
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        calls.append(args[2])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessRunnerSeam().run(branch="k8s-scratch", hypothesis=hyp, surface=surface, context=context)

    # Every executed subprocess call, in order, is exactly one planned step's
    # command -- the plan and the real run share one sequence builder, so
    # they can't silently diverge.
    assert calls == [p.command for p in plan]
    assert result.control is not None
    assert result.control.command == plan[-1].command
    assert plan[-1].step == "control"


def test_build_plan_k8s_scratch_prepare_flag_is_k8s_not_lxd():
    hyp = _hypothesis("k8s", [])
    plan = build_plan(branch="k8s-scratch", hypothesis=hyp, surface=None, context={})
    assert plan[0].command == "sudo concierge prepare --juju-channel 4.0/stable -p k8s"


def test_build_plan_lxd_scratch_prepare_flag_is_machine():
    """Renamed from `..._is_lxd`, which asserted a preset concierge does not
    have. Presets are crafts/dev/k8s/machine/microk8s; `-p lxd` failed with
    `unknown preset 'lxd'` on the first live run that dispatched here
    (2026-08-18). The old name and assertion encoded the bug."""
    hyp = _hypothesis("lxd", [])
    plan = build_plan(branch="lxd-scratch", hypothesis=hyp, surface=None, context={})
    assert plan[0].command == "sudo concierge prepare --juju-channel 4.0/stable -p machine"


def test_build_plan_unknown_branch_raises():
    hyp = _hypothesis("none", [])
    with pytest.raises(ValueError, match="unknown branch"):
        build_plan(branch="not-a-branch", hypothesis=hyp, surface=None, context={})


# --- the #2639 artefact: no placeholders, all runnable, matches choose_branch() ---


def test_2639_plan_routes_to_k8s_scratch():
    branch, _ = dry_run.generate_plan(2639, fixtures_dir=FIXTURES)
    assert branch == runner_stage.choose_branch(dry_run.load_hypothesis(2639, FIXTURES)) == "k8s-scratch"


def test_2639_plan_covers_the_whole_branch_including_control():
    _, plan = dry_run.generate_plan(2639, fixtures_dir=FIXTURES)
    steps = [p.step for p in plan]
    assert steps == ["prepare", "pack", "cleanup", "deploy", "wait", "stimulus", "status", "debug-log", "control"]


def test_2639_plan_has_no_unfilled_placeholders():
    # #2639's own hand extraction (fixtures/extractions/2639.json) contains
    # literal `<k8s-charm-with-workload-running-as-_daemon_>` / `<unit>`
    # placeholders -- the trap the task called out. The plan must not
    # contain any of that raw text, since it's built from `surface`, not
    # `hypothesis.commands`.
    hypothesis = dry_run.load_hypothesis(2639, FIXTURES)
    assert any(_PLACEHOLDER_RE.search(c) for c in hypothesis.commands), (
        "fixture no longer has the placeholder shape to guard against"
    )

    _, plan = dry_run.generate_plan(2639, fixtures_dir=FIXTURES)
    for step in plan:
        assert not _PLACEHOLDER_RE.search(step.command), step.command


def test_2639_plan_is_runnable():
    _, plan = dry_run.generate_plan(2639, fixtures_dir=FIXTURES)
    report = runnability.assess([step.command for step in plan])
    assert report.runnable, report.reason


def test_2639_plan_without_a_real_stimulus_omits_stimulus_and_control(tmp_path):
    # Guard against ever silently *inventing* a stimulus: when surface has no
    # resolved pebble command (the state fixtures/surface/2639.json was in
    # before this task -- command: null, per spike-step-3/FINDINGS.md's
    # "left out" note), the plan must only cover prepare/pack/deploy/status/
    # debug-log, not fabricate a stimulus/control from nothing.
    fixtures_dir = tmp_path / "fixtures"
    (fixtures_dir / "extractions").mkdir(parents=True)
    (fixtures_dir / "surface").mkdir(parents=True)
    (fixtures_dir / "extractions" / "2639.json").write_text(
        (FIXTURES / "extractions" / "2639.json").read_text()
    )
    (fixtures_dir / "surface" / "2639.json").write_text(
        """{"charm_name": "repro-i2639-pebble-notice", "pebble_service": {"container": "workload"}, "expected_signal": "observed notice:"}"""
    )
    _, plan = dry_run.generate_plan(2639, fixtures_dir=fixtures_dir)
    assert [p.step for p in plan] == ["prepare", "pack", "cleanup", "deploy", "wait", "status", "debug-log"]


# --- the committed artefact must match what the generator produces today ---


def test_committed_2639_artefact_matches_generator():
    branch, plan = dry_run.generate_plan(2639, fixtures_dir=FIXTURES)
    expected = dry_run.render_markdown(2639, branch, plan)
    assert ARTEFACT.exists(), f"expected a committed dry-run plan at {ARTEFACT}"
    assert ARTEFACT.read_text() == expected, (
        "spike-step-5/2639-k8s-scratch-dry-run-plan.md is stale -- regenerate with "
        "`uv run python dry_run.py --issue 2639 --out ../spike-step-5/2639-k8s-scratch-dry-run-plan.md`"
    )


# --- PlanNotRunnable: the gate this module exists to enforce ---


def test_generate_plan_raises_on_unrunnable_commands(tmp_path, monkeypatch):
    fixtures_dir = tmp_path / "fixtures"
    (fixtures_dir / "extractions").mkdir(parents=True)
    (fixtures_dir / "extractions" / "1.json").write_text(
        """{"in_scope": true, "moving_parts": {"substrate": "none"}, """
        """"commands": ["Run the reproduction steps described above"], "expected": "", "observed": "", "confidence": "medium"}"""
    )
    with pytest.raises(dry_run.PlanNotRunnable):
        dry_run.generate_plan(1, fixtures_dir=fixtures_dir)
