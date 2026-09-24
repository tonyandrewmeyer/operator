"""`snap install concierge` happens only when a branch is about to use it.

Measured cost of the unconditional workflow step it replaces: 14-21s on every
one of thirty dispatches, on well under a tenth of which anything reached a
concierge command, plus one outright job failure on a snap store 408 before
any harness code ran (`spike-step-5/seventh-dispatch/RESULT.md` §2.2, §5, and
`sixth-dispatch`'s 212 unused seconds before it).

Concierge is used by exactly one thing -- `sudo concierge prepare`, the first
command of `_scratch_sequence()` -- so the install goes immediately in front
of it, as a prerequisite step. That placement is what makes a failed install
an `INFRASTRUCTURE_FAILED` outcome (which composes nothing) rather than either
a red workflow step with no `stage=` line, or, worse, a verdict about the bug.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
from classifier import classify
from dry_run import build_plan
from models import (
    COMMENT_OUTCOMES,
    CommandResult,
    Hypothesis,
    MovingParts,
    Outcome,
    RunResult,
    SurfaceInference,
)
from seams.runner import INSTALL_CONCIERGE_COMMAND, PREREQUISITE_STEPS, SubprocessRunnerSeam

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "add-reproducer.yaml"


def _hypothesis(substrate: str, commands: list[str] | None = None, **moving_parts) -> Hypothesis:
    return Hypothesis(
        issue_number=9999,
        in_scope=True,
        confidence="medium",
        observed="o",
        expected="e",
        commands=["echo hello"] if commands is None else commands,
        moving_parts=MovingParts(substrate=substrate, **moving_parts),
    )


def _fake_run(calls: list[str], returncode: int = 0):
    def fake(args, **kwargs):
        calls.append(args[-1])
        return subprocess.CompletedProcess(args, returncode, stdout="", stderr="")

    return fake


# -- where the install goes ---------------------------------------------------


@pytest.mark.parametrize("branch,flag", [("k8s-scratch", "k8s"), ("lxd-scratch", "machine")])
def test_a_scratch_branch_installs_concierge_immediately_before_using_it(branch, flag):
    plan = build_plan(branch=branch, hypothesis=_hypothesis(branch.split("-")[0]), surface=None, context={})

    assert plan[0].step == "install-concierge"
    assert plan[0].command == INSTALL_CONCIERGE_COMMAND
    assert plan[1].command == f"sudo concierge prepare --juju-channel 4.0/stable -p {flag}"
    # Nothing before the install touches concierge, and nothing after the
    # install installs it again.
    concierge_uses = [i for i, step in enumerate(plan) if "concierge prepare" in step.command]
    assert concierge_uses and min(concierge_uses) == 1


def test_the_none_branch_never_installs_concierge(monkeypatch, tmp_path):
    """`none` runs the reporter's own commands in a scratch directory and
    never touches juju at all."""
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    SubprocessRunnerSeam().run(
        branch="none",
        hypothesis=_hypothesis("none", ["echo hello"]),
        surface=None,
        context={"workdir": str(tmp_path)},
    )

    assert calls
    assert not any("concierge" in call for call in calls)


def test_the_k8s_clone_branch_never_installs_concierge(monkeypatch, tmp_path):
    """`k8s-clone` runs the reporter's commands against a checkout; nothing it
    does is provisioned by concierge."""
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    SubprocessRunnerSeam().run(
        branch="k8s-clone",
        hypothesis=_hypothesis("k8s", ["uv run pytest"], ci_run_url="https://example/run/1"),
        surface=None,
        context={"workdir": str(tmp_path), "repo": "canonical/operator"},
    )

    assert calls
    assert not any("concierge" in call for call in calls)


# -- idempotence --------------------------------------------------------------


def test_the_install_is_skipped_when_concierge_is_already_on_path(tmp_path):
    """Run the real command string in a real shell, with a stub `concierge` on
    PATH and a `sudo` that fails loudly: if the guard did not short-circuit,
    this exits non-zero.

    Idempotence matters because `snap install` on an already-installed snap
    exits non-zero ("snap ... is already installed"), and as a prerequisite
    step that would abort a run on every developer box and VM that already has
    concierge."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("concierge", "#!/bin/sh\nexit 0\n"), ("sudo", "#!/bin/sh\necho 'should not run' >&2\nexit 9\n")):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    proc = subprocess.run(
        ["bash", "-c", INSTALL_CONCIERGE_COMMAND],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert "should not run" not in proc.stderr


def test_the_install_runs_when_concierge_is_absent(tmp_path):
    """The other half: with nothing on PATH the guard falls through to the
    install. `sudo` here is a stub that records rather than installs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "ran"
    sudo = bin_dir / "sudo"
    sudo.write_text(f"#!/bin/sh\necho \"$@\" > {marker}\n")
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC)

    # A PATH holding the stub `sudo` and the system tools, but no concierge.
    proc = subprocess.run(
        ["bash", "-c", INSTALL_CONCIERGE_COMMAND],
        env={**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert marker.read_text().strip() == "snap install --classic concierge"


# -- a failed install is infrastructure, not a verdict ------------------------


def test_a_failed_install_is_a_prerequisite_step():
    assert "install-concierge" in PREREQUISITE_STEPS


def test_a_failed_install_aborts_the_run_and_skips_everything_after_it(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls, returncode=1))

    result = SubprocessRunnerSeam().run(branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=None, context={})

    assert len(result.commands) == 1
    assert result.commands[0].step == "install-concierge"
    assert result.aborted_at_step == "install-concierge"
    assert result.skipped_steps[0] == "prepare"


def test_a_failed_install_is_classified_as_infrastructure_and_composes_nothing():
    """The store 408 that killed a dispatch on 2026-09-23 failed a workflow
    step, so the job was red with no `stage=` line and no artefact. Inside the
    sequence it is an outcome instead -- and one that never composes."""
    run_result = RunResult(
        hypothesis_number=9999,
        branch="k8s-scratch",
        commands=[
            CommandResult(
                command=INSTALL_CONCIERGE_COMMAND,
                exit_code=1,
                stderr='error: cannot perform the following tasks:\n- Fetch and check assertions for snap "snapd" '
                "(27738) (cannot get nonce from store: store server returned status 408)",
                step="install-concierge",
            )
        ],
        aborted_at_step="install-concierge",
        skipped_steps=["prepare", "pack", "cleanup", "deploy", "wait", "stimulus", "status", "debug-log"],
    )
    surface = SurfaceInference(
        charm_name="repro",
        pebble_service={"container": "workload", "service": "workload", "user": "_daemon_", "command": "pebble notify x"},
        expected_signal="observed notice:",
    )

    outcome, reason = classify(_hypothesis("k8s"), surface, run_result)

    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert outcome not in COMMENT_OUTCOMES
    assert "install-concierge" in reason


# -- and the workflow no longer does it unconditionally -----------------------


@pytest.mark.skipif(not WORKFLOW.exists(), reason="no workflow alongside this checkout")
def test_the_workflow_does_not_install_concierge_before_the_pipeline():
    lines = [
        line.strip()
        for line in WORKFLOW.read_text().splitlines()
        if line.strip().startswith("- run:") or line.strip().startswith("run:")
    ]
    assert not any("snap install" in line for line in lines), (
        "the workflow installs concierge for every dispatch again; the harness "
        "installs it on the branches that use it (seams/runner.py's "
        "INSTALL_CONCIERGE_COMMAND)"
    )
