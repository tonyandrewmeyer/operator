"""Scratch runs must not collide with, or read, a previous run's application.

Every `k8s-scratch`/`lxd-scratch` run used to deploy under the fixed name
`repro` into the same juju model, with no teardown. On a reused substrate the
second run to reach `deploy` died with `ERROR cannot add application "repro":
application already exists`, and the previous run's unit was left `active`
with a signal-bearing status message that the `status` step would read as
evidence about this run's bug. `spike-step-5/gate-substrate/RESULT.md` §6.

No fixture run could reach this: it needs two scratch runs to reach `deploy`
on one substrate, which had never happened before 2026-08-21.
"""

from __future__ import annotations

import subprocess

from models import Hypothesis, MovingParts, SurfaceInference
from seams.runner import SubprocessRunnerSeam, build_plan


def _hypothesis(issue_number: int) -> Hypothesis:
    return Hypothesis(
        issue_number=issue_number,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s"),
        commands=[],
        expected="expected",
        observed="observed",
        confidence="medium",
    )


_SURFACE = SurfaceInference(
    charm_name="repro-i2639-pebble-notice",
    pebble_service={"container": "workload", "service": "s", "command": "pebble notify x", "user": "_daemon_"},
    expected_signal="observed notice:",
)


def _plan(issue_number: int):
    return build_plan(
        branch="k8s-scratch",
        hypothesis=_hypothesis(issue_number),
        surface=_SURFACE,
        context={"charm_dir": "/w/c"},
    )


def test_application_name_is_per_issue():
    deploy_2639 = next(p.command for p in _plan(2639) if p.step == "deploy")
    deploy_1329 = next(p.command for p in _plan(1329) if p.step == "deploy")

    assert deploy_2639.endswith("repro-i2639 --resource workload-image=ubuntu:24.04")
    assert deploy_1329.endswith("repro-i1329 --resource workload-image=ubuntu:24.04")


def test_every_step_addresses_the_same_application():
    """A name used by `deploy` but not by `status` would read a *different*
    application -- which is the failure this fix exists to prevent, inverted."""
    commands = " ".join(p.command for p in _plan(2639))

    assert "repro-i2639" in commands
    # No bare `repro` application left addressed anywhere.
    assert " repro " not in commands
    assert " repro/0" not in commands


def test_no_step_hardcodes_a_unit_index():
    """juju continues an application's unit sequence across
    remove-and-redeploy, so `<app>/0` is only right on a first deploy. The
    `cleanup` step above makes redeploy the normal case; on 2026-08-21 the
    live unit was `repro-i2639/4` while `wait` sat on `/0` until it timed
    out, and reported that as an infrastructure failure about a charm that
    had deployed perfectly well."""
    for planned in _plan(2639):
        assert "repro-i2639/0" not in planned.command, planned.step

    # Each unit-taking step resolves the name at execution instead.
    for step in ("stimulus", "status", "debug-log", "control"):
        command = next(p.command for p in _plan(2639) if p.step == step)
        assert '["applications"]["repro-i2639"]["units"]' in command

    # And `wait` needs no unit at all.
    wait = next(p.command for p in _plan(2639) if p.step == "wait")
    assert "repro-i2639/0" not in wait


def test_every_juju_command_names_its_controller_and_model():
    """`concierge prepare` switches the active controller. A run whose
    extraction came back `substrate: lxd` bootstraps `concierge-lxd` and makes
    it current, so the next run -- however clearly `substrate: k8s` -- packed a
    k8s charm and deployed it to LXD. Seen on 2026-08-21, where a `k8s-scratch`
    run's `juju status` reported `Controller: concierge-lxd`."""
    for planned in _plan(2639):
        for fragment in planned.command.split(";"):
            if "juju " not in fragment:
                continue
            assert "-m concierge-k8s:testing" in fragment, (planned.step, fragment)


def test_the_lxd_branch_targets_the_lxd_controller():
    from models import SurfaceInference as _S

    plan = build_plan(
        branch="lxd-scratch", hypothesis=_hypothesis(2639), surface=_S(charm_name="x"), context={"charm_dir": "/w/c"}
    )
    deploy = next(p.command for p in plan if p.step == "deploy")

    assert "-m concierge-lxd:testing" in deploy


def test_cleanup_precedes_deploy_and_is_not_a_prerequisite():
    steps = [p.step for p in _plan(2639)]

    assert steps.index("cleanup") < steps.index("deploy")

    from seams.runner import PREREQUISITE_STEPS

    # Removing an application that was never deployed is the normal case.
    assert "cleanup" not in PREREQUISITE_STEPS


def test_cleanup_always_exits_zero_even_when_nothing_to_remove(monkeypatch):
    calls: list[str] = []

    def fake(args, **kwargs):
        if args[:2] != ["bash", "-c"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        command = args[2]
        calls.append(command)
        # `juju remove-application` on an absent application exits non-zero;
        # the step swallows that, so the run must continue.
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake)
    result = SubprocessRunnerSeam().run(
        branch="k8s-scratch", hypothesis=_hypothesis(2639), surface=_SURFACE, context={"charm_dir": "/w/c"}
    )

    cleanup = next(c for c in result.commands if c.step == "cleanup")
    assert cleanup.command.startswith(
        "juju remove-application -m concierge-k8s:testing repro-i2639 --destroy-storage --no-prompt"
    )
    assert cleanup.command.rstrip().endswith("true")  # never propagates a failure
    assert result.aborted_at_step is None
