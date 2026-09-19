"""Tests for the reproduction runner seam (PLAN.md Approach §4), focused on
the `lxd-scratch` branch added to close a real pipeline crash: `choose_branch()`
could return `"lxd-scratch"` (see the real #2107 corpus extraction,
`substrate: "lxd"`, in fixtures/extractions/2107.json) while `SubprocessRunnerSeam.run()`
only dispatched `"none"`/`"k8s-scratch"`/`"k8s-clone"`, so any substrate:lxd
hypothesis hit `raise ValueError(f"unknown branch {branch!r}")` and crashed the
pipeline.

None of this exercises real juju/concierge/charmcraft -- `subprocess.run` is
monkeypatched throughout. What's pinned here is the *decisions*: the exact
command sequence and substitutions `_run_lxd_scratch` builds, that a
non-zero exit or a timeout is captured as a `CommandResult` rather than
raised, and that every branch `choose_branch()` can return is actually
dispatchable (the failure mode this file exists to make hard to reintroduce
silently).
"""

from __future__ import annotations

import os
import subprocess

import pytest
import runner_stage
import seams.runner
from models import CommandResult, Hypothesis, MovingParts, RunResult, SurfaceInference
from seams.runner import (
    CONTROL_SIGNAL_POLL_INTERVAL_S,
    CONTROL_SIGNAL_POLLS,
    CONTROL_SIGNAL_TIMEOUT_S,
    PER_COMMAND_TIMEOUT_S,
    STEP_TIMEOUTS_S,
    SubprocessRunnerSeam,
    _control_signal_command,
    _juju_track,
    _packed_ops_version,
    _resolved_ops_version,
    _unit_expr,
    _wait_command,
    as_user_in_container_command,
    build_plan,
)


def _hypothesis(substrate: str, *, juju_version: str | None = None, **mp_kwargs) -> Hypothesis:
    return Hypothesis(
        issue_number=9999,
        in_scope=True,
        moving_parts=MovingParts(substrate=substrate, juju_version=juju_version, **mp_kwargs),
        commands=[],
        expected="expected",
        observed="observed",
        confidence="medium",
    )


# The inline substitution every unit-taking step uses. Hardcoding `<app>/0`
# held only for an application's first deploy -- juju continues the unit
# sequence across remove-and-redeploy, so `wait` sat on `repro-i2639/0` while
# the live unit was `repro-i2639/4` (2026-08-21).
_UNIT = (
    "$(juju status -m concierge-lxd:testing --format=json | python3 -c "
    "'import json,sys;u=json.load(sys.stdin)[\"applications\"][\"repro-i9999\"][\"units\"];"
    "print(sorted(u,key=lambda n:int(n.split(\"/\")[1]))[0])')"
)


def _fake_run(calls: list[str], *, returncode: int = 0, stdout: str = "ok", stderr: str = ""):
    """Records every `["bash", "-c", command]` call into `calls`, same as
    before. `_run_scratch_branch` now also shells out to a bare `["juju",
    "version"]` (`_observed_juju_version()`) after the sequence itself, which
    doesn't match that shape -- answered here with a fixed version string
    rather than appended to `calls`, so every existing exact-sequence
    assertion in this file keeps meaning what it always meant."""

    def fake(args, **kwargs):
        if args[:2] == ["bash", "-c"]:
            calls.append(args[2])
            return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)
        if args == ["juju", "version"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="3.6.27-ubuntu-amd64\n", stderr="")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    return fake


# --- as_user_in_container_command: the container-optional generalisation ---


def test_as_user_in_container_command_with_container_unchanged():
    cmd = as_user_in_container_command(
        unit="repro/0", container="workload", user="_daemon_", juju_track="4", pebble_command="pebble notify x"
    )
    # No `--` on the container path -- juju forwards it into the container's
    # shell (measured live, 2026-08-18); see the dedicated test below.
    assert cmd.startswith('juju ssh --container workload repro/0 "')
    assert "/charm/container/pebble.socket" in cmd


def test_as_user_in_container_command_without_container_for_machine_units():
    # lxd/machine units have no k8s-style Pebble container -- the
    # conservative reading (seams/runner.py's docstring) is to omit
    # `--container` entirely rather than invent one.
    cmd = as_user_in_container_command(
        unit="repro/0", container=None, user="_daemon_", juju_track="4", pebble_command="pebble notify x"
    )
    assert cmd.startswith('juju ssh repro/0 -- "')
    assert "--container" not in cmd


def test_as_user_in_container_command_juju_3_6_uses_new_socket_path():
    """`spike-step-5/wallclock-substrate/RESULT.md` §6: juju 3.6.27 measured
    using the same `/charm/container/` path as juju 4, not the legacy one
    the old `{4: NEW}` map would have sent it to."""
    cmd = as_user_in_container_command(
        unit="repro/0", container="workload", user="_daemon_", juju_track="3.6", pebble_command="pebble notify x"
    )
    assert "/charm/container/pebble.socket" in cmd
    assert "/var/lib/pebble/default/.pebble.socket" not in cmd


def test_as_user_in_container_command_unmeasured_juju_falls_back_to_legacy():
    """`2.9` is the only track below 3.6 the juju snap still publishes, and
    nothing has run this harness against it -- this pins today's fallback
    continued, not a claim that 2.9 is confirmed legacy."""
    cmd = as_user_in_container_command(
        unit="repro/0", container="workload", user="_daemon_", juju_track="2.9", pebble_command="pebble notify x"
    )
    assert "/var/lib/pebble/default/.pebble.socket" in cmd


# --- _run_lxd_scratch: command sequence and substitutions ---


def test_run_lxd_scratch_builds_prepare_pack_deploy_diagnostics_sequence(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    hyp = _hypothesis("lxd")
    surface = SurfaceInference(charm_name="repro-i9999-x", pebble_service={})
    result = SubprocessRunnerSeam().run(
        branch="lxd-scratch", hypothesis=hyp, surface=surface, context={"charm_dir": "charm-9999"}
    )

    # No pebble user/command in `surface` -> no stimulus step (Approach §4
    # delta docstring: this seam does not invent an action from nothing).
    assert calls == [
        "sudo concierge prepare --juju-channel 4.0/stable -p machine",  # concierge has no `lxd` preset
        # `-o` matters: without it charmcraft writes the .charm to cwd and
        # the deploy glob below never matches (first successful pack,
        # 2026-08-18).
        "rm -f charm-9999/*.charm && charmcraft pack -p charm-9999 -o charm-9999",
        # Per-issue application name plus a tolerant pre-deploy removal: the
        # fixed "repro" collided with a previous run's application on any
        # reused substrate, and left that run's unit `active` and
        # signal-bearing for the `status` step to read (2026-08-21).
        "juju remove-application -m concierge-lxd:testing repro-i9999 --destroy-storage --no-prompt "
        ">/dev/null 2>&1; for _ in $(seq 1 60); do "
        "juju status -m concierge-lxd:testing --format=json 2>/dev/null | grep -q '\"repro-i9999\"' || break; "
        "sleep 5; done; true",
        # `-m <controller>:<model>` on every juju command: `concierge prepare`
        # switches the active controller, so without this an lxd run silently
        # retargeted the next k8s one (2026-08-21).
        "juju deploy -m concierge-lxd:testing charm-9999/*.charm repro-i9999",
        # `juju deploy` returns before the unit exists; stimulating or even
        # diagnosing before it settles reads an empty/allocating unit. Waits
        # on the *application*, because the unit index is not predictable
        # once an application has been redeployed. A `juju status` poll rather
        # than `juju wait-for`, which juju 4 does not have -- see
        # `_wait_command`.
        _wait_command("concierge-lxd:testing", "repro-i9999"),
        f"juju status -m concierge-lxd:testing {_UNIT}",
        f"juju debug-log -m concierge-lxd:testing --include {_UNIT} --replay",
    ]
    assert result.branch == "lxd-scratch"
    assert result.hypothesis_number == 9999
    assert result.control is None
    assert all(c.exit_code == 0 for c in result.commands)


def test_run_lxd_scratch_falls_back_without_charm_dir(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    result = SubprocessRunnerSeam().run(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={})

    assert calls[1] == "charmcraft pack"
    assert calls[2].startswith("juju remove-application -m concierge-lxd:testing repro-i9999 ")
    assert calls[3] == "juju deploy -m concierge-lxd:testing ./*.charm repro-i9999"
    assert result.branch == "lxd-scratch"


def test_run_lxd_scratch_includes_stimulus_when_pebble_info_present(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={
            "service": "workload",
            "command": "pebble notify canonical.com/repro/notice key=value",
            "user": "_daemon_",
        },
    )
    SubprocessRunnerSeam().run(
        branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=surface, context={"charm_dir": "charm-9999"}
    )

    # cleanup joined the sequence in 2026-08-21's per-issue-app-name fix.
    assert len(calls) == 8  # prepare, pack, cleanup, deploy, wait, stimulus, status, debug-log
    stimulus = calls[5]
    assert stimulus.startswith(f'juju ssh -m concierge-lxd:testing {_UNIT} -- "')  # no k8s --container on lxd
    assert "_daemon_" in stimulus
    assert "pebble notify canonical.com/repro/notice key=value" in stimulus


def test_run_lxd_scratch_runs_control_when_expected_signal_and_pebble_info_set(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={
            "service": "workload",
            "command": "pebble notify canonical.com/repro/notice key=value",
            "user": "_daemon_",
        },
        expected_signal="observed notice:",
    )
    result = SubprocessRunnerSeam().run(
        branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=surface, context={"charm_dir": "charm-9999"}
    )

    assert result.control is not None
    assert "root" in result.control.command
    assert "_daemon_" not in result.control.command  # control reruns as root, not the suspect user
    # The notify, then a poll for its effect -- not the notify and a bare
    # status read joined by `;`, which is what this was until 2026-09-16.
    poll = _control_signal_command("concierge-lxd:testing", _UNIT, "observed notice:")
    assert result.control.command.endswith(f"; {poll}")
    assert result.control.command[: -len(f"; {poll}")] == as_user_in_container_command(
        unit=_UNIT,
        container=None,
        user="root",
        juju_track="4",
        pebble_command="pebble notify canonical.com/repro/notice key=value",
        model="concierge-lxd:testing",
    )


@pytest.mark.parametrize(
    "surface",
    [
        None,
        SurfaceInference(charm_name="repro-i9999-x", pebble_service={}, expected_signal="observed notice:"),
    ],
)
def test_run_lxd_scratch_no_control_without_pebble_info(monkeypatch, surface):
    # expected_signal alone isn't enough to build a control: without a real
    # pebble command to rerun as a different user, there's nothing to
    # control against (Approach §4 delta docstring's conservative-reading
    # note) -- no control is invented from nothing.
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    result = SubprocessRunnerSeam().run(
        branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=surface, context={}
    )
    assert result.control is None


def test_run_lxd_scratch_juju_3_6_selects_new_pebble_socket(monkeypatch):
    """`spike-step-5/wallclock-substrate/RESULT.md` §6: juju 3.6.27 measured
    on `/charm/container/`, not the legacy path a `{4: NEW}`-only map used
    to send every non-4 pin to (this exact hypothesis, `juju_version:
    "3.6.1"`, used to assert legacy here -- that assertion was the bug)."""
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={"service": "workload", "command": "pebble notify x", "user": "_daemon_"},
    )
    SubprocessRunnerSeam().run(
        branch="lxd-scratch",
        hypothesis=_hypothesis("lxd", juju_version="3.6.1"),
        surface=surface,
        context={},
    )
    stimulus = next(c for c in calls if "pebble notify x" in c)
    assert "/charm/container/pebble.socket" in stimulus


def test_run_lxd_scratch_unmeasured_juju_falls_back_to_legacy_pebble_socket(monkeypatch):
    """`2.9` is the only sub-3.6 track the juju snap still publishes, and it
    is unmeasured -- this is the map's existing fallback continued, not a
    second confirmed data point. A `3.5` pin no longer reaches here at all:
    it resolves to the `3` track (juju 3.6.28) and gets the new path."""
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={"service": "workload", "command": "pebble notify x", "user": "_daemon_"},
    )
    SubprocessRunnerSeam().run(
        branch="lxd-scratch",
        hypothesis=_hypothesis("lxd", juju_version="2.9.60"),
        surface=surface,
        context={},
    )
    stimulus = next(c for c in calls if "pebble notify x" in c)
    assert "/var/lib/pebble/default/.pebble.socket" in stimulus


def test_run_lxd_scratch_nonzero_exit_captured_not_raised(monkeypatch):
    """A failing command is captured rather than raised -- and, since
    `prepare` is a prerequisite, the run stops there instead of executing
    the other four against a substrate that never came up (first real k8s
    run, 2026-08-18: a failed `pack` was followed by a doomed `deploy` and
    two diagnostics that exit 0 on an empty model)."""
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls, returncode=1, stderr="boom"))

    result = SubprocessRunnerSeam().run(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={})

    assert len(result.commands) == 1
    assert result.commands[0].exit_code == 1
    assert result.commands[0].stderr == "boom"
    assert result.commands[0].step == "prepare"
    assert result.aborted_at_step == "prepare"
    assert result.skipped_steps == ["pack", "cleanup", "deploy", "wait", "status", "debug-log"]


def test_run_lxd_scratch_timeout_captured_not_raised(monkeypatch):
    def raises_timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", raises_timeout)

    result = SubprocessRunnerSeam().run(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={})

    assert len(result.commands) == 1
    assert result.commands[0].exit_code == 124
    assert "timed out" in result.commands[0].stderr
    assert result.aborted_at_step == "prepare"


def test_build_steps_get_a_longer_timeout_than_the_flat_default(monkeypatch):
    """`concierge prepare` measured 13m20s cold and `charmcraft pack` blew
    the flat 5-minute budget outright on the first real run, so both would
    time out on every cold runner."""
    timeouts: dict[str, int] = {}

    def record(args, **kwargs):
        if args[:2] != ["bash", "-c"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        timeouts[args[2]] = kwargs.get("timeout")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", record)
    SubprocessRunnerSeam().run(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={})

    prepare = next(v for k, v in timeouts.items() if "concierge prepare" in k)
    pack = next(v for k, v in timeouts.items() if "charmcraft pack" in k)
    # The diagnostics `juju status`, not the `cleanup` step's poll or the
    # `$UNIT` prelude -- both of which also run `juju status`, and cleanup has
    # its own budget.
    status = next(v for k, v in timeouts.items() if k.startswith("juju status -m concierge-lxd:testing $("))
    assert prepare > 13 * 60  # the measured cold-boot figure
    assert pack > 5 * 60
    assert status == 5 * 60  # unchanged: a status call has no reason to be slow


# --- dispatch completeness: the failure mode this branch fixes ---

# Deliberately a manually-kept mirror of `choose_branch()`'s four possible
# return values (its own docstring and `RunnerSeam.run()`'s Protocol
# docstring both enumerate the same four) -- not derived from source, so a
# fifth branch added to `choose_branch()` without a matching case here won't
# be caught automatically, but any of the existing four silently losing
# their dispatch (the bug this task fixes) will be.
_BRANCH_HYPOTHESES = {
    "none": lambda: _hypothesis("none"),
    "k8s-scratch": lambda: _hypothesis("k8s", **{}),
    "lxd-scratch": lambda: _hypothesis("lxd"),
    "k8s-clone": lambda: Hypothesis(
        issue_number=9999,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s", ci_run_url="https://github.com/canonical/operator/actions/runs/1"),
        commands=["echo see the linked CI run for the failure, nothing importable here"],
        expected="expected",
        observed="observed",
        confidence="medium",
    ),
}


@pytest.mark.parametrize("branch", sorted(_BRANCH_HYPOTHESES))
def test_every_choose_branch_output_is_dispatchable(branch, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    hyp = _BRANCH_HYPOTHESES[branch]()
    assert runner_stage.choose_branch(hyp) == branch

    result = SubprocessRunnerSeam().run(branch=branch, hypothesis=hyp, surface=None, context={})
    assert result.branch == branch
    assert result.hypothesis_number == 9999


def test_pack_writes_the_charm_where_deploy_looks_for_it():
    """`charmcraft pack -p <dir>` writes to the *current working
    directory*, not `<dir>`. The deploy step globs `<dir>/*.charm`, so
    without `-o` pack exits 0, the artefact lands somewhere else entirely,
    and deploy fails with "no charm was found" -- what happened on the
    first run that ever got as far as a successful pack (2026-08-18)."""
    plan = build_plan(branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=None, context={"charm_dir": "/w/c"})
    pack = next(p.command for p in plan if p.step == "pack")
    deploy = next(p.command for p in plan if p.step == "deploy")

    assert "-o /w/c" in pack
    assert deploy.startswith("juju deploy -m concierge-k8s:testing /w/c/*.charm")


def test_k8s_deploy_supplies_the_oci_resource_render_declares():
    """`render.py` emits a `<container>-image` oci-image resource for every
    k8s scratch charm, and `juju deploy` refuses a local charm whose OCI
    resources aren't supplied ("ERROR local charm missing OCI images for:
    workload-image") -- the failure on the first run that ever reached a
    successful deploy attempt (2026-08-18)."""
    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={"container": "workload", "service": "s", "command": "pebble notify x", "user": "_daemon_"},
    )
    plan = build_plan(branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=surface, context={"charm_dir": "/w/c"})
    deploy = next(p.command for p in plan if p.step == "deploy")
    assert "--resource workload-image=" in deploy


def test_machine_deploy_supplies_no_oci_resource():
    """A machine charm declares no containers and no oci-image resources;
    passing one would be an error, not a no-op."""
    surface = SurfaceInference(charm_name="repro-i9999-x", pebble_service={"service": "s"})
    plan = build_plan(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=surface, context={"charm_dir": "/w/c"})
    deploy = next(p.command for p in plan if p.step == "deploy")
    assert "--resource" not in deploy


def test_sequence_waits_for_the_unit_before_stimulating_it():
    """`juju deploy` returns once the deployment is *requested*; the unit is
    still allocating for a minute or more. The first successful deploy
    (2026-08-18) stimulated immediately and got `ERROR container for unit
    "repro/0" is not ready yet`."""
    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={"container": "workload", "service": "s", "command": "pebble notify x", "user": "_daemon_"},
        expected_signal="observed notice:",
    )
    plan = build_plan(branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=surface, context={"charm_dir": "/w/c"})
    steps = [p.step for p in plan]

    assert "wait" in steps
    assert steps.index("deploy") < steps.index("wait") < steps.index("stimulus")
    wait = next(p.command for p in plan if p.step == "wait")
    assert "juju status -m concierge-k8s:testing repro-i9999 --format=json" in wait
    assert "juju wait-for" not in wait


def test_a_failed_stimulus_stops_the_run(monkeypatch):
    """If the one command that provokes the bug didn't run, the diagnostics
    after it describe a charm nothing was done to."""
    calls: list[str] = []

    def fail_on_stimulus(args, **kwargs):
        if args[:2] != ["bash", "-c"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        cmd = args[2]
        calls.append(cmd)
        rc = 1 if "su - _daemon_" in cmd else 0
        return subprocess.CompletedProcess(args=args, returncode=rc, stdout="", stderr="not ready yet")

    monkeypatch.setattr(subprocess, "run", fail_on_stimulus)
    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={"container": "workload", "service": "s", "command": "pebble notify x", "user": "_daemon_"},
        expected_signal="observed notice:",
    )
    result = SubprocessRunnerSeam().run(
        branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=surface, context={"charm_dir": "/w/c"}
    )

    assert result.aborted_at_step == "stimulus"
    assert "status" in result.skipped_steps
    assert result.control is None


def test_k8s_container_form_omits_the_double_dash_separator():
    """`juju ssh --container c unit -- "cmd"` forwards the `--` into the
    container's shell, which dies with `sh: 0: Illegal option --` before
    running anything. Measured against a live k8s unit (2026-08-18): the
    same command without `--` runs and records the notice."""
    cmd = as_user_in_container_command(
        unit="repro/0", container="workload", user="_daemon_", juju_track="4", pebble_command="pebble notify x"
    )
    assert cmd.startswith('juju ssh --container workload repro/0 "')
    assert " -- " not in cmd


def test_machine_form_keeps_the_double_dash_separator():
    """Plain ssh consumes `--` as the documented separator; only the k8s
    exec path leaks it."""
    cmd = as_user_in_container_command(
        unit="repro/0", container=None, user="_daemon_", juju_track="4", pebble_command="pebble notify x"
    )
    assert cmd.startswith('juju ssh repro/0 -- "')


def test_pack_clears_stale_charms_so_the_deploy_glob_stays_unambiguous():
    """The deploy step globs `<dir>/*.charm`. A second pack into the same
    work dir leaves two artefacts, the glob matches both, and juju fails
    with `ERROR unrecognized args: ["repro"]` (2026-08-18)."""
    plan = build_plan(branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=None, context={"charm_dir": "/w/c"})
    pack = next(p.command for p in plan if p.step == "pack")
    assert pack.startswith("rm -f /w/c/*.charm &&")


def test_lxd_branch_uses_a_concierge_preset_that_exists():
    """concierge 1.7.0 presets: crafts, dev, k8s, machine, microk8s. There
    is no `lxd` preset, so this branch's prepare step failed with `unknown
    preset 'lxd'` every time it ran -- undetected because the branch had
    only ever been exercised with subprocess mocked, and a mock accepts any
    preset name."""
    plan = build_plan(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={})
    prepare = next(p.command for p in plan if p.step == "prepare")
    assert prepare == "sudo concierge prepare --juju-channel 4.0/stable -p machine"


# --- moving_parts.juju_version drives the substrate, not just a socket path ---
#
# `spike-step-5/wallclock-substrate/RESULT.md` §5: the extractor's pin used to
# reach only `_pebble_socket_path()`, so `prepare` provisioned whatever
# concierge defaulted to and `#2639` -- which reproduces on 4.0.5 and not on
# 3.6.27 -- returned `DID_NOT_REPRODUCE` for a reason that was about the
# substrate, not about the bug.
#
# The real `#2639` extraction pins nothing (`fixtures/extractions/2639.json`,
# recorded LLM output -- not edited to suit this), so the corpus can only
# exercise the default. The pinned paths are proven here with synthetic
# hypotheses, the same way the classifier's unreachable rungs are.


def _prepare_command(juju_version: str | None) -> str:
    plan = build_plan(
        branch="k8s-scratch",
        hypothesis=_hypothesis("k8s", juju_version=juju_version),
        surface=None,
        context={},
    )
    return next(p.command for p in plan if p.step == "prepare")


def test_prepare_channel_defaults_when_the_extraction_pins_nothing():
    """The common case per Approach §3's step-2 finding. What matters is that
    the default is the harness's own named constant rather than concierge's,
    which is what silently changed a verdict when concierge 1.7.0 defaulted to
    `3/stable`."""
    assert _prepare_command(None) == "sudo concierge prepare --juju-channel 4.0/stable -p k8s"


def test_prepare_channel_uses_a_pinned_minor_track():
    """`3.6` is a track the juju snap publishes, so an accurate pin selects it
    rather than being widened to `3/stable`."""
    assert _prepare_command("3.6") == "sudo concierge prepare --juju-channel 3.6/stable -p k8s"


def test_prepare_channel_narrows_a_full_version_to_its_track():
    """Extractions quote whole versions (`3.6.27` is what `juju version`
    printed on the wallclock-substrate run); `3.6.27/stable` is not a channel."""
    assert _prepare_command("3.6.27") == "sudo concierge prepare --juju-channel 3.6/stable -p k8s"


def test_prepare_channel_falls_back_to_the_major_track_for_an_unpublished_minor():
    """`3.4` is a real juju version but not its own snap track. Falling back to
    `3/stable` costs one version segment of precision; passing `3.4/stable`
    costs the whole run, because `prepare` fails outright on a channel that has
    never existed."""
    assert _prepare_command("3.4") == "sudo concierge prepare --juju-channel 3/stable -p k8s"


def test_prepare_channel_falls_back_to_the_default_for_an_unparseable_pin():
    """The pin is LLM output, so it can be prose. Anything the version regex
    can't read is treated as no pin at all, not as a channel name."""
    assert _prepare_command("latest") == "sudo concierge prepare --juju-channel 4.0/stable -p k8s"


def test_pinned_juju_version_reaches_both_the_channel_and_the_pebble_socket():
    """The two consumers of the pin agree: a `3.6` hypothesis provisions a 3.6
    substrate *and* addresses the socket path measured on 3.6.27. Before the
    channel existed these could disagree -- §6's latent defect was exactly a
    pin routing the stimulus to a path the substrate did not have."""
    hyp = _hypothesis("k8s", juju_version="3.6")
    plan = build_plan(branch="k8s-scratch", hypothesis=hyp, surface=None, context={})
    prepare = next(p.command for p in plan if p.step == "prepare")
    assert "--juju-channel 3.6/stable" in prepare
    assert (
        as_user_in_container_command(
            pebble_command="pebble notify canonical.com/x/y",
            user="_daemon_",
            unit="i2639/0",
            container="workload",
            juju_track="3.6",
        ).count("/charm/container/pebble.socket")
        == 1
    )


def test_wait_also_waits_for_the_container_to_accept_exec():
    """An active unit is not an exec-able container. The first run ever to get
    past `wait` (2026-09-04, juju 4.0.14) went active after 20.5s and its
    stimulus died 0.4s later on `ERROR container "workload" not running`.
    Waiting on the charm's status cannot see that, so `wait` polls the
    operation the stimulus is about to perform."""
    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={"container": "workload", "service": "s", "command": "pebble notify x", "user": "_daemon_"},
    )
    plan = build_plan(branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=surface, context={"charm_dir": "/w/c"})
    wait = next(p.command for p in plan if p.step == "wait")

    assert "juju ssh -m concierge-k8s:testing --container workload" in wait
    assert wait.index("juju status") < wait.index("juju ssh")


def test_wait_without_a_container_stops_at_the_status_poll():
    """A machine hypothesis has no workload container to exec into, so there
    is nothing to add -- and adding a `juju ssh --container` poll there would
    fail every lxd run at `wait`."""
    plan = build_plan(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={"charm_dir": "/w/c"})
    wait = next(p.command for p in plan if p.step == "wait")

    assert "juju status" in wait
    assert "--container" not in wait


def test_pin_on_a_retired_track_gets_the_socket_of_the_juju_it_actually_installs():
    """A `3.4` pin selects `3/stable`, which the juju snap now publishes as
    **3.6.28** -- there is no 3.0-3.5 track left. Keyed on the pin, the socket
    lookup read "3.4", missed the "3.6" entry and sent the stimulus to the
    legacy path, which on that substrate fails with `cannot communicate with
    server: ... socket "/var/lib/pebble/default/.pebble.socket" not found`
    (measured on 3.6.28, `spike-step-5/substrate-2026-09-04/RESULT.md`).
    Keyed on the resolved track, both halves say `3`."""
    hyp = _hypothesis("k8s", juju_version="3.4")
    plan = build_plan(branch="k8s-scratch", hypothesis=hyp, surface=None, context={})
    prepare = next(p.command for p in plan if p.step == "prepare")

    assert "--juju-channel 3/stable" in prepare
    assert (
        "/charm/container/pebble.socket"
        in as_user_in_container_command(
            pebble_command="pebble notify canonical.com/x/y",
            user="_daemon_",
            unit="i2639/0",
            container="workload",
            juju_track=_juju_track(hyp),
        )
    )


# --- RunResult.observed_juju_version: which juju actually ran this ---


def test_run_lxd_scratch_records_the_observed_juju_version(monkeypatch):
    """`spike-step-5/wallclock-substrate/RESULT.md` §5: nothing recorded
    which juju actually produced a run's verdict. `_fake_run` answers the
    seam's own `juju version` call with a fixed string; this pins that the
    scratch branch actually captures it into `RunResult`."""
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    result = SubprocessRunnerSeam().run(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={})

    assert result.observed_juju_version == "3.6.27-ubuntu-amd64"


def test_run_k8s_scratch_records_the_observed_juju_version(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    result = SubprocessRunnerSeam().run(branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=None, context={})

    assert result.observed_juju_version == "3.6.27-ubuntu-amd64"


def test_run_lxd_scratch_juju_version_absent_is_none_not_raised(monkeypatch):
    """`juju` missing entirely (e.g. `concierge prepare` itself failed) must
    not take the whole run down for want of a version string -- every other
    seam here treats infrastructure absence as data, not a crash."""

    def fake(args, **kwargs):
        if args == ["juju", "version"]:
            raise FileNotFoundError("juju: command not found")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake)

    result = SubprocessRunnerSeam().run(branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={})

    assert result.observed_juju_version is None


def test_run_none_does_not_record_an_observed_juju_version(monkeypatch):
    """The `none` branch (ops.testing) never touches a juju substrate at
    all -- recording a host-wide `juju version` there would attribute a
    substrate to a run that didn't ask for one."""
    hyp = Hypothesis(
        issue_number=9999,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["true"],
        expected="expected",
        observed="observed",
        confidence="medium",
    )
    monkeypatch.setattr(subprocess, "run", _fake_run([]))

    result = SubprocessRunnerSeam().run(branch="none", hypothesis=hyp, surface=None, context={})

    assert result.observed_juju_version is None


# --- the control's status read: a poll, not a race ---
#
# `spike-step-5/first-dispatch/RESULT.md` §6.5. The control was
# `pebble notify ...; juju status <unit>` -- one shell command, 0.960s for
# both halves -- against a measured notify-to-visible latency of under a
# second. Every other step in the sequence that needs the cluster to catch up
# polls for it; this one read straight away. Nothing had ever lost the race,
# and nothing could have: rung 5 reads the control only when
# `expected_signal` is *absent* from the main run, and it was present on
# every run to date. The path where it is load-bearing is a genuine
# `REPRODUCED_POSITIVE_SIGNAL_ABSENT`, and there an early read turns a
# comment-worthy outcome into a silent `DID_NOT_REPRODUCE` reported as "no
# control confirmed the observer works at all".


def test_control_signal_command_polls_with_the_modules_one_idiom():
    """The poll is `_wait_command()`'s shape -- a bounded `for _ in $(seq 1
    N)` loop with a `sleep`, then a check that `echo`es and `exit 1`s -- and
    not a second polling idiom invented for this step."""
    cmd = _control_signal_command("c:testing", "app/0", "observed notice:")

    assert cmd.startswith(f"for _ in $(seq 1 {CONTROL_SIGNAL_POLLS}); do ")
    assert f"sleep {CONTROL_SIGNAL_POLL_INTERVAL_S}; done; " in cmd
    assert cmd.endswith(">&2; exit 1; }")
    assert str(CONTROL_SIGNAL_TIMEOUT_S) in cmd


def test_control_signal_command_greps_the_signal_as_a_fixed_string():
    """`expected_signal` is free text an LLM wrote, so it is data in this
    command and never shell or regex: `shlex.quote`d, and matched with
    `grep -F` so a `.` or a `*` in a signal cannot widen the match past what
    `classifier.classify()`'s rung 5 substring-matches."""
    cmd = _control_signal_command("c:testing", "app/0", "it's a *signal*")

    assert "grep -qF -- 'it'\"'\"'s a *signal*'" in cmd


def test_control_step_is_the_notify_then_the_poll(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={
            "container": "workload",
            "service": "workload",
            "command": "pebble notify canonical.com/repro/notice key=value",
            "user": "_daemon_",
        },
        expected_signal="observed notice:",
    )

    plan = build_plan(
        branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=surface, context={"charm_dir": "charm-9999"}
    )

    control = plan[-1]
    assert control.step == "control"
    unit = _unit_expr("repro-i9999", "concierge-k8s:testing")
    assert control.command.endswith(
        "; " + _control_signal_command("concierge-k8s:testing", unit, "observed notice:")
    )
    # The old shape, pinned as gone: the notify's own `;` used to be followed
    # by nothing but a single status read.
    assert not control.command.endswith(f"; juju status -m concierge-k8s:testing {unit}")


def test_control_step_budget_outlasts_its_own_poll():
    """Same rule `wait` is in `STEP_TIMEOUTS_S` for: the subprocess budget
    has to outlast the timeout embedded in the command, or the inner one is
    dead code and a slow-but-working control comes back as
    `'control' failed (timed out)` instead of as a verdict."""
    assert STEP_TIMEOUTS_S["control"] > CONTROL_SIGNAL_TIMEOUT_S
    assert STEP_TIMEOUTS_S["control"] > PER_COMMAND_TIMEOUT_S - 1  # not silently smaller than the default


def _juju_stub(tmp_path, appear_at: int) -> dict:
    """A `juju` on PATH that answers `--format=json` with a unit listing and
    every other call with a `juju status` table -- carrying the signal from
    its `appear_at`-th call onwards. `appear_at` larger than the poll count
    is a signal that never arrives."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    stub = bin_dir / "juju"
    stub.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        '  if [ "$a" = "--format=json" ]; then\n'
        """    echo '{"applications":{"repro-i9999":{"units":{"repro-i9999/0":{}}}}}'\n"""
        "    exit 0\n"
        "  fi\n"
        "done\n"
        "n=0\n"
        '[ -f "$COUNT_FILE" ] && n=$(cat "$COUNT_FILE")\n'
        'n=$((n+1)); echo "$n" > "$COUNT_FILE"\n'
        "echo 'Unit             Workload  Agent  Address    Ports  Message'\n"
        f'if [ "$n" -ge {appear_at} ]; then\n'
        "  echo 'repro-i9999/0*   active    idle   10.1.0.72         "
        "observed notice: canonical.com/repro/notice in workload'\n"
        "else\n"
        "  echo 'repro-i9999/0*   active    idle   10.1.0.72         ready'\n"
        "fi\n"
    )
    stub.chmod(0o755)
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "COUNT_FILE": str(tmp_path / "count"),
    }


def _run_poll_for_real(tmp_path, monkeypatch, *, appear_at: int, polls: int = 3):
    """Build the control's poll and actually execute it in a shell, against a
    `juju` stub. Everything else in this file mocks `subprocess.run`, which
    pins the *decision* and not the shell -- and this step's whole point is a
    loop, a `sleep`, a `grep` and an exit code, none of which a mocked
    `subprocess.run` can be wrong about. The module constants are patched
    down so a real timeout costs no wall clock: they are read when the string
    is built, not when it runs."""
    monkeypatch.setattr(seams.runner, "CONTROL_SIGNAL_POLLS", polls)
    monkeypatch.setattr(seams.runner, "CONTROL_SIGNAL_POLL_INTERVAL_S", 0)
    env = dict(os.environ, **_juju_stub(tmp_path, appear_at))
    command = seams.runner._control_signal_command(
        "c:testing", _unit_expr("repro-i9999", "c:testing"), "observed notice:"
    )
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=env)


def test_control_poll_waits_for_a_signal_that_arrives_late(tmp_path, monkeypatch):
    """The whole point: a status read that would have missed the signal on
    its first attempt now waits for it."""
    proc = _run_poll_for_real(tmp_path, monkeypatch, appear_at=3)

    assert proc.returncode == 0
    # Not just the matched line -- the full status table, because
    # `composer._control_output()` renders this text as the evidence for the
    # outcome's claim, and a reader is being asked to check it.
    assert "observed notice: canonical.com/repro/notice in workload" in proc.stdout
    assert "Workload  Agent  Address" in proc.stdout


def test_control_poll_that_times_out_fails_loudly(tmp_path, monkeypatch):
    """A poll that times out is a control that did not fire. It has to exit
    non-zero and say so, because rung 5 reads a control's success as "the
    observer demonstrably works" -- exiting 0 on a signal that never arrived
    would hand that conclusion to a run that earned the opposite one."""
    proc = _run_poll_for_real(tmp_path, monkeypatch, appear_at=99)

    assert proc.returncode != 0
    assert "timed out" in proc.stderr
    assert "observed notice:" in proc.stderr  # says which signal it waited for
    # Still prints what it did see, so the failure is diagnosable.
    assert "Workload  Agent  Address" in proc.stdout
    assert "observed notice: canonical.com/repro/notice in workload" not in proc.stdout


def test_control_poll_exit_code_and_printed_text_cannot_disagree(tmp_path, monkeypatch):
    """`exit 0` and "the printed text contains the signal" are one fact, not
    two: the tail greps the text it just printed rather than re-reading. Rung
    5 checks both, so a command where they could differ would make the two
    checks answer different questions about the same control."""
    for appear_at, expected_exit in ((2, 0), (99, 1)):
        proc = _run_poll_for_real(tmp_path / str(appear_at), monkeypatch, appear_at=appear_at)
        assert (proc.returncode == 0) is ("observed notice:" in proc.stdout)
        assert proc.returncode == expected_exit


def test_a_failed_control_is_recorded_and_aborts_nothing(monkeypatch):
    """`control` is deliberately not in `PREREQUISITE_STEPS`: a control that
    failed is evidence the classifier needs, not a reason to throw the run
    away. It is the last step in the sequence anyway, so there is nothing
    after it to protect -- what matters is that the non-zero exit reaches
    `RunResult.control` rather than being swallowed."""
    surface = SurfaceInference(
        charm_name="repro-i9999-x",
        pebble_service={
            "container": "workload",
            "service": "workload",
            "command": "pebble notify canonical.com/repro/notice key=value",
            "user": "_daemon_",
        },
        expected_signal="observed notice:",
    )

    def fake(args, **kwargs):
        if args[:2] != ["bash", "-c"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        failed = "waiting for the control signal" in args[2]
        return subprocess.CompletedProcess(
            args=args,
            returncode=1 if failed else 0,
            stdout="",
            stderr="timed out after 120s waiting for the control signal" if failed else "",
        )

    monkeypatch.setattr(subprocess, "run", fake)
    result = SubprocessRunnerSeam().run(
        branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=surface, context={"charm_dir": "charm-9999"}
    )

    assert result.control is not None
    assert result.control.exit_code == 1
    assert result.control.step == "control"
    assert result.aborted_at_step is None
    assert result.skipped_steps == []


# --- RunResult.observed_ops_version: which ops was actually packed ---
#
# `spike-step-5/first-dispatch/RESULT.md` §6.1. The scratch charm pins
# `ops~=3.8` and `charmcraft pack` resolves that from PyPI inside its own
# managed LXD instance, so the `ops` in the checked-out tree is never under
# test. The first dispatch packed `ops==3.8.2`; the fact survived only as one
# line of charmcraft's stderr that no code read.

# The shape §6.1 quotes, in the surrounding noise a real pack log carries.
_PACK_LOG = """\
Packing charm
:: Installed 34 packages in 96ms
:: + ops==3.8.2
:: + ops-scenario==8.8.0
:: + pyyaml==6.0.3
Packed repro-i9999_amd64.charm
"""


def test_packed_ops_version_reads_the_charmcraft_pack_log():
    commands = [
        CommandResult(command="sudo concierge prepare ...", exit_code=0, step="prepare"),
        CommandResult(command="charmcraft pack ...", exit_code=0, stderr=_PACK_LOG, step="pack"),
    ]
    assert _packed_ops_version(commands) == "3.8.2"


def test_packed_ops_version_is_not_fooled_by_a_neighbouring_package():
    """Two different near-misses. `ops-scenario==`/`ops-testing==` are the
    lines that actually sit next to it in a real pack log, and requiring the
    literal `ops==` is enough for those. `charmops==` is the one that needs
    the lookbehind: it contains `ops==` as a substring, so without it the
    wrong package's version is reported as the packed `ops`."""
    log = ":: + charmops==1.2.3\n:: + ops-scenario==8.8.0\n:: + ops-testing==3.8.2\n"
    commands = [CommandResult(command="charmcraft pack", exit_code=0, stderr=log, step="pack")]
    assert _packed_ops_version(commands) is None


def test_packed_ops_version_takes_the_installed_version_not_the_replaced_one():
    """uv prints a package's removal before its addition, so the last `ops==`
    in the log is the one that ended up in the charm."""
    log = ":: - ops==3.8.1\n:: + ops==3.8.2\n"
    commands = [CommandResult(command="charmcraft pack", exit_code=0, stdout=log, step="pack")]
    assert _packed_ops_version(commands) == "3.8.2"


@pytest.mark.parametrize(
    "commands",
    [
        pytest.param([], id="no commands at all"),
        pytest.param(
            [CommandResult(command="git clone ...", exit_code=0, stdout="x", step="clone")],
            id="a branch with no pack step",
        ),
        pytest.param(
            [CommandResult(command="charmcraft pack", exit_code=0, stdout="", stderr="", step="pack")],
            id="a pack step that captured nothing",
        ),
        pytest.param(
            [CommandResult(command="charmcraft pack", exit_code=1, stderr="Failed to pack.", step="pack")],
            id="a pack that failed",
        ),
    ],
)
def test_packed_ops_version_fails_soft(commands):
    """Every way of not finding a version gives `None`, never an exception:
    same discipline as `_observed_juju_version()`, and for the same reason --
    a version string nobody could read must not take down a run that
    otherwise completed."""
    assert _packed_ops_version(commands) is None


def test_run_k8s_scratch_records_the_packed_ops_version(monkeypatch):
    def fake(args, **kwargs):
        if args[:2] != ["bash", "-c"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="3.6.27-ubuntu-amd64\n", stderr="")
        stderr = _PACK_LOG if "charmcraft pack" in args[2] else ""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake)
    result = SubprocessRunnerSeam().run(
        branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=None, context={"charm_dir": "charm-9999"}
    )

    assert result.observed_ops_version == "3.8.2"


def test_run_lxd_scratch_records_the_packed_ops_version(monkeypatch):
    def fake(args, **kwargs):
        if args[:2] != ["bash", "-c"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="3.6.27-ubuntu-amd64\n", stderr="")
        stderr = _PACK_LOG if "charmcraft pack" in args[2] else ""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake)
    result = SubprocessRunnerSeam().run(
        branch="lxd-scratch", hypothesis=_hypothesis("lxd"), surface=None, context={"charm_dir": "charm-9999"}
    )

    assert result.observed_ops_version == "3.8.2"


def test_run_k8s_scratch_ops_version_absent_is_none_not_raised(monkeypatch):
    """A pack log with nothing resolvable in it -- charmcraft's output format
    changing, or a cached build that prints no venv listing -- is `None`, the
    same way a missing `juju version` is."""
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    result = SubprocessRunnerSeam().run(
        branch="k8s-scratch", hypothesis=_hypothesis("k8s"), surface=None, context={"charm_dir": "charm-9999"}
    )

    assert result.observed_ops_version is None


def test_run_k8s_clone_does_not_record_a_packed_ops_version(monkeypatch):
    """`k8s-clone` runs the reporter's own commands against a checkout and
    packs no scratch charm, so there is no pack log to read -- same scoping
    as `observed_juju_version`, which that branch also leaves unset."""
    hyp = Hypothesis(
        issue_number=9999,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s", ci_run_url="https://github.com/canonical/operator/actions/runs/1"),
        commands=["git clone x", "charmcraft pack"],
        expected="expected",
        observed="observed",
        confidence="medium",
    )

    def fake(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=_PACK_LOG, stderr="")

    monkeypatch.setattr(subprocess, "run", fake)
    result = SubprocessRunnerSeam().run(
        branch="k8s-clone", hypothesis=hyp, surface=None, context={"repo": "canonical/operator"}
    )

    assert result.observed_ops_version is None


# --- _KNOWN_JUJU_TRACKS: every entry has to be buildable into a channel ---
#
# `spike-step-5/first-dispatch/RESULT.md` §3 measured the store and found that
# `4.1`, alone of the five entries the tuple then carried, publishes no
# `stable` risk -- so `_juju_channel()` built `4.1/stable` for any `4.1*` pin
# and `prepare` failed on a channel that has never existed. The tuple's own
# docstring claims the opposite property, and nothing asserted it.


def test_every_known_juju_track_has_a_populated_stable_risk():
    """The invariant the tuple is *for*. `_juju_channel()` only ever appends
    `/stable`, so an entry whose `stable` risk is empty is worse than an
    absent one: absent falls back safely, present is built into a 404."""
    assert set(seams.runner._KNOWN_JUJU_TRACKS) == set(seams.runner._KNOWN_JUJU_TRACK_STABLE_VERSIONS)
    for track, version in seams.runner._KNOWN_JUJU_TRACK_STABLE_VERSIONS.items():
        assert version, f"{track} is listed with no stable version behind it"


@pytest.mark.parametrize("pin", ["4.1", "4.1.0", "4.1-beta1"])
def test_prepare_channel_falls_back_for_a_track_with_no_stable_risk(pin):
    """`4.1` exists as a snap track and publishes `beta` and `edge` only
    (store, 2026-09-17). Falling back to the default costs one version
    segment; `4.1/stable` costs the whole runner slot and says nothing about
    the bug -- `prepare` is `commands[0]`, so a non-zero exit lands on rung 0
    (`INFRASTRUCTURE_FAILED`), which composes nothing."""
    assert _prepare_command(pin) == "sudo concierge prepare --juju-channel 4.0/stable -p k8s"


def test_a_four_one_pin_gets_the_socket_of_the_juju_it_actually_installs():
    """The other half of `test_pin_on_a_retired_track_...`: dropping `4.1`
    from the tuple moves the *socket* lookup too, because it is keyed on the
    resolved track rather than on the pin. Both halves now say `4.0`."""
    hyp = _hypothesis("k8s", juju_version="4.1")
    assert _juju_track(hyp) == "4.0"
    assert (
        as_user_in_container_command(
            pebble_command="pebble notify canonical.com/x/y",
            user="_daemon_",
            unit="i2639/0",
            container="workload",
            juju_track=_juju_track(hyp),
        ).count("/charm/container/pebble.socket")
        == 1
    )


# --- resolve_symbol: absence of evidence is not evidence of absence ---
#
# `runner_stage.is_stale()`'s second trigger skips a hypothesis outright and
# composes nothing, with no downstream appeal, so a `False` here has to mean
# "this API surface is gone". It used to also mean "nothing here imports",
# which on a GHA runner is true of every `ops.*` anchor -- the workflow runs
# the harness under `uv run` in a venv holding pyyaml and pytest and no `ops`.
# These run the real probe in a real subprocess: the whole behaviour is an
# interpreter's import machinery and an exit code, and a mocked
# `subprocess.run` can say nothing about either.


def test_resolve_symbol_resolves_a_real_dotted_path():
    """A module prefix plus attribute segments, which is the shape every
    anchor has. `os.path` imports, `join` is on it."""
    assert SubprocessRunnerSeam().resolve_symbol("os.path.join", {})


def test_resolve_symbol_reports_a_symbol_that_is_gone():
    """The one case that legitimately trips the gate: the package is right
    here and the attribute path is not on it."""
    assert not SubprocessRunnerSeam().resolve_symbol("os.path.no_such_function_here", {})


def test_resolve_symbol_abstains_when_no_prefix_imports_at_all():
    """Nothing importable means nothing was asked, not that the symbol has
    gone away. This is the GHA case: `ops` is not in the harness's venv, so
    reported as absence it skipped every anchor-carrying hypothesis before it
    could reach a runner."""
    assert SubprocessRunnerSeam().resolve_symbol("not_a_real_package_2f9c1a.mod.Thing", {})


def test_resolve_symbol_abstains_on_a_partially_importable_prefix():
    """`os` imports and `os.not_a_module` does not, so no prefix longer than
    `os` is importable -- but `os` itself is, and `not_a_module` is not an
    attribute of it. That is a real absence, not an abstention: the
    distinction is about whether anything answered, not about how many
    segments matched."""
    assert not SubprocessRunnerSeam().resolve_symbol("os.not_a_module_or_attribute", {})


def test_resolve_symbol_abstains_when_the_probe_cannot_run(monkeypatch):
    """No interpreter on PATH, or one that hung. Treated as data rather than
    as a crash, the same way `_observed_juju_version()` treats a missing
    `juju` -- and abstaining rather than skipping, because a gate that cannot
    look has learned nothing."""

    def boom(args, **kwargs):
        raise OSError("no python3 here")

    monkeypatch.setattr(subprocess, "run", boom)
    assert SubprocessRunnerSeam().resolve_symbol("ops.model.Model.get_relation", {})

    def hang(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=30)

    monkeypatch.setattr(subprocess, "run", hang)
    assert SubprocessRunnerSeam().resolve_symbol("ops.model.Model.get_relation", {})


def test_resolve_symbol_reads_only_the_absent_exit_code():
    """`SystemExit(2)` from the probe is "could not tell", and the wrapper has
    to read the code rather than `returncode == 0`. Pinned separately from the
    probe itself because the two halves are what got out of step: the old
    probe collapsed both non-resolving cases onto exit 1."""
    seam = SubprocessRunnerSeam()
    codes = {seam._SYMBOL_RESOLVED, seam._SYMBOL_ABSENT, seam._SYMBOL_UNKNOWABLE}
    assert len(codes) == 3

    for code in codes:

        def fake(args, _code=code, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=_code, stdout=b"", stderr=b"")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(subprocess, "run", fake)
            assert SubprocessRunnerSeam().resolve_symbol("a.b.C", {}) is (code != seam._SYMBOL_ABSENT)


def test_resolve_symbol_abstains_when_the_package_raises_on_import(tmp_path, monkeypatch):
    """A package whose import blows up (a missing transitive dependency is the
    common one -- `import ops` from the repo root dies on `import websocket`)
    has told us nothing about the symbol. It matters that the probe catches
    it: an unhandled exception in `python3 -c` exits 1, which is the code that
    means "gone"."""
    (tmp_path / "explodes_on_import_4b7e.py").write_text("raise RuntimeError('nope')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    assert SubprocessRunnerSeam().resolve_symbol("explodes_on_import_4b7e.Thing", {})


# --- RunResult.observed_ops_version on the `none` branch ---
#
# `spike-step-5/second-dispatch/RESULT.md` §6.2, closed in
# `fourth-dispatch/RESULT.md` §1. The field was scoped to the branches that
# `charmcraft pack`, on the reasoning that only they resolve an `ops`. The
# `none` branch resolves one too -- from PyPI, via the extraction's own `uv
# add 'ops[testing]'` -- and prints the version into output the harness was
# already capturing and already rendering into the composed comment. So a
# comment quoted the answer four lines above a versions line that declined to
# state it.

# Verbatim from run 35219949266's composed comment (`second-dispatch/
# RESULT.md` §5.1), which is the only recording of this branch's real output.
_UV_ADD_LOG = """\
Resolved 16 packages in 2ms
Installed 5 packages in 2ms
 + opentelemetry-api==1.44.0
 + ops==3.8.2
 + ops-scenario==8.8.2
 + typing-extensions==4.16.0
 + websocket-client==1.9.2
"""

_2045_COMMANDS = [
    CommandResult(command="uv init --bare .", exit_code=0, stdout="Initialized project `issue-2045`"),
    CommandResult(command="uv add 'ops[testing]'", exit_code=0, stdout=_UV_ADD_LOG),
    CommandResult(command="cat > test_cwd.py << 'PYEOF'\nimport ops\nPYEOF", exit_code=0),
    CommandResult(command="uv run pytest test_cwd.py -v", exit_code=1, stdout="1 failed"),
]


def test_the_pack_reader_cannot_see_the_none_branchs_ops_and_that_is_the_defect():
    """The measured shape, through the reader that was wired up for it: the
    `none` branch labels no steps at all, so `step == "pack"` matches nothing
    and the version in the log two commands earlier is invisible. This is
    `second-dispatch` §6.2's `null` reproduced, not described."""
    assert all(c.step is None for c in _2045_COMMANDS)
    assert _packed_ops_version(_2045_COMMANDS) is None


def test_resolved_ops_version_reads_the_none_branchs_own_install_log():
    assert _resolved_ops_version(_2045_COMMANDS) == "3.8.2"


def test_run_none_records_the_ops_it_resolved(monkeypatch):
    """End to end through the seam, on the four commands `#2045` really ran."""
    hyp = Hypothesis(
        issue_number=2045,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=[c.command for c in _2045_COMMANDS],
        expected="expected",
        observed="observed",
        confidence="medium",
    )

    def fake(args, **kwargs):
        stdout = _UV_ADD_LOG if "uv add" in args[2] else "ok"
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake)

    result = SubprocessRunnerSeam().run(branch="none", hypothesis=hyp, surface=None, context={})

    assert result.observed_ops_version == "3.8.2"
    # Unchanged, and the reason the two fields are not symmetric: this branch
    # resolves a library but never provisions a substrate.
    assert result.observed_juju_version is None


def test_resolved_ops_version_only_reads_install_commands():
    """A pytest failure dump can quote a requirements line or a traceback
    frame carrying `ops==`. Reading that as the resolved version would
    disclose a number nothing installed, so only an installer's own output
    counts as evidence about what is importable."""
    commands = [
        CommandResult(command="uv run pytest test_cwd.py -v", exit_code=1, stdout="E   assert 'ops==9.9.9' in reqs"),
    ]
    assert _resolved_ops_version(commands) is None


def test_resolved_ops_version_prefers_the_installed_version_over_the_replaced_one():
    log = "Uninstalled 1 package\n - ops==3.8.1\nInstalled 1 package\n + ops==3.8.2\n"
    commands = [CommandResult(command="uv pip install 'ops[testing]'", exit_code=0, stdout=log)]
    assert _resolved_ops_version(commands) == "3.8.2"


def test_resolved_ops_version_is_not_fooled_by_a_neighbouring_package():
    """The same lookbehind `_packed_ops_version()` needs, on the other
    reader: `charmops==` contains `ops==` as a substring."""
    log = " + charmops==1.2.3\n + ops-scenario==8.8.2\n + ops-testing==3.8.2\n"
    commands = [CommandResult(command="uv add 'ops[testing]'", exit_code=0, stdout=log)]
    assert _resolved_ops_version(commands) is None


@pytest.mark.parametrize(
    "commands",
    [
        pytest.param([], id="no commands at all"),
        pytest.param(
            [CommandResult(command="uv run pytest -v", exit_code=0, stdout="1 passed")],
            id="commands that install nothing",
        ),
        pytest.param(
            [CommandResult(command="uv add 'ops[testing]'", exit_code=0, stdout="", stderr="")],
            id="an install that captured nothing",
        ),
        pytest.param(
            [CommandResult(command="uv add 'ops[testing]'", exit_code=1, stderr="No solution found")],
            id="an install that failed",
        ),
    ],
)
def test_resolved_ops_version_fails_soft(commands):
    """Same discipline as `_packed_ops_version()`: every way of not finding a
    version is `None`, never an exception."""
    assert _resolved_ops_version(commands) is None


def test_run_k8s_clone_records_no_ops_version(monkeypatch):
    """`k8s-clone` runs the reporter's own commands against a checkout and
    resolves nothing of its own, so it keeps the `None` the widening does not
    touch."""
    hyp = Hypothesis(
        issue_number=2484,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s"),
        commands=["uv add 'ops[testing]'"],
        expected="expected",
        observed="observed",
        confidence="medium",
    )
    monkeypatch.setattr(subprocess, "run", _fake_run([], stdout=_UV_ADD_LOG))

    result = SubprocessRunnerSeam().run(branch="k8s-clone", hypothesis=hyp, surface=None, context={})

    assert result.observed_ops_version is None
