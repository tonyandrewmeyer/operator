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

import subprocess

import pytest
import runner_stage
from models import CommandResult, Hypothesis, MovingParts, RunResult, SurfaceInference
from seams.runner import (
    PER_COMMAND_TIMEOUT_S,
    SubprocessRunnerSeam,
    _juju_track,
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
    assert result.control.command.endswith(f"; juju status -m concierge-lxd:testing {_UNIT}")


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
