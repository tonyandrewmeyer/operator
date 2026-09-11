"""Runs that are not evidence about the bug must not be scored as though
they were.

All three cases here are lifted from the first real end-to-end run against
a live k8s substrate (2026-08-18, multipass `repro-2639` + concierge/juju
3.6.27/charmcraft 4.3.1), where every one of them fired:

- `charmcraft pack` failed on a base/platforms conflict, `juju deploy` then
  failed for want of a `.charm`, and `juju status`/`juju debug-log` both
  exited **0** against an empty model. The classifier returned
  `did_not_reproduce` -- a verdict about a bug nothing had tested.
- On the following run, `pack` blew the flat 5-minute per-command timeout
  (exit 124) fetching its LXD build base, producing the same shape.
- Live surface inference returned `pebble_service.command: null` on 3 of 3
  runs, so `_scratch_sequence()` built neither stimulus nor control and the
  run "reproduced nothing" without ever poking the charm.
"""

import runner_stage
from classifier import classify
from models import (
    COMMENT_OUTCOMES,
    CommandResult,
    Hypothesis,
    MovingParts,
    Outcome,
    RunResult,
    SurfaceInference,
)


def _hypothesis():
    return Hypothesis(
        issue_number=2639,
        in_scope=True,
        confidence="medium",
        observed='Custom notice from "_daemon_" never reaches the charm',
        expected="the charm receives a pebble-custom-notice event",
        commands=[],
        moving_parts=MovingParts(substrate="k8s"),
    )


def _surface(**overrides):
    pebble = {"container": "workload", "service": "workload", "user": "_daemon_", "command": "pebble notify x"}
    pebble.update(overrides.pop("pebble_service", {}))
    return SurfaceInference(
        charm_name="repro",
        relation={},
        storage={},
        pebble_service=pebble,
        ops_api_surface="pebble-custom-notice",
        expected_signal="Custom notice received",
        **overrides,
    )


def _aborted_run(step="pack", exit_code=1, stderr="Platform 'ubuntu@24.04:amd64' declares a base"):
    """The real shape: prerequisite fails, everything after it is skipped."""
    return RunResult(
        hypothesis_number=2639,
        branch="k8s-scratch",
        commands=[
            CommandResult(command="sudo concierge prepare -p k8s", exit_code=0, step="prepare"),
            CommandResult(command="charmcraft pack -p /w/charm-2639", exit_code=exit_code, stderr=stderr, step=step),
        ],
        aborted_at_step=step,
        skipped_steps=["deploy", "stimulus", "status", "debug-log", "control"],
    )


def test_failed_pack_is_infrastructure_failure_not_a_verdict():
    outcome, reason = classify(_hypothesis(), _surface(), _aborted_run())
    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert "never ran" in reason
    assert "pack" in reason


def test_infrastructure_failure_never_composes_a_comment():
    outcome, _ = classify(_hypothesis(), _surface(), _aborted_run())
    assert outcome not in COMMENT_OUTCOMES


def test_timed_out_pack_is_reported_as_a_timeout():
    outcome, reason = classify(
        _hypothesis(), _surface(), _aborted_run(exit_code=124, stderr="timed out after 1200s")
    )
    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert "timed out" in reason


def test_a_run_that_never_deployed_cannot_reach_the_positive_signal_absent_rung():
    """The dangerous case: `reproduced_positive_signal_absent` composes a
    comment. Before rung 0 existed, a failed pack plus a control that also
    failed to do anything landed there and asserted a reproduction."""
    run = _aborted_run()
    run.control = CommandResult(command="juju ssh ... root", exit_code=0, stdout="", step="control")
    outcome, _ = classify(_hypothesis(), _surface(), run)
    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert outcome is not Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT


def test_a_clean_run_is_still_classified_normally():
    """Rung 0 must not swallow runs that actually happened."""
    run = RunResult(
        hypothesis_number=2639,
        branch="k8s-scratch",
        commands=[
            CommandResult(command="sudo concierge prepare -p k8s", exit_code=0, step="prepare"),
            CommandResult(command="charmcraft pack", exit_code=0, step="pack"),
            CommandResult(command="juju deploy", exit_code=0, step="deploy"),
            CommandResult(command="juju ssh ... _daemon_", exit_code=0, step="stimulus"),
            CommandResult(command="juju status repro/0", exit_code=0, stdout="active", step="status"),
        ],
    )
    outcome, _ = classify(_hypothesis(), _surface(), run)
    assert outcome is not Outcome.INFRASTRUCTURE_FAILED


# -- the no-stimulus gate ------------------------------------------------


def test_null_pebble_command_is_gated_before_the_substrate_spend():
    """`pebble_service.command: null` -- what live inference actually
    returned for #2639, 3/3 runs."""
    ok, reason = runner_stage.check_stimulus("k8s-scratch", _surface(pebble_service={"command": None}))
    assert not ok
    assert "command" in reason


def test_missing_pebble_user_is_gated_too():
    ok, reason = runner_stage.check_stimulus("k8s-scratch", _surface(pebble_service={"user": None}))
    assert not ok
    assert "user" in reason


def test_lxd_scratch_is_gated_on_the_same_terms():
    ok, _ = runner_stage.check_stimulus("lxd-scratch", _surface(pebble_service={"command": None}))
    assert not ok


def test_a_complete_surface_passes_the_gate():
    ok, reason = runner_stage.check_stimulus("k8s-scratch", _surface())
    assert ok
    assert reason is None


def test_non_scratch_branches_are_not_gated():
    """`none`/`k8s-clone` build their sequence from `commands[]`, and have
    no pebble stimulus to be missing."""
    for branch in ("none", "k8s-clone"):
        ok, _ = runner_stage.check_stimulus(branch, None)
        assert ok


def test_no_stimulus_outcome_is_silent():
    assert Outcome.UNRUNNABLE_NO_STIMULUS not in COMMENT_OUTCOMES


# -- the gate is keyed on expected_signal, not on pebble_service alone ----


def test_deploy_only_run_with_no_expected_signal_is_not_gated():
    """#2107's real shape: a machine charm that errors during
    update-status, `pebble_service: {}`, `expected_signal: null`. Nothing
    is promised, so nothing is missing -- the deploy *is* the experiment,
    and only the positive-signal-absent rung (which needs
    `expected_signal`) could misread it."""
    surface = SurfaceInference(
        charm_name="repro-i2107-machine-id",
        pebble_service={},
        ops_api_surface="update-status",
        expected_signal=None,
    )
    ok, reason = runner_stage.check_stimulus("lxd-scratch", surface)
    assert ok
    assert reason is None


def test_promising_a_signal_without_a_stimulus_is_gated():
    surface = _surface(pebble_service={"command": None})
    assert surface.expected_signal
    ok, reason = runner_stage.check_stimulus("k8s-scratch", surface)
    assert not ok
    assert "stimulate nothing" in reason


# -- collection ImportError is the machine's fault, not the selector's ----


def _pytest_run(stdout, exit_code=2):
    return RunResult(
        hypothesis_number=2639,
        branch="none",
        commands=[CommandResult(command="uv run pytest test_issue_2639_repro.py -v", exit_code=exit_code, stdout=stdout)],
    )


_MISSING_OPS = """\
collecting ... collected 0 items / 1 error
==================================== ERRORS ====================================
ImportError while importing test module '/w/test_issue_2639_repro.py'.
test_issue_2639_repro.py:1: in <module>
    import ops
E   ModuleNotFoundError: No module named 'ops'
"""

_STALE_SELECTOR = """\
collected 0 items
ERROR: file or directory not found: tests/test_gone.py
"""


def test_missing_dependency_is_infrastructure_not_a_stale_selector():
    """The real 2026-08-18 `substrate: none` run: the synthesised test
    imports ops, nothing installed it, and the classifier called it a stale
    test selector -- a statement about the reproduction, for what is a
    statement about the machine."""
    outcome, reason = classify(_hypothesis(), None, _pytest_run(_MISSING_OPS))
    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert "ops" in reason
    assert outcome is not Outcome.UNRUNNABLE_TEST_SELECTOR_STALE


def test_missing_dependency_never_composes_a_comment():
    outcome, _ = classify(_hypothesis(), None, _pytest_run(_MISSING_OPS))
    assert outcome not in COMMENT_OUTCOMES


def test_a_genuinely_stale_selector_still_reaches_rung_1():
    """#2484's shape must keep its own rung -- the new check is narrower,
    not a replacement."""
    outcome, _ = classify(_hypothesis(), None, _pytest_run(_STALE_SELECTOR))
    assert outcome is Outcome.UNRUNNABLE_TEST_SELECTOR_STALE


# -- only an assertion is evidence, on the synthesised-test path ----------


_FROZEN_STATE_FAILURE = """\
    def test_pebble_custom_notice():
        ctx = testing.Context(ProbeCharm, meta=META)
        state = testing.State()
>       state.pebble = {
E   dataclasses.FrozenInstanceError: cannot assign to field 'pebble'
=========================== short test summary info ============================
FAILED test_issue_2639_repro.py::test_pebble_custom_notice - dataclasses.Froz...
"""

_REAL_ASSERTION_FAILURE = """\
    def test_pebble_custom_notice():
        ctx.run(ctx.on.pebble_custom_notice(), state)
>       assert captured.get('notice_received', False)
E       assert False
=========================== short test summary info ============================
FAILED test_issue_2639_repro.py::test_pebble_custom_notice - assert False
"""

_ASSERTION_WITH_MESSAGE = """\
>       assert captured.get('notice_received'), "charm did not react"
E       AssertionError: charm did not react
"""


def _synthesised_hypothesis():
    from models import TestFile

    hyp = _hypothesis()
    hyp.synthesized_test_file = TestFile(path="test_issue_2639_repro.py", body="...")
    return hyp


def _pytest_result(stdout):
    return RunResult(
        hypothesis_number=2639,
        branch="none",
        commands=[CommandResult(command="uv run pytest test_issue_2639_repro.py -v", exit_code=1, stdout=stdout)],
    )


def test_a_broken_synthesised_test_is_not_a_reproduction():
    """The real 2026-08-18 false comment: `state.pebble = {...}` on a frozen
    `State`. The pipeline composed "The bug reproduced." off this."""
    outcome, reason = classify(_synthesised_hypothesis(), None, _pytest_result(_FROZEN_STATE_FAILURE))
    assert outcome is Outcome.UNRUNNABLE_SYNTHESIS_INVALID
    assert "FrozenInstanceError" in reason


def test_a_broken_synthesised_test_never_composes_a_comment():
    outcome, _ = classify(_synthesised_hypothesis(), None, _pytest_result(_FROZEN_STATE_FAILURE))
    assert outcome not in COMMENT_OUTCOMES


def test_a_bare_assert_failure_is_still_evidence():
    """A plain `assert` renders as `E   assert False` with no exception
    name -- that is the reproduction signal, and must survive."""
    outcome, _ = classify(_synthesised_hypothesis(), None, _pytest_result(_REAL_ASSERTION_FAILURE))
    assert outcome is not Outcome.UNRUNNABLE_SYNTHESIS_INVALID


def test_an_assertion_with_a_message_is_still_evidence():
    outcome, _ = classify(_synthesised_hypothesis(), None, _pytest_result(_ASSERTION_WITH_MESSAGE))
    assert outcome is not Outcome.UNRUNNABLE_SYNTHESIS_INVALID


def test_a_reporter_supplied_script_may_fail_with_any_exception():
    """Only tests *this project generated* are held to the assertion rule.
    A repro script the reporter wrote is allowed to raise -- that exception
    may well be the bug."""
    hyp = _hypothesis()
    assert hyp.synthesized_test_file is None
    outcome, _ = classify(hyp, None, _pytest_result(_FROZEN_STATE_FAILURE))
    assert outcome is not Outcome.UNRUNNABLE_SYNTHESIS_INVALID


def test_synthesis_invalid_outcome_is_silent():
    assert Outcome.UNRUNNABLE_SYNTHESIS_INVALID not in COMMENT_OUTCOMES


# -- a failed environment build is not a partial reproduction ------------


def test_failed_uv_init_is_infrastructure_not_partial():
    """40-issue batch, 2026-08-18: every issue shared one workdir, so
    `uv init --bare` exited 2 ("Project is already initialized") for all
    but the first. Rung 6 scored that `partial` and composed a comment --
    for issues whose test then *passed*. 5 of 9 comments in that batch."""
    run = RunResult(
        hypothesis_number=2513,
        branch="none",
        commands=[
            CommandResult(
                command="uv init --bare",
                exit_code=2,
                stderr="error: Project is already initialized in `/w/work` (`pyproject.toml` file exists)",
            ),
            CommandResult(command="uv add 'ops[testing]' pytest", exit_code=0),
            CommandResult(command="uv run pytest test_issue_2513_repro.py -v", exit_code=0, stdout="1 passed"),
        ],
    )
    outcome, reason = classify(_hypothesis(), None, run)
    assert outcome is Outcome.INFRASTRUCTURE_FAILED
    assert "already initialized" in reason
    assert outcome not in COMMENT_OUTCOMES


def test_failed_uv_add_is_infrastructure_too():
    run = RunResult(
        hypothesis_number=1,
        branch="none",
        commands=[
            CommandResult(command="uv add 'ops[testing]' pytest", exit_code=1, stderr="network unreachable"),
            CommandResult(command="uv run pytest t.py -v", exit_code=1, stdout="E   assert False"),
        ],
    )
    outcome, _ = classify(_hypothesis(), None, run)
    assert outcome is Outcome.INFRASTRUCTURE_FAILED


def test_a_clean_setup_does_not_trip_the_env_rung():
    run = RunResult(
        hypothesis_number=1,
        branch="none",
        commands=[
            CommandResult(command="uv init --bare", exit_code=0),
            CommandResult(command="uv add 'ops[testing]' pytest", exit_code=0),
            CommandResult(command="uv run pytest t.py -v", exit_code=1, stdout="E   assert False"),
        ],
    )
    outcome, _ = classify(_hypothesis(), None, run)
    assert outcome is not Outcome.INFRASTRUCTURE_FAILED
