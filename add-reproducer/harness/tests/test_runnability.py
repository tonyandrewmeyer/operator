"""Runnability gate (`spike-step-5/live-llm/RESULT.md` Finding 2).

The strings in the `live_*` tests are the **real** output of the live
`deepseek/deepseek-chat` extraction run recorded in
`spike-step-5/live-llm/out/`, not invented examples.
"""

import json
from pathlib import Path

import pytest
import runnability
from runnability import Shape, assess, classify_command

FIXTURES = Path(__file__).parent.parent / "fixtures"
_LIVE_LLM = Path(__file__).parent.parent.parent / "spike-step-5" / "live-llm"
LIVE_OUT = _LIVE_LLM / "out"
LIVE_SUBSTRATE = _LIVE_LLM / "out-before-substrate-fix"
LIVE_COMMANDS = _LIVE_LLM / "out-before-commands-fix"
LIVE_INSCOPE = _LIVE_LLM / "out-before-inscope-fix"


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest test_cwd.py -v",
        "uv init /tmp/repro-2045-cwd",
        "cd /tmp/x && uv add 'ops[testing]'",
        "python3 -c 'import ops; print(ops.__version__)'",
        "sudo concierge prepare -p k8s",
        "juju debug-log --include app/0 --replay | grep notice",
        "PYTHONPATH=. uv run pytest -k relation",
        "./run.sh",
        "cat > repro_test.py << 'EOF'\nimport ops\nEOF",
    ],
)
def test_shell_commands_are_recognised(command):
    assert classify_command(command) is Shape.SHELL


@pytest.mark.parametrize(
    "command",
    [
        "Run a charm test that checks the default relation settings with Juju 4.0+",
        "Run the `test_deployment` integration test",
        "test_scenario_two",
        "Deploy the charm and observe the warning",
    ],
)
def test_prose_is_not_a_command(command):
    assert classify_command(command) is Shape.PROSE


def test_python_fragment_is_not_a_command():
    # Both are real #2327 live output. They fail the check for different
    # reasons -- the first has parens, so bash rejects it outright; the second
    # parses as a bare word and fails the first-token check instead. Which of
    # PROSE/CODE_FRAGMENT they land on isn't meaningful, so assert the thing
    # the gate actually consumes: neither is runnable.
    for fragment in (
        "relation = self.model.get_relation('not-integrated', 1)",
        "relation.data[self.app]",
    ):
        assert classify_command(fragment) is not Shape.SHELL
        assert not assess([fragment]).runnable


def test_unfilled_placeholder_is_not_runnable():
    # The hand extraction for #2639 -- the only hypothesis that has ever
    # reproduced -- carries these. A shell reads <...> as a redirect, so the
    # step-4 walk must have filled them in by hand.
    assert classify_command("juju deploy <k8s-charm-with-workload>") is Shape.CODE_FRAGMENT


def test_comment_placeholder_and_empty():
    assert classify_command("# write test_cwd.py: minimal Scenario test") is Shape.COMMENT
    assert classify_command("   ") is Shape.EMPTY


def test_extra_tools_widen_the_allowlist():
    assert classify_command("mycustomtool --flag") is Shape.PROSE
    assert classify_command("mycustomtool --flag", extra_tools=frozenset({"mycustomtool"})) is Shape.SHELL


def test_assess_requires_every_command_to_be_runnable():
    report = assess(["uv init .", "Run the test and observe the failure"])
    assert not report.runnable
    assert "prose" in report.reason
    assert report.unrunnable_commands == [1]


def test_assess_rejects_empty_commands():
    assert not assess([]).runnable
    assert "empty" in assess([]).reason


def test_assess_rejects_comment_only_sequence():
    # No executable command at all: nothing would run, so a zero exit here
    # would otherwise read as "all commands succeeded, did not reproduce".
    report = assess(["# write a test that does the thing"])
    assert not report.runnable
    assert "no executable command" in report.reason


def test_assess_accepts_a_real_hand_extraction():
    raw = json.loads((FIXTURES / "extractions" / "2045.json").read_text())
    assert assess(raw["commands"]).runnable


def test_hand_corpus_runnability_is_measured_not_assumed():
    # 5 of the 6 in_scope hand extractions are runnable; #2639 is not,
    # because of the unfilled placeholders above. #2107 (added alongside
    # the lxd-scratch branch, see fixtures/extractions/2107.json's
    # _provenance) is a real live extraction whose commands[] never
    # actually touch juju/lxd -- runnable here just means "a shell accepts
    # it", not "this is the right recipe for its substrate". Pinned so a
    # corpus edit that breaks one of these is visible rather than silent.
    results = {}
    for path in sorted((FIXTURES / "extractions").glob("*.json")):
        raw = json.loads(path.read_text())
        if raw["in_scope"]:
            results[int(path.stem)] = assess(raw["commands"]).runnable
    assert results == {2045: True, 2107: True, 2327: True, 2341: True, 2484: True, 2639: False}


def _in_scope_command_lists(directory: Path) -> list[list[str]]:
    """`commands[]` of the in-scope extractions in a recorded live run.

    In-scope is the right population for Approach §3's `commands_runnable`
    bar: an out-of-scope extraction carries an empty `commands[]` by design
    and never reaches a runner, so counting it as "not runnable" understates
    the metric.
    """
    lists = []
    for path in sorted(directory.glob("*.json")):
        if path.stem == "summary":
            continue
        record = json.loads(path.read_text())
        extraction = record.get("live_extraction") or {}
        if extraction.get("in_scope"):
            lists.append(extraction["commands"])
    return lists


@pytest.mark.parametrize(
    "run_dir,expected",
    [
        # Original run: prose, bare test names, Python fragments. The one
        # runnable entry is #2341's `python3 -m pytest tests/test_scenario.py`.
        (LIVE_SUBSTRATE, (1, 7)),
        # After the substrate fix (Finding 1). Unchanged fraction -- that fix
        # targeted routing, not commands -- though the one runnable issue
        # changed to #2045.
        (LIVE_COMMANDS, (1, 7)),
        # After the commands-prompt hardening (Finding 2): five issues emit
        # uv init -> uv add -> heredoc -> pytest.
        (LIVE_INSCOPE, (5, 7)),
        # After the in_scope definition (Finding 3): #2185 and #2304 are now
        # correctly out of scope, so they leave the denominator rather than
        # counting as failures, and #2639 gained a runnable recipe. 4/5 = 80%,
        # over Approach §3's >=70% bar.
        (LIVE_OUT, (4, 5)),
    ],
)
def test_live_commands_runnable_fraction(run_dir, expected):
    """Approach §3's commands_runnable metric, pinned across all four real runs.

    Every directory is committed, so this documents the progression rather
    than asserting it from memory. If a later prompt change moves a number,
    this test is what should fail -- update it and record the new fraction in
    `spike-step-5/live-llm/RESULT.md`.
    """
    if not run_dir.exists():  # pragma: no cover - spike output not always present
        pytest.skip(f"{run_dir.name} not available")
    assert runnability.runnable_fraction(_in_scope_command_lists(run_dir)) == expected


# --- the false-comment path this gate exists to close -----------------------


def _prose_hypothesis(number: int, command: str):
    from models import Hypothesis

    return Hypothesis.from_dict(
        number,
        {
            "in_scope": True,
            "moving_parts": {"substrate": "none"},
            "commands": [command],
            "expected": "the documented behaviour",
            "observed": "the bug happens",
            "confidence": "medium",
        },
    )


def test_prose_command_would_be_read_as_a_reproduction_without_the_gate():
    """Why the gate exists, pinned end to end.

    A prose command exits 127 (`Run: command not found`). The classifier's
    rung 6 turns any non-zero last command into `reproduced (weaker)`, which
    is in COMMENT_OUTCOMES -- i.e. a maintainer-visible comment claiming the
    bug reproduced, produced by a command that never ran. This test asserts
    that misclassification still happens if the gate is bypassed, so nobody
    "simplifies" the gate away believing the classifier would catch it.
    """
    import classifier
    from models import COMMENT_OUTCOMES, CommandResult, Outcome, RunResult

    command = "Run a charm test that checks the default relation settings with Juju 4.0+"
    hyp = _prose_hypothesis(2185, command)
    run = RunResult(
        hypothesis_number=2185,
        branch="none",
        commands=[CommandResult(command=command, exit_code=127, stderr="bash: Run: command not found")],
    )
    outcome, _ = classifier.classify(hyp, None, run)
    assert outcome is Outcome.REPRODUCED_WEAKER
    assert outcome in COMMENT_OUTCOMES  # the false comment


def test_substrate_none_prose_is_dropped_by_synthesis_not_run(tmp_path):
    """For `substrate: none` the protection is the synthesis filter, not the gate.

    `needs_test_file()` is true for any substrate-none hypothesis with no
    pytest invocation, so a prose-only extraction gets a synthesised test
    appended -- and the prose itself is now dropped from the setup commands
    rather than kept and executed. That's a better outcome than gating: a real
    reproduction is attempted from expected/observed, and nothing exits 127.
    """
    import runner_stage
    from models import Issue
    from seams.llm import FixtureLLM

    command = "Run a charm test that checks the default relation settings with Juju 4.0+"
    hyp = _prose_hypothesis(2185, command)
    issue = Issue(number=2185, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    repaired = runner_stage.write_synthesized_test_file_if_needed(
        hyp, issue, FixtureLLM(FIXTURES), {"workdir": str(tmp_path)}
    )
    assert command not in repaired.commands, "prose must not survive as a setup command"
    assert any("pytest" in c for c in repaired.commands)
    assert assess(repaired.commands).runnable


def test_gate_blocks_prose_on_the_clone_branch(tmp_path):
    """`k8s-clone` has no synthesis step, so the gate is the only protection.

    Shape per spike-step-4's third-branch heuristic: a ci_run_url plus no
    self-contained snippet.
    """
    import runner_stage
    from models import Hypothesis, Issue
    from seams.llm import FixtureLLM
    from seams.runner import FixtureRunnerSeam

    command = "Run the `test_deployment` integration test"  # real #2484 live output
    hyp = Hypothesis.from_dict(
        2484,
        {
            "in_scope": True,
            "moving_parts": {
                "substrate": "k8s",
                "ci_run_url": "https://github.com/canonical/operator/actions/runs/1",
            },
            "commands": [command],
            "expected": "the test passes",
            "observed": "the test fails",
            "confidence": "medium",
        },
    )
    assert runner_stage.choose_branch(hyp) == "k8s-clone"
    issue = Issue(
        number=2484, title="t", body="b", labels=[], state="OPEN", created_at="",
        author="a", repo="canonical/operator",
    )
    result = runner_stage.run_hypothesis(
        hyp, issue, None, FixtureRunnerSeam(FIXTURES), FixtureLLM(FIXTURES), {"issue_number": 2484, "workdir": str(tmp_path)}
    )
    assert result.run_result is None, "the runner must never see a prose command"
    assert result.unrunnable_reason is not None
    assert "prose" in result.unrunnable_reason


def test_gate_does_not_apply_to_k8s_scratch():
    # `_run_k8s_scratch` builds its own sequence from the rendered scaffold,
    # so commands[] isn't the recipe there -- gating it would kill #2639, the
    # only hypothesis that has ever reproduced.
    import runner_stage

    hyp = _prose_hypothesis(2639, "Run the thing")
    ok, reason = runner_stage.check_runnable(hyp, "k8s-scratch")
    assert ok and reason is None
    ok, reason = runner_stage.check_runnable(hyp, "none")
    assert not ok
