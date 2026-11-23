"""Comment composition tests.

`compose_template()` renders every comment (there is no LLM path;
`Composer` wraps the renderer and makes no call). These tests cover the
renderer directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from composer import (
    AUTOMATION_PREFIX,
    Composer,
    _resolved_test_file,
    _test_file_to_render,
    _trim,
    _when,
    compose_template,
)
from models import (
    COMMENT_OUTCOMES,
    CommandResult,
    Hypothesis,
    Issue,
    MovingParts,
    Outcome,
    RunResult,
    TestFile,
)


def after_prefix(text: str) -> str:
    """The comment with `AUTOMATION_PREFIX` stripped.

    Every composed comment opens with the prefix (the false-comment
    mitigation), so assertions about what a *renderer* or the *model* wrote
    start one line further down.
    """
    assert text.startswith(AUTOMATION_PREFIX + "\n\n"), text[:200]
    return text[len(AUTOMATION_PREFIX) + 2 :]


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def _load_hypothesis(number: int) -> Hypothesis:
    raw = json.loads((FIXTURES / "extractions" / f"{number}.json").read_text())
    return Hypothesis.from_dict(number, raw)


def _load_run(number: int) -> RunResult:
    return RunResult.from_dict(json.loads((FIXTURES / "runs" / f"{number}.json").read_text()))


def test_template_silent_outcome_returns_none():
    hyp = _load_hypothesis(2341)
    run = _load_run(2341)
    assert compose_template(hyp, _load_issue(2341), run, Outcome.DID_NOT_REPRODUCE, "n/a", run_id="r", timestamp="t") is None


def test_template_partial_wording():
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(hypothesis_number=3, branch="none", commands=[CommandResult(command="uv venv", exit_code=1)])
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.PARTIAL, "diverged early", run_id="r1", timestamp="2026-01-01T00:00:00Z")
    assert after_prefix(body).startswith("**Partial reproduction attempt**")
    assert "<!-- add-reproducer:issue=3:run=r1 -->" in body


def test_template_partial_says_attempted_not_reproduced():
    # A `partial` outcome did not reproduce the bug -- the trailing summary
    # line must not contradict that.
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none", base="ubuntu@24.04"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(hypothesis_number=3, branch="none", commands=[CommandResult(command="uv venv", exit_code=1)])
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.PARTIAL, "diverged early", run_id="r1", timestamp="2026-01-01T00:00:00Z")
    assert "Attempted with Scenario (no Juju) at 2026-01-01 00:00 UTC." in body
    assert "Reproduced" not in body


def test_template_versions_line_includes_observed_juju_version_when_present():
    """The pinned `moving_parts.juju_version` (often nothing) and the juju that actually
    produced the verdict can differ -- a reader trusting a "Reproduced"
    comment needs both, not just the pin."""
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s", base="ubuntu@24.04"),
        commands=["true"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=3,
        branch="k8s-scratch",
        commands=[CommandResult(command="true", exit_code=0)],
        observed_juju_version="3.6.27-ubuntu-amd64",
    )
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "Versions: Juju 3.6.27-ubuntu-amd64, k8s substrate, base ubuntu@24.04\n" in body


def test_template_versions_line_omits_observed_juju_version_when_absent():
    """The `none` branch never touches a juju substrate, so
    `observed_juju_version` stays `None` -- the versions line must not claim
    an "observed_juju=" value nobody measured."""
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none", base="ubuntu@24.04"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(hypothesis_number=3, branch="none", commands=[CommandResult(command="uv venv", exit_code=0)])
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "Juju " not in body.split("Commands run:")[0]
    # Scenario only: no Juju and no base to state.
    assert "Versions: Scenario only (no Juju)\n" in body


def test_template_2639_includes_control_run():
    # The control run is the entire basis for a
    # `reproduced_positive_signal_absent` verdict, so the comment must show it.
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    assert run.control is not None
    assert run.control.command in body
    assert "Recorded notice 2" in body  # control's stdout
    assert "observed notice: canonical.com/repro/notice-root in workload" in body


def test_template_control_label_explains_what_it_rules_out():
    # Read cold, a bare "Control" label is unclear -- it separates
    # reproduced_positive_signal_absent from a broken observer, and the
    # label has to say so. One clause, not a new section.
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="t")
    assert "rules out a broken check" in body


def _log_only_run():
    # #2341's shape: exit 0, no stdout/stderr worth reading, the
    # bug's only evidence is a captured log record.
    return RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[
            CommandResult(
                command="uv run pytest repro_test.py -v --log-cli-level=DEBUG",
                exit_code=0,
                stdout="1 passed in 0.08s",
                log_records=["WARNING consistency_checker: \"remote_unit=0 implicit\" (matches observed)"],
            )
        ],
    )


def test_template_log_only_includes_log_records():
    hyp = _load_hypothesis(2341)
    run = _log_only_run()
    issue = _load_issue(2341)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_LOG_ONLY, "log record matched", run_id="run-x", timestamp="t")
    assert "remote_unit=0 implicit" in body
    # And the clean-looking stdout is still shown too -- this is additive,
    # not a replacement for the (now potentially misleading-on-its-own)
    # stdout/stderr section.
    assert "1 passed in 0.08s" in body


def test_template_2639_contains_marker_and_versions():
    hyp = _load_hypothesis(2639)
    run = _load_run(2639)
    issue = _load_issue(2639)
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT, "control confirmed", run_id="run-x", timestamp="2026-07-27T00:00:00Z")
    assert "k8s substrate" in body
    assert "add-reproducer:issue=2639:run=run-x" in body
    for c in run.commands:
        assert c.command in body


# --- Composer: gating ----------------------------------------------------


class _CountingLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        self.calls += 1
        return {"comment_body": "should never be seen"}


def test_composer_never_calls_llm_for_silent_outcome():
    llm = _CountingLLM()
    composer = Composer(llm)
    hyp = _load_hypothesis(2341)
    run = _load_run(2341)
    result = composer.compose(hyp, _load_issue(2341), run, Outcome.DID_NOT_REPRODUCE, "n/a", run_id="r", timestamp="t")
    assert result is None
    assert llm.calls == 0


# --- Test files -----------------------------------------------------------


def _hypothesis_with_test_file(*, path="test_issue_9999_repro.py", body="def test_x():\n    assert False\n"):
    return Hypothesis(
        issue_number=9999,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv init --bare .", "uv add 'ops[testing]'", f"uv run pytest {path} -v"],
        expected="passes",
        observed="fails",
        confidence="medium",
        synthesized_test_file=TestFile(path=path, body=body),
    )


def _issue_9999():
    return Issue(number=9999, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")


def test_template_includes_synthesized_test_file_body():
    hyp = _hypothesis_with_test_file()
    run = RunResult(
        hypothesis_number=9999,
        branch="none",
        commands=[CommandResult(command=c, exit_code=0) for c in hyp.commands],
    )
    body = compose_template(hyp, _issue_9999(), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "test_issue_9999_repro.py" in body
    assert "def test_x():" in body
    # The file body must appear before the commands section: the reader
    # needs the file to exist before the commands that reference it make
    # sense.
    assert body.index("def test_x():") < body.index("Commands run:")
    # Inline it collapsed, since these bodies are long.
    assert "<details>" in body
    assert "<summary>Synthesized test file: test_issue_9999_repro.py" in body
    assert body.index("<summary>") < body.index("def test_x():") < body.index("</details>")


# --- Heredoc-embedded test file ---------------------------------------------
#
# Some extractions write their test file with a heredoc in commands[], and the
# executed run may not include that heredoc, so the composer must show it.


def test_template_includes_heredoc_embedded_test_file():
    hyp = _load_hypothesis(2341)  # real extraction; commands[2] is the heredoc
    # The executed run replays only the final pytest invocation, not the setup
    # or the heredoc that wrote the file.
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed in 0.08s")],
    )
    body = compose_template(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    # "Commands run" renders only run_result.commands, which here lacks the
    # heredoc that creates the file.
    commands_section = body[body.index("Commands run:") :]
    assert "cat >" not in commands_section
    assert "class DummyCharm(ops.CharmBase):" in body  # the heredoc's actual body, shown elsewhere
    assert "<details>" in body
    assert "<summary>Test file: repro_test.py" in body
    assert body.index("<summary>") < body.index("class DummyCharm") < body.index("</details>") < body.index("Commands run:")


def test_resolved_test_file_prefers_synthesized_over_embedded_heredoc():
    # The two sources shouldn't both be set, but if they are, the synthesised
    # file wins because it is a copy of what was actually written for the run.
    hyp = _load_hypothesis(2341)
    hyp.synthesized_test_file = TestFile(path="different.py", body="def test_y(): pass\n")
    resolved = _resolved_test_file(hyp)
    assert resolved.path == "different.py"


def test_template_omits_test_file_section_when_none():
    hyp = _load_hypothesis(2341)
    run = _load_run(2341)
    hyp.observed = 'crashes with "boom"'
    # Strip the heredoc so this case really has no test file.
    hyp.commands = ["uv run pytest -v"]
    run.commands[-1] = CommandResult(command="uv run pytest -v", exit_code=1, stderr='"boom"')
    body = compose_template(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "Synthesized test file" not in body
    assert "<details>" not in body


def _prefix_case() -> tuple[Hypothesis, Issue, RunResult]:
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none", base="ubuntu@24.04"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=3,
        branch="none",
        commands=[CommandResult(command="uv venv", exit_code=1)],
    )
    issue = Issue(
        number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator"
    )
    return hyp, issue, run


@pytest.mark.parametrize("outcome", sorted(COMMENT_OUTCOMES, key=lambda o: o.value))
def test_template_prefixes_every_comment_worthy_outcome(outcome):
    """The false-comment mitigation rests on this line being present, so it
    is asserted on every rung that can reach a maintainer -- including
    `partial`, which is why the wording says "attempt" rather than
    "reproduced"."""
    hyp, issue, run = _prefix_case()
    body = compose_template(hyp, issue, run, outcome, "reason", run_id="r1", timestamp="t")
    assert body.startswith(AUTOMATION_PREFIX + "\n\n")


def test_template_versions_line_includes_observed_ops_version_when_present():
    """`charmcraft pack` resolves `ops~=3.8` from PyPI inside its own managed LXD instance, so the
    `ops` a verdict was produced on is not the checked-out tree and is not
    what `repo=` names either -- `repo=` is the extraction's pin, usually
    nothing. Disclosed for the same reason `observed_juju` is."""
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="k8s", base="ubuntu@24.04"),
        commands=["true"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=3,
        branch="k8s-scratch",
        commands=[CommandResult(command="true", exit_code=0)],
        observed_juju_version="4.0.14-genericlinux-amd64",
        observed_ops_version="3.8.2",
    )
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "Versions: ops 3.8.2, Juju 4.0.14-genericlinux-amd64, k8s substrate, base ubuntu@24.04\n" in body


def test_template_versions_line_omits_observed_ops_version_when_absent():
    """A branch that packs no charm has no pack log to read, and an
    unparseable one gives `None` -- either way the line must not claim an
    "observed_ops=" value nobody measured."""
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="none", base="ubuntu@24.04"),
        commands=["uv venv"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(hypothesis_number=3, branch="none", commands=[CommandResult(command="uv venv", exit_code=0)])
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "ops" not in body.split("Commands run:")[0].split("Versions:")[1]


# --- The `<summary>` contradiction ------------------------------------
#
# A test file shown in `<details>` must not claim to be "not shown in the
# commands" when the heredoc that wrote it is in the commands block. The
# commands and body below come from a real composed comment; the empty
# `expected`/`observed` fields are not read by these assertions.

_DISPATCH_TEST_FILE_BODY = """\
import os
import pytest
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, *args):
        super().__init__(*args)
        assert os.getcwd() == self.framework.charm_dir


def test_cwd_in_scenario():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.update_status(), testing.State()) as mgr:
        pass"""

_DISPATCH_HEREDOC = f"cat > test_cwd.py << 'PYEOF'\n{_DISPATCH_TEST_FILE_BODY}\nPYEOF"

_DISPATCH_COMMANDS = [
    "uv init --bare .",
    "uv add 'ops[testing]'",
    _DISPATCH_HEREDOC,
    "uv run pytest test_cwd.py -v",
]

_DISPATCH_PYTEST_OUTPUT = """\
test_cwd.py::test_cwd_in_scenario FAILED                                 [100%]

=================================== FAILURES ===================================
_____________________________ test_cwd_in_scenario _____________________________

    def __init__(self, *args):
        super().__init__(*args)
>       assert os.getcwd() == self.framework.charm_dir
E       AssertionError: assert '/home/runner/work/operator/operator' == PosixPath('/tmp/tmph5kel9et')
"""


def _dispatch_hypothesis() -> Hypothesis:
    return Hypothesis(
        issue_number=2045,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=list(_DISPATCH_COMMANDS),
        expected="",
        observed="",
        confidence="medium",
    )


def _dispatch_run() -> RunResult:
    """The `none` branch runs `commands[]` one `bash -c` at a time, so the
    heredoc that writes the file is itself one of the executed commands --
    which is exactly why the comment showed the body twice."""
    return RunResult(
        hypothesis_number=2045,
        branch="none",
        commands=[
            CommandResult(command=_DISPATCH_COMMANDS[0], exit_code=0),
            CommandResult(command=_DISPATCH_COMMANDS[1], exit_code=0),
            CommandResult(command=_DISPATCH_COMMANDS[2], exit_code=0),
            CommandResult(command=_DISPATCH_COMMANDS[3], exit_code=1, stdout=_DISPATCH_PYTEST_OUTPUT),
        ],
    )


def _dispatch_issue() -> Issue:
    return Issue(
        number=2045,
        title="os.getcwd() is not the charm root in ops[testing]",
        body="b",
        labels=[],
        state="OPEN",
        created_at="",
        author="a",
        repo="canonical/operator",
    )


def test_composed_comment_does_not_render_the_heredoc_body_twice():
    """The defect itself, against the real composed text: the body appeared
    once in the `<details>` block and once in the commands block."""
    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        _dispatch_run(),
        Outcome.REPRODUCED_WEAKER,
        "'uv run pytest test_cwd.py -v' (last command) exited non-zero, no substring match",
        run_id="35219949266",
        timestamp="2026-09-17T12:14:36.483167+00:00",
    )
    assert body.count("def test_cwd_in_scenario():") == 1


def test_composed_comment_does_not_deny_showing_a_file_it_shows():
    """The reader-visible half: the `<summary>` said the file was "not shown
    in the commands actually run below" directly above the commands that
    show it."""
    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        _dispatch_run(),
        Outcome.REPRODUCED_WEAKER,
        "reason",
        run_id="r1",
        timestamp="t",
    )
    assert "not shown in the commands actually run below" not in body
    assert "<details>" not in body


def test_the_heredoc_body_survives_in_the_commands_block():
    """Dropping the duplicate must not drop the file. The reader still sees
    every line of it, in the `cat` heredoc that writes it."""
    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        _dispatch_run(),
        Outcome.REPRODUCED_WEAKER,
        "reason",
        run_id="r1",
        timestamp="t",
    )
    commands_section = body[body.index("Commands run:") :]
    for line in _DISPATCH_TEST_FILE_BODY.splitlines():
        assert line in commands_section


def test_a_heredoc_the_executed_commands_drop_is_still_rendered():
    """The other side of the same question, and the reason it is asked about
    visibility rather than provenance: `#2341`'s extraction embeds its test
    file the same way, but its run replays only the final `pytest`
    invocation, so the body really is absent from the commands block and the
    `<details>` section -- and its "not shown" wording -- are correct."""
    hyp = _load_hypothesis(2341)
    run = RunResult(
        hypothesis_number=2341,
        branch="none",
        commands=[CommandResult(command="uv run pytest repro_test.py -v", exit_code=0, stdout="1 passed")],
    )
    body = compose_template(hyp, _load_issue(2341), run, Outcome.REPRODUCED, "matched", run_id="r1", timestamp="t")
    assert "<summary>Test file: repro_test.py" in body
    assert "not shown in the commands actually run below" in body


def test_a_synthesized_file_is_always_rendered():
    """Synthesis writes the file straight to disk, so no command ever shows
    it -- this path is untouched by the visibility question."""
    hyp = _dispatch_hypothesis()
    hyp.commands = ["uv run pytest test_x.py -v"]
    hyp.synthesized_test_file = TestFile(path="test_x.py", body="def test_x():\n    assert False\n")
    run = RunResult(
        hypothesis_number=2045,
        branch="none",
        commands=[CommandResult(command="uv run pytest test_x.py -v", exit_code=1, stdout="E assert False")],
    )
    body = compose_template(hyp, _dispatch_issue(), run, Outcome.REPRODUCED_WEAKER, "r", run_id="r1", timestamp="t")
    assert "<summary>Synthesized test file: test_x.py" in body


def test_a_none_branch_comment_now_discloses_the_ops_it_resolved():
    """A `none`-branch comment shows ` + ops==3.8.2` in its observed output,
    so the versions line four lines above it must name that `ops` too. Same
    commands, same branch, with the version the seam captures: the versions
    line states what the observed output already shows, and the two halves
    of the comment agree."""
    run = _dispatch_run()
    run.observed_ops_version = "3.8.2"

    body = compose_template(
        _dispatch_hypothesis(),
        _dispatch_issue(),
        run,
        Outcome.REPRODUCED_WEAKER,
        "'uv run pytest test_cwd.py -v' (last command) exited non-zero, no substring match",
        run_id="35219949266",
        timestamp="2026-09-17T12:14:36.483167+00:00",
    )

    assert "Versions: ops 3.8.2, Scenario only (no Juju)\n" in body


# --- The wrapper is fixed text -----------------------------------------
#
# The comment is rendered from the run record alone, so these pin the
# wrapper's exact text.


def _cwd_case(timestamp: str = "2026-11-19T12:46:00.843074+00:00") -> str:
    hyp = _dispatch_hypothesis()
    run = _dispatch_run()
    run.observed_ops_version = "3.8.3"
    return compose_template(
        hyp, _dispatch_issue(), run, Outcome.REPRODUCED_WEAKER, "r", run_id="0d0d", timestamp=timestamp
    )


def test_the_wrapper_is_fixed():
    body = _cwd_case()
    head, rest = body.split("Commands run:\n", 1)
    assert head == f"{AUTOMATION_PREFIX}\n\nVersions: ops 3.8.3, Scenario only (no Juju)\n\n"
    assert rest.startswith("```shell\n$ ")
    assert "\n\nOutput:\n```\n" in rest
    assert body.endswith(
        "\n\nReproduced with Scenario (no Juju) at 2026-11-19 12:46 UTC.\n\n"
        "<!-- add-reproducer:issue=2045:run=0d0d -->"
    )


def test_no_second_sentence_repeating_the_banner():
    body = _cwd_case()
    assert "should verify" not in body
    assert "The bug reproduced" not in body
    assert body.count("verify") == 1  # the banner's


def test_no_juju_or_base_for_a_scenario_only_run():
    hyp = _dispatch_hypothesis()
    hyp.moving_parts.base = "ubuntu@24.04"
    hyp.moving_parts.juju_version = "3.6"
    run = _dispatch_run()
    body = compose_template(hyp, _dispatch_issue(), run, Outcome.REPRODUCED_WEAKER, "r", run_id="r", timestamp="t")
    assert "Juju 3.6" not in body and "ubuntu@24.04" not in body
    assert "unpinned" not in body


def test_the_commands_and_output_are_the_records():
    run = _dispatch_run()
    body = _cwd_case()
    commands = "\n".join(f"$ {c.command}" for c in run.commands)
    assert f"```shell\n{commands}\n```" in body
    last = run.commands[-1]
    output = f"{last.stdout}{last.stderr}"
    assert f"Output:\n```\n{output[:200]}" in body


def _real_workflow_case() -> tuple[RunResult, str]:
    """A run record from a real workflow run on `#2045`, setup output and all."""
    raw = json.loads((FIXTURES / "composed" / "2045-35442613976-run.json").read_text())
    run = RunResult.from_dict(raw)
    body = compose_template(
        _load_hypothesis(2045),
        _load_issue(2045),
        run,
        Outcome.REPRODUCED_WEAKER,
        "r",
        run_id="35442613976",
        timestamp="2026-11-21T00:00:00Z",
    )
    assert body is not None
    return run, body


def test_output_is_the_last_command_only():
    run, body = _real_workflow_case()
    shown = body.split("Output:\n```\n", 1)[1].split("\n```", 1)[0]
    last = run.commands[-1]
    assert shown == _trim(f"{last.stdout}{last.stderr}")
    assert shown.startswith("=" * 29 + " test session starts")


def test_setup_output_is_not_shown():
    run, body = _real_workflow_case()
    # The setup commands are listed, but nothing they printed is.
    for c in run.commands[:-1]:
        assert f"$ {c.command.splitlines()[0]}" in body
    assert "Adding `repro` as member of workspace" not in body
    assert "Resolved 16 packages" not in body
    assert " + ops-scenario==" not in body


def test_a_run_with_no_commands_says_so():
    run = RunResult(hypothesis_number=2045, branch="none", commands=[])
    body = compose_template(
        _dispatch_hypothesis(), _dispatch_issue(), run, Outcome.REPRODUCED_WEAKER, "r", run_id="r", timestamp="t"
    )
    assert body is None or "Output:\n```\n(no output captured)\n```" in body


@pytest.mark.parametrize(
    ("timestamp", "shown"),
    [
        ("2026-11-19T12:46:00.843074+00:00", "2026-11-19 12:46 UTC"),
        ("2026-11-19T12:46:59Z", "2026-11-19 12:46 UTC"),
        ("2026-11-20T01:16:00+13:00", "2026-11-19 12:16 UTC"),
        ("2026-11-19T12:46:00", "2026-11-19 12:46 UTC"),
        ("t", "t"),
    ],
)
def test_the_timestamp_is_trimmed(timestamp: str, shown: str):
    assert _when(timestamp) == shown


def test_a_fence_in_the_output_does_not_close_the_block():
    hyp = _dispatch_hypothesis()
    run = _dispatch_run()
    run.commands[-1].stdout = "before\n```\nafter\n"
    body = compose_template(hyp, _dispatch_issue(), run, Outcome.REPRODUCED_WEAKER, "r", run_id="r", timestamp="t")
    assert "Output:\n````\n" in body
    assert "after\n" in body.split("Output:\n````\n", 1)[1].split("\n````", 1)[0]


def test_composer_renders_the_template_without_calling_the_llm():
    llm = _CountingLLM()
    hyp, run, issue = _dispatch_hypothesis(), _dispatch_run(), _dispatch_issue()
    composed = Composer(llm).compose(hyp, issue, run, Outcome.REPRODUCED_WEAKER, "r", run_id="r", timestamp="t")
    assert composed == compose_template(hyp, issue, run, Outcome.REPRODUCED_WEAKER, "r", run_id="r", timestamp="t")
    assert llm.calls == 0


def test_juju_pin_shown_beside_what_ran_when_they_differ():
    hyp = Hypothesis(
        issue_number=3,
        in_scope=True,
        moving_parts=MovingParts(substrate="lxd", juju_version="3.6", base="ubuntu@22.04"),
        commands=["true"],
        expected="passes",
        observed="fails",
        confidence="medium",
    )
    run = RunResult(
        hypothesis_number=3,
        branch="lxd-scratch",
        commands=[CommandResult(command="true", exit_code=0)],
        observed_juju_version="3.6.27",
    )
    issue = Issue(number=3, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED, "m", run_id="r", timestamp="t")
    assert "Versions: Juju 3.6.27 (asked for 3.6), lxd substrate, base ubuntu@22.04\n" in body
    assert "Reproduced on lxd substrate, ubuntu@22.04 at t." in body
