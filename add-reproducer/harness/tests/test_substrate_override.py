"""The substrate override: a fake-hook-tool `_ModelBackend` test runs as
`substrate: none`, whatever the answer's `substrate` field says.

On `#2709` 13 of 14 live answers wrote that test and none said `none`, so
every one stopped at the stimulus gate without running it
(`spike-step-5/static-retry/RESULT.md` §30). These tests drive the whole
pipeline with a static extraction and a runner that records what it was
asked to run.
"""

from __future__ import annotations

import pytest

import extraction_record
from models import CommandResult, Hypothesis, Issue, MovingParts, Outcome, RunResult
from pipeline import Pipeline, override_substrate
from static_test_check import check, faked_hook_tools_in_commands
from tests.test_hook_tool_fakes import _FAKE_TOOL_TEST

_COMMANDS = [
    "uv init --bare .",
    "uv add 'ops==3.8.3' pytest",
    f"cat > test_gone.py << 'PYEOF'\n{_FAKE_TOOL_TEST}PYEOF",
    "uv run pytest test_gone.py -v",
]

_TOOLS = ["is-leader", "relation-get", "relation-ids", "relation-list"]

# A `testing.Context` test: no `_ModelBackend`, no fakes.
_CONTEXT_TEST = """\
import ops
from ops import testing


class MyCharm(ops.CharmBase):
    pass


def test_start():
    ctx = testing.Context(MyCharm, meta={'name': 'myapp'})
    ctx.run(ctx.on.start(), testing.State())
"""


class _StaticLLM:
    """Returns the extraction for the extraction call and `{}` otherwise,
    which sends the composer to its template."""

    def __init__(self, extraction: dict):
        self.extraction = extraction
        self.calls: list[dict] = []

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        self.calls.append({"purpose": purpose})
        if purpose.startswith("extraction"):
            return self.extraction
        return {}


class _RecordingRunner:
    """Runs nothing: records the branch and the commands, and answers as a
    3.8.3 run of `_FAKE_TOOL_TEST` does (its assertion fails)."""

    def __init__(self):
        self.calls: list[dict] = []

    def resolve_symbol(self, symbol, context) -> bool:
        return True

    def run(self, *, branch, hypothesis, surface, context) -> RunResult:
        self.calls.append({"branch": branch, "commands": list(hypothesis.commands), "surface": surface})
        results = [CommandResult(command=c, exit_code=0) for c in hypothesis.commands[:-1]]
        results.append(
            CommandResult(
                command=hypothesis.commands[-1],
                exit_code=1,
                stdout=(
                    "test_gone.py::test_gone_relation_databag_reads_empty FAILED\n"
                    "E       AssertionError: assert ModelError('ERROR permission denied\\n') == {}\n"
                    "FAILED test_gone.py::test_gone_relation_databag_reads_empty - AssertionError\n"
                    "1 failed in 0.05s\n"
                ),
            )
        )
        return RunResult(hypothesis_number=hypothesis.issue_number, branch=branch, commands=results)


class _NoRunner:
    runs = 0

    def resolve_symbol(self, symbol, context) -> bool:
        return True

    def run(self, **kwargs) -> RunResult:
        self.runs += 1
        raise AssertionError("the runner should not be reached")


def _issue() -> Issue:
    return Issue(
        number=2709,
        title="t",
        body="b",
        labels=[],
        state="OPEN",
        created_at="",
        author="a",
        repo="canonical/operator",
    )


def _extraction(substrate: str, commands: list[str], *, other: dict | None = None, confidence="high") -> dict:
    return {
        "in_scope": True,
        "moving_parts": {"substrate": substrate, "repo_version": "3.8.3", "other": dict(other or {})},
        "commands": commands,
        "expected": "a gone relation's databag reads as empty",
        "observed": "ModelError: ERROR permission denied",
        "confidence": confidence,
    }


def _hypothesis(substrate: str, commands: list[str]) -> Hypothesis:
    return Hypothesis(
        issue_number=2709,
        in_scope=True,
        moving_parts=MovingParts(substrate=substrate, repo_version="3.8.3"),
        commands=commands,
        expected="e",
        observed="o",
        confidence="high",
    )


# --- the recogniser ------------------------------------------------------------


def test_the_fake_tool_test_passes_the_check():
    """So that the pipeline runs it, rather than stopping at the check gate."""
    assert check(_FAKE_TOOL_TEST).passed


def test_faked_hook_tools_in_commands_reads_the_test_heredoc():
    assert faked_hook_tools_in_commands(_COMMANDS) == _TOOLS


def test_faked_hook_tools_in_commands_without_a_heredoc():
    assert faked_hook_tools_in_commands(["juju deploy ./my.charm", "juju status"]) == []
    assert faked_hook_tools_in_commands([]) == []


def test_faked_hook_tools_in_commands_for_a_testing_context_test():
    commands = [f"cat > test_ctx.py << 'PYEOF'\n{_CONTEXT_TEST}PYEOF", "uv run pytest test_ctx.py"]
    assert faked_hook_tools_in_commands(commands) == []


# --- override_substrate() --------------------------------------------------------


@pytest.mark.parametrize("substrate", ["lxd", "k8s"])
def test_override_substrate_moves_a_fake_tool_test_to_none(substrate):
    original = _hypothesis(substrate, _COMMANDS)
    overridden, record = override_substrate(original)
    assert overridden.moving_parts.substrate == "none"
    assert record == {"from": substrate, "to": "none", "faked_hook_tools": _TOOLS}
    # The extraction's own hypothesis is left as it was.
    assert original.moving_parts.substrate == substrate
    assert overridden.commands == original.commands


def test_override_substrate_leaves_a_native_none_alone():
    original = _hypothesis("none", _COMMANDS)
    overridden, record = override_substrate(original)
    assert overridden is original
    assert record is None


def test_override_substrate_leaves_lxd_without_fakes_alone():
    original = _hypothesis("lxd", ["juju deploy ./my.charm", "juju status"])
    overridden, record = override_substrate(original)
    assert overridden is original
    assert record is None


# --- the pipeline ------------------------------------------------------------------


@pytest.mark.parametrize("substrate", ["lxd", "k8s"])
def test_a_fake_tool_test_runs_as_none(tmp_path, substrate):
    runner = _RecordingRunner()
    pipeline = Pipeline(_StaticLLM(_extraction(substrate, _COMMANDS)), runner, work_dir=tmp_path)
    result = pipeline.run_for_issue(_issue())
    assert [call["branch"] for call in runner.calls] == ["none"]
    assert runner.calls[0]["commands"] == _COMMANDS
    assert runner.calls[0]["surface"] is None
    assert result.branch == "none"
    assert result.hypothesis.moving_parts.substrate == "none"
    assert result.stage_reached == "classifier"
    assert result.outcome == Outcome.REPRODUCED_WEAKER
    assert result.substrate_override == {"from": substrate, "to": "none", "faked_hook_tools": _TOOLS}
    assert result.extraction_record["substrate_override"] == result.substrate_override
    # The record still says what the extraction answered.
    assert result.extraction_record["first_pass"]["substrate"] == substrate
    # Composed as `none`, not as the field's substrate.
    assert "/none at" in result.comment
    assert f"/{substrate} at" not in result.comment
    assert "are fakes that print what the issue reports" in result.comment


def test_the_override_is_rendered_in_the_record(tmp_path):
    pipeline = Pipeline(_StaticLLM(_extraction("lxd", _COMMANDS)), _RecordingRunner(), work_dir=tmp_path)
    result = pipeline.run_for_issue(_issue())
    lines = extraction_record.render(result.extraction_record)
    assert (
        "  substrate override: lxd -> none (the test fakes hook tools: "
        "is-leader, relation-get, relation-ids, relation-list)"
    ) in lines


def test_an_overridden_answer_goes_through_needs_juju(tmp_path):
    """`needs_juju` is scoped to `none`, and an overridden answer is `none`."""
    extraction = _extraction("lxd", _COMMANDS, other={"needs_juju": "relation-ids output not stated"})
    runner = _NoRunner()
    pipeline = Pipeline(_StaticLLM(extraction), runner, work_dir=tmp_path)
    result = pipeline.run_for_issue(_issue())
    assert runner.runs == 0
    assert result.outcome == Outcome.SKIPPED_NEEDS_JUJU
    assert result.substrate_override["from"] == "lxd"


def test_an_overridden_answer_goes_through_the_check_gate(tmp_path):
    """A failing fake-tool test is stopped by the check gate, as on `none`."""
    lines = _FAKE_TOOL_TEST.splitlines(keepends=True)
    broken = "".join(line for line in lines if "'relation-ids'" not in line)
    assert not check(broken).passed
    commands = [_COMMANDS[0], _COMMANDS[1], f"cat > test_gone.py << 'PYEOF'\n{broken}PYEOF", _COMMANDS[3]]
    runner = _NoRunner()
    pipeline = Pipeline(_StaticLLM(_extraction("lxd", commands)), runner, work_dir=tmp_path)
    result = pipeline.run_for_issue(_issue())
    assert runner.runs == 0
    assert result.stage_reached == "extraction:test_check_failed"
    assert result.substrate_override is not None


def test_lxd_without_fakes_is_unchanged(tmp_path):
    """No fake-hook-tool test: still `lxd`, still to surface inference, no override."""
    runner = _NoRunner()
    pipeline = Pipeline(
        _StaticLLM(_extraction("lxd", ["juju deploy ./my.charm", "juju status"])), runner, work_dir=tmp_path
    )
    result = pipeline.run_for_issue(_issue())
    assert runner.runs == 0
    assert result.stage_reached == "surface_inference"
    assert result.substrate_override is None
    assert result.extraction_record["substrate_override"] is None


def test_lxd_with_a_testing_context_test_is_unchanged(tmp_path):
    commands = [f"cat > test_ctx.py << 'PYEOF'\n{_CONTEXT_TEST}PYEOF", "uv run pytest test_ctx.py"]
    runner = _NoRunner()
    pipeline = Pipeline(_StaticLLM(_extraction("lxd", commands)), runner, work_dir=tmp_path)
    result = pipeline.run_for_issue(_issue())
    assert runner.runs == 0
    assert result.stage_reached == "surface_inference"
    assert result.substrate_override is None


def test_a_native_none_shows_no_override(tmp_path):
    runner = _RecordingRunner()
    pipeline = Pipeline(_StaticLLM(_extraction("none", _COMMANDS)), runner, work_dir=tmp_path)
    result = pipeline.run_for_issue(_issue())
    assert [call["branch"] for call in runner.calls] == ["none"]
    assert result.outcome == Outcome.REPRODUCED_WEAKER
    assert result.substrate_override is None
    assert result.extraction_record["substrate_override"] is None
    assert not any("substrate override" in line for line in extraction_record.render(result.extraction_record))
