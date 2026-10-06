"""Bugs in how ops handles a hook tool's output (`#2709`).

`ops.testing` replaces the backend that runs hook tools, so it cannot
produce what Juju's hook tools return. On `#2709` the extraction wrote an
`ops.testing` test expecting Juju's "permission denied" in 2 of 24 live runs,
and each failed on its own assertion with or without the fix, which rung 6
reads as a reproduction (`spike-step-5/static-retry/RESULT.md` §10, §11).
These tests cover the three parts of the change: the extraction's fake hook
tool shape and its "needs_juju" decline, the static check's `_backend` rule,
and the composer's caveat when a test fakes hook tools.

The calibration corpus is the 24 saved `#2709` extractions under
`fixtures/static_test_check/2709.json` (§10, §11, §13).
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

import extraction
from composer import AUTOMATION_PREFIX, Composer, compose_template
from models import CommandResult, Hypothesis, Issue, MovingParts, Outcome, RunResult
from pipeline import Pipeline
from static_test_check import check, faked_hook_tools

FIXTURES = Path(__file__).parent.parent / "fixtures"

# The test from `#2709`'s thread, in the shape the prompt asks for: it fails
# on an assertion on ops 3.8.3 and passes with the fix.
_FAKE_TOOL_TEST = '''\
import os

import ops
from ops.model import _ModelBackend


def fake_hook_tool(bin_dir, name, script):
    path = bin_dir / name
    path.write_text('#!/bin/sh\\n' + script + '\\n')
    path.chmod(0o755)


def test_gone_relation_databag_reads_empty(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('JUJU_VERSION', '3.6.27')
    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'["db:2"]\\'')
    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'[]\\'')
    fake_hook_tool(bin_dir, 'relation-get', 'echo "ERROR permission denied" >&2; exit 1')
    fake_hook_tool(bin_dir, 'is-leader', 'echo false')
    meta = ops.CharmMeta.from_yaml('name: myapp\\nrequires:\\n  db:\\n    interface: db\\n')
    model = ops.Model(meta, _ModelBackend('myapp/0'))
    relation = model.relations['db'][0]
    try:
        data = dict(relation.data[relation.app])
    except ops.ModelError as e:
        data = e
    assert data == {}
'''


def _corpus() -> list[dict]:
    return json.loads((FIXTURES / "static_test_check" / "2709.json").read_text())


def _prompt_example() -> str:
    """The fake hook tool example in the extraction prompt, as a file."""
    text = extraction._SCHEMA_INSTRUCTIONS
    start = text.index("  import os\n")
    end = text.index("      assert config == {}\n") + len("      assert config == {}\n")
    return textwrap.dedent(text[start:end])


# --- the static check --------------------------------------------------------


def test_the_backend_rule_rejects_only_the_backend_test_in_the_corpus():
    """§10 run 3 is the only one of the 24 that reaches into the backend; §11
    run 6, the other false reproduction, uses public API and is left to the
    prompt."""
    rejected = []
    for record in _corpus():
        if record["test_file"] is None:
            continue
        reasons = check(record["test_file"]["body"]).reasons
        if any("replaces the model backend" in reason for reason in reasons):
            rejected.append((record["source"], record["run"]))
    assert rejected == [("retry-2709.json", 3)]


def test_the_backend_rule_names_the_line_and_the_alternative():
    body = textwrap.dedent("""\
        import ops
        from ops import testing


        class MyCharm(ops.CharmBase):
            def __init__(self, framework):
                super().__init__(framework)
                framework.observe(self.on.start, self._on_start)

            def _on_start(self, event):
                self.model._backend.relation_get(1, 'unit/0', False)


        def test_start():
            ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
            ctx.run(ctx.on.start(), testing.State())
    """)
    assert check(body).reasons == [
        "line 11: `self.model._backend` under `ops.testing`, which replaces the model backend, "
        "so no hook tool runs and nothing Juju's hook tools return can happen; for a bug in how "
        "ops handles a hook tool's output, test `ops.model._ModelBackend` with fake hook tools "
        "instead"
    ]


def test_the_backend_rule_is_off_without_a_context():
    body = "import ops\n\n\ndef test_x():\n    model = object()\n    model._backend\n"
    assert check(body).passed


def test_the_fake_tool_test_passes_the_check():
    assert check(_FAKE_TOOL_TEST).passed, check(_FAKE_TOOL_TEST).reasons


def test_the_prompt_example_passes_the_check():
    example = _prompt_example()
    assert check(example).passed, check(example).reasons


# --- faked_hook_tools() ------------------------------------------------------


def test_faked_hook_tools_names_each_tool_once():
    assert faked_hook_tools(_FAKE_TOOL_TEST) == ["is-leader", "relation-get", "relation-ids", "relation-list"]


def test_faked_hook_tools_in_the_prompt_example():
    assert faked_hook_tools(_prompt_example()) == ["config-get"]


def test_an_ops_testing_test_fakes_no_hook_tools():
    """A hook tool's name in an `ops.testing` test is not a fake: nothing
    runs it."""
    body = "from ops import testing\n\n\ndef test_x():\n    assert 'relation-get'\n"
    assert faked_hook_tools(body) == []


def test_the_corpus_fakes_no_hook_tools():
    assert all(
        faked_hook_tools(record["test_file"]["body"]) == [] for record in _corpus() if record["test_file"]
    )


def test_faked_hook_tools_on_a_file_that_does_not_parse():
    assert faked_hook_tools("def (") == []


# --- the extraction prompt ---------------------------------------------------


def test_the_prompt_says_how_to_decline():
    text = extraction._SCHEMA_INSTRUCTIONS
    assert 'moving_parts.other["needs_juju"]' in text
    assert "Install `ops` and `pytest`, not\n  `ops[testing]`." in text


def test_the_prompt_example_has_no_doubled_escapes():
    """The instructions are a plain string, so `\\n` in the source is `\n`
    in what the model reads, as it would be in the test file."""
    assert re.search(r'"#!/bin/sh\\n" \+ script \+ "\\n"', extraction._SCHEMA_INSTRUCTIONS)


# --- the pipeline ------------------------------------------------------------


class _StaticLLM:
    def __init__(self, extraction: dict):
        self.extraction = extraction
        self.calls: list[dict] = []

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        self.calls.append({"purpose": purpose})
        return self.extraction


class _NoRunner:
    runs = 0

    def run(self, **kwargs) -> RunResult:
        self.runs += 1
        raise AssertionError("the runner should not be reached")


def _issue() -> Issue:
    return Issue(
        number=1, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator"
    )


def _needs_juju_extraction(substrate: str = "none") -> dict:
    return {
        "in_scope": True,
        "moving_parts": {
            "substrate": substrate,
            "repo_version": "main",
            "other": {"needs_juju": "relation-get fails, and the issue does not say with what"},
        },
        "commands": [],
        "expected": "the databag reads as empty",
        "observed": "ModelError",
        "confidence": "low",
    }


@pytest.mark.parametrize("calibration_mode", [False, True])
def test_needs_juju_stops_with_its_own_outcome(tmp_path, calibration_mode):
    runner = _NoRunner()
    pipeline = Pipeline(_StaticLLM(_needs_juju_extraction()), runner, work_dir=tmp_path)
    result = pipeline.run_for_issue(_issue(), calibration_mode=calibration_mode)
    assert runner.runs == 0
    assert result.stage_reached == "extraction:needs_juju"
    assert result.outcome == Outcome.SKIPPED_NEEDS_JUJU
    assert result.comment is None
    assert "relation-get fails, and the issue does not say with what" in result.reason


# --- the composer ------------------------------------------------------------


def _fake_tool_case() -> tuple[Hypothesis, Issue, RunResult]:
    commands = [
        "uv init --bare .",
        "uv add ops pytest",
        f"cat > test_gone.py << 'PYEOF'\n{_FAKE_TOOL_TEST}PYEOF",
        "uv run pytest test_gone.py -v",
    ]
    hyp = Hypothesis(
        issue_number=1,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=commands,
        expected="the databag reads as empty",
        observed="ModelError",
        confidence="high",
    )
    run = RunResult(
        hypothesis_number=1,
        branch="none",
        commands=[CommandResult(command=c, exit_code=0 if i < 3 else 1) for i, c in enumerate(commands)],
    )
    return hyp, _issue(), run


_CAVEAT = (
    "> The hook tools this test runs (`is-leader`, `relation-get`, `relation-ids`, "
    "`relation-list`) are fakes that print what the issue reports Juju returning. The run "
    "shows how ops handles that output, not that Juju returns it."
)


def test_template_carries_the_caveat_under_the_prefix():
    hyp, issue, run = _fake_tool_case()
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_WEAKER, "r", run_id="r1", timestamp="t")
    assert body.startswith(f"{AUTOMATION_PREFIX}\n>\n{_CAVEAT}\n\n")


def test_llm_path_carries_the_caveat_under_the_prefix():
    hyp, issue, run = _fake_tool_case()
    block = "\n".join(f"$ {c.command}" for c in run.commands)
    llm = _StaticLLM({"comment_body": f"Reproduced.\n\n```shell\n{block}\n```\n"})
    body = Composer(llm).compose(hyp, issue, run, Outcome.REPRODUCED_WEAKER, "r", run_id="r1", timestamp="t")
    assert body.startswith(f"{AUTOMATION_PREFIX}\n>\n{_CAVEAT}\n\nReproduced.")


def test_no_caveat_without_fakes():
    hyp, issue, run = _fake_tool_case()
    hyp.commands = ["uv init --bare .", "uv run pytest -v"]
    body = compose_template(hyp, issue, run, Outcome.REPRODUCED_WEAKER, "r", run_id="r1", timestamp="t")
    assert body.startswith(f"{AUTOMATION_PREFIX}\n\n")
    assert "fakes" not in body
