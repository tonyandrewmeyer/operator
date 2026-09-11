import json
from pathlib import Path

import pytest

from models import CommandResult, Outcome, RunResult
from pipeline import Pipeline, build_pipeline

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int):
    from models import Issue

    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


@pytest.fixture
def pipeline(tmp_path) -> Pipeline:
    return build_pipeline(fixture_mode=True, fixtures_dir=FIXTURES, work_dir=tmp_path)


def test_9001_dropped_at_filter(pipeline):
    result = pipeline.run_for_issue(_load_issue(9001))
    assert result.stage_reached == "filter"
    assert result.outcome is None
    assert result.comment is None


def test_2304_dropped_at_extraction_in_scope_false(pipeline):
    result = pipeline.run_for_issue(_load_issue(2304))
    assert result.stage_reached == "extraction"
    assert result.outcome is None
    assert "in_scope" in result.reason


def test_2341_skipped_stale_full_pipeline(pipeline):
    # confidence=high, so it clears the confidence gate and reaches the
    # Approach §4 skip-when-stale gate on its own (repo_version null + CLOSED).
    result = pipeline.run_for_issue(_load_issue(2341))
    assert result.outcome == Outcome.SKIPPED_STALE
    assert result.comment is None


def test_2327_skipped_stale_full_pipeline(pipeline):
    # confidence=medium, clears the gate the same way.
    result = pipeline.run_for_issue(_load_issue(2327))
    assert result.outcome == Outcome.SKIPPED_STALE
    assert result.comment is None


def test_2484_low_confidence_stays_silent_in_production_mode(pipeline):
    # #2484's fixture carries a `ci_run_url` with no self-contained snippet,
    # so it routes to `k8s-clone` -- a branch that DOES execute `commands[]`
    # (`runner_stage.COMMAND_EXECUTING_BRANCHES`), unlike `k8s-scratch`
    # (see the #2639 tests below). The confidence gate still has to fire
    # here: this pins that the scratch-branch exemption is scoped to
    # branches that don't execute `commands[]`, not to "any k8s hypothesis".
    result = pipeline.run_for_issue(_load_issue(2484))
    assert result.stage_reached == "extraction"
    assert result.outcome is None
    assert "confidence=low" in result.reason


def test_2484_calibration_mode_reaches_classifier(pipeline):
    result = pipeline.run_for_issue(_load_issue(2484), calibration_mode=True)
    assert result.outcome == Outcome.UNRUNNABLE_TEST_SELECTOR_STALE
    assert result.comment is None


def test_2639_confidence_gate_exempt_on_scratch_branch(pipeline):
    # #2639 is `substrate: k8s` with no `ci_run_url`, so it routes to
    # `k8s-scratch` -- a branch that builds its reproduction sequence from
    # `surface`, never from `commands[]`, which is exactly what the
    # extractor's `confidence` field is calibrated against (extraction.py:
    # "if you cannot produce a sequence like that ... set confidence to
    # low"). `pipeline.py` no longer gates that branch on confidence, so
    # production mode now behaves the same as calibration mode for this
    # case. Fixed per spike-step-5/confidence-gate-2639/RESULT.md: this test
    # used to assert the opposite ("stays silent in production mode"), which
    # was the bug -- #2639 is the one hypothesis this project has ever
    # confirmed reproduces end-to-end, and the confidence gate silenced it.
    result = pipeline.run_for_issue(_load_issue(2639), run_id="test-run", timestamp="2026-07-23T00:00:00Z")
    if result.stage_reached == "scaffold":
        pytest.skip(f"scratch-charm render needs network access for `uv lock`, unavailable here: {result.reason}")
    assert result.outcome == Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT
    assert result.comment is not None
    assert "add-reproducer:issue=2639:run=test-run" in result.comment


def test_2639_calibration_mode_reproduces(pipeline):
    result = pipeline.run_for_issue(_load_issue(2639), calibration_mode=True, run_id="test-run", timestamp="2026-07-23T00:00:00Z")
    if result.stage_reached == "scaffold":
        pytest.skip(f"scratch-charm render needs network access for `uv lock`, unavailable here: {result.reason}")
    assert result.outcome == Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT
    assert result.comment is not None
    assert "add-reproducer:issue=2639:run=test-run" in result.comment


def test_confidence_gate_still_silences_none_substrate(pipeline, monkeypatch):
    # The scratch-branch exemption above is scoped to branches that don't
    # execute `commands[]` at all. A `substrate: none` hypothesis (where
    # `commands[]` IS the whole reproduction) must still stay silent at
    # `confidence: low` -- this pins that the exemption didn't quietly widen
    # into "confidence gate does nothing".
    from models import Hypothesis

    hyp = Hypothesis.from_dict(
        2304,
        {
            "in_scope": True,
            "moving_parts": {"substrate": "none", "other": {}},
            "commands": [],
            "expected": "x",
            "observed": "y",
            "confidence": "low",
        },
    )
    monkeypatch.setattr(pipeline.extractor, "extract", lambda issue: hyp)
    result = pipeline.run_for_issue(_load_issue(2304))
    assert result.stage_reached == "extraction"
    assert result.outcome is None
    assert "confidence=low" in result.reason


def test_2045_calibration_mode_composed_comment_includes_synthesized_test_file(pipeline):
    # spike-step-5/composer-live/RESULT.md Finding 4, closed: #2045 is the
    # real case whose commands[] has no runnable pytest invocation, so
    # run_for_issue() synthesizes a test file before running. The composed
    # comment (compose_template, since no compositions/2045.json fixture
    # exists) must show that file's body, not just reference its name in
    # the commands block.
    result = pipeline.run_for_issue(_load_issue(2045), calibration_mode=True, run_id="test-run", timestamp="2026-07-30T00:00:00Z")
    assert result.comment is not None
    assert "test_issue_2045_repro.py" in result.comment
    assert "Synthesized test file" in result.comment
    assert result.comment.index("Synthesized test file") < result.comment.index("Commands run:")


def test_2107_skipped_stale_full_pipeline(pipeline):
    # The real #2107 (the only substrate:lxd extraction in the corpus,
    # see fixtures/extractions/2107.json's _provenance) is CLOSED with
    # repo_version null -- caught by the Approach §4 stale gate before ever
    # reaching branch dispatch, same as #2341/#2327. Confirms the full
    # pipeline doesn't crash on a real substrate:lxd issue and that the
    # gate short-circuits it exactly as it does for k8s/none hypotheses.
    result = pipeline.run_for_issue(_load_issue(2107), calibration_mode=True)
    assert result.outcome == Outcome.SKIPPED_STALE
    assert result.comment is None


class _StaticLLM:
    """Minimal LLMSeam stub returning the same canned dict regardless of
    prompt -- lets a test drive `Pipeline.run_for_issue()` through the
    lxd-scratch branch with a synthetic (not a real GitHub) issue, without
    adding a new on-disk issue fixture."""

    def __init__(self, extraction: dict, surface: dict):
        self._extraction = extraction
        self._surface = surface

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        return self._extraction if purpose == "extraction" else self._surface


class _StaticRunner:
    def __init__(self, result: RunResult):
        self._result = result

    def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
        return True

    def run(self, **kwargs) -> RunResult:
        return self._result


def test_lxd_scratch_reaches_classifier_via_full_pipeline(tmp_path):
    # Demonstrates the fix end-to-end at the Pipeline level: before
    # seams/runner.py grew `_run_lxd_scratch`, any substrate:lxd hypothesis
    # that made it this far crashed with `ValueError: unknown branch
    # 'lxd-scratch'`. Issue number 1 here is deliberately synthetic (not a
    # real GitHub issue, per the task's instruction not to invent one for
    # the fixture corpus) -- OPEN with repo_version set so it clears the
    # stale gate, which #2107 (the one real lxd extraction) does not.
    llm = _StaticLLM(
        extraction={
            "in_scope": True,
            "moving_parts": {"substrate": "lxd", "repo_version": "main"},
            "commands": [],
            "expected": "the charm reacts to the reported condition",
            "observed": "it does not",
            "confidence": "high",
        },
        # A complete `pebble_service`: with `user`/`command` missing, the
        # no-stimulus gate now stops this before the runner (correctly --
        # such a run deploys a charm and pokes nothing), and this test is
        # about lxd-scratch *dispatch*, not that gate.
        surface={
            "charm_name": "repro-i1-x",
            "pebble_service": {"service": "workload", "user": "ubuntu", "command": "pebble notify x"},
        },
    )
    run_result = RunResult(
        hypothesis_number=1,
        branch="lxd-scratch",
        commands=[
            CommandResult(command="sudo concierge prepare -p lxd", exit_code=0, stdout="ready"),
            CommandResult(command="juju status repro/0", exit_code=0, stdout="Workload: active"),
        ],
    )
    pipeline = Pipeline(llm, _StaticRunner(run_result), work_dir=tmp_path)
    from models import Issue

    issue = Issue(
        number=1, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator"
    )

    result = pipeline.run_for_issue(issue, calibration_mode=True)

    if result.stage_reached == "scaffold":
        pytest.skip(f"scratch-charm render needs network access for `uv lock`, unavailable here: {result.reason}")
    assert result.stage_reached == "classifier"
    assert result.outcome == Outcome.DID_NOT_REPRODUCE


def test_ignore_in_scope_runs_an_out_of_scope_hypothesis(tmp_path):
    """`--ignore-in-scope` is a measurement tool: production drops these,
    and a pipeline that composes nothing cannot be measured for false
    comments (40 real issues, 2026-08-18: 0 comments)."""
    extraction = {
        "in_scope": False,
        "moving_parts": {"substrate": "none", "repo_version": "main"},
        "commands": ["uv venv", "cat > t.py << 'EOF'\nimport ops\nEOF", "uv run pytest t.py -v"],
        "expected": "x",
        "observed": "y",
        "confidence": "high",
    }

    class _OutOfScopeLLM:
        """First pass says out of scope; the second pass agrees
        (`concrete_defect: False`), which is the shape production drops."""

        def complete_json(self, *, purpose, prompt, context):
            if purpose == "inscope_second_pass":
                return {**extraction, "concrete_defect": False}
            return extraction

    llm = _OutOfScopeLLM()
    run_result = RunResult(
        hypothesis_number=1,
        branch="none",
        commands=[CommandResult(command="uv run pytest t.py -v", exit_code=0, stdout="1 passed")],
    )
    from models import Issue

    issue = Issue(
        number=1, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator"
    )
    pipeline = Pipeline(llm, _StaticRunner(run_result), work_dir=tmp_path)

    dropped = pipeline.run_for_issue(issue, calibration_mode=True)
    assert dropped.stage_reached == "extraction"
    assert "in_scope=false" in dropped.reason

    forced = pipeline.run_for_issue(issue, calibration_mode=True, ignore_in_scope=True)
    assert forced.stage_reached == "classifier"


def test_each_issue_gets_its_own_scratch_dir(tmp_path):
    """A shared workdir made `uv init` collide across issues and put every
    synthesised test in one pytest rootdir."""
    p = build_pipeline(fixture_mode=True, fixtures_dir=FIXTURES, work_dir=tmp_path)
    p.run_for_issue(_load_issue(2045), calibration_mode=True)
    p.run_for_issue(_load_issue(2484), calibration_mode=True)
    assert (tmp_path / "issue-2045").is_dir()
    assert (tmp_path / "issue-2484").is_dir()


def test_fixture_llm_splits_the_seams(tmp_path):
    """`--fixture-llm` is how a substrate measurement runs the real runner
    without spending an OpenRouter call on an extraction already recorded in
    `fixtures/`. Neither `fixture_mode` nor `--live` can express that on its
    own -- both set the two seams together."""
    from seams.llm import FixtureLLM
    from seams.runner import SubprocessRunnerSeam

    p = build_pipeline(fixture_mode=False, fixtures_dir=FIXTURES, work_dir=tmp_path, fixture_llm=True)
    assert isinstance(p.llm, FixtureLLM)
    assert isinstance(p.runner, SubprocessRunnerSeam)


def test_fixture_llm_defaults_off_for_live(tmp_path, monkeypatch):
    """Without the flag, `--live` still means live on both seams -- the split
    must never be something a production run can fall into by accident."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-not-a-real-key")
    from seams.llm import LiveOpenRouterLLM

    p = build_pipeline(fixture_mode=False, fixtures_dir=FIXTURES, work_dir=tmp_path)
    assert isinstance(p.llm, LiveOpenRouterLLM)
