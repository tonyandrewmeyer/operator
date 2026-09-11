import json
from pathlib import Path

import pytest
import runner_stage
from models import Hypothesis, Issue
from seams.llm import FixtureLLM
from seams.runner import FixtureRunnerSeam

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def _load_hypothesis(number: int) -> Hypothesis:
    raw = json.loads((FIXTURES / "extractions" / f"{number}.json").read_text())
    return Hypothesis.from_dict(number, raw)


def _runner():
    return FixtureRunnerSeam(FIXTURES)


def test_2341_skipped_stale_repo_version_null_and_closed():
    hyp, issue = _load_hypothesis(2341), _load_issue(2341)
    assert issue.state == "CLOSED"
    assert hyp.moving_parts.repo_version is None
    stale, reason = runner_stage.is_stale(hyp, issue, _runner(), {})
    assert stale
    assert "CLOSED" in reason


def test_2327_skipped_stale_repo_version_null_and_closed():
    hyp, issue = _load_hypothesis(2327), _load_issue(2327)
    stale, reason = runner_stage.is_stale(hyp, issue, _runner(), {})
    assert stale


def test_2484_not_stale_open_issue():
    hyp, issue = _load_hypothesis(2484), _load_issue(2484)
    assert issue.state == "OPEN"
    stale, reason = runner_stage.is_stale(hyp, issue, _runner(), {})
    assert not stale


def test_2639_not_stale_open_issue():
    hyp, issue = _load_hypothesis(2639), _load_issue(2639)
    stale, reason = runner_stage.is_stale(hyp, issue, _runner(), {})
    assert not stale


def test_symbol_anchor_unresolved_triggers_stale_gate():
    hyp = Hypothesis.from_dict(
        1,
        {
            "in_scope": True,
            "moving_parts": {"substrate": "none", "symbol_anchor": "ops.testing.scenario.Relation.relation_id"},
            "commands": [],
            "expected": "",
            "observed": "",
            "confidence": "medium",
        },
    )
    issue = Issue(number=1, title="t", body="", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    stale, reason = runner_stage.is_stale(hyp, issue, _runner(), {})
    assert stale
    assert "symbol_anchor" in reason


def test_choose_branch_none_for_substrate_none():
    assert runner_stage.choose_branch(_load_hypothesis(2341)) == "none"
    assert runner_stage.choose_branch(_load_hypothesis(2327)) == "none"


def test_choose_branch_k8s_scratch_for_2639():
    # No ci_run_url, k8s substrate, no CI-only signal -> deployed scratch charm.
    assert runner_stage.choose_branch(_load_hypothesis(2639)) == "k8s-scratch"


@pytest.mark.parametrize("substrate", [None, "kubernetes", ""])
def test_choose_branch_refuses_to_guess_a_branch(substrate):
    # Previously fell through to "k8s-scratch", silently provisioning a
    # cluster for an extraction that never named one
    # (`spike-step-5/live-llm/RESULT.md` Finding 1). validate() rejects these
    # first; this is the belt-and-braces for any path that skips validation.
    hyp = Hypothesis.from_dict(
        1,
        {
            "in_scope": True,
            "moving_parts": {"substrate": substrate},
            "commands": [],
            "expected": "",
            "observed": "",
            "confidence": "medium",
        },
    )
    with pytest.raises(ValueError, match="substrate"):
        runner_stage.choose_branch(hyp)


def test_2107_skipped_stale_repo_version_null_and_closed():
    # The one real substrate:lxd extraction in the corpus (see
    # fixtures/extractions/2107.json's _provenance) is also CLOSED with
    # repo_version null, same 2/2-plus-one pattern as #2341/#2327 --
    # meaning this real case never actually reaches choose_branch()/the new
    # lxd-scratch runner via the full pipeline. Recorded plainly rather than
    # routed around: see test_choose_branch_lxd_scratch_for_2107 below and
    # test_seams_runner.py for how lxd-scratch dispatch is actually
    # exercised.
    hyp, issue = _load_hypothesis(2107), _load_issue(2107)
    assert issue.state == "CLOSED"
    assert hyp.moving_parts.repo_version is None
    stale, reason = runner_stage.is_stale(hyp, issue, _runner(), {})
    assert stale
    assert "CLOSED" in reason


def test_choose_branch_lxd_scratch_for_2107():
    # Bypassing the stale gate on purpose (as the test above's docstring
    # explains) to confirm branch dispatch itself -- this is the extraction
    # that used to crash the pipeline with `ValueError: unknown branch
    # 'lxd-scratch'` before seams/runner.py grew `_run_lxd_scratch`.
    assert runner_stage.choose_branch(_load_hypothesis(2107)) == "lxd-scratch"


def test_choose_branch_lxd_scratch_for_lxd_substrate():
    hyp = Hypothesis.from_dict(
        1,
        {
            "in_scope": True,
            "moving_parts": {"substrate": "lxd"},
            "commands": ["juju deploy ./x.charm"],
            "expected": "",
            "observed": "",
            "confidence": "medium",
        },
    )
    assert runner_stage.choose_branch(hyp) == "lxd-scratch"


def test_choose_branch_k8s_clone_for_2484_third_branch_heuristic():
    # ci_run_url set AND no self-contained python snippet in commands.
    hyp = _load_hypothesis(2484)
    assert hyp.moving_parts.ci_run_url is not None
    assert not runner_stage.has_self_contained_snippet(hyp.commands)
    assert runner_stage.choose_branch(hyp) == "k8s-clone"


def test_choose_branch_does_not_misfire_clone_on_self_contained_snippet():
    # A hypothesis with both a ci_run_url and a self-contained repro
    # snippet should NOT route to the clone branch (spike-step-4's
    # trigger heuristic: ci_run_url AND no snippet).
    hyp = Hypothesis.from_dict(
        1,
        {
            "in_scope": True,
            "moving_parts": {"substrate": "k8s", "ci_run_url": "https://github.com/canonical/operator/actions/runs/1"},
            "commands": ["cat > repro.py << 'EOF'\nimport ops\nEOF", "uv run pytest repro.py"],
            "expected": "",
            "observed": "",
            "confidence": "medium",
        },
    )
    assert runner_stage.choose_branch(hyp) == "k8s-scratch"


def test_run_hypothesis_full_flow_2639():
    # branch is "k8s-scratch", not "none", so test-file synthesis (and
    # therefore the llm argument) is never touched here -- a stub that
    # raises if called would prove it, but FixtureLLM is simplest.
    hyp, issue = _load_hypothesis(2639), _load_issue(2639)
    result = runner_stage.run_hypothesis(hyp, issue, None, _runner(), FixtureLLM(FIXTURES), {})
    assert not result.skipped_stale
    assert result.branch == "k8s-scratch"
    assert result.run_result.hypothesis_number == 2639


def test_run_hypothesis_full_flow_2341_stops_at_gate():
    # Stale gate fires before branch dispatch, so the llm argument is never
    # touched -- None is fine here.
    hyp, issue = _load_hypothesis(2341), _load_issue(2341)
    result = runner_stage.run_hypothesis(hyp, issue, None, _runner(), None, {})
    assert result.skipped_stale
    assert result.run_result is None
