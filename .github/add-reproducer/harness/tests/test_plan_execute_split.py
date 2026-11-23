"""The pipeline split into `plan_for_issue()` and `execute_plan()`.

The workflow runs the two halves in separate jobs, so that the job holding
the OpenRouter key never runs a command taken from the issue, and the job
running those commands holds no secrets. These tests pin the two properties
that rely on: the plan survives JSON, and executing it needs no model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models import Issue
from pipeline import Pipeline, PipelineResult, RunPlan, build_pipeline
from seams.llm import LLMError, NoLLM
from seams.runner import FixtureRunnerSeam

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def _plan(tmp_path: Path, number: int) -> RunPlan:
    pipeline = build_pipeline(fixture_mode=True, fixtures_dir=FIXTURES, work_dir=tmp_path / "plan")
    planned = pipeline.plan_for_issue(
        _load_issue(number), calibration_mode=True, run_id="test-run", timestamp="2026-07-30T00:00:00Z"
    )
    assert isinstance(planned, RunPlan)
    return planned


@pytest.mark.parametrize("number", [2045, 2639])
def test_a_plan_survives_json(tmp_path, number):
    planned = _plan(tmp_path, number)
    assert RunPlan.from_dict(json.loads(json.dumps(planned.to_dict()))) == planned


def test_planning_writes_nothing_the_run_needs(tmp_path):
    # The synthesised test file travels in the plan, not on disk: the job
    # that plans is not the job that runs.
    planned = _plan(tmp_path, 2045)
    assert planned.hypothesis.synthesized_test_file is not None
    assert not list(tmp_path.rglob(planned.hypothesis.synthesized_test_file.path))


def test_a_plan_executes_without_a_model(tmp_path):
    planned = RunPlan.from_dict(json.loads(json.dumps(_plan(tmp_path, 2045).to_dict())))
    pipeline = Pipeline(NoLLM(), FixtureRunnerSeam(FIXTURES), work_dir=tmp_path / "execute")
    result = pipeline.execute_plan(planned)
    assert result.comment is not None
    assert "test_issue_2045_repro.py" in result.comment
    assert list((tmp_path / "execute").rglob("test_issue_2045_repro.py"))
    assert result.extraction_record == planned.extraction_record


def test_split_and_whole_runs_compose_the_same_comment(tmp_path):
    whole = build_pipeline(fixture_mode=True, fixtures_dir=FIXTURES, work_dir=tmp_path / "whole").run_for_issue(
        _load_issue(2045), calibration_mode=True, run_id="test-run", timestamp="2026-07-30T00:00:00Z"
    )
    planned = _plan(tmp_path, 2045)
    split = Pipeline(NoLLM(), FixtureRunnerSeam(FIXTURES), work_dir=tmp_path / "execute").execute_plan(planned)
    assert split.comment == whole.comment
    assert split.outcome == whole.outcome


def test_a_gate_stop_is_a_result_not_a_plan(tmp_path):
    pipeline = build_pipeline(fixture_mode=True, fixtures_dir=FIXTURES, work_dir=tmp_path)
    assert isinstance(pipeline.plan_for_issue(_load_issue(9001)), PipelineResult)


def test_no_llm_refuses_every_call():
    with pytest.raises(LLMError):
        NoLLM().complete_json(purpose="extraction", prompt="", context={})
