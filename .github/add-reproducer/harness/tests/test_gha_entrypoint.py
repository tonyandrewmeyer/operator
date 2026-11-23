"""Tests for `../run.py`, the GitHub Actions entry point.

Mostly the field mapping, which is easy to get wrong: the REST API and the
shape `models.Issue.from_dict` reads disagree about nearly every
field name. `created_at` vs `createdAt`, a nested `user.login` vs a bare
`author`, lower-case `open` vs upper-case `OPEN` -- and a mistake in any of
them is silent, because `from_dict` fills a default rather than raising. An
issue whose `state` arrives as `open` instead of `OPEN` is not an error
anywhere; it just quietly stops matching.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_RUN_PY = Path(__file__).parent.parent.parent / "run.py"


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("add_reproducer_run", _RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RAW = {
    "number": 2639,
    "title": "Pebble custom notice is not fired",
    "body": "It does not fire.",
    "labels": [{"id": 1, "name": "bug"}, {"id": 2, "name": "kind/needs-triage"}],
    "state": "open",
    "created_at": "2026-07-01T09:00:00Z",
    "user": {"login": "reporter"},
    "type": {"id": 7, "name": "Bug"},
    "comments": 2,
}

COMMENTS = [
    {"user": {"login": "reporter"}, "body": "Actually it is worse than that.", "created_at": "2026-07-01T10:00:00Z"},
    {"user": {"login": "maintainer"}, "body": "Which juju?", "created_at": "2026-07-02T11:00:00Z"},
]


def test_rest_fields_map_onto_the_harness_issue(run):
    issue = run.to_issue(RAW, COMMENTS, "canonical/operator")
    assert issue.number == 2639
    assert issue.title == "Pebble custom notice is not fired"
    assert issue.body == "It does not fire."
    # Label *objects*, not names, are what the API sends.
    assert issue.labels == ["bug", "kind/needs-triage"]
    # `open` -> `OPEN`: the harness's corpus and `gh` both use upper case.
    assert issue.state == "OPEN"
    assert issue.created_at == "2026-07-01T09:00:00Z"
    assert issue.author == "reporter"
    assert issue.repo == "canonical/operator"
    # The issue type is GitHub's own field, and one of the filter's drop
    # rules reads it.
    assert issue.issue_type == "Bug"


def test_comments_are_carried_with_their_authors(run):
    issue = run.to_issue(RAW, COMMENTS, "canonical/operator")
    assert [(c.author, c.created_at) for c in issue.comments] == [
        ("reporter", "2026-07-01T10:00:00Z"),
        ("maintainer", "2026-07-02T11:00:00Z"),
    ]
    assert issue.comments[0].body == "Actually it is worse than that."


def test_missing_optional_fields_do_not_raise(run):
    issue = run.to_issue({"number": 1, "title": "t", "state": "closed"}, [], "canonical/operator")
    assert issue.body == ""
    assert issue.labels == []
    assert issue.state == "CLOSED"
    assert issue.author == ""
    assert issue.issue_type is None


def test_a_pull_request_is_refused(run):
    with pytest.raises(SystemExit):
        run.to_issue({**RAW, "pull_request": {"url": "..."}}, [], "canonical/operator")


def test_marker_prefix_matches_what_the_composer_appends(run):
    from composer import _marker
    from models import Issue

    issue = Issue(
        number=2639,
        title="t",
        body="b",
        labels=[],
        state="OPEN",
        created_at="",
        author="a",
        repo="canonical/operator",
    )
    assert run.MARKER_PREFIX.format(number=2639) in _marker(issue, "34100431032")


# -- the extraction record reaches the artefact, the log and the summary -----
#
# A run that stops at the in-scope gate has no `RunResult`, so the
# extraction record is all `upload-artifact` has to upload.


def _gate_stop_result(number: int = 2484):
    from pipeline import PipelineResult

    return PipelineResult(
        number,
        "extraction",
        None,
        "in_scope=false (second pass ran and confirmed the drop)",
        None,
        extraction_record={
            "schema_version": 1,
            "issue_number": number,
            "first_pass": {"in_scope": False, "confidence": "low", "substrate": "none"},
            "second_pass": {"ran": True, "concrete_defect": False, "recovered": False},
            "final_in_scope": False,
            "llm_calls": [
                {
                    "purpose": "extraction",
                    "model": "deepseek/deepseek-chat",
                    "provider": "DeepSeek",
                    "usage": {"total_tokens": 11},
                }
            ],
        },
    )


@pytest.fixture
def gate_stop(run, monkeypatch, tmp_path):
    from models import Issue

    issue = Issue(
        number=2484,
        title="t",
        body="b",
        labels=[],
        state="OPEN",
        created_at="",
        author="a",
        repo="canonical/operator",
    )

    class _Pipeline:
        def plan_for_issue(self, issue, **kwargs):
            return _gate_stop_result()

    monkeypatch.setattr(run, "fetch_issue", lambda repo, number, token: issue)
    monkeypatch.setattr(run, "build_pipeline", lambda **kwargs: _Pipeline())
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["run.py", "plan", "--repo", "canonical/operator", "--issue", "2484", "--out-dir", str(out_dir)]
    )
    return out_dir, summary


def test_a_gate_stop_writes_an_extraction_artefact(run, gate_stop, capsys):
    out_dir, _ = gate_stop
    assert run.main() == 0

    import json as _json

    record = _json.loads((out_dir / "2484-extraction.json").read_text())
    assert record["first_pass"]["substrate"] == "none"
    assert record["llm_calls"][0]["provider"] == "DeepSeek"


def test_a_gate_stop_prints_the_record_to_the_job_log(run, gate_stop, capsys):
    assert run.main() == 0
    out = capsys.readouterr().out
    # The `stage=` line keeps its shape apart from its parenthetical, so logs
    # read the same way across runs and a grep across them still works.
    assert "stage=extraction" in out
    assert "reason=in_scope=false (second pass ran and confirmed the drop)" in out
    assert "provider=DeepSeek" in out
    assert "substrate=none" in out


def test_a_gate_stop_puts_the_record_in_the_job_summary(run, gate_stop):
    _, summary = gate_stop
    assert run.main() == 0
    text = summary.read_text()
    assert "second pass: ran" in text
    assert "provider=DeepSeek" in text


# -- one subcommand per job ----------------------------------------------


def test_execute_refuses_to_start_with_a_secret_set(run, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["run.py", "execute", "--plan", str(tmp_path / "plan.json"), "--out-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        run.main()


def test_plan_writes_the_plan_and_says_so(run, monkeypatch, tmp_path):
    from pipeline import RunPlan
    from test_plan_execute_split import _plan

    planned = _plan(tmp_path, 2045)

    class _Pipeline:
        def plan_for_issue(self, issue, **kwargs):
            return planned

    monkeypatch.setattr(run, "fetch_issue", lambda repo, number, token: planned.issue)
    monkeypatch.setattr(run, "build_pipeline", lambda **kwargs: _Pipeline())
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["run.py", "plan", "--repo", "canonical/operator", "--issue", "2045", "--out-dir", str(out_dir)]
    )
    assert run.main() == 0
    import json as _json

    assert RunPlan.from_dict(_json.loads((out_dir / "2045-plan.json").read_text())) == planned
    assert output.read_text() == "planned=true\n"


def test_post_with_no_comment_posts_nothing(run, monkeypatch, tmp_path):
    def _no_network(*args, **kwargs):
        raise AssertionError("post reached the network with nothing to post")

    monkeypatch.setattr(run, "_request", _no_network)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "post", "--repo", "canonical/operator", "--issue", "2045", "--comment", str(tmp_path / "2045.md")],
    )
    assert run.main() == 0
