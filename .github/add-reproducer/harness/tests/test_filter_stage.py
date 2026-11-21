import json
from pathlib import Path

import filter_stage
from models import Issue

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def test_keeps_bug_shaped_issue():
    issue = _load_issue(2639)
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "KEEP"


def test_drops_feature_request_label():
    issue = _load_issue(9001)
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "DROP"
    assert "label=" in reason


def test_drops_repo_outside_v1_allowlist():
    issue = Issue(
        number=1,
        title="some bug",
        body="",
        labels=[],
        state="OPEN",
        created_at="",
        author="x",
        repo="canonical/hyrum",
    )
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "DROP"
    assert "v1 allowlist" in reason


def test_extract_ci_run_url_from_2484_body():
    issue = _load_issue(2484)
    url = filter_stage.extract_ci_run_url(issue.body)
    assert url == "https://github.com/canonical/operator/actions/runs/26018272398"


def test_extract_ci_run_url_absent():
    issue = _load_issue(2639)
    assert filter_stage.extract_ci_run_url(issue.body) is None


def test_body_has_reproducer_fenced_block():
    body = "steps:\n```\n$ juju deploy foo\n$ juju status\n$ juju debug-log\n```"
    has_repro, why = filter_stage.body_has_reproducer(body)
    assert has_repro
    assert "fenced" in why


# PLAN.md Approach §2 gap 2 (spike-step-5/corpus-v2/RESULT.md "Two filter
# findings" #1): `gh` has no `issueType` field, so nothing ever supplied the
# `classify_issue(issue, issue_type=...)` argument in practice. Production
# gets the type from the `issues.opened` webhook payload's `issue.type.name`.


def _webhook_issue(number: int, type_name) -> Issue:
    d = json.loads((FIXTURES / "issues" / f"{number}.json").read_text())
    d["type"] = {"name": type_name} if type_name is not None else None
    return Issue.from_dict(d)


def test_webhook_shaped_issue_type_feature_is_dropped():
    # #9001 also carries the "enhancement" label, so use a bug-shaped body/
    # labels but stamp a Feature type to isolate the issue-type rule.
    issue = _webhook_issue(2639, "Feature")
    assert issue.issue_type == "Feature"
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "DROP"
    assert reason == "issue-type=Feature"


def test_issue_type_match_is_case_insensitive():
    issue = _webhook_issue(2639, "feature")
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "DROP"


def test_webhook_shaped_issue_type_bug_is_kept():
    issue = _webhook_issue(2639, "Bug")
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "KEEP"


def test_missing_issue_type_key_defaults_to_none():
    d = json.loads((FIXTURES / "issues" / "2639.json").read_text())
    assert "type" not in d
    issue = Issue.from_dict(d)
    assert issue.issue_type is None


def test_plain_string_type_is_tolerated():
    issue = Issue.from_dict(
        {
            "number": 1,
            "title": "x",
            "body": "",
            "labels": [],
            "state": "OPEN",
            "author": "x",
            "repo": "canonical/operator",
            "type": "Task",
        }
    )
    assert issue.issue_type == "Task"
    verdict, _ = filter_stage.classify_issue(issue)
    assert verdict == "DROP"


def test_explicit_issue_type_argument_still_overrides():
    # An explicit argument (still unused by any real caller, but the
    # parameter stays for callers that DO have a type from elsewhere) wins
    # over issue.issue_type.
    issue = _webhook_issue(2639, "Bug")
    verdict, reason = filter_stage.classify_issue(issue, issue_type="Feature")
    assert verdict == "DROP"
    assert reason == "issue-type=Feature"


# PLAN.md Approach §2 gap 1 (spike-step-5/corpus-v2/RESULT.md "Two filter
# findings" #2): `operator`'s real label vocabulary barely overlaps
# DROP_LABELS.


def test_refactoring_label_drops():
    issue = Issue(
        number=1,
        title="Simplify the retry loop",
        body="",
        labels=["refactoring"],
        state="OPEN",
        created_at="",
        author="x",
        repo="canonical/operator",
    )
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "DROP"
    assert "label=refactoring" in reason


def test_needs_design_and_roadmap_labels_do_not_drop_a_real_bug():
    # fixtures/issues/2639.json is labelled ["needs design", "roadmap"] and is
    # the one hypothesis this project has ever confirmed reproduces
    # end-to-end (spike-step-4/2639/RESULT.md). Both labels mark a
    # triage/planning state, not bug-vs-not-bug (hand-labels.json's #1109 is a
    # real bug carrying "roadmap" alone) -- see filter_stage.py's comment
    # above DROP_LABELS for the full reasoning. This test pins that decision:
    # neither label may be added to DROP_LABELS without re-breaking this case.
    issue = _load_issue(2639)
    assert set(issue.labels) == {"needs design", "roadmap"}
    verdict, reason = filter_stage.classify_issue(issue)
    assert verdict == "KEEP"
