"""Tests for `../run.py`, the GitHub Actions entry point.

Only one thing in it is worth pinning and easy to get wrong: the REST API
and the shape `models.Issue.from_dict` reads disagree about nearly every
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
