"""Tests for comment-threading in extraction (the `#2185` and `#2045`
shapes).
"""

import json
from pathlib import Path

from extraction import DEFAULT_MAX_COMMENT_CHARS, Extractor, _format_comments
from models import Comment, Issue

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


# -- Comment / Issue model -----------------------------------------------


def test_comment_from_dict_unwraps_nested_author_login():
    # gh issue view --json ...,comments (and the raw REST/GraphQL API) nest
    # the author as {"login": ...}.
    c = Comment.from_dict({"author": {"login": "james-garner-canonical"}, "body": "hi", "createdAt": "2026-01-01T00:00:00Z"})
    assert c.author == "james-garner-canonical"


def test_comment_from_dict_accepts_plain_string_author():
    c = Comment.from_dict({"author": "someone", "body": "hi", "createdAt": ""})
    assert c.author == "someone"


def test_issue_from_dict_parses_comments_list():
    d = {
        "number": 1, "title": "t", "body": "b", "labels": [], "state": "OPEN",
        "createdAt": "", "author": "a", "repo": "canonical/operator",
        "comments": [{"author": {"login": "x"}, "body": "y", "createdAt": "z"}],
    }
    issue = Issue.from_dict(d)
    assert len(issue.comments) == 1
    assert issue.comments[0].author == "x"


def test_issue_from_dict_defaults_comments_to_empty_list():
    # Regression: fixtures/payloads with no "comments" key at all must
    # still parse.
    d = {"number": 1, "title": "t", "body": "b", "labels": [], "state": "OPEN"}
    issue = Issue.from_dict(d)
    assert issue.comments == []


# -- Prompt formatting -----------------------------------------------------


def test_no_comments_produces_empty_section_regression():
    # #2639 has no comments, so the prompt gets no comments section.
    issue = _load_issue(2639)
    assert issue.comments == []
    assert _format_comments(issue, DEFAULT_MAX_COMMENT_CHARS) == ""
    prompt = Extractor._build_prompt(issue, DEFAULT_MAX_COMMENT_CHARS)
    assert "Comments" not in prompt


def test_real_2045_comment_is_attributed_as_other():
    # A verbatim API capture with a follow-up from someone other than the
    # reporter.
    issue = _load_issue(2045)
    assert len(issue.comments) == 1
    comment = issue.comments[0]
    assert comment.author == "james-garner-canonical"
    assert issue.author == "tonyandrewmeyer"
    prompt = Extractor._build_prompt(issue, DEFAULT_MAX_COMMENT_CHARS)
    assert "[1] james-garner-canonical (other)" in prompt
    assert "friction to charms accessing the files" in prompt


def test_scope_redirecting_reply_is_flagged_as_issue_author():
    # Modelled on canonical/operator#2185, where the reporter narrows the
    # issue's scope in the first reply. The reply body here is invented.
    issue = Issue(
        number=99101,
        title="Don't include private-address in the default Scenario database when mocking Juju 4",
        body=(
            "EDIT: see the first reply for the new scope of this issue\n\n---\n\n"
            "The `private-address` key is not included in the default relation "
            "settings (databag) as of Juju 4.0, so we should not include it in "
            "the mock one either, if the mock Juju version is 4.0+."
        ),
        labels=["rainy day", "small item"],
        state="OPEN",
        created_at="2025-11-19T00:00:00Z",
        author="tonyandrewmeyer",
        repo="canonical/operator",
        comments=[
            Comment(
                author="tonyandrewmeyer",
                body=(
                    "[SYNTHETIC -- not real GitHub content] Narrowing scope: let's only handle the lxd-substrate "
                    "case for now; the k8s mock can stay as-is until a separate "
                    "issue tracks it."
                ),
                created_at="2025-11-20T00:00:00Z",
            )
        ],
    )
    prompt = Extractor._build_prompt(issue, DEFAULT_MAX_COMMENT_CHARS)
    assert "[1] tonyandrewmeyer (issue author)" in prompt
    assert "Narrowing scope" in prompt


def test_irrelevant_chatty_reply_is_still_included_verbatim():
    # Extraction doesn't pre-filter comment relevance; the model decides, so
    # even an irrelevant reply must reach the prompt.
    issue = Issue(
        number=99102,
        title="some bug",
        body="steps: run X, see Y crash",
        labels=[],
        state="OPEN",
        created_at="",
        author="reporter1",
        repo="canonical/operator",
        comments=[
            Comment(author="bystander42", body="+1, hit this too!", created_at="2026-01-01T00:00:00Z"),
        ],
    )
    prompt = Extractor._build_prompt(issue, DEFAULT_MAX_COMMENT_CHARS)
    assert "[1] bystander42 (other)" in prompt
    assert "+1, hit this too!" in prompt


def test_truncation_keeps_first_comment_and_marks_omission_visibly():
    long_body = "x" * 3000
    issue = Issue(
        number=99103,
        title="some bug",
        body="body",
        labels=[],
        state="OPEN",
        created_at="",
        author="reporter1",
        repo="canonical/operator",
        comments=[
            Comment(author="reporter1", body="first reply: " + long_body, created_at="2026-01-01T00:00:00Z"),
            Comment(author="other1", body="second reply: " + long_body, created_at="2026-01-02T00:00:00Z"),
            Comment(author="other2", body="third reply, short", created_at="2026-01-03T00:00:00Z"),
        ],
    )
    prompt = Extractor._build_prompt(issue, max_comment_chars=1000)
    assert "first reply:" in prompt  # first comment always included...
    assert long_body not in prompt  # ...but truncated in place, not in full
    assert "further comment(s) omitted" in prompt  # truncation is visible
    assert "second reply:" not in prompt
    assert "third reply, short" not in prompt


def test_extractor_max_comment_chars_is_configurable():
    issue = Issue(
        number=99104,
        title="t",
        body="b",
        labels=[],
        state="OPEN",
        created_at="",
        author="a",
        repo="canonical/operator",
        comments=[Comment(author="a", body="y" * 50, created_at="")],
    )
    tight = Extractor(llm=None, max_comment_chars=10)
    assert tight.max_comment_chars == 10
    default = Extractor(llm=None)
    assert default.max_comment_chars == DEFAULT_MAX_COMMENT_CHARS
