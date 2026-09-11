"""`Issue.from_dict` against the shapes `gh` actually emits.

The whole fixture corpus in `fixtures/issues/` hand-writes `labels` as
bare strings. Real `gh issue list --json labels` emits label *objects*.
Nothing in the suite noticed until the first live run against
`canonical/operator#2639` died in `filter_stage.classify_issue` on
`label.lower()` -- stage 2 of 8, before any LLM call. These tests pin
both shapes so the corpus's convenience shape can't drift back into
being the only one that works.
"""

from models import Issue, RunResult


def _issue_dict(**overrides):
    d = {
        "number": 2639,
        "title": "Juju/ops only reacts to public Pebble custom notices",
        "body": "body",
        "labels": [],
        "state": "OPEN",
        "createdAt": "2026-05-01T00:00:00Z",
        "author": "someone",
        "repo": "canonical/operator",
    }
    d.update(overrides)
    return d


def test_labels_from_real_gh_output_are_reduced_to_names():
    """The shape `gh issue list --json number,...,labels` really returns."""
    issue = Issue.from_dict(
        _issue_dict(
            labels=[
                {
                    "id": "MDU6TGFiZWwyMDA1OTI3MDk5",
                    "name": "needs design",
                    "description": "Needs more thought or a spec",
                    "color": "7df2ee",
                },
                {
                    "id": "LA_kwDODKRcgM8AAAACA-M-jQ",
                    "name": "roadmap",
                    "description": "An official roadmap item",
                    "color": "fbca04",
                },
            ]
        )
    )
    assert issue.labels == ["needs design", "roadmap"]


def test_labels_from_fixture_style_strings_still_work():
    """The shape every `fixtures/issues/*.json` uses."""
    issue = Issue.from_dict(_issue_dict(labels=["needs design", "roadmap"]))
    assert issue.labels == ["needs design", "roadmap"]


def test_labels_survive_a_mixed_list():
    issue = Issue.from_dict(_issue_dict(labels=[{"name": "bug"}, "roadmap"]))
    assert issue.labels == ["bug", "roadmap"]


def test_missing_and_null_labels_are_empty():
    assert Issue.from_dict(_issue_dict(labels=None)).labels == []
    d = _issue_dict()
    del d["labels"]
    assert Issue.from_dict(d).labels == []


def test_label_object_without_a_name_does_not_crash_the_filter():
    """A malformed label must not take the run down -- the filter
    lowercases every entry, so a `None` here would raise the same
    AttributeError the real shape did."""
    issue = Issue.from_dict(_issue_dict(labels=[{"color": "ffffff"}]))
    assert issue.labels == [""]
    assert all(isinstance(label, str) for label in issue.labels)


def _run_result_dict(**overrides):
    d = {
        "hypothesis_number": 2639,
        "branch": "k8s-scratch",
        "commands": [],
    }
    d.update(overrides)
    return d


def test_run_result_records_the_observed_juju_version():
    """`spike-step-5/wallclock-substrate/RESULT.md` §5: nothing recorded
    which juju actually produced a run's verdict, which is how `#2639`
    reproducing on 4.0.5 and not on 3.6.27 went unnoticed."""
    run = RunResult.from_dict(_run_result_dict(observed_juju_version="3.6.27-ubuntu-amd64"))
    assert run.observed_juju_version == "3.6.27-ubuntu-amd64"


def test_run_result_observed_juju_version_is_optional():
    """Every fixture recorded before this field existed has no
    `observed_juju_version` key at all -- it must still load, as `None`,
    not raise a KeyError."""
    run = RunResult.from_dict(_run_result_dict())
    assert run.observed_juju_version is None
