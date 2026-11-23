import json
from pathlib import Path

import pytest

from extraction import Extractor, ExtractionInvalid, validate
from models import Issue
from seams.llm import FixtureLLM

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


@pytest.fixture
def extractor():
    return Extractor(FixtureLLM(FIXTURES))


def test_extract_2639_low_confidence_k8s(extractor):
    hyp = extractor.extract(_load_issue(2639))
    assert hyp.in_scope is True
    assert hyp.confidence == "low"
    assert hyp.moving_parts.substrate == "k8s"
    assert hyp.moving_parts.other.get("pebble_notify_user") == "_daemon_"


def test_extract_2341_high_confidence_none(extractor):
    hyp = extractor.extract(_load_issue(2341))
    assert hyp.confidence == "high"
    assert hyp.moving_parts.substrate == "none"


def test_extract_2327_carries_symbol_anchor(extractor):
    hyp = extractor.extract(_load_issue(2327))
    # The permalink (ops/model.py#L970 at deba324) points inside
    # `RelationMapping._get_unique`, not its caller `Model.get_relation`.
    assert hyp.moving_parts.symbol_anchor == "ops.model.RelationMapping._get_unique"


def test_extract_2484_ci_run_url_is_deterministic_not_llm(extractor):
    # extraction fixture doesn't set ci_run_url; Extractor.extract()
    # fills it in from the issue body regardless of what the fixture said.
    issue = _load_issue(2484)
    raw = json.loads((FIXTURES / "extractions" / "2484.json").read_text())
    assert raw["moving_parts"]["ci_run_url"] == "https://github.com/canonical/operator/actions/runs/26018272398"
    hyp = extractor.extract(issue)
    assert hyp.moving_parts.ci_run_url == "https://github.com/canonical/operator/actions/runs/26018272398"


def test_extract_2304_in_scope_false(extractor):
    hyp = extractor.extract(_load_issue(2304))
    assert hyp.in_scope is False


def test_validate_rejects_missing_key():
    with pytest.raises(ExtractionInvalid):
        validate({"in_scope": True, "moving_parts": {}, "commands": [], "expected": "x", "observed": "y"})


def test_validate_rejects_bad_confidence():
    with pytest.raises(ExtractionInvalid):
        validate(
            {
                "in_scope": True,
                "moving_parts": {},
                "commands": [],
                "expected": "x",
                "observed": "y",
                "confidence": "extremely-high",
            }
        )


def _raw(**overrides) -> dict:
    raw = {
        "in_scope": True,
        "moving_parts": {"substrate": "none"},
        "commands": [],
        "expected": "x",
        "observed": "y",
        "confidence": "medium",
    }
    raw.update(overrides)
    return raw


# `choose_branch` would route a null or unrecognised substrate to
# `k8s-scratch`, so an invalid substrate must fail extraction.
@pytest.mark.parametrize("substrate", [None, "kubernetes", "", "K8s", "microk8s"])
def test_validate_rejects_unusable_substrate_when_in_scope(substrate):
    with pytest.raises(ExtractionInvalid, match="substrate"):
        validate(_raw(moving_parts={"substrate": substrate}))


def test_validate_rejects_omitted_substrate_when_in_scope():
    with pytest.raises(ExtractionInvalid, match="substrate"):
        validate(_raw(moving_parts={}))


@pytest.mark.parametrize("substrate", ["none", "lxd", "k8s"])
def test_validate_accepts_the_three_substrates(substrate):
    validate(_raw(moving_parts={"substrate": substrate}))


def test_validate_allows_null_substrate_when_out_of_scope():
    # A dropped issue never reaches a runner, so it has no substrate to name.
    # This is the shape the hand extraction for #2304 actually carries.
    validate(_raw(in_scope=False, moving_parts={"substrate": None}))
    validate(_raw(in_scope=False, moving_parts={}))


def test_validate_still_rejects_a_wrong_substrate_when_out_of_scope():
    with pytest.raises(ExtractionInvalid, match="substrate"):
        validate(_raw(in_scope=False, moving_parts={"substrate": "kubernetes"}))


def test_hand_extraction_fixtures_all_validate():
    # The corpus itself must satisfy the tightened rule: five in_scope:true
    # extractions naming a real substrate, plus #2304 (in_scope:false, null).
    for path in sorted((FIXTURES / "extractions").glob("*.json")):
        validate(json.loads(path.read_text()))


def test_extract_missing_fixture_raises():
    extractor = Extractor(FixtureLLM(FIXTURES))
    issue = Issue(number=99999, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator")
    with pytest.raises(Exception):
        extractor.extract(issue)


# An out-of-scope extraction may leave expected/observed null, since they
# only describe a defect.
def test_out_of_scope_may_omit_expected_and_observed():
    validate(_raw(in_scope=False, moving_parts={}, expected=None, observed=None))
    raw = _raw(in_scope=False, moving_parts={})
    del raw["expected"], raw["observed"]
    validate(raw)


def test_in_scope_still_requires_expected_and_observed():
    with pytest.raises(ExtractionInvalid, match="expected"):
        validate(_raw(expected=None))
    with pytest.raises(ExtractionInvalid, match="observed"):
        validate(_raw(observed=None))


def test_out_of_scope_still_rejects_a_wrong_type():
    with pytest.raises(ExtractionInvalid, match="expected"):
        validate(_raw(in_scope=False, moving_parts={}, expected=42))


def test_null_expected_becomes_empty_string_not_none():
    # The classifier and composer do string work on these fields.
    from models import Hypothesis

    hyp = Hypothesis.from_dict(1, _raw(in_scope=False, moving_parts={}, expected=None, observed=None))
    assert hyp.expected == ""
    assert hyp.observed == ""
