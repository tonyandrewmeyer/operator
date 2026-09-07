"""Tests for the second `in_scope` classification pass (PLAN.md Approach §3
delta, `spike-step-5/inscope-instrument/RESULT.md`).

Fixture-driven, same pattern as `test_extraction.py`: no network, no key.
`fixtures/issues/9002.json` and its paired `fixtures/extractions/9002.json` /
`fixtures/inscope_second_pass/9002.json` are fully synthetic (clearly marked
inline), written to exercise the recovery path -- no real GitHub issue
carries this content. `#2304` and `#2639` are real, existing fixtures.
"""

import json
from pathlib import Path

import pytest

from extraction import ExtractionInvalid
from inscope_second_pass import (
    SecondPassInvalid,
    TwoPassExtractor,
    needs_second_pass,
    validate_second_pass,
)
from models import Hypothesis, Issue, MovingParts
from seams.llm import FixtureLLM

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def _hyp(*, in_scope: bool, observed: str = "") -> Hypothesis:
    return Hypothesis(
        issue_number=1,
        in_scope=in_scope,
        moving_parts=MovingParts(),
        commands=[],
        expected="",
        observed=observed,
        confidence="low",
    )


# -- needs_second_pass gate --------------------------------------------------


def test_needs_second_pass_fires_on_any_drop():
    assert needs_second_pass(_hyp(in_scope=False)) is True
    assert needs_second_pass(_hyp(in_scope=False, observed="")) is True


def test_needs_second_pass_skips_a_keep():
    assert needs_second_pass(_hyp(in_scope=True)) is False


# -- validate_second_pass -----------------------------------------------------


def test_validate_second_pass_accepts_no_defect_with_minimal_fields():
    validate_second_pass({"concrete_defect": False, "reason": "not a bug"})


def test_validate_second_pass_rejects_missing_concrete_defect():
    with pytest.raises(SecondPassInvalid, match="concrete_defect"):
        validate_second_pass({"reason": "x"})


def test_validate_second_pass_reuses_extraction_validate_when_defect_true():
    # No substrate named -- extraction.validate()'s existing rule (never
    # guess a substrate) applies here exactly as it does to the first pass.
    with pytest.raises(SecondPassInvalid, match="substrate"):
        validate_second_pass(
            {
                "concrete_defect": True,
                "reason": "x",
                "moving_parts": {},
                "commands": [],
                "expected": "e",
                "observed": "o",
                "confidence": "low",
            }
        )


def test_validate_second_pass_accepts_a_complete_recovery():
    validate_second_pass(
        {
            "concrete_defect": True,
            "reason": "x",
            "moving_parts": {"substrate": "none"},
            "commands": [],
            "expected": "e",
            "observed": "o",
            "confidence": "low",
        }
    )


# -- TwoPassExtractor ---------------------------------------------------------


@pytest.fixture
def extractor():
    return TwoPassExtractor(FixtureLLM(FIXTURES))


def test_skips_second_pass_when_first_pass_already_keeps(extractor):
    # #2639's extraction fixture has in_scope: true. If TwoPassExtractor
    # ran the second pass anyway, this would raise (no
    # fixtures/inscope_second_pass/2639.json exists) -- so a passing test
    # is itself proof the gate was honoured.
    hyp = extractor.extract(_load_issue(2639))
    assert hyp.in_scope is True
    assert extractor.last_second_pass is None


def test_stays_dropped_when_second_pass_finds_no_concrete_defect(extractor):
    # #2304: first pass drops it (a "remove this now-obsolete check"
    # proposal); fixtures/inscope_second_pass/2304.json agrees.
    hyp = extractor.extract(_load_issue(2304))
    assert hyp.in_scope is False
    assert extractor.last_second_pass["concrete_defect"] is False


def test_recovers_when_second_pass_finds_a_concrete_defect(extractor):
    # #9002 (fully synthetic): first pass drops with blank observed/expected
    # -- the exact failure mode this module exists for -- and the second
    # pass, given the full issue body, finds the concrete defect and
    # re-extracts a full hypothesis.
    hyp = extractor.extract(_load_issue(9002))
    assert hyp.in_scope is True
    assert extractor.last_second_pass["concrete_defect"] is True
    assert hyp.moving_parts.substrate == "k8s"
    assert hyp.confidence == "medium"
    assert hyp.commands  # re-extracted, not just a flipped boolean


def test_recovery_carries_over_the_deterministic_ci_run_url(extractor):
    # ci_run_url is regex-extracted from the issue body, not model output
    # (extraction.py's own docstring); a recovery must not lose it.
    hyp = extractor.extract(_load_issue(9002))
    assert hyp.moving_parts.ci_run_url is None  # #9002's body has no CI link
    issue_with_ci = _load_issue(9002)
    issue_with_ci.body += "\nSee https://github.com/canonical/operator/actions/runs/1"
    hyp2 = extractor.extract(issue_with_ci)
    assert hyp2.moving_parts.ci_run_url == "https://github.com/canonical/operator/actions/runs/1"


def test_duck_types_extractor_max_comment_chars_and_build_prompt(extractor):
    # run_live_extraction.run_one() and Extractor's own callers read these
    # directly; TwoPassExtractor must expose them to be a drop-in swap.
    assert isinstance(extractor.max_comment_chars, int)
    prompt = extractor._build_prompt(_load_issue(2304), extractor.max_comment_chars)
    assert "Title: " in prompt


def test_second_pass_invalid_is_an_extraction_invalid():
    # pipeline.py's stay-silent fallback catches ExtractionInvalid; this
    # must be one so a broken second-pass response falls back the same way
    # a broken first-pass response already does, with no extra clause.
    assert issubclass(SecondPassInvalid, ExtractionInvalid)
