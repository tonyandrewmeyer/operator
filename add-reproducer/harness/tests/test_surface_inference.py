import json
from pathlib import Path

import pytest

from extraction import Extractor
from models import Issue
from seams.llm import FixtureLLM
from surface_inference import SurfaceInferenceInvalid, SurfaceInferrer, validate

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def test_infer_2639_surface():
    llm = FixtureLLM(FIXTURES)
    issue = _load_issue(2639)
    hypothesis = Extractor(llm).extract(issue)
    surface = SurfaceInferrer(llm).infer(issue, hypothesis)
    assert surface.charm_name == "repro-i2639-pebble-notice"
    assert surface.pebble_service["container"] == "workload"
    assert surface.ops_api_surface == "pebble-custom-notice"
    assert surface.expected_signal == "observed notice:"


def test_validate_rejects_digit_after_hyphen_charm_name():
    # Steps §3 fix (2026-07-22): juju's local-charm URL parser rejects a
    # numeric segment directly after a hyphen. Must never regress.
    with pytest.raises(SurfaceInferenceInvalid, match="digit after a hyphen"):
        validate({"charm_name": "repro-2639-pebble-notice"})


def test_validate_accepts_prefixed_charm_name():
    validate({"charm_name": "repro-i2639-pebble-notice"})


def test_validate_rejects_bad_surface():
    with pytest.raises(SurfaceInferenceInvalid):
        validate({"charm_name": "repro-i1-x", "ops_api_surface": "not-a-real-surface"})


# --- the stimulus half: pebble_service.command ---------------------------
#
# Live inference returned `pebble_service.command: null` on 3 of 3 calls
# for #2639 (`spike-step-5/first-real-substrate/RESULT.md` finding 5) and
# `validate()` never looked at it, so the run deployed a charm, poked
# nothing, and still produced a verdict. The fixture had been hand-filled
# from step 4's by-hand walk -- `fixtures/surface/2639.json`'s own
# `_command_note` records that -- which is why no fixture run saw it.


def _valid(**overrides):
    raw = {
        "charm_name": "repro-i2639-pebble-notice",
        "pebble_service": {
            "container": "workload",
            "service": "notice-source",
            "command": "/charm/bin/pebble notify canonical.com/repro/notice-1 key=value",
            "user": "_daemon_",
        },
        "ops_api_surface": "pebble-custom-notice",
        "expected_signal": "observed notice:",
    }
    raw.update(overrides)
    return raw


def test_validate_rejects_null_command_when_a_signal_is_promised():
    """The exact shape live inference produced, 3/3."""
    raw = _valid()
    raw["pebble_service"]["command"] = None
    with pytest.raises(SurfaceInferenceInvalid, match="never did anything to produce"):
        validate(raw)


def test_validate_rejects_empty_command():
    raw = _valid()
    raw["pebble_service"]["command"] = "   "
    with pytest.raises(SurfaceInferenceInvalid, match="command"):
        validate(raw)


def test_validate_rejects_missing_user_when_a_signal_is_promised():
    raw = _valid()
    del raw["pebble_service"]["user"]
    with pytest.raises(SurfaceInferenceInvalid, match="user"):
        validate(raw)


def test_validate_names_both_missing_fields():
    raw = _valid(pebble_service={"container": "workload", "service": "s"})
    with pytest.raises(SurfaceInferenceInvalid, match="command, user"):
        validate(raw)


def test_validate_rejects_placeholder_in_command():
    """`<workload-container>`-style placeholders are what #2639's own hand
    extraction carried; a shell reads them as redirects."""
    raw = _valid()
    raw["pebble_service"]["command"] = "juju ssh --container <workload-container> <unit>/0"
    with pytest.raises(SurfaceInferenceInvalid, match="placeholder"):
        validate(raw)


def test_validate_accepts_a_complete_stimulus():
    validate(_valid())


def test_validate_allows_no_stimulus_when_nothing_is_promised():
    """#2107: a machine charm that errors on update-status. No container,
    no signal, no stimulus needed -- the deploy is the experiment."""
    validate(
        {
            "charm_name": "repro-i2107-machine-id",
            "pebble_service": {},
            "ops_api_surface": "update-status",
            "expected_signal": None,
        }
    )


class _ScriptedLLM:
    """Returns each queued response in turn, recording the prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete_json(self, *, purpose, prompt, context=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _hypothesis_for_reask():
    from models import Hypothesis, MovingParts

    return Hypothesis(
        issue_number=2639,
        in_scope=True,
        confidence="medium",
        expected="the charm receives the notice",
        observed="it does not",
        commands=[],
        moving_parts=MovingParts(substrate="k8s"),
    )


def test_infer_re_asks_once_when_the_first_answer_is_invalid():
    bad = _valid()
    bad["pebble_service"]["command"] = None
    llm = _ScriptedLLM([bad, _valid()])

    surface = SurfaceInferrer(llm).infer(_load_issue(2639), _hypothesis_for_reask())

    assert len(llm.prompts) == 2
    assert "was rejected" in llm.prompts[1]
    assert "never did anything to produce" in llm.prompts[1]
    assert surface.pebble_service["command"].endswith("key=value")


def test_infer_does_not_re_ask_when_the_first_answer_is_valid():
    llm = _ScriptedLLM([_valid()])
    SurfaceInferrer(llm).infer(_load_issue(2639), _hypothesis_for_reask())
    assert len(llm.prompts) == 1


def test_infer_gives_up_after_one_re_ask():
    """A prompt-comprehension failure, not a transient -- a third attempt
    buys nothing the second didn't."""
    bad = _valid()
    bad["pebble_service"]["command"] = None
    llm = _ScriptedLLM([bad, bad])
    with pytest.raises(SurfaceInferenceInvalid):
        SurfaceInferrer(llm).infer(_load_issue(2639), _hypothesis_for_reask())
    assert len(llm.prompts) == 2


def test_validate_rejects_a_user_flag_pebble_does_not_have():
    """A live call produced `pebble notify --user=_daemon_ ...`
    (2026-08-18). `pebble notify` has no such flag -- it rejects it, so the
    stimulus never fires. The user belongs in `pebble_service.user`, which
    the runner applies by executing the command as that account (which is
    the whole reason `as_user_in_container_command` exists)."""
    raw = _valid()
    raw["pebble_service"]["command"] = "/charm/bin/pebble notify --user=_daemon_ canonical.com/repro/n key=v"
    with pytest.raises(SurfaceInferenceInvalid, match="--user"):
        validate(raw)
