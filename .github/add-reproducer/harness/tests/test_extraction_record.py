"""The extraction record: what a run that reached extraction actually did.

The gap this closes is measured, not hypothetical. On 2026-09-23 four live
GitHub Actions dispatches stopped at `stage=extraction ... reason=in_scope=
false (second opinion)` on `#2639` and `#2484`; on 2026-09-24 a local run of
the same tip, the same model and the same `run.fetch_issue()` put both in
scope 24 of 24, and a re-dispatch of `#2639` came back in scope. Nothing
either side recorded could say what differed
(`spike-step-5/comments-check/RESULT.md` §4), because the job log printed one
`stage=` line, the run wrote no artefact at that stop
(`spike-step-5/seventh-dispatch/RESULT.md` §4.2), and the seam kept only the
*last* call's raw content and usage -- never the `provider` OpenRouter
returns, which is the field a routing question turns on.

Two further things the old reason string got wrong, both fixed here:
`(second opinion)` was printed for any first-pass drop whether or not the
second pass ran, and the same stop reported no substrate.

No network: `urlopen` is monkeypatched, and everything else runs on
`FixtureLLM`.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import extraction_record
import pytest
from inscope_second_pass import TwoPassExtractor
from models import Hypothesis, Issue, MovingParts
from pipeline import Pipeline
from seams import llm as llm_module
from seams.llm import FixtureLLM, LiveOpenRouterLLM, LLMError
from seams.runner import FixtureRunnerSeam

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_issue(number: int) -> Issue:
    return Issue.from_dict(json.loads((FIXTURES / "issues" / f"{number}.json").read_text()))


def _response(payload: dict):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Resp(json.dumps(payload).encode())


def _payload(content: str = '{"ok": true}', **extra) -> dict:
    payload = {"choices": [{"message": {"content": content}}]}
    payload.update(extra)
    return payload


# -- per-call metadata on the live seam --------------------------------------


def test_every_call_is_recorded_with_its_purpose_model_provider_and_usage(monkeypatch):
    """The whole point: three calls, three entries, each naming which provider
    served it. `last_raw_content`/`last_usage` describe only the third."""
    payloads = [
        _payload(model="deepseek/deepseek-chat", provider="DeepSeek", usage={"total_tokens": 10}),
        _payload(model="deepseek/deepseek-chat", provider="Fireworks", usage={"total_tokens": 20}),
        _payload(model="deepseek/deepseek-chat", provider="Novita", usage={"total_tokens": 30}),
    ]
    monkeypatch.setattr(
        llm_module.urllib.request, "urlopen", lambda req, timeout=None: _response(payloads.pop(0))
    )
    client = LiveOpenRouterLLM(api_key="sk-test")

    client.complete_json(purpose="extraction", prompt="p", context={})
    client.complete_json(purpose="inscope_second_pass", prompt="p", context={})
    client.complete_json(purpose="surface_inference", prompt="p", context={})

    assert client.calls == [
        {
            "purpose": "extraction",
            "model": "deepseek/deepseek-chat",
            "provider": "DeepSeek",
            "usage": {"total_tokens": 10},
        },
        {
            "purpose": "inscope_second_pass",
            "model": "deepseek/deepseek-chat",
            "provider": "Fireworks",
            "usage": {"total_tokens": 20},
        },
        {
            "purpose": "surface_inference",
            "model": "deepseek/deepseek-chat",
            "provider": "Novita",
            "usage": {"total_tokens": 30},
        },
    ]


def test_a_payload_without_a_provider_still_records_the_call(monkeypatch):
    """`provider` is not in every OpenRouter response. A missing field must
    leave the entry present with a null, never drop the call -- a dropped call
    is exactly the hole this record exists to fill."""
    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        lambda req, timeout=None: _response(_payload(model="deepseek/deepseek-chat", usage={"total_tokens": 7})),
    )
    client = LiveOpenRouterLLM(api_key="sk-test")

    client.complete_json(purpose="extraction", prompt="p", context={})

    assert client.calls == [
        {
            "purpose": "extraction",
            "model": "deepseek/deepseek-chat",
            "provider": None,
            "usage": {"total_tokens": 7},
        }
    ]


def test_the_existing_last_raw_content_and_last_usage_still_work(monkeypatch):
    """These are what the calibration rounds read; the per-call list is an
    addition, not a replacement."""
    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        lambda req, timeout=None: _response(_payload(content='{"a": 1}', usage={"total_tokens": 3})),
    )
    client = LiveOpenRouterLLM(api_key="sk-test")

    assert client.complete_json(purpose="extraction", prompt="p", context={}) == {"a": 1}
    assert client.last_raw_content == '{"a": 1}'
    assert client.last_usage == {"total_tokens": 3}


def test_a_call_whose_content_is_not_json_is_recorded_before_it_raises(monkeypatch):
    """A response this seam rejects is precisely one whose provider you want
    named."""
    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        lambda req, timeout=None: _response(_payload(content="I'm sorry, but", provider="Chutes")),
    )
    client = LiveOpenRouterLLM(api_key="sk-test")

    with pytest.raises(LLMError):
        client.complete_json(purpose="extraction", prompt="p", context={})

    assert [call["provider"] for call in client.calls] == ["Chutes"]


def test_an_http_failure_records_no_call(monkeypatch):
    """Nothing came back, so there is nothing to attribute."""

    def fails(req, timeout=None):
        raise urllib.error.HTTPError("url", 401, "no", {}, None)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fails)
    client = LiveOpenRouterLLM(api_key="sk-test", sleep=lambda _: None)

    with pytest.raises(LLMError):
        client.complete_json(purpose="extraction", prompt="p", context={})

    assert client.calls == []


# -- the record itself, through the pipeline ---------------------------------


def _pipeline(tmp_path, llm=None):
    llm = llm or FixtureLLM(FIXTURES)
    return Pipeline(llm, FixtureRunnerSeam(FIXTURES), work_dir=tmp_path / "work")


def test_an_in_scope_drop_records_the_second_pass_that_ran(tmp_path):
    """`#2304`: first pass drops, second pass runs and agrees. Both facts are
    now in the record, and the `stage=` reason says which of them happened."""
    result = _pipeline(tmp_path).run_for_issue(_load_issue(2304))

    assert result.stage_reached == "extraction"
    assert result.reason == "in_scope=false (second pass ran and confirmed the drop)"
    record = result.extraction_record
    assert record is not None
    assert record["first_pass"] == {"in_scope": False, "confidence": "low", "substrate": None}
    assert record["second_pass"]["ran"] is True
    assert record["second_pass"]["concrete_defect"] is False
    assert record["second_pass"]["recovered"] is False
    assert record["final_in_scope"] is False
    assert [call["purpose"] for call in record["llm_calls"]] == ["extraction", "inscope_second_pass"]


def test_an_in_scope_drop_with_no_second_pass_says_so(tmp_path):
    """The distinction the old string could not make. An extractor that never
    ran a second pass -- a plain `Extractor`, or a `TwoPassExtractor` whose
    gate changed -- must not have a "second opinion" attributed to it.
    `seventh-dispatch` §4.1 read four such lines as evidence of four second
    passes that recovered nothing; nothing in the log supported that."""

    class FirstPassOnly:
        def extract(self, issue):
            return Hypothesis(
                issue_number=issue.number,
                in_scope=False,
                moving_parts=MovingParts(),
                commands=[],
                expected="",
                observed="",
                confidence="low",
            )

    pipeline = _pipeline(tmp_path)
    pipeline.extractor = FirstPassOnly()
    result = pipeline.run_for_issue(_load_issue(2304))

    assert result.reason == "in_scope=false (no second pass ran)"
    assert result.extraction_record["second_pass"] == {
        "ran": False,
        "concrete_defect": None,
        "recovered": False,
    }


def test_the_record_keeps_the_first_pass_verdict_through_a_recovery():
    """`#9002` is the synthetic recovery fixture. `extract()` returns the
    *recovered* hypothesis, so without `last_first_pass` the first pass's own
    verdict would be gone by the time anything could record it."""
    llm = FixtureLLM(FIXTURES)
    extractor = TwoPassExtractor(llm)
    issue = _load_issue(9002)
    hypothesis = extractor.extract(issue)

    record = extraction_record.build(9002, extractor=extractor, hypothesis=hypothesis, llm=llm)

    assert record["first_pass"]["in_scope"] is False
    assert record["second_pass"]["ran"] is True
    assert record["second_pass"]["concrete_defect"] is True
    assert record["second_pass"]["recovered"] is True
    assert record["final_in_scope"] is True


def test_a_filter_drop_has_no_extraction_record(tmp_path):
    """The record says "extraction ran and this is what it did". A run that
    never got there must not carry one."""
    issue = Issue.from_dict(
        {
            "number": 9100,
            "title": "Add a --verbose flag",
            "body": "It would be nice.",
            "labels": [{"name": "docs"}],
            "state": "OPEN",
            "repo": "canonical/operator",
        }
    )
    result = _pipeline(tmp_path).run_for_issue(issue)

    assert result.stage_reached == "filter"
    assert result.extraction_record is None


def test_one_issues_record_does_not_leak_into_the_next(tmp_path):
    """One `Pipeline` runs a whole batch. A filter drop after an extraction
    must not inherit the previous issue's record."""
    pipeline = _pipeline(tmp_path)
    assert pipeline.run_for_issue(_load_issue(2304)).extraction_record is not None
    filtered = Issue.from_dict(
        {
            "number": 9101,
            "title": "Please document the new flag",
            "body": "-",
            "labels": [{"name": "docs"}],
            "state": "OPEN",
            "repo": "canonical/operator",
        }
    )
    assert pipeline.run_for_issue(filtered).extraction_record is None


def test_the_substrate_is_recorded_on_a_gate_stop(tmp_path):
    """`seventh-dispatch` §4.2: substrate was unobservable on the commonest
    non-filter stop, which made substrate stability unmeasurable live. It is
    in the record now, whatever the first pass put there."""
    result = _pipeline(tmp_path).run_for_issue(_load_issue(2304))

    assert "substrate" in result.extraction_record["first_pass"]


# -- rendering ---------------------------------------------------------------


def test_the_rendered_block_names_every_call_with_its_provider():
    record = {
        "issue_number": 2484,
        "first_pass": {"in_scope": False, "confidence": "low", "substrate": "none"},
        "second_pass": {"ran": True, "concrete_defect": False, "recovered": False},
        "final_in_scope": False,
        "llm_calls": [
            {
                "purpose": "extraction",
                "model": "deepseek/deepseek-chat",
                "provider": "DeepSeek",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
            {"purpose": "inscope_second_pass", "model": "deepseek/deepseek-chat", "provider": None, "usage": None},
        ],
    }

    block = "\n".join(extraction_record.render(record))

    assert "in_scope=False" in block
    assert "substrate=none" in block
    assert "second pass: ran" in block
    assert "provider=DeepSeek" in block
    assert "provider=None" in block
    assert "total_tokens=3" in block


def test_the_rendered_block_says_when_no_second_pass_ran():
    record = extraction_record.build(1, extractor=object(), hypothesis=None, llm=object())

    assert "second pass: did not run" in "\n".join(extraction_record.render(record))


def test_building_a_record_never_raises_on_an_unfamiliar_seam_or_extractor():
    """The record is diagnostics. It must not be the thing that turns a
    working run into a crash."""
    record = extraction_record.build(1, extractor=object(), hypothesis=None, llm=object(), error="boom")

    assert record["issue_number"] == 1
    assert record["first_pass"] is None
    assert record["llm_calls"] == []
    assert record["error"] == "boom"
