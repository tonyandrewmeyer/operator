"""What extraction actually did, on every run that reaches it.

Until this module existed, a dispatch that stopped at the in-scope gate
recorded one line -- `stage=extraction ... reason=in_scope=false (second
opinion)` -- and that line was not even true: `pipeline.py` printed it for
*any* first-pass drop, so it said "second opinion" whether or not the second
pass had run (`spike-step-5/comments-check/RESULT.md` §3.1). The run wrote no
artefact either, because `pipeline.py` returns before the record-writing step,
so `upload-artifact` reported `No files were found`
(`spike-step-5/seventh-dispatch/RESULT.md` §4.2).

The consequence was measured on 2026-09-23/24: four live runs dropped `#2639`
and `#2484` at that gate, a local run of the same tip, the same model and the
same fetch path put both in scope 24 of 24, and nothing either side recorded
could say what differed. The candidates were the Actions key's account
(OpenRouter routes per account), provider variance on the day, and the runner
-- and separating them needs the per-call `provider`/`model` OpenRouter
returns, which nothing kept.

So: one record, built after extraction whether or not the issue is in scope,
carrying the first pass's verdict, whether the second pass ran and what it
said, and one entry per LLM call. `pipeline.py` attaches it to every
`PipelineResult` from extraction onwards; `run.py` prints it, writes it to
`<out-dir>/<issue>-extraction.json` and puts it in the job summary.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1


def build(
    issue_number: int,
    *,
    extractor: Any,
    hypothesis: Any,
    llm: Any,
    error: str | None = None,
) -> dict:
    """Assemble the record.

    `getattr` throughout rather than attribute access: the record must never
    be the thing that turns a working run into a crash, and both seams
    (`FixtureLLM`, `LiveOpenRouterLLM`) and both extractors (`Extractor`,
    `TwoPassExtractor`) can legitimately reach here. An extractor with no
    `last_second_pass` reports "the second pass did not run", which is the
    truth for a plain `Extractor`.

    `hypothesis` is the hypothesis `extract()` returned, or `None` when it
    raised -- an invalid extraction is a run that reached extraction, and the
    calls it made are exactly what says why it was invalid.
    """
    first_pass = getattr(extractor, "last_first_pass", None)
    # Falling back to the returned hypothesis matters for a plain `Extractor`:
    # it has no `last_first_pass`, and there its own output *is* the first
    # pass. On a `TwoPassExtractor` recovery the two differ, which is why the
    # attribute exists.
    if first_pass is None:
        first_pass = hypothesis
    second = getattr(extractor, "last_second_pass", None)
    record: dict = {
        "schema_version": SCHEMA_VERSION,
        "issue_number": issue_number,
        "first_pass": _first_pass_fields(first_pass),
        "second_pass": _second_pass_fields(second),
        "final_in_scope": getattr(hypothesis, "in_scope", None),
        "llm_calls": list(getattr(llm, "calls", []) or []),
    }
    if error is not None:
        record["error"] = error
    return record


def refresh_calls(record: dict | None, llm: Any) -> dict | None:
    """Re-read the seam's call list into an already-built record.

    The record is built the moment extraction returns, which is the only
    moment the first/second-pass fields are unambiguous -- but surface
    inference, test-file synthesis and composition all call the same seam
    afterwards, and "every call this run made" is what a spend or a
    provider-variance question actually needs. So the record is stamped onto
    the result at the end of the run, with the call list re-read then.
    """
    if record is not None:
        record["llm_calls"] = list(getattr(llm, "calls", []) or [])
    return record


def _first_pass_fields(hypothesis: Any) -> dict | None:
    if hypothesis is None:
        return None
    moving_parts = getattr(hypothesis, "moving_parts", None)
    return {
        "in_scope": getattr(hypothesis, "in_scope", None),
        "confidence": getattr(hypothesis, "confidence", None),
        "substrate": getattr(moving_parts, "substrate", None),
    }


def _second_pass_fields(raw: dict | None) -> dict:
    """`ran` is the field the old `(second opinion)` reason implied and could
    not support. `recovered` is `concrete_defect` under the name that says
    what it did: `inscope_second_pass.TwoPassExtractor.extract()` re-extracts
    the hypothesis exactly when `concrete_defect` is true."""
    if raw is None:
        return {"ran": False, "concrete_defect": None, "recovered": False}
    concrete_defect = raw.get("concrete_defect")
    return {
        "ran": True,
        "concrete_defect": concrete_defect,
        "recovered": bool(concrete_defect),
        "reason": raw.get("reason"),
    }


def in_scope_drop_reason(record: dict | None) -> str:
    """The `stage=` reason for a run stopped at the in-scope gate.

    Keeps the `in_scope=false` prefix every earlier round's logs carry, so
    those logs still read the same way and a grep across rounds still finds
    this stop. What changes is the parenthetical: it now names what happened
    rather than what the code's structure implies. The old text --
    `(second opinion)` -- was read by `seventh-dispatch` §4.1 as evidence the
    second pass had run and recovered nothing, and it was never evidence of
    that.
    """
    second = (record or {}).get("second_pass") or {}
    if second.get("ran"):
        return "in_scope=false (second pass ran and confirmed the drop)"
    return "in_scope=false (no second pass ran)"


def render(record: dict) -> list[str]:
    """The block printed to the job log and appended to the job summary.

    Deliberately flat lines rather than the JSON: this is read in a job log,
    where the JSON of the same thing is the artefact's job.
    """
    first = record.get("first_pass") or {}
    second = record.get("second_pass") or {}
    lines = [
        f"extraction record for #{record.get('issue_number')}:",
        f"  first pass: in_scope={first.get('in_scope')} "
        f"confidence={first.get('confidence')} substrate={first.get('substrate')}",
    ]
    if second.get("ran"):
        lines.append(
            f"  second pass: ran, concrete_defect={second.get('concrete_defect')} "
            f"recovered={second.get('recovered')}"
        )
    else:
        lines.append("  second pass: did not run")
    lines.append(f"  final in_scope: {record.get('final_in_scope')}")
    if record.get("error"):
        lines.append(f"  error: {record['error']}")
    calls = record.get("llm_calls") or []
    lines.append(f"  llm calls: {len(calls)}")
    for index, call in enumerate(calls, start=1):
        usage = call.get("usage") or {}
        lines.append(
            f"    {index}. purpose={call.get('purpose')} model={call.get('model')} "
            f"provider={call.get('provider')} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"total_tokens={usage.get('total_tokens')}"
        )
    return lines
