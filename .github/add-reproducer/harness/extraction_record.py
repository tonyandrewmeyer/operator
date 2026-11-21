"""What extraction actually did, on every run that reaches it.

A run that stops at the in-scope gate otherwise records one line --
`stage=extraction ... reason=in_scope=...` -- and writes no artefact, because
`pipeline.py` returns before the record-writing step (so `upload-artifact`
reports `No files were found`).

That is not enough to explain a drop. CI runs can drop an issue at that gate
that a local run of the same code, the same model and the same fetch path
puts in scope every time, and nothing either side records can say what
differed. The candidates are the Actions key's account (OpenRouter routes per
account), provider variance on the day, and the runner -- and separating them
needs the per-call `provider`/`model` OpenRouter returns.

So: one record, built after extraction whether or not the issue is in scope,
carrying the first pass's verdict, whether the second pass ran and what it
said, and one entry per LLM call. `pipeline.py` attaches it to every
`PipelineResult` from extraction onwards; `run.py` prints it, writes it to
`<out-dir>/<issue>-extraction.json` and puts it in the job summary.
"""

from __future__ import annotations

from typing import Any

# 2: adds `test_check`, the first pass's static check of its embedded test
# file and whether that check sent the extraction back for one more attempt.
# 3: adds `substrate_override`, set by the pipeline when a fake-hook-tool
# test ran as `substrate: none` against the answer's own field
# (`pipeline.override_substrate()`); `None` otherwise.
SCHEMA_VERSION = 3


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
        "test_check": _test_check_fields(extractor),
        "final_in_scope": getattr(hypothesis, "in_scope", None),
        "substrate_override": None,
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
    """`ran` says whether a second pass actually happened, which a bare
    "second opinion" label cannot. `recovered` is `concrete_defect` under the name that says
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


def _test_check_fields(extractor: Any) -> dict:
    """`checks` holds one entry per in-scope extraction that embedded a test
    file, in call order, so a re-ask shows the reasons both times.
    `both_failed` is the case that carries on to the runner exactly as before
    the check existed, for rung 1a or 1c to keep silent; it is false, not
    unknown, when the re-ask came back with nothing to check (out of scope,
    no test file, or a schema failure), because then there is no second
    verdict to have failed."""
    checks = [check.to_dict() for check in getattr(extractor, "last_test_checks", None) or []]
    retried = bool(getattr(extractor, "last_test_retried", False))
    return {
        "ran": bool(checks),
        "checks": checks,
        "retried": retried,
        "both_failed": retried and len(checks) == 2 and not any(c["passed"] for c in checks),
    }


def in_scope_drop_reason(record: dict | None) -> str:
    """The `stage=` reason for a run stopped at the in-scope gate.

    Keeps the `in_scope=false` prefix so a grep across older logs still
    finds this stop. The parenthetical names what actually happened -- whether
    the second pass ran -- so a drop is never read as "the second pass ran and
    recovered nothing" when it did not run at all.
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
    test_check = record.get("test_check") or {}
    if test_check.get("ran"):
        for index, check in enumerate(test_check.get("checks") or [], start=1):
            verdict = "passed" if check.get("passed") else "failed"
            lines.append(f"  test check {index}: {verdict} ({check.get('path')})")
            lines.extend(f"    + added `{line}`" for line in check.get("added_imports") or [])
            lines.extend(f"    - {reason}" for reason in check.get("reasons") or [])
        lines.append(
            f"  test check: retried={test_check.get('retried')} "
            f"both_failed={test_check.get('both_failed')}"
        )
    else:
        lines.append("  test check: nothing to check")
    lines.append(f"  final in_scope: {record.get('final_in_scope')}")
    override = record.get("substrate_override")
    if override:
        tools = ", ".join(override.get("faked_hook_tools") or [])
        lines.append(
            f"  substrate override: {override.get('from')} -> {override.get('to')} "
            f"(the test fakes hook tools: {tools})"
        )
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
