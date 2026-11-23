"""Outcome classification.

The rung ladder. A simple four-rung ladder (traceback+substring -> non-zero
exit -> success -> partial) misclassifies most real hypotheses, so this one is
checked most-specific/safest first: a noisy nonzero exit (a stale test
selector, an API-shape crash) can't masquerade as a reproduction just because
*something* in its output happens to look like a match.
"""

from __future__ import annotations

import re

from models import (
    CommandResult,
    Hypothesis,
    Outcome,
    RunResult,
    SurfaceInference,
    embedded_test_file,
)
from surface_inference import SYNTHESIS_INCOMPLETE_MARKER

_QUOTED_SNIPPET_RE = re.compile(r'"([^"]{8,})"')
_COLLECTION_ERROR_RE = re.compile(r"collected 0 items|no tests collected", re.I)
# Separate from _IMPORT_ERROR_RE because pytest prints the generic banner
# above the ModuleNotFoundError line, and an alternation would match the
# banner first and lose the module name.
_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"](?P<module>[^'\"]+)['\"]")
# Commands that build the environment a `substrate: none` run needs. Kept
# in step with `runner_stage._ESTABLISHES_ENVIRONMENT_RE`.
_ENV_SETUP_RE = re.compile(r"\buv\s+(init|venv|add|pip\s+install)\b|\bpip\s+install\b")
# pytest shows a failed assertion as `E    assert ...` or, for an assert with
# a message or a raised AssertionError, `E    AssertionError: ...`. Only these
# count as evidence.
_ASSERTION_FAILURE_RE = re.compile(r"^E\s+(assert\b|AssertionError\b)", re.M)
# Any other exception surfacing out of the test body, as pytest prints it
# in the FAILURES section: `E   dataclasses.FrozenInstanceError: ...`.
# Dotted prefixes are kept so the reason names what the reader will see.
_RAISED_EXCEPTION_RE = re.compile(r"^E\s+(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Exit))\b", re.M)
# A `pytest.raises(...)` whose body raised nothing, as pytest prints it:
# `E   Failed: DID NOT RAISE <class 'ops.model.RelationNotFoundError'>` on
# older pytest, `E   Failed: DID NOT RAISE RelationNotFoundError` on newer.
_DID_NOT_RAISE_RE = re.compile(r"^E\s+Failed: DID NOT RAISE\b(?P<what>.*)$", re.M)
_IMPORT_ERROR_RE = re.compile(r"ImportError while importing test module|ModuleNotFoundError")
_API_SHAPE_ERROR_RE = re.compile(r"\b(AttributeError|TypeError|ImportError)\b")
# juju writes progress to stderr alongside errors, so the first line of a
# failed command is often informational. Prefer the first line that looks like
# an error, falling back to the first line.
_ERROR_LINE_RE = re.compile(r"^\s*(error\b|.*\berror:)", re.I)


def _first_error_line(lines: list[str]) -> str:
    for line in lines:
        if _ERROR_LINE_RE.search(line):
            return line.strip()
    return lines[0].strip() if lines else ""


def _quoted_snippets(text: str) -> list[str]:
    """Literal substrings quoted in the hypothesis's free-text `observed`
    field -- the closest thing to a "message substring to match" the
    extraction schema carries. Requires >=8 chars to avoid trivial
    accidental matches."""
    return _QUOTED_SNIPPET_RE.findall(text or "")


def _contains_any(haystack: str, snippets: list[str]) -> bool:
    return bool(snippets) and any(snippet in haystack for snippet in snippets)


def _command_text(c: CommandResult) -> str:
    return f"{c.stdout}\n{c.stderr}"


def classify(
    hypothesis: Hypothesis, surface: SurfaceInference | None, run_result: RunResult
) -> tuple[Outcome, str]:
    """Return (outcome, reason). Never called for `skipped_stale` --
    that's runner_stage's pre-run gate, not a classifier rung."""
    commands = run_result.commands
    observed_snippets = _quoted_snippets(hypothesis.observed)
    all_log_text = "\n".join("\n".join(c.log_records) for c in commands)

    # Rung 0: infrastructure failed, so the run is not evidence about the bug.
    # Every later rung assumes the reproduction actually ran, and commands like
    # `juju status` still exit 0 against a model where nothing deployed.
    if run_result.aborted_at_step:
        failed = next(
            (c for c in commands if c.step == run_result.aborted_at_step),
            None,
        )
        detail = ""
        if failed is not None:
            reason_text = (failed.stderr or failed.stdout or "").strip().splitlines()
            chosen = _first_error_line(reason_text)
            if chosen:
                detail = f": {chosen}"
            if failed.exit_code == 124:
                detail = f" (timed out){detail}"
        return (
            Outcome.INFRASTRUCTURE_FAILED,
            f"{run_result.aborted_at_step!r} failed{detail} -- skipped "
            f"{', '.join(run_result.skipped_steps) or 'nothing'}; the reproduction "
            "never ran, so this run says nothing about the bug",
        )

    # Rung 0b: a collection ImportError means the environment was never built.
    # Checked before rung 1 because pytest also reports it as "collected 0
    # items", which rung 1 would read as a stale selector.
    for c in commands:
        if c.exit_code != 0 and _IMPORT_ERROR_RE.search(_command_text(c)):
            named = _MISSING_MODULE_RE.search(_command_text(c))
            module = f"'{named.group('module')}'" if named else "a dependency"
            return (
                Outcome.INFRASTRUCTURE_FAILED,
                f"{c.command!r} could not import {module} -- the test environment was "
                "never prepared, so nothing was tested; this says nothing about the bug",
            )

    # Rung 0c: if the environment build failed, later commands ran somewhere
    # other than the environment the hypothesis described. Rung 6 would
    # otherwise read this as `partial`, which composes a comment.
    for c in commands:
        if c.exit_code != 0 and _ENV_SETUP_RE.search(c.command):
            detail = (c.stderr or c.stdout or "").strip().splitlines()
            return (
                Outcome.INFRASTRUCTURE_FAILED,
                f"{c.command!r} exited {c.exit_code}"
                + (f": {detail[0]}" if detail else "")
                + " -- the test environment was never built, so nothing after it is "
                "evidence about the bug",
            )

    # Rung 1a: the model-written test file errored during collection, so the
    # generated test is malformed. Checked before rung 1, which would read
    # "collected 0 items" as a stale selector.
    written = hypothesis.synthesized_test_file or embedded_test_file(hypothesis.commands)
    if written is not None:
        name = written.path.rsplit("/", 1)[-1]
        collecting_re = re.compile(rf"ERROR collecting \S*{re.escape(name)}\b")
        for c in commands:
            text = _command_text(c)
            if c.exit_code != 0 and _COLLECTION_ERROR_RE.search(text) and collecting_re.search(text):
                return (
                    Outcome.UNRUNNABLE_SYNTHESIS_INVALID,
                    f"the model-written test file {written.path} failed while pytest was "
                    "collecting it (0 items collected) -- the generated test is malformed, so "
                    "this is evidence about the generator and not about the reported bug",
                )

    # Rung 1: stale test selector. pytest fails on unrelated collection errors
    # and `-k` matches nothing; checked before the substring rungs so a match
    # somewhere in that noisy output can't shadow it.
    for c in commands:
        if c.exit_code != 0 and _COLLECTION_ERROR_RE.search(_command_text(c)):
            return (
                Outcome.UNRUNNABLE_TEST_SELECTOR_STALE,
                f"{c.command!r} exited {c.exit_code} with a stale test selector "
                "(0 items collected, unrelated collection errors) -- not a reproduction",
            )

    # Rung 1b: test-file synthesis fell back to the stub, which fails loudly
    # with SYNTHESIS_INCOMPLETE_MARKER. Checked before rungs 2 and 6 so the stub
    # can never be read as a reproduction.
    for c in commands:
        if c.exit_code != 0 and SYNTHESIS_INCOMPLETE_MARKER in _command_text(c):
            return (
                Outcome.UNRUNNABLE_SYNTHESIS_INCOMPLETE,
                f"{c.command!r} ran a test-file synthesis fallback stub with no real "
                "assertion (LLM synthesis was unavailable or produced an unusable body) "
                "-- not a reproduction, not runnable",
            )

    # Rung 1c: a model-written test (synthesised, or a heredoc in the
    # commands) that dies with anything but an assertion failure is a broken
    # test, not a reproduction. Output containing text the reporter quoted as
    # observed is still evidence, so that case falls through to rung 2.
    if hypothesis.synthesized_test_file is not None or embedded_test_file(hypothesis.commands):
        for c in commands:
            if c.exit_code == 0:
                continue
            text = _command_text(c)
            if _ASSERTION_FAILURE_RE.search(text):
                break
            if _contains_any(text, observed_snippets):
                break
            raised = _RAISED_EXCEPTION_RE.search(text)
            if raised:
                return (
                    Outcome.UNRUNNABLE_SYNTHESIS_INVALID,
                    f"the model-written test raised {raised.group('exc')} rather than failing an "
                    "assertion -- the generated test is not valid, so its failure is evidence "
                    "about the generator and not about the reported bug",
                )

    # Rung 1d: a model-written test that failed only because its own
    # `pytest.raises(...)` did not hold. That expectation is the model's, not
    # the reporter's symptom, so it is not evidence. Checked before rung 2 so a
    # quoted snippet in the printed test source can't lift it, and skipped when
    # the same output also has a failed assertion.
    if written is not None:
        for c in commands:
            if c.exit_code == 0:
                continue
            text = _command_text(c)
            not_raised = _DID_NOT_RAISE_RE.search(text)
            if not_raised and not _ASSERTION_FAILURE_RE.search(text):
                return (
                    Outcome.UNRUNNABLE_SYNTHESIS_INVALID,
                    f"the model-written test failed only because its own `pytest.raises(...)` did not "
                    f"hold (DID NOT RAISE{not_raised.group('what').rstrip()}) -- that is the generated "
                    "test's expectation, not the reported symptom, so it is not evidence about the bug",
                )

    # Rung 2: reproduced. A non-zero exit whose output contains text quoted in
    # `observed`.
    for c in commands:
        if c.exit_code != 0 and _contains_any(_command_text(c), observed_snippets):
            return Outcome.REPRODUCED, f"{c.command!r} (exit {c.exit_code}) matched observed output"

    # Rung 3: reproduced (log-only). Some symptoms are only a logged warning,
    # with nothing on stdout/stderr and a zero exit.
    if _contains_any(all_log_text, observed_snippets):
        return Outcome.REPRODUCED_LOG_ONLY, "observed substring matched captured log records (exit was zero)"

    # Rung 4: API-shape mismatch. The extraction targets the current API but
    # the bug lived on an older one. Checked after the positive-match rungs so a
    # real reproduction whose symptom is one of these exceptions wins.
    for c in commands:
        if c.exit_code != 0 and _API_SHAPE_ERROR_RE.search(_command_text(c)):
            anchor = f" (symbol_anchor={hypothesis.moving_parts.symbol_anchor!r})" if hypothesis.moving_parts.symbol_anchor else ""
            return (
                Outcome.UNRUNNABLE_API_SHAPE_MISMATCH,
                f"{c.command!r} hit an API-shape error{anchor} -- extraction is pinned to a "
                "different API surface than the one the bug lived on",
            )

    # Rung 5: the bug is a signal that never appears. Silence only counts if a
    # control stimulus that should produce the signal actually did.
    if surface is not None and surface.expected_signal:
        main_text = "\n".join(_command_text(c) for c in commands)
        if surface.expected_signal not in main_text:
            control = run_result.control
            # The control must have succeeded, not merely run: its command is
            # a bounded poll for the signal, and on a timeout (exit 124, no
            # output) the exit code is the only evidence.
            if control is not None and control.exit_code == 0 and surface.expected_signal in _command_text(control):
                return (
                    Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT,
                    f"expected signal {surface.expected_signal!r} absent from the main run but "
                    "present in the control run -- confirms a broken observer isn't the explanation",
                )
            if control is None:
                detail = "no control was run"
            elif control.exit_code != 0:
                detail = f"the control exited {control.exit_code} without ever showing the signal"
            else:
                detail = "the control ran but did not produce the signal either"
            return (
                Outcome.DID_NOT_REPRODUCE,
                f"expected signal {surface.expected_signal!r} absent, and {detail} -- nothing "
                "confirmed the observer works at all, so a reproduced bug can't be distinguished "
                "from a broken observer",
            )

    # Rung 6: reproduced (weaker) -- non-zero exit on the last command,
    # nothing more specific matched.
    if commands and commands[-1].exit_code != 0:
        return Outcome.REPRODUCED_WEAKER, f"{commands[-1].command!r} (last command) exited non-zero, no substring match"

    # Rung 7: partial -- failed at an earlier command with a different
    # error than expected.
    if any(c.exit_code != 0 for c in commands[:-1]):
        failed = next(c for c in commands[:-1] if c.exit_code != 0)
        return Outcome.PARTIAL, f"{failed.command!r} exited {failed.exit_code}, later commands did not run as hypothesised"

    return Outcome.DID_NOT_REPRODUCE, "all commands succeeded, no observed-substring match"
