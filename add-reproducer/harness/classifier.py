"""Outcome classification (PLAN.md Approach §5).

The rung ladder, folded per `spike-step-4/FINDINGS.md`: the original
four-rung ladder (traceback+substring -> non-zero exit -> success ->
partial) misclassified 3 of 4 calibration hypotheses. This is the
nine-outcome ladder that replaces it -- checked most-specific/safest
first, so a noisy nonzero exit (a stale test selector, an API-shape
crash) can't masquerade as a reproduction just because *something* in
its output happens to look like a match.
"""

from __future__ import annotations

import re

from models import CommandResult, Hypothesis, Outcome, RunResult, SurfaceInference
from surface_inference import SYNTHESIS_INCOMPLETE_MARKER

_QUOTED_SNIPPET_RE = re.compile(r'"([^"]{8,})"')
_COLLECTION_ERROR_RE = re.compile(r"collected 0 items|no tests collected", re.I)
# A collection-time import failure. Two patterns rather than one
# alternation: pytest prints the generic "ImportError while importing test
# module" banner *above* the specific ModuleNotFoundError, so a single
# alternation matches the banner first and loses the module name.
_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"](?P<module>[^'\"]+)['\"]")
# Commands that build the environment a `substrate: none` run needs. Kept
# in step with `runner_stage._ESTABLISHES_ENVIRONMENT_RE`.
_ENV_SETUP_RE = re.compile(r"\buv\s+(init|venv|add|pip\s+install)\b|\bpip\s+install\b")
# pytest renders a failed assertion either as a bare `E    assert ...`
# (a plain `assert` statement, the common case) or as an explicit
# `E    AssertionError: ...` (an `assert x, "msg"` with a message, or a
# raised AssertionError). Both are evidence; nothing else is.
_ASSERTION_FAILURE_RE = re.compile(r"^E\s+(assert\b|AssertionError\b)", re.M)
# Any other exception surfacing out of the test body, as pytest prints it
# in the FAILURES section: `E   dataclasses.FrozenInstanceError: ...`.
# Dotted prefixes are kept so the reason names what the reader will see.
_RAISED_EXCEPTION_RE = re.compile(r"^E\s+(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Exit))\b", re.M)
_IMPORT_ERROR_RE = re.compile(r"ImportError while importing test module|ModuleNotFoundError")
_API_SHAPE_ERROR_RE = re.compile(r"\b(AttributeError|TypeError|ImportError)\b")
# juju writes progress to stderr alongside errors, so the *first* line of a
# failed command is routinely informational. A collided `juju deploy` reported
# `'deploy' failed: Located local charm "i2639-pebble-notice", revision 0` --
# which reads as though nothing went wrong -- while the actual
# `ERROR cannot add application ...` sat three lines further down. Prefer the
# first line that looks like an error; fall back to the first line when none
# does. See `spike-step-5/gate-substrate/RESULT.md` §6.
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

    # Rung 0: infrastructure failed, so the run is not evidence about the
    # bug at all. Every rung below reads exit codes and output as though
    # the reproduction attempt actually happened; that precondition was
    # never checked. First real k8s run (2026-08-18) hit it immediately --
    # `charmcraft pack` failed, `juju deploy` then failed for want of a
    # charm, and `juju status`/`juju debug-log` both exited **0** against
    # an empty model, so the ladder scored a healthy-looking run and
    # returned `did_not_reproduce` about a bug nothing had tested. With a
    # control in the sequence the same shape reaches
    # `reproduced_positive_signal_absent`, which composes a comment.
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

    # Rung 0b: the test module could not be imported at all. A collection
    # ImportError means the environment was never built -- the run is not
    # evidence about the bug, same as a failed pack. Checked *before* rung
    # 1 because pytest reports both as "collected 0 items", so rung 1's
    # pattern matches this too and used to claim a stale `-k` selector for
    # what is really a missing dependency: the first live `substrate: none`
    # run (2026-08-18) synthesised a test importing `ops`, ran it with no
    # `uv add 'ops[testing]'` ahead of it, and was scored
    # `unrunnable_test_selector_stale`. Distinct outcome, not a rung: the
    # selector rung is a statement about the reproduction attempt, and this
    # is a statement about the machine.
    for c in commands:
        if c.exit_code != 0 and _IMPORT_ERROR_RE.search(_command_text(c)):
            named = _MISSING_MODULE_RE.search(_command_text(c))
            module = f"'{named.group('module')}'" if named else "a dependency"
            return (
                Outcome.INFRASTRUCTURE_FAILED,
                f"{c.command!r} could not import {module} -- the test environment was "
                "never prepared, so nothing was tested; this says nothing about the bug",
            )

    # Rung 0c: the environment build itself failed. Same principle as rung
    # 0 on the scratch branches: if `uv init`/`uv venv`/`uv add`/`pip
    # install` did not succeed, whatever ran afterwards ran somewhere other
    # than the environment the hypothesis described, and is not evidence
    # about the bug. Rung 6 previously read such a failure as `partial`,
    # which composes a comment -- 5 of the 9 comments in the first
    # loosened-gate batch (2026-08-18) were exactly that.
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

    # Rung 1: un-runnable (test selector stale). spike-step-4/2484 --
    # pytest exits non-zero purely from unrelated collection errors, with
    # the `-k` selector matching zero tests. Checked first: this pattern
    # is specific enough that it should never be shadowed by a generic
    # substring match inside the same noisy output.
    for c in commands:
        if c.exit_code != 0 and _COLLECTION_ERROR_RE.search(_command_text(c)):
            return (
                Outcome.UNRUNNABLE_TEST_SELECTOR_STALE,
                f"{c.command!r} exited {c.exit_code} with a stale test selector "
                "(0 items collected, unrelated collection errors) -- not a reproduction",
            )

    # Rung 1b: un-runnable (synthesis incomplete). spike-step-5/composer-live/
    # RESULT.md Finding 6 -- a substrate:none hypothesis whose test-file
    # synthesis fell back to the loud-failing stub (LLM synthesis unavailable
    # or produced an unusable body, see surface_inference.TestFileSynthesizer)
    # raises an AssertionError carrying this marker. That's a genuine
    # non-zero exit, but it says nothing about the reported bug -- checked
    # before rung 2's substring match and rung 6's generic "last command
    # failed" so the fallback stub can never be misread as a reproduction.
    for c in commands:
        if c.exit_code != 0 and SYNTHESIS_INCOMPLETE_MARKER in _command_text(c):
            return (
                Outcome.UNRUNNABLE_SYNTHESIS_INCOMPLETE,
                f"{c.command!r} ran a test-file synthesis fallback stub with no real "
                "assertion (LLM synthesis was unavailable or produced an unusable body) "
                "-- not a reproduction, not runnable",
            )

    # Rung 1c: the synthesised test is broken rather than failing. Only an
    # *assertion* failure is evidence about the bug: the assertion is the
    # thing that encodes the hypothesis's expected-vs-observed claim. Any
    # other exception out of a test this project generated means synthesis
    # wrote a test that does not run -- a statement about the generator,
    # not about the issue.
    #
    # First real `substrate: none` run, 2026-08-18: the synthesised test did
    # `state = testing.State()` then `state.pebble = {...}`. `State` is
    # frozen, so pytest reported `dataclasses.FrozenInstanceError: cannot
    # assign to field 'pebble'`. Rung 6 read the non-zero exit as
    # `reproduced_weaker` and the pipeline **composed a comment opening
    # "The bug reproduced."** for a test that had tested nothing. Rung 1b
    # above only catches the loud-failing fallback stub, which is a
    # different failure: that one is a stub with no assertion, this one is
    # a real attempt at a test that is not valid `ops.testing` code.
    #
    # Scoped to `synthesized_test_file` deliberately -- a repro script the
    # *reporter* supplied is allowed to fail with any exception it likes,
    # since that exception may well be the bug.
    if hypothesis.synthesized_test_file is not None:
        for c in commands:
            if c.exit_code == 0:
                continue
            text = _command_text(c)
            if _ASSERTION_FAILURE_RE.search(text):
                break
            raised = _RAISED_EXCEPTION_RE.search(text)
            if raised:
                return (
                    Outcome.UNRUNNABLE_SYNTHESIS_INVALID,
                    f"the synthesised test raised {raised.group('exc')} rather than failing an "
                    "assertion -- the generated test is not valid, so its failure is evidence "
                    "about the generator and not about the reported bug",
                )

    # Rung 2: reproduced -- traceback class + observed substring match,
    # non-zero exit. The original ladder's one rung that still holds.
    for c in commands:
        if c.exit_code != 0 and _contains_any(_command_text(c), observed_snippets):
            return Outcome.REPRODUCED, f"{c.command!r} (exit {c.exit_code}) matched observed output"

    # Rung 3: reproduced (log-only). spike-step-4/2341 -- a
    # logger.warning() from ops-scenario's consistency checker that never
    # touches stdout/stderr or a non-zero exit code.
    if _contains_any(all_log_text, observed_snippets):
        return Outcome.REPRODUCED_LOG_ONLY, "observed substring matched captured log records (exit was zero)"

    # Rung 4: un-runnable (API-shape mismatch). spike-step-4/2327 -- the
    # extraction is written against the *current* API; the bug lived only
    # on an older one. Checked after the positive-match rungs above so a
    # real reproduction whose expected symptom *is* one of these
    # exceptions isn't swallowed here.
    for c in commands:
        if c.exit_code != 0 and _API_SHAPE_ERROR_RE.search(_command_text(c)):
            anchor = f" (symbol_anchor={hypothesis.moving_parts.symbol_anchor!r})" if hypothesis.moving_parts.symbol_anchor else ""
            return (
                Outcome.UNRUNNABLE_API_SHAPE_MISMATCH,
                f"{c.command!r} hit an API-shape error{anchor} -- extraction is pinned to a "
                "different API surface than the one the bug lived on",
            )

    # Rung 5: reproduced (positive-signal absent) + control case.
    # spike-step-4/2639 -- the reproduction *is* "a specific hook does not
    # fire." Silence is only meaningful evidence if a control stimulus
    # that *should* produce the signal actually did.
    if surface is not None and surface.expected_signal:
        main_text = "\n".join(_command_text(c) for c in commands)
        if surface.expected_signal not in main_text:
            if run_result.control is not None and surface.expected_signal in _command_text(run_result.control):
                return (
                    Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT,
                    f"expected signal {surface.expected_signal!r} absent from the main run but "
                    "present in the control run -- confirms a broken observer isn't the explanation",
                )
            return (
                Outcome.DID_NOT_REPRODUCE,
                f"expected signal {surface.expected_signal!r} absent, but no control confirmed the "
                "observer works at all -- can't distinguish a reproduced bug from a broken observer",
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
