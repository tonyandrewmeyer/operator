"""Comment composition (PLAN.md Approach §6).

Two paths, mirroring the seam pattern `extraction.py`/`surface_inference.py`
already use for their own LLM calls:

- `compose_template()` -- the original deterministic renderer (module
  docstring used to call this a "minimal stub"). Fixed shape, no LLM. Kept
  as-is so the existing fixture-driven suite keeps passing unchanged, and
  reused below as `Composer`'s own fallback.
- `Composer` -- Approach §6's actual design: "one final LLM call to turn the
  run log into a short, readable steps-to-reproduce comment." Behind an
  `LLMSeam`, exactly like `Extractor`/`SurfaceInferrer`. Any failure of that
  call (no key, transport error, invalid JSON, malformed response) falls
  back to `compose_template()` rather than raising or leaving the issue
  silent -- a comment-worthy outcome always gets *a* comment.

`COMMENT_OUTCOMES` gates both paths identically and first: a silent outcome
returns `None` before either path ever builds a prompt, so a live key is
never billed for an issue the pipeline was always going to stay quiet on
(PLAN.md Goal §4: "Nothing useful produced -> stay silent").

The trailing `<!-- add-reproducer:issue=<n>:run=<uuid> -->` idempotency
marker (Approach §7) is always appended by this module's own code, in both
paths, never left to the model to reproduce -- Approach §7's "skip if a
comment carrying the same marker exists" check depends on that exact,
literal shape, and an LLM asked to emit it verbatim is a needless place to
risk a paraphrase.

The leading `AUTOMATION_PREFIX` disclaimer is prepended the same way, and
for the same reason turned up one notch: it is the mitigation criterion 2
rests on (`spike-step-5/EXIT-CRITERIA.md`, criterion 2), so it must be
present on every composed comment whatever the model returns. Both the
prefix and the marker are therefore structural, not instructed -- the
prompt tells the model to omit both.
"""

from __future__ import annotations

import re

from models import COMMENT_OUTCOMES, Hypothesis, Issue, Outcome, RunResult, TestFile
from seams.llm import LLMError, LLMSeam

_TRIM_CHARS = 1500

#: Prepended verbatim to every composed comment, on both paths. PLAN.md's
#: Open Questions section named this as mitigation (b) for the
#: false-comment cost -- "prefix every comment with a visible 'reproduced by
#: automation, please verify' line", noting that the trailing HTML marker
#: isn't reader-visible. It is worded as an *attempt* so the one string
#: covers the `partial` rung as honestly as the reproduced ones, and it is
#: a blockquote so it reads as chrome above the comment rather than as the
#: comment's first claim.
AUTOMATION_PREFIX = (
    "> **Automated reproduction attempt.** These steps were produced by a bot "
    "that ran the commands below; please verify before relying on them."
)


def _trim(text: str, limit: int = _TRIM_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [trimmed, {len(text) - limit} more characters]"


def _versions_line(hypothesis: Hypothesis, run_result: RunResult) -> str:
    mp = hypothesis.moving_parts
    parts = [
        f"repo={mp.repo_version or 'unpinned'}",
        f"juju={mp.juju_version or 'unpinned'}",
        f"base={mp.base or 'unpinned'}",
        f"substrate={mp.substrate or 'unknown'}",
    ]
    # `mp.juju_version` is only ever what the extraction *pinned* -- often
    # nothing at all. `run_result.observed_juju_version` is what actually
    # produced this verdict (`RunResult.observed_juju_version`'s docstring:
    # `#2639` reproduces on juju 4.0.5 and returns `DID_NOT_REPRODUCE` on
    # 3.6.27), and a reader trusting a "Reproduced" comment needs to be able
    # to tell the two apart rather than assume the pinned value describes
    # what actually ran. Omitted, not "observed_juju=unknown", when the
    # runner seam couldn't determine it (e.g. the `none` branch, which never
    # touches a juju substrate) -- unlike the fields above, this one has no
    # pinned/unpinned distinction to fall back on.
    if run_result.observed_juju_version:
        parts.append(f"observed_juju={run_result.observed_juju_version}")
    return ", ".join(parts)


def _observed_output(run_result: RunResult) -> str:
    combined = "\n".join(f"{c.stdout}{c.stderr}" for c in run_result.commands)
    return _trim(combined) or "(no output captured)"


def _log_output(run_result: RunResult) -> str | None:
    """Rendered captured-log-record text, or `None` if no command recorded
    any.

    `spike-step-5/composer-scale/RESULT.md` Finding 1: `classifier.classify()`'s
    rung 3 (`reproduced_log_only`, spike-step-4/2341's shape -- a
    `logger.warning()` that never touches stdout/stderr) matches against
    `CommandResult.log_records`, a channel `_observed_output()` never reads
    (it only ever walks `stdout`/`stderr`). Before this fix, every composed
    `reproduced_log_only` comment rendered "Observed output" from stdout/
    stderr alone -- for this rung specifically, that is the evidence the
    classifier did *not* use, and it can read as a clean pass (e.g. "1 passed
    in 0.08s") sitting directly under a "the bug reproduced" opening line,
    self-contradictory on its face. Same shape as Finding 1's control-run gap
    in `composer-live/RESULT.md`: the one thing a reader would need to
    verify the claim was never shown.
    """
    lines = [record for c in run_result.commands for record in c.log_records]
    if not lines:
        return None
    return _trim("\n".join(lines))


# `spike-step-5/maintainer-review/FOLLOWUPS.md` §1: `hypothesis.synthesized_test_file`
# (below) is the ONLY source `_resolved_test_file()` used to check -- but a
# `substrate: none` extraction can also embed the file it needs directly in
# `commands[]` as a shell heredoc (`cat > <path> << 'EOF' ... EOF`, the
# `#2341` shape -- `surface_inference._heredoc_writes()` already recognises
# this exact form to decide `needs_test_file()` is False for it). That body
# is real data already sitting in the hypothesis; the composer just never
# read it. Matches whatever `cat`/`tee` heredoc `_heredoc_writes()` accepts,
# so the two never drift on what counts as "the file is really there".
_HEREDOC_RE = re.compile(
    r"(?:cat|tee)\s*>\s*(?P<path>\S+)\s*<<\s*'?(?P<delim>\w+)'?\n(?P<body>.*?)\n(?P=delim)",
    re.DOTALL,
)


def _embedded_test_file(commands: list[str]) -> TestFile | None:
    """A test file written by a heredoc already present in `commands[]`,
    or `None` if no command matches. See `_HEREDOC_RE`'s comment above."""
    for command in commands:
        match = _HEREDOC_RE.search(command)
        if match:
            return TestFile(path=match.group("path"), body=match.group("body"))
    return None


def _resolved_test_file(hypothesis: Hypothesis) -> TestFile | None:
    """The test file this hypothesis's commands depend on, from whichever of
    the two sources produced it -- LLM synthesis
    (`runner_stage.write_synthesized_test_file_if_needed()`) or a heredoc
    already embedded in the extracted `commands[]`. `synthesized_test_file`
    takes priority since it is a copy of the file actually written to disk
    for the run that happened, where an embedded heredoc is only ever a
    plan for one.

    `spike-step-5/maintainer-review/FOLLOWUPS.md` §1: closing only the
    `synthesized_test_file` gap (`composer-live/RESULT.md` Finding 4,
    2026-07-30) left every hypothesis whose test file arrived the other way
    with no rendered file at all -- three of `maintainer-review/RESULT.md`
    §7's five passing ratings were conditional on seeing this content.
    """
    return hypothesis.synthesized_test_file or _embedded_test_file(hypothesis.commands)


def _test_file_heading(hypothesis: Hypothesis, test_file: TestFile) -> str:
    """Label text for the rendered/prompted test-file section. Kept
    provenance-specific: "Synthesized" is only true of the
    `synthesized_test_file` source (`test_pipeline_e2e.py`'s
    `test_2045_calibration_mode_composed_comment_includes_synthesized_test_file`
    pins this exact string for that path) -- a heredoc-embedded file was
    never synthesised by anything, it was already part of the extraction,
    and saying otherwise would overclaim what happened, same as any other
    wording this module is careful not to overclaim."""
    if hypothesis.synthesized_test_file is not None:
        return f"Synthesized test file: {test_file.path} (written directly, not shown as a command below)"
    return f"Test file: {test_file.path} (from the extracted commands, not shown in the commands actually run below)"


def _control_output(run_result: RunResult) -> str | None:
    """Rendered control-case command + output, or `None` if this run has no
    control (most outcomes don't).

    Found missing from both render paths while measuring #2639's live
    composition (spike-step-5/composer-live/RESULT.md): `_observed_output()`
    only ever walked `run_result.commands`, so a `reproduced_positive_signal_
    absent` comment stated the classifier's conclusion ("absent from the
    main run but present in the control run") without ever showing the
    control run itself. PLAN.md's own control-case concept
    (spike-step-4/2639/RESULT.md) exists specifically so "silence" is
    checked against a stimulus that *should* produce the signal -- a
    comment that asserts the check happened without showing it asks the
    reader to trust exactly the step this pipeline is supposed to make
    verifiable.
    """
    if run_result.control is None:
        return None
    c = run_result.control
    return _trim(f"$ {c.command}\n{c.stdout}{c.stderr}")


def _reproduced_line(hypothesis: Hypothesis, outcome: Outcome, timestamp: str) -> str:
    # PLAN.md Approach §6 literally specifies "a one-line 'reproduced on
    # <base>/<substrate> at <timestamp>'" for both `reproduced` and
    # `partial` outcomes. Composing a live `partial` comment during this
    # task's own dry run (see spike-step-5/composer-live/RESULT.md) showed
    # why that's a false-comment risk worth not shipping literally: the
    # opening line correctly says "got partway, then diverged", but an
    # unconditional trailing "Reproduced on ..." line contradicts it --
    # exactly the kind of overclaiming PLAN.md's own Open Questions section
    # warns costs a maintainer a skipped verification. Verb is outcome-aware;
    # the base/substrate/timestamp payload PLAN.md asks for is unchanged.
    verb = "Attempted" if outcome == Outcome.PARTIAL else "Reproduced"
    return (
        f"{verb} on {hypothesis.moving_parts.base or 'unknown base'}/"
        f"{hypothesis.moving_parts.substrate or 'unknown substrate'} at {timestamp}."
    )


def _marker(issue: Issue, run_id: str) -> str:
    return f"<!-- add-reproducer:issue={issue.number}:run={run_id} -->"


def compose_template(
    hypothesis: Hypothesis,
    issue: Issue,
    run_result: RunResult,
    outcome: Outcome,
    reason: str,
    *,
    run_id: str,
    timestamp: str,
) -> str | None:
    """Deterministic renderer -- Approach §6's fallback path. Behaviour is
    unchanged from before this module grew an LLM path."""
    if outcome not in COMMENT_OUTCOMES:
        return None

    lines = [AUTOMATION_PREFIX, ""]
    if outcome == Outcome.PARTIAL:
        lines.append("**Partial reproduction attempt** -- got partway, then diverged.")
    else:
        lines.append("**Reproduced** -- automated attempt, please verify before relying on it.")
    lines.append("")
    lines.append(f"Versions: {_versions_line(hypothesis, run_result)}")
    lines.append(f"Outcome: `{outcome.value}` -- {reason}")
    lines.append("")
    test_file = _resolved_test_file(hypothesis)
    if test_file is not None:
        # Collapsed <details> block, not a bare fenced code block --
        # `maintainer-review/RESULT.md` §7's cross-cutting note: these bodies
        # are long enough that inlining one uncollapsed pushes the rest of
        # the comment below the fold. GFM only renders markdown inside
        # <details> when there's a blank line straight after <summary>, so
        # that blank line isn't cosmetic.
        lines.append("<details>")
        lines.append(f"<summary>{_test_file_heading(hypothesis, test_file)}</summary>")
        lines.append("")
        lines.append("```python")
        lines.append(test_file.body)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    lines.append("Commands run:")
    lines.append("```shell")
    for c in run_result.commands:
        lines.append(f"$ {c.command}")
    lines.append("```")
    lines.append("")
    lines.append("Observed output:")
    lines.append("```")
    lines.append(_observed_output(run_result))
    lines.append("```")
    lines.append("")
    log_output = _log_output(run_result)
    if log_output is not None:
        lines.append("Captured log records (not part of stdout/stderr above):")
        lines.append("```")
        lines.append(log_output)
        lines.append("```")
        lines.append("")
    control_output = _control_output(run_result)
    if control_output is not None:
        lines.append(
            "Control (re-run to confirm the observer itself works: this rules out a "
            "broken check that would never have detected anything, not a reproduction "
            "of the bug):"
        )
        lines.append("```")
        lines.append(control_output)
        lines.append("```")
        lines.append("")
    lines.append(_reproduced_line(hypothesis, outcome, timestamp))
    lines.append("")
    lines.append(_marker(issue, run_id))
    return "\n".join(lines)


# Backward-compatible alias: `pipeline.py` (and this module's own former
# docstring) referred to the template renderer as plain `compose`. Keep the
# name usable for anyone who wants the fixed template with no LLM involved.
compose = compose_template


class ComposerInvalid(ValueError):
    """Raised when the LLM's composition output doesn't match the expected
    shape. `Composer.compose()` catches this itself and falls back to
    `compose_template()` -- it never propagates to a caller."""


_SCHEMA_INSTRUCTIONS = """\
Given the reproduction run below, write a short, readable GitHub issue
comment as JSON matching exactly this shape (PLAN.md Approach §6):

{
  "comment_body": str
}

comment_body is markdown. It MUST include, in this order:

1. One line stating the outcome plainly: for `partial`, say the attempt got
   partway and diverged, do NOT say the bug reproduced; for anything else,
   say it reproduced and that this is an automated attempt the reader
   should verify.
2. A "Versions:" line, exactly as given below -- copy it verbatim, do not
   reformat or guess at any value it doesn't contain.
3. The exact commands that were run, in a fenced shell code block, in the
   given order, verbatim -- do not add, remove, reorder, or "clean up" any
   command.
4. The observed output, trimmed if long, in a fenced code block, taken only
   from the "Observed output (actual run)" text below -- never invent or
   embellish output that isn't there.
5. One line: "Reproduced on <base>/<substrate> at <timestamp>." using
   exactly the base/substrate/timestamp values given below -- EXCEPT when
   the outcome is `partial`, where this line must instead read "Attempted
   on <base>/<substrate> at <timestamp>.": a partial attempt did not
   reproduce the bug, and saying "Reproduced" on that trailing line would
   contradict line 1's own "diverged" wording and overclaim what happened.

If a "Synthesized test file" or "Test file" section is given below (exactly
one of the two, matching its own heading verbatim), the commands section
alone does not show it -- either it was written directly to disk rather than
run as a shell command, or it was embedded in the extraction's own commands
but is not part of the commands this run actually executed -- so a reader
copy-pasting only the commands below would hit FileNotFoundError. Include it
verbatim, positioned BEFORE the commands section (the reader needs the file
to exist before the commands that reference it make sense), as a collapsed
section so it doesn't push the rest of the comment down: an HTML `<details>`
block whose `<summary>` is the given heading text verbatim, with a blank
line after the `<summary>` line, then the file's contents in their own
fenced python code block, then `</details>`. Do not paraphrase, summarise,
or truncate the file's contents.

If a "Captured log records" section is given below, include it too,
immediately after the "Observed output" block, under its own "Captured log
records (not part of stdout/stderr):" heading, in a fenced code block,
verbatim. For a `reproduced_log_only` outcome this is the *only* evidence
of the bug -- the commands above may show a clean pass (exit 0, no
stdout/stderr output worth reading) with nothing wrong-looking about it on
its own, and the log records are what actually grounds the "reproduced"
claim. Never say the bug reproduced without showing this section when it is
given; a reader who sees only a clean-looking stdout/stderr block under a
"reproduced" claim has no way to tell the claim is true.

If a "Control run" section is given below, include it too, between steps 4
and 5, under its own "Control (re-run to confirm the observer itself works:
this rules out a broken check that would never have detected anything, not
a reproduction of the bug):" heading, in a fenced code block, verbatim. This
outcome's whole claim rests on that control (the signal was absent in the
main run but present here, which is what rules out a broken observer rather
than a reproduced bug) -- never state or imply that a control confirmed
anything without showing it, and never fabricate one if this section is
absent.

Do not include a trailing HTML comment marker, and do not open the comment
with an "automated"/"generated by a bot"/"please verify" disclaimer line --
both are added separately, after your response, and are not part of
comment_body. Line 1 above is the outcome statement, nothing before it.

This comment will be posted on a real bug report and a maintainer will act
on it without re-running anything themselves. Do not state or imply
anything the commands/output below don't actually show: do not claim a fix
works, do not claim a root cause beyond what the hypothesis's own
`expected`/`observed` text says, and do not soften or omit a `partial` or
weaker outcome's caveats. A confidently wrong comment is worse than no
comment at all, because the reader trusts it and skips verifying.
"""


_DISCLAIMER_LINE = re.compile(r"^\s*>?\s*\**\s*(automated|generated)\b.*", re.IGNORECASE)


def _strip_leading_disclaimer(body: str) -> str:
    """Drop a disclaimer the model opened with anyway.

    `AUTOMATION_PREFIX` is prepended structurally, so a model that ignores
    the prompt's "do not open with a disclaimer" line would give the reader
    the same sentence twice. That is cosmetic, not misleading, so it is
    repaired here rather than rejected in `_validate()` -- falling back to
    the template would throw away a good comment over a duplicated line.
    """
    lines = body.split("\n")
    if lines and _DISCLAIMER_LINE.match(lines[0]):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)


def _validate(
    raw: dict, *, control_output: str | None, test_file_path: str | None, log_output: str | None
) -> None:
    if not isinstance(raw, dict) or "comment_body" not in raw:
        raise ComposerInvalid(f"missing 'comment_body' key, got {raw!r}")
    body = raw["comment_body"]
    if not isinstance(body, str) or not body.strip():
        raise ComposerInvalid(f"'comment_body' must be a non-empty string, got {body!r}")
    if log_output is not None and "log record" not in body.lower():
        # `_log_output()`'s docstring: for `reproduced_log_only`, the log
        # records ARE the evidence -- stdout/stderr alone can read as a
        # clean pass. Same shape as the control check below: a response
        # that drops this isn't incomplete, it's a "reproduced" claim with
        # no visible support, indistinguishable from "did not reproduce" to
        # a reader who only looks at the commands/output blocks.
        raise ComposerInvalid("comment_body omits the log records this outcome's claim depends on")
    if control_output is not None and "control" not in body.lower():
        # This outcome's entire claim rests on the control run (see
        # `_control_output()`'s docstring) -- a response that drops it isn't
        # just incomplete, it's the exact "confidently wrong" shape PLAN.md's
        # Open Questions section warns about. Falling back to
        # `compose_template()`, which always renders the control section,
        # is safer than posting a comment that asserts a signal was absent
        # "but present in the control run" with no control shown.
        raise ComposerInvalid("comment_body omits the control run this outcome's claim depends on")
    if test_file_path is not None and test_file_path not in body:
        # Same shape as the control check above, for Finding 4
        # (`spike-step-5/composer-live/RESULT.md`) and its
        # `spike-step-5/maintainer-review/FOLLOWUPS.md` §1 extension to a
        # heredoc-embedded file: a response that drops the file leaves the
        # commands section referencing one the reader was never shown -- not
        # incomplete, actively misleading, since "Commands run" reads as
        # self-contained when it isn't.
        raise ComposerInvalid(
            f"comment_body omits the test file {test_file_path!r} its commands reference"
        )


class Composer:
    """Approach §6's LLM composition path.

    `compose_template()` is this class's own fallback on any failure (no
    key, transport error, invalid JSON, malformed response) -- a
    comment-worthy outcome always yields *a* comment, never an exception and
    never a silently dropped one.
    """

    def __init__(self, llm: LLMSeam):
        self.llm = llm

    def compose(
        self,
        hypothesis: Hypothesis,
        issue: Issue,
        run_result: RunResult,
        outcome: Outcome,
        reason: str,
        *,
        run_id: str,
        timestamp: str,
    ) -> str | None:
        if outcome not in COMMENT_OUTCOMES:
            return None
        test_file = _resolved_test_file(hypothesis)
        try:
            prompt = self._build_prompt(hypothesis, issue, run_result, outcome, reason, timestamp)
            raw = self.llm.complete_json(
                purpose="composition", prompt=prompt, context={"issue_number": issue.number}
            )
            _validate(
                raw,
                control_output=_control_output(run_result),
                test_file_path=test_file.path if test_file is not None else None,
                log_output=_log_output(run_result),
            )
        except (LLMError, ComposerInvalid):
            # compose_template() already appends the marker -- return
            # directly rather than falling through to the shared
            # marker-append below, which would otherwise double it up.
            return compose_template(
                hypothesis, issue, run_result, outcome, reason, run_id=run_id, timestamp=timestamp
            )
        body = _strip_leading_disclaimer(raw["comment_body"].rstrip())
        # Prepended here, not asked for in the prompt, for the same reason
        # the marker is: criterion 2's accepted-risk mitigation is only
        # worth anything if it cannot be dropped by a model that decided
        # the comment read better without it.
        return f"{AUTOMATION_PREFIX}\n\n{body}\n\n{_marker(issue, run_id)}"

    @staticmethod
    def _build_prompt(
        hypothesis: Hypothesis,
        issue: Issue,
        run_result: RunResult,
        outcome: Outcome,
        reason: str,
        timestamp: str,
    ) -> str:
        commands_block = "\n".join(f"$ {c.command}" for c in run_result.commands)
        control_output = _control_output(run_result)
        control_section = f"Control run (must appear in your comment, verbatim):\n{control_output}\n" if control_output is not None else ""
        log_output = _log_output(run_result)
        log_section = (
            f"Captured log records (must appear in your comment, verbatim, immediately after "
            f"Observed output -- this outcome's evidence is NOT in stdout/stderr):\n{log_output}\n"
            if log_output is not None
            else ""
        )
        test_file = _resolved_test_file(hypothesis)
        test_file_prompt_section = (
            f"{_test_file_heading(hypothesis, test_file)} (heading must appear verbatim as an "
            f"HTML <details> <summary>, BEFORE the commands section -- see the instructions "
            f"above):\n{test_file.path}:\n{test_file.body}\n"
            if test_file is not None
            else ""
        )
        return (
            f"{_SCHEMA_INSTRUCTIONS}\n"
            f"Issue #{issue.number}: {issue.title}\n"
            f"Outcome: {outcome.value} -- {reason}\n"
            f"Versions: {_versions_line(hypothesis, run_result)}\n"
            f"Timestamp: {timestamp}\n"
            f"Hypothesis expected: {hypothesis.expected}\n"
            f"Hypothesis observed (from the issue): {hypothesis.observed}\n"
            f"{test_file_prompt_section}"
            f"Commands run, in order:\n{commands_block}\n"
            f"Observed output (actual run):\n{_observed_output(run_result)}\n"
            f"{log_section}"
            f"{control_section}"
        )
