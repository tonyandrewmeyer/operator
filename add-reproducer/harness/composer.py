"""Comment composition (PLAN.md Approach §6).

Every comment is rendered by this module's own code from the run record:
the automation banner, the versions line, the commands that ran (with the
test file they write), their output, any captured log records or control
run, the closing "Reproduced ... at ..." line and the idempotency marker.

Until `spike-step-5/static-retry/RESULT.md` §42 an LLM wrote the middle of
the comment (`Composer`, behind an `LLMSeam`), with `compose_template()` as
its fallback. But every part it wrote was a copy of something in the run
record that the prompt told it to copy verbatim, and `_validate()` rejected
a response whose commands were not exactly the ones that ran; the only
freedom left to the model was the wrapper, and that is where the composed
comments differed. A human read of eleven composed `#2045` comments found
one (`composed-41/2045-prbar2045d-14.md`) whose wrapper alone differed from
the other ten: an unlabelled fence, a reworded opening sentence, and a
closing `Reproduced on unpinned/unpinned` that contradicted its own
`substrate=none`. With nothing left for the model to supply, the comment is
rendered here and no LLM call is made. `Composer` keeps its constructor so
the pipeline's wiring is unchanged.

`COMMENT_OUTCOMES` gates rendering first: a silent outcome returns `None`
(PLAN.md Goal §4: "Nothing useful produced -> stay silent").

The leading `AUTOMATION_PREFIX` disclaimer is the mitigation criterion 2
rests on (`spike-step-5/EXIT-CRITERIA.md`, criterion 2), and the trailing
`<!-- add-reproducer:issue=<n>:run=<uuid> -->` marker is what Approach §7's
"skip if a comment carrying the same marker exists" check matches, so both
are on every comment.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import static_test_check
from models import (
    COMMENT_OUTCOMES,
    Hypothesis,
    Issue,
    Outcome,
    RunResult,
    TestFile,
    embedded_test_file,
)
from seams.llm import LLMSeam

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


def _prefix(hypothesis: Hypothesis) -> str:
    """`AUTOMATION_PREFIX`, plus a caveat when the test fakes hook tools.

    A fake hook tool prints what the issue says Juju returned, so a run shows
    how ops handles that output and nothing about whether Juju returns it. On
    `#2709` part of the reporter's account of Juju was wrong
    (`relation-list` succeeds), and a test built from it would have
    "reproduced" that part too. Structural for the same reason as the
    prefix: the claim the comment can make depends on it.
    """
    test_file = _resolved_test_file(hypothesis)
    tools = static_test_check.faked_hook_tools(test_file.body) if test_file is not None else []
    if not tools:
        return AUTOMATION_PREFIX
    names = ", ".join(f"`{tool}`" for tool in tools)
    return (
        f"{AUTOMATION_PREFIX}\n>\n"
        f"> The hook tools this test runs ({names}) are fakes that print what the issue "
        "reports Juju returning. The run shows how ops handles that output, not that "
        "Juju returns it."
    )


def _trim(text: str, limit: int = _TRIM_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [trimmed, {len(text) - limit} more characters]"


def _versions_line(hypothesis: Hypothesis, run_result: RunResult) -> str:
    """What the reproduction ran on, in words.

    Before `spike-step-5/static-retry/RESULT.md` §42 this was
    `repo=unpinned, juju=unpinned, base=unpinned, substrate=none`: four
    pinned-or-not fields, two of them meaningless for a Scenario-only run,
    which has no Juju and no base. A reader wants what was used, so this
    leads with what the run record measured and leaves out what does not
    apply.

    - `ops`: `run_result.observed_ops_version`, the `ops` the run resolved
      (`seams/runner.py:_resolved_ops_version()` on the `none` branch, the
      packed one on the scratch-charm branches). The extraction's
      `repo_version` pin is added only when there is one.
    - With `substrate: none`, "Scenario only (no Juju)" and nothing about
      Juju or a base.
    - Otherwise Juju (`run_result.observed_juju_version`, which is what
      produced the verdict, and the pin when it says something else), the
      substrate and the base, each only when known: an unknown value is
      left out rather than printed as `unpinned`.
    """
    mp = hypothesis.moving_parts
    parts = []
    if run_result.observed_ops_version:
        parts.append(f"ops {run_result.observed_ops_version}")
    if mp.repo_version:
        parts.append(f"operator repository at {mp.repo_version}")
    if _scenario_only(hypothesis):
        parts.append("Scenario only (no Juju)")
    else:
        juju = run_result.observed_juju_version or mp.juju_version
        if juju:
            pinned = (
                f" (asked for {mp.juju_version})"
                if run_result.observed_juju_version and mp.juju_version and mp.juju_version != juju
                else ""
            )
            parts.append(f"Juju {juju}{pinned}")
        if mp.substrate:
            parts.append(f"{mp.substrate} substrate")
        if mp.base:
            parts.append(f"base {mp.base}")
    return ", ".join(parts) if parts else "not recorded"


def _scenario_only(hypothesis: Hypothesis) -> bool:
    return (hypothesis.moving_parts.substrate or "").strip().lower() == "none"


def _when(timestamp: str) -> str:
    """`2026-11-19T12:46:00.843074+00:00` as `2026-11-19 12:46 UTC`: a
    reader needs the minute, not the microsecond. A timestamp that does not
    parse is shown as given."""
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return f"{moment:%Y-%m-%d %H:%M} UTC"


def _fence(text: str, info: str = "") -> str:
    """`text` in a fenced code block whose fence no line of `text` can close
    (output that prints a markdown fence of its own would otherwise end the
    block early)."""
    longest = max((len(m) for m in re.findall(r"`{3,}", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{info}\n{text}\n{fence}"


def _observed_output(run_result: RunResult) -> str:
    """The output of the last command that ran, which is the one the outcome
    rests on: the runner stops at the first failing command, so this is
    either that command or the reproduction itself.

    The setup commands stay in "Commands run", but their output is left out.
    It is `uv`'s install chatter, and its first lines name the scratch
    workspace the runner isolates each issue in, which a reader running the
    same commands would never see.
    """
    if not run_result.commands:
        return "(no output captured)"
    last = run_result.commands[-1]
    return _trim(f"{last.stdout}{last.stderr}") or "(no output captured)"


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
    return hypothesis.synthesized_test_file or embedded_test_file(hypothesis.commands)


def _test_file_to_render(hypothesis: Hypothesis, run_result: RunResult) -> TestFile | None:
    """The test file this comment has to show for itself, or `None` when it
    does not have to show one -- either because there is no test file at all,
    or because the commands this run actually executed already show it.

    `spike-step-5/second-dispatch/RESULT.md` §5.2: the first composed comment
    anywhere rendered `#2045`'s fourteen-line test file twice -- once in a
    `<details>` block whose `<summary>` said it was "not shown in the commands
    actually run below", and again, four lines later, as the `cat > ... <<
    'PYEOF'` heredoc that wrote it. Both halves came from
    `_resolved_test_file()` alone, which asks where the file *came from* and
    never asks whether the reader can already see it.

    Provenance is the wrong question, and `#2341` and `#2045` are the two
    answers that prove it. Both embed their test file as a heredoc in the
    extraction's `commands[]`, so provenance cannot tell them apart -- but
    `#2341`'s executed run replays only the final `pytest` invocation, so the
    heredoc never reaches the "Commands run" block and the file genuinely has
    to be shown separately, while `#2045`'s `substrate: none` run executes the
    heredoc itself, so the block shows the file in full. The question that
    separates them is visibility, which is what this asks.

    Not a reword of `_test_file_heading()`, which was the other available fix:
    that heading's claim is the *reason* the section exists, so a heading that
    had to hedge about whether the body appears below would be describing a
    section that should not have been rendered. Deciding not to render it
    keeps the existing wording true by construction on every path, and drops
    the duplicated body rather than annotating it.
    """
    test_file = _resolved_test_file(hypothesis)
    if test_file is None:
        return None
    # `CommandResult.command` is the whole command string, heredoc body and
    # all, and it is what the "Commands run" block renders verbatim -- so a
    # substring test against it is exactly the question "will the reader see
    # this body below?", not an approximation of it.
    if any(test_file.body in c.command for c in run_result.commands):
        return None
    return test_file


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
    # PLAN.md Approach §6 specifies "a one-line 'reproduced on
    # <base>/<substrate> at <timestamp>'" for both `reproduced` and
    # `partial` outcomes. The verb is outcome-aware
    # (`spike-step-5/composer-live/RESULT.md`): a `partial` attempt did not
    # reproduce the bug, so its line says "Attempted". Where it ran is said
    # in words (§42): a Scenario-only run names no base or substrate, and an
    # unknown one is left out rather than shown as `unpinned/unpinned`, the
    # form a model once wrote under its own `substrate=none`.
    verb = "Attempted" if outcome == Outcome.PARTIAL else "Reproduced"
    mp = hypothesis.moving_parts
    if _scenario_only(hypothesis):
        where = " with Scenario (no Juju)"
    else:
        on = ", ".join(x for x in (f"{mp.substrate} substrate" if mp.substrate else "", mp.base or "") if x)
        where = f" on {on}" if on else ""
    return f"{verb}{where} at {_when(timestamp)}."


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
    """The comment for a comment-worthy outcome, or `None`.

    Only the commands, the output, the test file, the log records and the
    control run vary, and each comes from `run_result` (or the hypothesis's
    test file) unchanged. Everything around them is fixed text: the banner,
    a `partial` caveat, the versions line, the closing line and the marker.
    There is no opening "the bug reproduced, please verify" sentence: the
    banner says that already (§42), and the closing line says what
    reproduced and where. `reason` (the classifier's) is not rendered; it
    is kept in the signature for the pipeline's call.
    """
    if outcome not in COMMENT_OUTCOMES:
        return None

    blocks = [_prefix(hypothesis)]
    if outcome == Outcome.PARTIAL:
        blocks.append("**Partial reproduction attempt**: it got partway, then diverged.")
    blocks.append(f"Versions: {_versions_line(hypothesis, run_result)}")
    test_file = _test_file_to_render(hypothesis, run_result)
    if test_file is not None:
        # Collapsed: `maintainer-review/RESULT.md` §7, these bodies are long
        # enough that an uncollapsed one pushes the rest below the fold. GFM
        # only renders markdown inside <details> when there's a blank line
        # straight after <summary>, so that blank line isn't cosmetic.
        blocks.append(
            f"<details>\n<summary>{_test_file_heading(hypothesis, test_file)}</summary>\n\n"
            f"{_fence(test_file.body, 'python')}\n\n</details>"
        )
    commands = "\n".join(f"$ {c.command}" for c in run_result.commands)
    blocks.append(f"Commands run:\n{_fence(commands, 'shell')}")
    blocks.append(f"Output:\n{_fence(_observed_output(run_result))}")
    log_output = _log_output(run_result)
    if log_output is not None:
        blocks.append(f"Captured log records (not part of the output above):\n{_fence(log_output)}")
    control_output = _control_output(run_result)
    if control_output is not None:
        blocks.append(
            "Control (re-run to confirm the observer itself works: this rules out a "
            "broken check that would never have detected anything, not a reproduction "
            f"of the bug):\n{_fence(control_output)}"
        )
    blocks.append(_reproduced_line(hypothesis, outcome, timestamp))
    blocks.append(_marker(issue, run_id))
    return "\n\n".join(blocks)


#: The renderer under its other names: `pipeline.py` once called it plain
#: `compose`, and `render` is what it does.
compose = compose_template
render = compose_template


class Composer:
    """The pipeline's composer: `compose_template()` behind the interface the
    pipeline already calls.

    It takes an `LLMSeam` because it used to make one call to it; since §42
    it makes none (see the module docstring), and the seam is kept only so
    `Pipeline` and its tests construct it as before.
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
        return compose_template(
            hypothesis, issue, run_result, outcome, reason, run_id=run_id, timestamp=timestamp
        )
