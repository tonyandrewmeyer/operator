"""Shared data shapes for the add-reproducer pipeline.

Mirrors PLAN.md Approach §3's `moving_parts` schema (plus the `ci_run_url` /
`symbol_anchor` delta), the surface-inference pass's output (Approach §3's
"distinct surface inference pass" note), and the reproduction runner's
captured output (Approach §4/§5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# `#1685`/`#1552` (spike-step-5/criterion-1-live/RESULT.md §5.1): a live
# extraction sometimes opens a heredoc (`cat > file.py << 'PYEOF'`) and then
# puts every line of the heredoc's body -- and its closing terminator -- in
# its own `commands[]` element, instead of one element carrying the whole
# heredoc the way `#2327`/`#2341`'s extractions do. Two consumers read
# `commands[]` two different, individually-correct, jointly-incompatible
# ways: `surface_inference.needs_test_file()` joins the whole list with
# newlines and reads it as a script (sees the heredoc, declines to
# synthesise a replacement test file), while `runnability.assess()` and
# `seams/runner.py`'s `_run_none` (one `bash -c` per element) both read it as
# a list of independent commands (the body lines look like bare Python
# and get rejected/never execute). Re-joining a split heredoc back into one
# element here, at `Hypothesis` construction, is the single point both
# consumers -- and the runner -- read from, so they agree by construction
# rather than needing to separately learn the same detection logic.
_HEREDOC_OPENER_RE = re.compile(r"<<-?\s*(['\"]?)(?P<delim>[A-Za-z_][A-Za-z0-9_]*)\1\s*$")


def _join_split_heredocs(commands: list[str]) -> list[str]:
    """Fold a heredoc opener and its following elements back into one
    `commands[]` entry, when the body/terminator were split across separate
    elements instead of embedded in the opener's own string.

    An element only counts as a split-heredoc *opener* if it ends in
    `<< DELIM` (optionally `<<-`, optionally quoted) with nothing after that
    on the same element -- a self-contained heredoc (the shape
    `surface_inference._heredoc_writes()` already handles) carries its body
    and terminator in that same element and is left untouched here. When an
    opener is found, every following element up to and including the first
    one that is exactly the bare delimiter is folded into it, newline-joined
    (mirroring how that heredoc would have looked as one `cat << 'EOF' ...
    EOF` string). If the terminator never turns up, the opener is left as-is
    rather than silently swallowing the rest of `commands[]`.
    """
    out: list[str] = []
    i = 0
    while i < len(commands):
        command = commands[i]
        match = _HEREDOC_OPENER_RE.search(command.rstrip())
        if match is None:
            out.append(command)
            i += 1
            continue
        delim = match.group("delim")
        end = next((j for j in range(i + 1, len(commands)) if commands[j].strip() == delim), None)
        if end is None:
            out.append(command)
            i += 1
            continue
        out.append("\n".join(commands[i : end + 1]))
        i = end + 1
    return out


@dataclass
class Comment:
    """A single issue comment (PLAN.md Approach §3 delta,
    `spike-step-5/2185/RESULT.md` / `spike-step-5/2045/RESULT.md`).

    Deliberately minimal: author login, body, and created-at timestamp are
    all extraction needs to attribute a comment and place it in chronological
    order. `gh issue view --json ...,comments` (and the raw GitHub REST/GraphQL
    API, per `spike-step-5/2045/issue.json`) nest the author as
    `{"login": ...}`; `from_dict` unwraps that so hand-authored fixtures can
    also just pass a plain string.
    """

    author: str
    body: str
    created_at: str

    @classmethod
    def from_dict(cls, d: dict) -> "Comment":
        author = d.get("author") or ""
        if isinstance(author, dict):
            author = author.get("login", "")
        return cls(author=author, body=d.get("body") or "", created_at=d.get("createdAt", ""))


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    state: str  # "OPEN" | "CLOSED"
    created_at: str
    author: str
    repo: str
    comments: list[Comment] = field(default_factory=list)
    # PLAN.md Approach §2 gap (spike-step-5/corpus-v2/RESULT.md "Two filter
    # findings" #1): production reads this from the `issues.opened` webhook
    # payload's `issue.type.name`, not from `gh`, which has no `issueType`
    # JSON field at all. `None` when the webhook carried no type (or a
    # hand-built fixture doesn't set one).
    issue_type: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Issue":
        raw_type = d.get("type")
        if isinstance(raw_type, dict):
            issue_type = raw_type.get("name")
        elif isinstance(raw_type, str):
            issue_type = raw_type or None
        else:
            issue_type = None
        # `gh issue list --json labels` -- the exact command the harness
        # README documents for building the dump -- emits a list of label
        # *objects* (`{id, name, description, color}`), not a list of
        # names. Every fixture in `fixtures/issues/` hand-writes bare
        # strings, so the whole test suite ran against a shape `gh` never
        # produces, and the first live run died in `filter_stage` on
        # `label.lower()` before reaching a single LLM call. Accept both,
        # same as `type` above.
        labels = [
            label.get("name", "") if isinstance(label, dict) else label
            for label in (d.get("labels") or [])
        ]
        return cls(
            number=d["number"],
            title=d["title"],
            body=d.get("body") or "",
            labels=labels,
            state=d["state"],
            created_at=d.get("createdAt", ""),
            author=d.get("author", ""),
            repo=d.get("repo", ""),
            comments=[Comment.from_dict(c) for c in (d.get("comments") or [])],
            issue_type=issue_type,
        )


@dataclass
class MovingParts:
    repo_version: str | None = None
    juju_version: str | None = None
    base: str | None = None
    substrate: str | None = None  # "lxd" | "k8s" | "none" | None
    other: dict = field(default_factory=dict)
    ci_run_url: str | None = None
    symbol_anchor: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "MovingParts":
        return cls(
            repo_version=d.get("repo_version"),
            juju_version=d.get("juju_version"),
            base=d.get("base"),
            substrate=d.get("substrate"),
            other=dict(d.get("other") or {}),
            ci_run_url=d.get("ci_run_url"),
            symbol_anchor=d.get("symbol_anchor"),
        )


@dataclass
class Hypothesis:
    issue_number: int
    in_scope: bool
    moving_parts: MovingParts
    commands: list[str]
    expected: str
    observed: str
    confidence: str  # "high" | "medium" | "low"
    # Set by `runner_stage.write_synthesized_test_file_if_needed()` when
    # synthesis fires (Approach §3/§4's `#2045`-shaped delta). Carries the
    # synthesised file itself through to the composer
    # (`spike-step-5/composer-live/RESULT.md` Finding 4: the file gets
    # written to the scratch workdir and referenced by `commands[]`, but
    # until this field existed nothing downstream kept a copy of its body,
    # so a composed comment could reference a file it never showed).
    synthesized_test_file: "TestFile | None" = None

    @classmethod
    def from_dict(cls, issue_number: int, d: dict) -> "Hypothesis":
        return cls(
            issue_number=issue_number,
            in_scope=d["in_scope"],
            moving_parts=MovingParts.from_dict(d.get("moving_parts") or {}),
            commands=_join_split_heredocs(list(d.get("commands") or [])),
            # `or ""` not `get(..., "")`: an explicit null is what a model
            # sends for an out-of-scope extraction, and downstream code
            # (classifier's `_quoted_snippets`, the composer) does string work
            # on these.
            expected=d.get("expected") or "",
            observed=d.get("observed") or "",
            confidence=d.get("confidence", "low"),
        )


@dataclass
class TestFile:
    """A synthesised pytest file (PLAN.md Approach §3/§4 delta,
    `spike-step-5/2045/RESULT.md` "PLAN deltas surfaced" §1).

    `substrate: none` hypotheses whose `commands[]` contain no runnable
    pytest invocation (the `#2045` shape: the extraction has enough
    `expected`/`observed` text to reason about, but no self-contained repro
    snippet) get a synthesised test body instead of staying un-runnable.
    `path` is relative to the scratch working directory the runner executes
    `commands[]` in.
    """

    path: str
    body: str

    @classmethod
    def from_dict(cls, d: dict) -> "TestFile":
        return cls(path=d["path"], body=d["body"])


@dataclass
class SurfaceInference:
    """Approach §3's "distinct surface inference pass" output.

    Consumed by `scaffold.py` (relation/storage/pebble_service/ops_api_surface,
    matching spike-step-3/charm/render.py's params.yaml shape) plus
    `expected_signal`, which the classifier's *positive-signal-absent* rung
    needs (Approach §5 / spike-step-4/2639/RESULT.md) and which has no home
    in render.py's params schema since it's a classification concern, not a
    scaffolding one.
    """

    charm_name: str
    relation: dict = field(default_factory=dict)
    storage: dict = field(default_factory=dict)
    pebble_service: dict = field(default_factory=dict)
    ops_api_surface: str | None = None
    expected_signal: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SurfaceInference":
        return cls(
            charm_name=d["charm_name"],
            relation=dict(d.get("relation") or {}),
            storage=dict(d.get("storage") or {}),
            pebble_service=dict(d.get("pebble_service") or {}),
            ops_api_surface=d.get("ops_api_surface"),
            expected_signal=d.get("expected_signal"),
        )


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    log_records: list[str] = field(default_factory=list)
    # Which part of the branch this command belongs to ("prepare", "pack",
    # "deploy", "stimulus", "status", "debug-log", "control"). The runner
    # built this from `PlannedCommand.step` and then threw it away, so
    # nothing downstream could tell a failed *prerequisite* from a failed
    # *reproduction attempt* -- the classifier saw seven anonymous commands
    # and read the last exit code. `None` for branches whose commands come
    # from `hypothesis.commands` rather than a planned sequence.
    step: str | None = None
    # Wall-clock seconds this command took. Criterion 7 (EXIT-CRITERIA.md)
    # is an end-to-end budget, but nothing recorded per-step durations, so a
    # 418s warm run could not be attributed across pack/deploy/wait -- see
    # `spike-step-5/gate-substrate/RESULT.md` §8. `None` for replayed
    # fixtures, which have no timing to report.
    elapsed_s: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "CommandResult":
        return cls(
            command=d["command"],
            exit_code=d["exit_code"],
            stdout=d.get("stdout", ""),
            stderr=d.get("stderr", ""),
            log_records=list(d.get("log_records") or []),
            step=d.get("step"),
            elapsed_s=d.get("elapsed_s"),
        )


@dataclass
class RunResult:
    """Captured output of the reproduction runner (Approach §4)."""

    hypothesis_number: int
    branch: str  # "none" | "k8s-scratch" | "lxd-scratch" | "k8s-clone"
    commands: list[CommandResult]
    control: CommandResult | None = None
    # Set when a prerequisite step failed and the rest of the sequence was
    # abandoned rather than executed against a substrate that never came
    # up. Before this, a failed `pack` was followed by a doomed `deploy`
    # and then `juju status`/`juju debug-log`, both of which exit **0**
    # against an empty model -- so the run looked healthy from the outside
    # and the classifier scored it as a statement about the bug.
    aborted_at_step: str | None = None
    skipped_steps: list[str] = field(default_factory=list)
    # The juju version that actually produced this run's verdict, distinct
    # from `hypothesis.moving_parts.juju_version` (what the extraction
    # *pinned*, if anything). `spike-step-5/wallclock-substrate/RESULT.md`
    # §5: `#2639` reproduces on juju 4.0.5 and returns `DID_NOT_REPRODUCE`
    # on 3.6.27, and nothing recorded which one a given run actually used --
    # the two can differ even when the hypothesis pins nothing, because
    # concierge's own default channel decides it. `None` for branches that
    # never touch a juju substrate (`none`) and for fixtures recorded before
    # this field existed -- optional so they keep loading unchanged.
    observed_juju_version: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        return cls(
            hypothesis_number=d["hypothesis_number"],
            branch=d["branch"],
            commands=[CommandResult.from_dict(c) for c in d["commands"]],
            control=CommandResult.from_dict(d["control"]) if d.get("control") else None,
            aborted_at_step=d.get("aborted_at_step"),
            skipped_steps=list(d.get("skipped_steps") or []),
            observed_juju_version=d.get("observed_juju_version"),
        )


class Outcome(str, Enum):
    """Approach §5's rung ladder, plus Approach §4's `skipped_stale` outcome."""

    SKIPPED_STALE = "skipped_stale"
    REPRODUCED = "reproduced"
    REPRODUCED_WEAKER = "reproduced_weaker"
    REPRODUCED_LOG_ONLY = "reproduced_log_only"
    REPRODUCED_POSITIVE_SIGNAL_ABSENT = "reproduced_positive_signal_absent"
    PARTIAL = "partial"
    DID_NOT_REPRODUCE = "did_not_reproduce"
    UNRUNNABLE_TEST_SELECTOR_STALE = "unrunnable_test_selector_stale"
    UNRUNNABLE_API_SHAPE_MISMATCH = "unrunnable_api_shape_mismatch"
    # `spike-step-5/live-llm/RESULT.md` Finding 2: commands[] that aren't
    # shell at all (prose, bare Python, test names). Caught by
    # `runnability.assess()` before the runner, because handing them to a
    # shell produces exit 127 and the classifier's rung 6 reads any non-zero
    # last command as `reproduced (weaker)` -- a comment-worthy outcome.
    UNRUNNABLE_COMMANDS_NOT_SHELL = "unrunnable_commands_not_shell"
    # `spike-step-5/composer-live/RESULT.md` Finding 6: a synthesized test
    # file whose LLM-driven synthesis (`surface_inference.TestFileSynthesizer`)
    # failed and fell back to the loud-failing stub has no real assertion --
    # its non-zero exit is not evidence about the reported bug, so it must
    # never be read as `reproduced_weaker` the way an always-passing stub's
    # zero exit used to be misread as `did_not_reproduce`.
    UNRUNNABLE_SYNTHESIS_INCOMPLETE = "unrunnable_synthesis_incomplete"
    # First real live run against k8s, 2026-08-18. A prerequisite step
    # (`concierge prepare` / `charmcraft pack` / `juju deploy`) failed, so
    # nothing was ever deployed and no stimulus ever ran. The rung ladder
    # below assumes the run got far enough to be evidence about the bug;
    # nothing used to state that precondition, so a charm that failed to
    # pack still produced `did_not_reproduce` -- a verdict about the bug
    # drawn from a run that never tested it. Worse with a control present:
    # the same shape reaches `reproduced_positive_signal_absent`, which
    # *composes a comment*.
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    # Same run. `_scratch_sequence()` builds a stimulus only when surface
    # inference hands over both a pebble user *and* command, and silently
    # built neither stimulus nor control when it didn't -- the run deployed
    # a charm, poked nothing, and still got scored. A scratch-charm run with
    # no stimulus is not evidence either way.
    UNRUNNABLE_NO_STIMULUS = "unrunnable_no_stimulus"
    # First real `substrate: none` run, 2026-08-18. The synthesised test
    # raised something other than an AssertionError -- it was not valid
    # `ops.testing` code (`state.pebble = {...}` on a frozen `State`), so
    # its non-zero exit says nothing about the reported bug. Distinct from
    # UNRUNNABLE_SYNTHESIS_INCOMPLETE, which is the *fallback stub* with no
    # assertion at all: this one is a real attempt that does not run.
    # Before this rung existed the pipeline composed a comment opening
    # "The bug reproduced." off exactly that failure.
    UNRUNNABLE_SYNTHESIS_INVALID = "unrunnable_synthesis_invalid"


# Outcomes that Approach §6 composes a comment for. Every other outcome is
# silent (PLAN.md Goal §4: "Nothing useful produced → stay silent").
COMMENT_OUTCOMES = frozenset(
    {
        Outcome.REPRODUCED,
        Outcome.REPRODUCED_WEAKER,
        Outcome.REPRODUCED_LOG_ONLY,
        Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT,
        Outcome.PARTIAL,
    }
)
