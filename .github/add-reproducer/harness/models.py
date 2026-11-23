"""Shared data shapes for the add-reproducer pipeline.

The extraction's `moving_parts` schema (including `ci_run_url` and
`symbol_anchor`), the surface-inference pass's output, and the reproduction
runner's captured output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# Some extractions split a heredoc across several `commands[]` elements.
# `needs_test_file()` reads the list as one script while `runnability` and the
# runner run each element separately, so split heredocs are re-joined here, at
# construction, and every consumer sees the same commands.
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
    """A single issue comment, threaded into the extraction prompt.

    Deliberately minimal: author login, body, and created-at timestamp are
    all extraction needs to attribute a comment and place it in chronological
    order. `gh issue view --json ...,comments` (and the raw GitHub REST/GraphQL
    API) nest the author as
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
    # From the `issues.opened` webhook's `issue.type.name`, since `gh` has no
    # `issueType` JSON field. `None` when no type was given.
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
        # `gh issue list --json labels` emits label objects while the fixtures
        # use bare strings, so accept both.
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
    # Set when test-file synthesis fires, so the composer can show the body of
    # the file that `commands[]` references.
    synthesized_test_file: "TestFile | None" = None

    @classmethod
    def from_dict(cls, issue_number: int, d: dict) -> "Hypothesis":
        return cls(
            issue_number=issue_number,
            in_scope=d["in_scope"],
            moving_parts=MovingParts.from_dict(d.get("moving_parts") or {}),
            commands=_join_split_heredocs(list(d.get("commands") or [])),
            # `or ""` rather than a `get` default: an out-of-scope extraction
            # sends explicit nulls, and downstream code does string work on
            # these.
            expected=d.get("expected") or "",
            observed=d.get("observed") or "",
            confidence=d.get("confidence", "low"),
        )


@dataclass
class TestFile:
    """A synthesised pytest file.

    `substrate: none` hypotheses whose `commands[]` contain no runnable
    pytest invocation (the `#2045` shape: the extraction has enough
    `expected`/`observed` text to reason about, but no self-contained repro
    snippet) get a synthesised test body instead of staying un-runnable.
    `path` is relative to the scratch working directory the runner executes
    `commands[]` in.
    """

    path: str
    body: str

    # Not a pytest test class, despite the name; pytest collects `Test*` by
    # default and the repository's unit run turns that warning into an error.
    __test__ = False

    @classmethod
    def from_dict(cls, d: dict) -> "TestFile":
        return cls(path=d["path"], body=d["body"])


# A test file embedded in `commands[]` as a `cat`/`tee` heredoc, matching what
# `surface_inference._heredoc_writes()` accepts. Shared here so the composer
# and the classifier agree on whether a model wrote a test file.
_HEREDOC_RE = re.compile(
    r"(?:cat|tee)\s*>\s*(?P<path>\S+)\s*<<\s*'?(?P<delim>\w+)'?\n(?P<body>.*?)\n(?P=delim)",
    re.DOTALL,
)


def runs_under_pytest_after(path: str, commands: list[str]) -> bool:
    """Whether a command after the one that writes `path` as a heredoc runs
    pytest on it: names its file, or is a bare `pytest`."""
    name = path.rsplit("/", 1)[-1]
    after = False
    for command in commands:
        if not after:
            after = path in command and "<<" in command
            continue
        if "pytest" in command and (name in command or not command.split("pytest", 1)[1].strip()):
            return True
    return False


def _test_heredoc(commands: list[str]) -> "tuple[int, re.Match[str]] | None":
    """The heredoc `embedded_test_file()` reads, with its command's index.

    The first one whose path is a `.py` file that a later command passes to
    pytest, else the first one. An answer can write fake hook tools as
    `cat > bin/relation-ids << 'EOF'` heredocs before the test, and taking
    the first heredoc would then check a shell script."""
    found = [(i, m) for i, command in enumerate(commands) for m in _HEREDOC_RE.finditer(command)]
    for i, match in found:
        path = match.group("path")
        if path.endswith(".py") and runs_under_pytest_after(path, commands):
            return i, match
    return found[0] if found else None


def embedded_test_file(commands: list[str]) -> "TestFile | None":
    """A test file written by a heredoc already present in `commands[]`, or
    `None` if no command matches. See `_HEREDOC_RE`'s comment above, and
    `_test_heredoc()` for which heredoc it is when there are several."""
    found = _test_heredoc(commands)
    if found is None:
        return None
    _, match = found
    return TestFile(path=match.group("path"), body=match.group("body"))


def replace_embedded_test_body(commands: list[str], body: str) -> list[str]:
    """`commands` with the body of the heredoc `embedded_test_file()` reads
    replaced by `body`; unchanged if there is no such heredoc."""
    out = list(commands)
    found = _test_heredoc(out)
    if found is not None:
        i, match = found
        out[i] = out[i][: match.start("body")] + body + out[i][match.end("body") :]
    return out


@dataclass
class StaticCheckResult:
    """`static_test_check.check()`'s verdict on an embedded test file.

    `reasons` is empty exactly when `passed` is true. Each reason is a short,
    factual sentence, written to be fed back to the model as it stands when
    the extraction is re-asked, and recorded in the extraction record.
    """

    path: str | None
    passed: bool
    reasons: list[str] = field(default_factory=list)
    # Import lines the harness added before checking, for names the file
    # used and never imported (`static_test_check.add_missing_imports()`).
    added_imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "added_imports": list(self.added_imports),
        }


@dataclass
class SurfaceInference:
    """The surface inference pass's output.

    Consumed by `scaffold.py` (relation/storage/pebble_service/ops_api_surface,
    matching scratch-charm/charm/render.py's params.yaml shape) plus
    `expected_signal`, which the classifier's *positive-signal-absent* rung
    needs (the `#2639` shape) and which has no home
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
    # The step this command belongs to ("prepare", "pack", "deploy", ...), so
    # the classifier can tell a failed prerequisite from a failed reproduction.
    # `None` when the commands come from `hypothesis.commands`.
    step: str | None = None
    # Wall-clock seconds this command took, so an end-to-end run time can
    # be attributed across pack/deploy/wait. `None` for replayed fixtures,
    # which have no timing to report.
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
    """Captured output of the reproduction runner."""

    hypothesis_number: int
    branch: str  # "none" | "k8s-scratch" | "lxd-scratch" | "k8s-clone"
    commands: list[CommandResult]
    control: CommandResult | None = None
    # Set when a prerequisite step failed and the rest of the sequence was
    # skipped, so the classifier doesn't score commands run against a substrate
    # that never came up.
    aborted_at_step: str | None = None
    skipped_steps: list[str] = field(default_factory=list)
    # The juju version this run actually used, which the verdict can depend on
    # and which can differ from what the extraction pinned (concierge's default
    # channel decides it). `None` for the `none` branch and older fixtures.
    observed_juju_version: str | None = None
    # The `ops` version the reproduction was actually built against, read from
    # the pack or install log. Both branches resolve `ops` from PyPI rather than
    # testing the checked-out tree, and `ops~=3.8` is a range, so the comment
    # has to say which version ran. `None` for `k8s-clone`, when no log carries
    # a parseable version, and for older fixtures.
    observed_ops_version: str | None = None

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
            observed_ops_version=d.get("observed_ops_version"),
        )


class Outcome(str, Enum):
    """The classifier's rung ladder, plus the stale gate's `skipped_stale`."""

    SKIPPED_STALE = "skipped_stale"
    # The bug is in how ops handles a hook tool's output and the issue doesn't
    # say what the tool returned, so only a real Juju would show it.
    SKIPPED_NEEDS_JUJU = "skipped_needs_juju"
    REPRODUCED = "reproduced"
    REPRODUCED_WEAKER = "reproduced_weaker"
    REPRODUCED_LOG_ONLY = "reproduced_log_only"
    REPRODUCED_POSITIVE_SIGNAL_ABSENT = "reproduced_positive_signal_absent"
    PARTIAL = "partial"
    DID_NOT_REPRODUCE = "did_not_reproduce"
    UNRUNNABLE_TEST_SELECTOR_STALE = "unrunnable_test_selector_stale"
    UNRUNNABLE_API_SHAPE_MISMATCH = "unrunnable_api_shape_mismatch"
    # commands[] that aren't shell (prose, bare Python, test names). Caught
    # before the runner, because the resulting exit 127 would otherwise be read
    # as `reproduced_weaker`.
    UNRUNNABLE_COMMANDS_NOT_SHELL = "unrunnable_commands_not_shell"
    # Synthesis fell back to the loud-failing stub, which has no real
    # assertion, so its exit code is not evidence about the bug.
    UNRUNNABLE_SYNTHESIS_INCOMPLETE = "unrunnable_synthesis_incomplete"
    # A prerequisite step (prepare, pack, deploy) failed, so nothing was tested
    # and the run is not evidence about the bug.
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    # A scratch-charm run gets a stimulus only when surface inference provides
    # both a pebble user and a command; without one the run is not evidence
    # either way.
    UNRUNNABLE_NO_STIMULUS = "unrunnable_no_stimulus"
    # A model-written test raised something other than an assertion failure, so
    # it is not valid test code and its failure says nothing about the bug.
    # Distinct from UNRUNNABLE_SYNTHESIS_INCOMPLETE, the fallback stub.
    UNRUNNABLE_SYNTHESIS_INVALID = "unrunnable_synthesis_invalid"


# Outcomes the composer writes a comment for. Every other outcome is
# silent: nothing useful produced, so stay silent.
COMMENT_OUTCOMES = frozenset(
    {
        Outcome.REPRODUCED,
        Outcome.REPRODUCED_WEAKER,
        Outcome.REPRODUCED_LOG_ONLY,
        Outcome.REPRODUCED_POSITIVE_SIGNAL_ABSENT,
        Outcome.PARTIAL,
    }
)
