"""In-scope filter (PLAN.md Approach §2) plus the `ci_run_url` pre-extraction
(Approach §3 delta: "trivially regex-extractable at filter time... before
any LLM call").

The classification logic here is ported verbatim from
`spike-step-1/filter.py` / `spike-step-2/filter.py` (identical between the
two spikes) — deterministic, no LLM call.
"""

from __future__ import annotations

import re

from models import Issue

DROP_ISSUE_TYPES = {"Feature", "Task", "Question", "Documentation", "Epic", "Tracking"}
_DROP_ISSUE_TYPES_LOWER = {t.lower() for t in DROP_ISSUE_TYPES}

DROP_LABELS = {
    "docs", "documentation", "question", "discussion", "epic", "tracking",
    "enhancement", "feature", "feature-request", "meta", "chore",
    # spike-step-5/corpus-v2/RESULT.md "Two filter findings": `operator`'s
    # real label vocabulary barely overlaps the set above (only `docs`
    # matched across 400 issues). `refactoring` is added on that finding's
    # evidence -- every corpus-v2 issue carrying it is in `hand-labels.json`'s
    # `screened_only` list, i.e. accepted as a correct (non-bug) drop.
    "refactoring",
}

# `needs design` and `roadmap` were also named by the corpus-v2 finding as
# label-vocabulary candidates, but are deliberately NOT in DROP_LABELS:
# `hand-labels.json` records `#1109` as a real bug (`label: true`) carrying
# `roadmap` alone, and `fixtures/issues/2639.json` -- the one hypothesis this
# whole project has ever confirmed reproduces end-to-end
# (`spike-step-4/2639/RESULT.md`) -- carries *both* `needs design` and
# `roadmap`. Both labels mark a triage/planning state, not bug-vs-not-bug;
# adding either would false-drop a hand-verified real bug and the project's
# one validated success case. `next release` is excluded for the same reason:
# `#1775` is a hand-labelled real bug (`ops fails when a charm has no config
# options`) carrying only that label -- it marks scheduling, not scope.

TITLE_DROP_PATTERNS = [
    (re.compile(r"^\s*RFC[:\s]", re.I), "title RFC marker"),
    (re.compile(r"\[epic\]", re.I), "title [epic] marker"),
    (re.compile(r"^\s*feature\s*request", re.I), "title feature-request marker"),
    (re.compile(r"^\s*docs?:", re.I), "title docs: marker"),
    (re.compile(r"^\s*chore:", re.I), "title chore: marker"),
    (re.compile(r"^\s*(?:tracking|epic|meta):", re.I), "title tracking/epic/meta marker"),
    (re.compile(r"^Scheduled workflow ", re.I), "title = ai-failure-notifications output"),
]

FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
STEPS_HEADING_RE = re.compile(
    r"^\s*#{1,6}\s*(?:steps to reproduce|how to reproduce|reproduction steps|reproducer)\b",
    re.I | re.M,
)
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S", re.M)

RUNNABLE_MARKERS = re.compile(
    r"(\$ |> |# |juju\s+(?:deploy|bootstrap|add-|status|run)|pytest\b|uv\s+run\b|"
    r"charmcraft\s+\w+|python[0-9]*\s+-m\b|import\s+\w|Traceback\s*\(most\s+recent)",
    re.I,
)

# Approach §3 delta: github.com/canonical/<repo>/actions/runs/<id> --
# load-bearing for the third-branch trigger heuristic and the commit-sha
# fallback for stale test selectors (spike-step-4/2484/RESULT.md).
CI_RUN_URL_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+/actions/runs/(\d+)")

# v1 target list, PLAN.md "Where it lives" > "Target repos, v1".
V1_ALLOWLIST = {"canonical/operator", "canonical/charmlibs", "canonical/charm-ubuntu"}


def body_has_reproducer(body: str) -> tuple[bool, str]:
    """Return (has_repro, reason)."""
    if not body:
        return False, ""
    for _lang, content in FENCE_RE.findall(body):
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if len(lines) >= 3 and RUNNABLE_MARKERS.search(content):
            return True, "fenced block looks runnable"
    m = STEPS_HEADING_RE.search(body)
    if m:
        after = body[m.end():]
        section_end = STEPS_HEADING_RE.search(after)
        section = after[: section_end.start()] if section_end else after
        items = LIST_ITEM_RE.findall(section)
        if len(items) >= 3:
            return True, "steps-to-reproduce heading + 3+ list items"
    return False, ""


def title_drop_reason(title: str) -> str | None:
    for pat, reason in TITLE_DROP_PATTERNS:
        if pat.search(title):
            return reason
    return None


def extract_ci_run_url(body: str) -> str | None:
    m = CI_RUN_URL_RE.search(body or "")
    return m.group(0) if m else None


def classify_issue(issue: Issue, issue_type: str | None = None) -> tuple[str, str]:
    """Return (KEEP|DROP, reason).

    `issue_type` defaults to `issue.issue_type` (the webhook's
    `issue.type.name`) when not given explicitly, so callers that only have
    an `Issue` -- which is every real caller, per PLAN.md Approach §2 gap 2 --
    still get the issue-type drop rule instead of it silently never firing.
    """
    if issue.repo not in V1_ALLOWLIST:
        return "DROP", f"repo={issue.repo} not in v1 allowlist"
    effective_type = issue_type if issue_type is not None else issue.issue_type
    if effective_type and effective_type.lower() in _DROP_ISSUE_TYPES_LOWER:
        return "DROP", f"issue-type={effective_type}"
    label_names = {label.lower() for label in issue.labels}
    hit = label_names & DROP_LABELS
    if hit:
        return "DROP", f"label={','.join(sorted(hit))}"
    tdrop = title_drop_reason(issue.title)
    if tdrop:
        return "DROP", tdrop
    has_repro, why = body_has_reproducer(issue.body)
    if has_repro:
        return "DROP", f"already has reproducer ({why})"
    return "KEEP", "bug-shaped, no existing reproducer"
