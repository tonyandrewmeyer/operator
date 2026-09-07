"""Hypothesis extraction (PLAN.md Approach §3).

One LLM call per in-scope issue, structured output validated against the
`moving_parts` schema (extended with the `ci_run_url` / `symbol_anchor`
delta from `spike-step-4/FINDINGS.md`). Invalid output falls back to
stay-silent, per Approach §3.
"""

from __future__ import annotations

from filter_stage import extract_ci_run_url
from models import Hypothesis, Issue
from seams.llm import LLMSeam

CONFIDENCE_LEVELS = {"high", "medium", "low"}
SUBSTRATES = {"lxd", "k8s", "none"}

# Bounds prompt size on long comment threads (PLAN.md Approach §3 delta,
# `spike-step-5/2185/RESULT.md`). Configurable via
# `Extractor(max_comment_chars=...)`; this default comfortably fits a
# handful of ordinary review-style comments (the real #2045 follow-up is
# ~350 characters) while capping the pathological long-thread case.
DEFAULT_MAX_COMMENT_CHARS = 4000

_SCHEMA_INSTRUCTIONS = """\
Given the issue below, extract a reproduction hypothesis as JSON matching
exactly this shape (PLAN.md Approach §3):

{
  "in_scope": bool,
  "moving_parts": {
    "repo_version": str|null,
    "juju_version": str|null,
    "base": str|null,
    "substrate": "lxd"|"k8s"|"none",
    "other": {str: str}
  },
  "commands": [str],
  "expected": str,
  "observed": str,
  "confidence": "high"|"medium"|"low"
}

moving_parts.substrate is REQUIRED whenever in_scope is true, and must be
exactly one of these three strings -- never null, never omitted. It decides
which machinery gets provisioned to run the reproduction, so a wrong or
absent value is expensive:

- "none"  -- reproduces on the host in a plain directory: a unit test, an
             ops[testing]/Scenario test, an import-time error. No Juju
             controller, no cluster, no charm deployment. This is the most
             common case; prefer it whenever the bug is reachable without
             deploying anything.
- "k8s"   -- needs a real Kubernetes substrate and a deployed charm
             (Pebble behaviour, container/workload interaction, k8s-only
             Juju surface).
- "lxd"   -- needs a real machine substrate and a deployed charm.

If in_scope is true and you genuinely cannot tell which of the three
applies, pick the most likely one and additionally set
moving_parts.other["substrate_uncertain"] = "yes". Do not omit the field
and do not send null: an absent substrate is treated as a failed
extraction and the issue is dropped without comment.

"commands" is a list of POSIX shell commands, run in order, non-interactively,
by `bash -c` in a fresh empty directory. They are executed literally. The
directory has python3, uv, git and the usual shell utilities; for
substrate lxd/k8s it also has juju, charmcraft and concierge.

Each entry MUST be a command a shell can run. Specifically:

- Every entry must start with an executable or a shell builtin.
- The sequence must be self-contained: create every file it references
  (with a heredoc), and install every dependency it imports. It starts in
  an EMPTY directory -- there is no checkout, no test suite, no conftest,
  and no file from the issue lying around.
- Substitute real values. Never leave a placeholder like <charm-name> or
  <unit>: the shell reads angle brackets as redirects and the command dies.

Each entry MUST NOT be any of these (all of them exit non-zero and get
misread as evidence of the bug):

- An instruction to a human: "Run a charm test that checks ...", "Run the
  provided test code and observe the warning".
- A bare test name or selector: "test_relation_broken".
- A fragment of Python: "rel = self.model.get_relation('db', 1)".
- A reference to code you have not written into a file in an earlier entry.

Worked example. For an issue reporting that a hypothetical
`ops.model.Model.get_widget()` returns None instead of raising when the
widget is absent, on `substrate: none`:

[
  "uv init --bare .",
  "uv add 'ops[testing]'",
  "cat > test_widget.py << 'PYEOF'\nimport pytest\nimport ops\nfrom ops import testing\n\nclass MyCharm(ops.CharmBase):\n    pass\n\ndef test_absent_widget_raises():\n    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})\n    with ctx(ctx.on.update_status(), testing.State()) as mgr:\n        with pytest.raises(KeyError):\n            mgr.charm.model.get_widget('nope')\nPYEOF",
  "uv run pytest test_widget.py -v"
]

Note what that does: sets up a project, installs the dependency, writes the
test with a heredoc, then runs it. If you cannot produce a sequence like
that from the issue -- because it does not name a concrete API, error, or
snippet to build one from -- return an empty commands list and set
confidence to "low" rather than writing prose. An empty list is handled
(the harness may synthesise a test); prose is not.

"confidence" is about the reproduction hypothesis as a whole -- whether
these commands, run on this substrate, would actually show the reported
failure. It is not a measure of how sure you are about the substrate field
specifically, so do not raise it just because the substrate is obvious.
Use "low" when the issue does not really give you enough to reproduce
from (no concrete charm, no error text, no version), "medium" when you
have a plausible recipe with gaps you had to fill in, and "high" only
when the issue hands you a self-contained reproduction you have
transcribed rather than reconstructed.

"in_scope" is a second opinion on a deterministic pre-filter that already
dropped the obvious non-bugs. Set it true ONLY if both of these hold:

1. Someone describes a concrete symptom: an error, a traceback, a crash, a
   hang, a wrong or missing value -- something that happened, to them or in
   their CI, and that contradicts what the software promises (its docs, its
   type hints, its own error messages, or plain expectation).
2. The issue does not already contain a COMPLETE, RUNNABLE reproducer.

On (1): a reporter who also proposes a fix is still reporting a bug. "This
hangs, and I think the timeout handling is wrong" is a bug report. Do not
drop a concrete symptom just because a suggested fix is attached to it.

On (2): read this narrowly. It excludes only issues carrying a
self-contained recipe someone could run as-is. Partial information -- a CI
link, a traceback with no setup, a snippet that assumes a charm you don't
have -- is NOT a complete reproducer, and those issues are exactly the ones
worth reproducing. Do not reason "there is already some detail here, so a
reproduction adds nothing".

Set it false otherwise. In particular, false for all of these, even when
they show code or describe current behaviour:

- A request for behaviour the software never promised -- stricter
  validation, a new error, a nicer API: "X should fail if Y", "it would be
  better if Z", "should we warn about W?". The distinguishing test is
  whether the current behaviour breaks a promise (bug) or merely falls
  short of what the reporter would prefer (feature request). Showing a
  passing test to demonstrate today's behaviour does not make it a bug.
- A proposal to change the codebase: "X can be removed", "we should
  refactor Y". Someone reading the source and reasoning about it --
  typically citing a source permalink -- is not someone who hit a bug, and
  there is nothing to reproduce beyond re-asserting what the code plainly
  does.
- A design discussion or an open question about what the right behaviour
  would be, even a concrete one.
- A question about how to use the library.
- A task, chore, epic, tracking issue, or CI/infrastructure note.

When it is genuinely ambiguous -- when you cannot tell whether the reporter
hit a broken promise or is asking for something better -- set in_scope
false. A wrongly dropped issue costs little: a human triages it as they
would have anyway. A wrongly kept one spends a maintainer's attention on a
comment that should never have been posted. Those costs are not symmetric,
so do not treat this as a balanced judgement call.

But note what that tie-break is *for*. It applies to the bug-versus-feature
judgement, not to the presence of a symptom. An issue that plainly states
something failed is not ambiguous, and this rule is not a reason to drop
it.

Do not set moving_parts.ci_run_url -- that's filled in deterministically
before this call, not extracted by you (it's a literal URL in the issue
body; no interpretation needed).

If the issue body links to a source-line permalink (e.g. a
github.com/canonical/<repo>/blob/<sha>/<path>#L<n> URL), also set
moving_parts.symbol_anchor to the dotted symbol the linked line belongs to
(e.g. "ops.model.Model.get_relation") -- unlike ci_run_url this needs
interpreting the linked source, so it's part of your extraction, not a
pre-extraction step.

If comments are listed below the body, read them too: a comment sometimes
redirects or narrows the scope the body describes (for example, a body
that says "EDIT: see the first reply for the new scope of this issue")
rather than merely adding detail. When that happens, extract the current
(comment-redirected) scope, not the stale body text alone.
"""


class ExtractionInvalid(ValueError):
    """Raised when the LLM's output doesn't match the extraction schema.

    Approach §3: "Validate against the schema; on invalid output, fall
    back to stay-silent."
    """


def _require(d: dict, key: str, types: tuple) -> None:
    if key not in d:
        raise ExtractionInvalid(f"missing key {key!r}")
    if not isinstance(d[key], types):
        raise ExtractionInvalid(f"{key!r} must be {types}, got {type(d[key])}")


def validate(raw: dict) -> None:
    _require(raw, "in_scope", (bool,))
    _require(raw, "moving_parts", (dict,))
    _require(raw, "commands", (list,))
    # `expected`/`observed` describe a defect, so they only mean anything for
    # an in-scope extraction. Requiring them unconditionally turned two clean
    # drops into `ExtractionInvalid` on the corpus-v2 run (#1449, #1492: the
    # model returned null for both, reasonably, having just judged the issue
    # not to be a bug). Same shape as the `substrate` defect in
    # `spike-step-5/live-llm/RESULT.md` Finding 1: a field demanded of every
    # extraction that is only defined for some of them. The outcome was
    # stay-silent either way, but a drop recorded as a validation failure
    # pollutes the metrics and would read as an error in production logs.
    if raw["in_scope"]:
        _require(raw, "expected", (str,))
        _require(raw, "observed", (str,))
    else:
        for key in ("expected", "observed"):
            if raw.get(key) is not None and not isinstance(raw[key], str):
                raise ExtractionInvalid(f"{key!r} must be a string or null, got {type(raw[key])}")
    _require(raw, "confidence", (str,))
    if raw["confidence"] not in CONFIDENCE_LEVELS:
        raise ExtractionInvalid(f"confidence must be one of {CONFIDENCE_LEVELS}, got {raw['confidence']!r}")
    for i, cmd in enumerate(raw["commands"]):
        if not isinstance(cmd, str):
            raise ExtractionInvalid(f"commands[{i}] must be a string")
    mp = raw["moving_parts"]
    substrate = mp.get("substrate")
    if raw["in_scope"]:
        # Required, and never null, for anything that might reach a runner.
        # `choose_branch` dispatches the provisioning decision off this field,
        # so an absent value used to fall through to the most expensive branch
        # (`k8s-scratch`) by accident -- see `spike-step-5/live-llm/RESULT.md`
        # Finding 1, where a live model returned null for three host-only
        # hypotheses and for #2639, the one hypothesis that has ever
        # reproduced. Guessing either way is wrong: "none" strands a k8s bug
        # on the host and reports a false non-reproduction, while defaulting
        # to k8s buys a cluster for a unit test. Approach §3's stay-silent
        # fallback is the honest answer, and matches the project's
        # bias-toward-false-drop (Steps §1).
        if substrate not in SUBSTRATES:
            raise ExtractionInvalid(
                f"moving_parts.substrate must be one of {sorted(SUBSTRATES)} when "
                f"in_scope is true, got {substrate!r}"
            )
    elif substrate is not None and substrate not in SUBSTRATES:
        # Dropped issues never reach a runner, so null is fine here (it's what
        # the hand extraction for #2304 carries); a *wrong* value still isn't.
        raise ExtractionInvalid(f"moving_parts.substrate invalid: {substrate!r}")


def _format_comments(issue: Issue, max_chars: int) -> str:
    """Render `issue.comments` for the extraction prompt.

    Chronological order, each attributed by author and flagged
    reporter-vs-other (PLAN.md Approach §3 delta, `spike-step-5/2185/
    RESULT.md`'s reporter-redirect case vs `spike-step-5/2045/RESULT.md`'s
    third-party-follow-up case -- both matter, differently, so the label is
    surfaced rather than left for the model to infer from prose alone).

    Capped at `max_chars` total characters across all but the first comment,
    to bound prompt size on long threads. The first comment is always
    included (truncated in place if it alone exceeds the cap) because it is
    the highest-value one for the exact "EDIT: see the first reply" shape
    this cap exists to protect against silently dropping. Any further
    omission is a visible marker line, never silent truncation.
    """
    if not issue.comments:
        return ""
    lines = [
        "",
        "Comments, in chronological order (a later comment may redirect or "
        "narrow the scope described in the body above):",
    ]
    used_chars = 0
    for i, comment in enumerate(issue.comments):
        who = "issue author" if comment.author == issue.author else "other"
        block = f"[{i + 1}] {comment.author} ({who}), {comment.created_at}:\n{comment.body}"
        if i == 0:
            if len(block) > max_chars:
                block = block[: max_chars - 3] + "..."
            lines.append(block)
            used_chars += len(block)
            continue
        if used_chars + len(block) > max_chars:
            omitted = len(issue.comments) - i
            lines.append(
                f"[... {omitted} further comment(s) omitted, over the "
                f"{max_chars}-character comment cap ...]"
            )
            break
        lines.append(block)
        used_chars += len(block)
    return "\n\n".join(lines)


class Extractor:
    def __init__(self, llm: LLMSeam, *, max_comment_chars: int = DEFAULT_MAX_COMMENT_CHARS):
        self.llm = llm
        self.max_comment_chars = max_comment_chars

    def extract(self, issue: Issue) -> Hypothesis:
        prompt = self._build_prompt(issue, self.max_comment_chars)
        raw = self.llm.complete_json(
            purpose="extraction", prompt=prompt, context={"issue_number": issue.number}
        )
        validate(raw)
        hypothesis = Hypothesis.from_dict(issue.number, raw)
        # Deterministic pre-extraction (Approach §3 delta): ci_run_url is
        # regex-extractable at filter time, not an LLM output.
        hypothesis.moving_parts.ci_run_url = extract_ci_run_url(issue.body)
        return hypothesis

    @staticmethod
    def _build_prompt(issue: Issue, max_comment_chars: int) -> str:
        return (
            f"{_SCHEMA_INSTRUCTIONS}\n"
            f"Repo: {issue.repo}\n"
            f"Title: {issue.title}\n"
            f"Labels: {', '.join(issue.labels)}\n"
            f"Body:\n{issue.body}\n"
            f"{_format_comments(issue, max_comment_chars)}"
        )
