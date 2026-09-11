"""A second, separately-prompted `in_scope` classification pass (PLAN.md
Approach §3 delta).

`spike-step-5/inscope-retune/RESULT.md` measured a real `#2639` false-drop
under the single `in_scope` criterion and found that three structurally
different rewordings of that one prompt all failed `spike-step-5/corpus-v2/`'s
precision/recall floor -- concluding the single promise-based criterion
"may not have enough resolution to carry the bug-vs-feature-request
distinction via prompt wording alone" and naming a rationale field or a
second pass as the more promising levers.

`spike-step-5/inscope-instrument/RESULT.md` did both, in order: a rationale
field first (diagnostic only, not shipped -- see that file), then this
module, designed from what the rationale data showed. Full method, the
corpus/flagship measurements, and the numbers behind the ship decision are
in that RESULT.md; summarised here:

For every issue the first pass (`extraction.Extractor`, completely
unmodified) drops, this module asks a second, independent call a narrower
question: does the CURRENT-BEHAVIOUR text describe something concretely,
presently broken, independent of how the EXPECTED-BEHAVIOUR text happens to
be phrased? If yes, it re-extracts the full hypothesis (not just a flipped
boolean) so a recovered issue still has `moving_parts`/`commands` to run.

The second pass runs on every drop, not only ones whose `observed` field
still names a symptom -- the rationale instrumentation showed the failure
mode is exactly the first pass talking itself out of writing one down (two
of three false drops in that run had `observed: ""`), so gating on that
same field would be circular. The extra calls (most of a corpus, not a
handful) are cheap at this project's established sub-cent-per-call cost.

Measured on `spike-step-5/corpus-v2/`, two independent runs: precision 100%
/ 100% (zero false keeps in either run -- the extra calls did not cost
precision), recall 11/12 = 91.7% / 11/12 = 91.7% (both runs land on the
exact same fraction the corpus-v2 refinement's own first confirmation run
reported as "92%", with the exact same sole miss, `#1088`, unchanged from
before this module existed -- a tie with the established baseline, not a
regression, and not a strict >=92% by raw floating-point comparison; see
RESULT.md for the full discussion of that nuance). On the flagship `#2639`
anecdote (n=8, not part of the labelled corpus): false-drop rate fell from
the same-day single-pass baseline's 75% to 62.5% with this module -- a
real, partial improvement, explicitly not a fix (`#2639` is still dropped
more often than kept).
"""

from __future__ import annotations

from extraction import Extractor, ExtractionInvalid, _format_comments, validate
from models import Hypothesis, Issue
from seams.llm import LLMSeam

_SECOND_PASS_INSTRUCTIONS = """\
A first automated pass just read the GitHub issue below and judged it OUT
of scope (not a bug worth reproducing). Its own notes on the issue were:

  observed (what it read as the current behaviour): "{observed}"
  expected (what it read as the reporter's expectation): "{expected}"

Those notes may be blank -- the first pass sometimes fully commits to "not
a bug" and writes nothing down, which is exactly why this second, narrower
look exists. Read the issue body and comments below yourself; do not treat
a blank note above as evidence of anything.

Answer ONLY this question: does the CURRENT-BEHAVIOUR the issue describes
amount to something concretely, presently broken -- an error, a hang, a
crash, a wrong or missing value, a documented/typed/promised behaviour that
silently fails -- independent of how any paired expected-behaviour text
happens to be phrased? A "should"-phrased expected-behaviour sentence is
not, on its own, evidence either way; some bug-report templates always
phrase it that way. Labels like `needs design` or `roadmap` are not
evidence either way -- this project has confirmed real bugs carrying both.

Worked contrast, since this is the exact distinction that matters here:

- BUG-shaped: "Current behavior: a workload running as a non-root user
  sends a Pebble notice, and the charm's observer never fires -- nothing in
  the log, no hook runs. Expected behavior: charms should be able to react
  to notices regardless of which user sent them." The *current* half names
  a mechanism (notice -> observer) that silently does not fire. That is a
  concrete defect, in scope, however the expected half is phrased and
  whatever labels are attached.
- FEATURE-shaped: "Currently, action output shows a Python
  DeprecationWarning. In my view users shouldn't see internal warnings in
  action output." Nothing is failing -- the software is doing exactly what
  it does today, and the reporter would prefer it did something else. Not
  in scope.

If you catch yourself about to answer "no" mainly because of the
expected-behaviour wording or a `needs design`/`roadmap` label, stop and
re-read only the current-behaviour sentences: is there a mechanism that
fails to fire, a value that comes back wrong, or an error that occurs,
right now, when the described action happens? Answer from that alone.

Answer "no" (not a concrete defect) for any of these, even if the current
behaviour text sounds specific:
- A factual description of what happens today, when nothing says it should
  happen differently -- an investigation, not a symptom.
- A request for behaviour the software never promised, a design opinion, or
  a proposal to change the code.
- A design discussion, an open question, or a how-to question.
- An issue that already contains a complete, runnable reproducer (nothing
  left to add by reproducing it again).

Return JSON matching exactly this shape:

{{
  "concrete_defect": bool,
  "reason": str,
  "moving_parts": {{
    "repo_version": str|null,
    "juju_version": str|null,
    "base": str|null,
    "substrate": "lxd"|"k8s"|"none"|null,
    "other": {{}}
  }},
  "commands": [str],
  "expected": str,
  "observed": str,
  "confidence": "high"|"medium"|"low"
}}

If concrete_defect is true, moving_parts/commands/expected/observed/confidence
follow the same rules the first pass used (substrate is REQUIRED and must be
one of "lxd"/"k8s"/"none"; commands are POSIX shell commands runnable as-is
in a fresh empty directory; use "low" confidence and an empty commands list
if you cannot build a concrete recipe from the issue). If concrete_defect is
false, set moving_parts.substrate to null and leave commands empty --
nothing downstream will read them.

Repo: {repo}
Title: {title}
Labels: {labels}
Body:
{body}
{comments}
"""


class SecondPassInvalid(ExtractionInvalid):
    """Raised when the second pass's output doesn't match its schema.

    Subclasses `ExtractionInvalid` (rather than a bare `ValueError`) so
    `pipeline.py`'s existing `except ExtractionInvalid` -> stay-silent
    fallback catches it without needing its own clause.
    """


def needs_second_pass(hyp: Hypothesis) -> bool:
    """The gate: every first-pass drop.

    Originally gated on the first pass's own `observed` field being
    non-trivially populated (reusing `spike-step-5/corpus-v2/analyse.py`'s
    `OBSERVED_MIN_CHARS = 25` / `dropped_with_symptom` screen). Dropped
    after `spike-step-5/inscope-instrument/RESULT.md`'s corpus-instrumented
    run showed that gate would have missed 2 of that run's 3 real false
    drops outright (`observed: ""` on both) -- the failure mode is exactly
    the first pass talking itself out of writing a symptom down, so gating
    the second pass on that same field is circular.
    """
    return not hyp.in_scope


def validate_second_pass(raw: dict) -> None:
    if "concrete_defect" not in raw or not isinstance(raw["concrete_defect"], bool):
        raise SecondPassInvalid(f"'concrete_defect' must be bool, got {raw.get('concrete_defect')!r}")
    if not raw["concrete_defect"]:
        return
    # Reuse extraction.py's own schema validation for the recovered
    # hypothesis rather than duplicating its rules.
    as_extraction = dict(raw)
    as_extraction["in_scope"] = True
    try:
        validate(as_extraction)
    except ExtractionInvalid as exc:
        raise SecondPassInvalid(str(exc)) from exc


def _build_second_pass_prompt(issue: Issue, hyp: Hypothesis, max_comment_chars: int) -> str:
    comments = _format_comments(issue, max_comment_chars)
    return _SECOND_PASS_INSTRUCTIONS.format(
        observed=hyp.observed,
        expected=hyp.expected,
        repo=issue.repo,
        title=issue.title,
        labels=", ".join(issue.labels),
        body=issue.body,
        comments=comments,
    )


class TwoPassExtractor:
    """`Extractor`, plus a second, narrower-question pass on every drop.

    Wraps rather than subclasses `Extractor` -- the first pass's prompt and
    validation are used completely unmodified (`self._first.extract()`), and
    this class only adds the second call and the override logic on top.
    Duck-types `Extractor`'s `max_comment_chars` attribute and
    `_build_prompt` staticmethod so it's a drop-in replacement wherever
    `Extractor` was used (`pipeline.py`, `run_live_extraction.py`).
    """

    def __init__(self, llm: LLMSeam, *, max_comment_chars: int | None = None):
        kwargs = {} if max_comment_chars is None else {"max_comment_chars": max_comment_chars}
        self._first = Extractor(llm, **kwargs)
        self.llm = llm
        self.last_second_pass: dict | None = None

    @property
    def max_comment_chars(self) -> int:
        return self._first.max_comment_chars

    @staticmethod
    def _build_prompt(issue, max_comment_chars):
        return Extractor._build_prompt(issue, max_comment_chars)

    def extract(self, issue: Issue) -> Hypothesis:
        hyp = self._first.extract(issue)
        self.last_second_pass = None
        if not needs_second_pass(hyp):
            return hyp
        prompt = _build_second_pass_prompt(issue, hyp, self._first.max_comment_chars)
        raw = self.llm.complete_json(
            purpose="inscope_second_pass", prompt=prompt, context={"issue_number": issue.number}
        )
        validate_second_pass(raw)
        self.last_second_pass = raw
        if not raw["concrete_defect"]:
            return hyp
        # Recovery: re-extract the full hypothesis from the second pass's
        # own output rather than merely flipping the boolean, so a recovered
        # issue actually has moving_parts/commands to run downstream.
        recovered = dict(raw)
        recovered["in_scope"] = True
        recovered_hyp = Hypothesis.from_dict(issue.number, recovered)
        # ci_run_url is deterministic (regex on the issue body), not model
        # output -- carry over what the first pass already computed rather
        # than leaving it null on recovery.
        recovered_hyp.moving_parts.ci_run_url = hyp.moving_parts.ci_run_url
        return recovered_hyp
