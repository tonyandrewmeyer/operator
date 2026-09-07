"""Reproduction runner orchestration (PLAN.md Approach §4).

Three deterministic pieces, none of which need an LLM call:

- the skip-when-almost-certainly-stale gate (two triggers, both yield
  `skipped_stale`, never reaching the runner seam at all);
- the third-branch trigger heuristic (route to the clone-repo branch iff
  `ci_run_url` is set and there's no self-contained repro snippet);
- dispatch to the runner seam for whichever branch was chosen.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import runnability
from models import Hypothesis, Issue, RunResult, SurfaceInference
from seams.llm import LLMSeam
from seams.runner import RunnerSeam
from surface_inference import TestFileSynthesizer, is_pytest_invocation, needs_test_file

_SNIPPET_MARKERS = ("import ops", "import scenario")


def has_self_contained_snippet(commands: list[str]) -> bool:
    """A self-contained Python repro snippet in `commands` (the #2327/#2341/
    #2639 shape) vs. no such snippet, only a CI-run reference (#2484's
    shape) -- spike-step-4/2484/RESULT.md's trigger-heuristic table."""
    joined = "\n".join(commands)
    return any(marker in joined for marker in _SNIPPET_MARKERS)


def is_stale(hypothesis: Hypothesis, issue: Issue, runner_seam: RunnerSeam, context: dict) -> tuple[bool, str | None]:
    """Approach §4's skip-when-almost-certainly-stale gate.

    Both triggers yield the same `skipped_stale` verdict -- distinct from
    `did_not_reproduce` so Approach §6 never composes a "could not
    reproduce" comment for either case.
    """
    mp = hypothesis.moving_parts
    if mp.repo_version is None and issue.state == "CLOSED":
        return True, "repo_version is null and issue is CLOSED (spike-step-4/2341, /2327 -- 2/2 on this pattern)"
    if mp.symbol_anchor and not runner_seam.resolve_symbol(mp.symbol_anchor, context):
        return (
            True,
            f"symbol_anchor {mp.symbol_anchor!r} does not resolve against target main "
            "(extraction is pinned to an API surface that no longer exists)",
        )
    return False, None


def choose_branch(hypothesis: Hypothesis) -> str:
    """Approach §4's substrate/third-branch dispatch.

    Dispatch is explicit on all three substrates: an unrecognised or absent
    value raises rather than falling through. It used to fall through to
    `k8s-scratch`, which meant a `substrate: null` extraction silently
    provisioned a cluster it had no use for (`spike-step-5/live-llm/RESULT.md`
    Finding 1). `extraction.validate()` now rejects such extractions up
    front, so reaching this raise means a `Hypothesis` was built by some
    path that bypassed validation.
    """
    mp = hypothesis.moving_parts
    if mp.substrate == "none":
        return "none"
    if mp.substrate not in ("lxd", "k8s"):
        raise ValueError(
            f"cannot choose a runner branch for substrate {mp.substrate!r} "
            "(expected 'none', 'lxd' or 'k8s'); the extraction should have "
            "been rejected by extraction.validate()"
        )
    if mp.ci_run_url and not has_self_contained_snippet(hypothesis.commands):
        return "k8s-clone"
    if mp.substrate == "lxd":
        return "lxd-scratch"
    return "k8s-scratch"


# Branches whose runner actually executes `hypothesis.commands` through a
# shell, and which therefore need the runnability gate. `k8s-scratch` and
# `lxd-scratch` are excluded deliberately: `seams/runner.py`'s
# `_run_k8s_scratch`/`_run_lxd_scratch` synthesise their own sequence
# (`concierge prepare` -> `charmcraft pack` -> deploy -> stimulus) from the
# rendered scaffold and `surface.pebble_service`, so `commands[]` is not the
# recipe there. Note the latent issue that leaves behind: #2639's hand
# extraction carries unfilled `<k8s-charm-...>`/`<unit>` placeholders, which
# a shell reads as redirects -- if either scratch branch ever does start
# running `commands[]`, they need templating against the rendered charm
# first.
COMMAND_EXECUTING_BRANCHES = frozenset({"none", "k8s-clone"})


@dataclasses.dataclass
class RunnerStageResult:
    branch: str | None
    skipped_stale: bool
    skip_reason: str | None
    run_result: RunResult | None
    unrunnable_reason: str | None = None


def check_runnable(hypothesis: Hypothesis, branch: str) -> tuple[bool, str | None]:
    """Approach §4 pre-run gate (`spike-step-5/live-llm/RESULT.md` Finding 2).

    Returns `(ok, reason)`. Called after test-file synthesis, which can
    legitimately repair an otherwise un-runnable `commands[]`, and only for
    branches that actually shell out to `commands[]`.
    """
    if branch not in COMMAND_EXECUTING_BRANCHES:
        return True, None
    report = runnability.assess(hypothesis.commands)
    return report.runnable, None if report.runnable else report.reason


# Branches that build their command sequence from `surface`, not from
# `hypothesis.commands`, and whose whole point is the stimulus.
SCRATCH_BRANCHES = frozenset({"k8s-scratch", "lxd-scratch"})

# Does a setup command actually establish a Python environment to run in?
# `uv init`/`uv venv` create one; `uv add`/`uv pip install` populate one.
# Anything else (a `cd`, a `git clone`, an `export`) does not.
_ESTABLISHES_ENVIRONMENT_RE = re.compile(r"\buv\s+(init|venv|add|pip\s+install)\b|\bpip\s+install\b")


def check_stimulus(branch: str, surface: SurfaceInference | None) -> tuple[bool, str | None]:
    """Pre-run gate: a scratch-charm branch with no stimulus to run.

    `seams/runner.py`'s `_scratch_sequence()` builds the stimulus only when
    surface inference hands over both a pebble `user` and a pebble
    `command`, and the control only when there is a stimulus to control
    against. When either is missing it silently built neither -- and the
    run went on to `concierge prepare`, `charmcraft pack` and `juju deploy`
    (13+ minutes on the first real run), poked the deployed charm with
    nothing at all, then handed the classifier a run whose "expected signal
    absent" is a statement about an experiment that was never performed.

    Live surface inference returns `pebble_service.command: null` for
    `#2639` repeatably (3/3 runs, 2026-08-18), so this is the common case
    rather than a corner. Gated here, next to `check_runnable`, so it costs
    nothing instead of costing a substrate bootstrap.

    Keyed on `expected_signal`, matching the rung it protects: only
    `classifier.py`'s positive-signal-absent rung reasons from a signal's
    absence, and it fires only when `expected_signal` is set. A hypothesis
    with no expected signal is a deploy-only experiment -- `#2107`, a
    machine charm that errors during update-status with `pebble_service:
    {}`, is the real corpus case -- and gating that would reject a run
    whose stimulus *is* the deploy.
    """
    if branch not in SCRATCH_BRANCHES:
        return True, None
    if not (surface and surface.expected_signal):
        return True, None
    pebble = dict(surface.pebble_service) if surface else {}
    missing = [field for field in ("user", "command") if not pebble.get(field)]
    if not missing:
        return True, None
    return False, (
        f"surface inference promised the signal {surface.expected_signal!r} but gave no "
        f"pebble {' or '.join(missing)}, so the {branch} sequence would deploy a charm, "
        "stimulate nothing, and then check whether the signal appeared"
    )


def write_synthesized_test_file_if_needed(
    hypothesis: Hypothesis, issue: Issue, llm: LLMSeam, context: dict
) -> Hypothesis:
    """Approach §3/§4 delta (spike-step-5/2045/RESULT.md "PLAN deltas
    surfaced" §1): for `substrate: none` hypotheses with no runnable pytest
    invocation in `commands[]`, synthesise a test file, write it into the
    scratch working directory, and swap in the pytest invocation that runs
    it -- mirroring what step 4/5's by-hand walks did per-issue. Returns
    `hypothesis` unchanged when no synthesis is needed.

    `llm` is threaded through to `TestFileSynthesizer` (2026-07-30,
    `spike-step-5/composer-live/RESULT.md` Finding 6): synthesis is now an
    LLM call, not a fixed template, so this stage needs the same seam
    `Extractor`/`SurfaceInferrer`/`Composer` already use.

    Genuine setup commands (`uv venv`/`uv init`/`uv add`/`uv pip install`)
    are kept; comment-only placeholders and any pytest invocation are
    dropped before appending the real one -- #2045's actual extraction has
    both a `# write test_cwd.py: ...` comment *and* a `pytest test_cwd.py`
    invocation that would fail outright (no such file exists) if run
    as-is, since `test_cwd.py` was never anything but a comment.
    """
    if not needs_test_file(hypothesis):
        return hypothesis
    test_file = TestFileSynthesizer(llm).synthesize(issue, hypothesis)
    workdir = Path(context.get("workdir", "."))
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / test_file.path).write_text(test_file.body)
    # Keep only genuinely executable setup; drop comment placeholders, the
    # broken pytest invocation being replaced, and -- per
    # `spike-step-5/live-llm/RESULT.md` Finding 2 -- prose and bare code
    # fragments. Prose used to survive this filter as a "setup command" and
    # then exit 127 mid-sequence, which the classifier's rung 7 reads as
    # `partial`: another comment-worthy outcome from a command that never ran.
    # 2026-07-30 live measurement (spike-step-5/synthesis-live/RESULT.md): a
    # bare `"pytest" not in c` substring check here used to drop
    # `uv add 'ops[testing]' pytest` -- a legitimate setup command, not the
    # broken invocation this filter exists to remove -- silently stripping
    # pytest out of the venv before the appended, synthesized-test
    # invocation ever ran. `is_pytest_invocation()` only matches an actual
    # pytest invocation (bare or `uv run`), not "pytest" as a trailing
    # package name.
    setup_commands = [
        c
        for c in hypothesis.commands
        if not is_pytest_invocation(c) and runnability.classify_command(c) is runnability.Shape.SHELL
    ]
    # ...and if the extraction supplied no environment setup at all, build
    # one. This whole stage assumed `commands[]` already carried the
    # `uv init` -> `uv add` prologue, which is true of the corpus-v2
    # extraction shape but not of a live extraction that returns
    # `commands: []` -- the real 2026-08-18 case. With nothing prepended,
    # `uv run pytest` finds no project in the scratch dir, walks *up* to the
    # harness's own pyproject, runs against the harness venv (which has no
    # `ops`), and dies with ModuleNotFoundError at collection. A synthesised
    # test is always an `ops.testing` scaffold, so the dependency set is
    # known rather than guessed.
    if not any(_ESTABLISHES_ENVIRONMENT_RE.search(c) for c in setup_commands):
        setup_commands = [*setup_commands, "uv init --bare", "uv add 'ops[testing]' pytest"]
    commands = [*setup_commands, f"uv run pytest {test_file.path} -v"]
    return dataclasses.replace(hypothesis, commands=commands, synthesized_test_file=test_file)


def run_hypothesis(
    hypothesis: Hypothesis,
    issue: Issue,
    surface: SurfaceInference | None,
    runner_seam: RunnerSeam,
    llm: LLMSeam,
    context: dict,
) -> RunnerStageResult:
    stale, reason = is_stale(hypothesis, issue, runner_seam, context)
    if stale:
        return RunnerStageResult(branch=None, skipped_stale=True, skip_reason=reason, run_result=None)
    branch = choose_branch(hypothesis)
    if branch == "none":
        hypothesis = write_synthesized_test_file_if_needed(hypothesis, issue, llm, context)
    ok, unrunnable_reason = check_runnable(hypothesis, branch)
    if not ok:
        return RunnerStageResult(
            branch=branch,
            skipped_stale=False,
            skip_reason=None,
            run_result=None,
            unrunnable_reason=unrunnable_reason,
        )
    run_result = runner_seam.run(branch=branch, hypothesis=hypothesis, surface=surface, context=context)
    return RunnerStageResult(branch=branch, skipped_stale=False, skip_reason=None, run_result=run_result)
