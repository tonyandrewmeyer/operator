#!/usr/bin/env python3
"""Entry point: fetch issues -> in-scope filter -> hypothesis extraction ->
scratch-charm render -> reproduction run -> outcome classification ->
comment composition (PLAN.md Approach §1-7).

Fixture mode (default, what the test suite runs): no OPENROUTER_API_KEY, no
juju/concierge/charmcraft needed -- the LLM and runner seams replay
recorded fixtures under `fixtures/`.

Live mode (`--live`): point at a real `gh issue list --json ...` dump, set
OPENROUTER_API_KEY, and run on a host with concierge/juju/charmcraft
installed (the multipass VM per PLAN.md Steps §5). See harness/README.md.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
import dataclasses
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import classifier
import filter_stage
import runner_stage
from composer import Composer
from extraction import ExtractionInvalid
from inscope_second_pass import TwoPassExtractor
from models import Issue, Outcome, RunResult, SurfaceInference
from scaffold import ScaffoldError, expected_signal_for, render_charm
from seams.llm import FixtureLLM, LiveOpenRouterLLM, LLMSeam
from seams.runner import FixtureRunnerSeam, RunnerSeam, SubprocessRunnerSeam
from surface_inference import SurfaceInferenceInvalid, SurfaceInferrer

HERE = Path(__file__).parent
DEFAULT_FIXTURES_DIR = HERE / "fixtures"


@dataclass
class PipelineResult:
    issue_number: int
    stage_reached: str
    outcome: Outcome | None
    reason: str | None
    comment: str | None
    # The captured `RunResult` used to be dropped on the floor here, so a
    # live run's real exit codes and stderr -- the only evidence that says
    # whether a verdict came from the bug or from a broken deploy -- were
    # unrecoverable once `run_for_issue` returned. `--dump-run` writes this
    # to `<out-dir>/<issue>-run.json`.
    run_result: RunResult | None = None
    branch: str | None = None
    hypothesis: object | None = None
    surface: SurfaceInference | None = None


class Pipeline:
    def __init__(self, llm: LLMSeam, runner: RunnerSeam, *, work_dir: Path | None = None):
        self.llm = llm
        self.runner = runner
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="add-reproducer-"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # PLAN.md Approach §3 delta, `spike-step-5/inscope-instrument/
        # RESULT.md`: a second, narrower-question pass on every first-pass
        # drop, measured to hold the corpus-v2 precision/recall floor.
        self.extractor = TwoPassExtractor(llm)
        self.surface_inferrer = SurfaceInferrer(llm)
        self.composer = Composer(llm)

    def run_for_issue(
        self,
        issue: Issue,
        *,
        calibration_mode: bool = False,
        ignore_in_scope: bool = False,
        run_id: str | None = None,
        timestamp: str | None = None,
    ) -> PipelineResult:
        """`calibration_mode` bypasses Approach §3's confidence gate --
        this is exactly what spike-step-4 did by hand (walked all four
        `in_scope: true` extractions regardless of confidence, to
        calibrate the runner/classifier). Production runs leave it False."""
        run_id = run_id or str(uuid.uuid4())
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()

        verdict, reason = filter_stage.classify_issue(issue)
        if verdict == "DROP":
            return PipelineResult(issue.number, "filter", None, reason, None)

        try:
            hypothesis = self.extractor.extract(issue)
        except ExtractionInvalid as exc:
            return PipelineResult(issue.number, "extraction", None, f"invalid extraction: {exc}", None)

        if not hypothesis.in_scope and not ignore_in_scope:
            return PipelineResult(issue.number, "extraction", None, "in_scope=false (second opinion)", None)

        # Branch is decided here, ahead of its other use below, because the
        # confidence gate needs it: `extraction.py`'s own instructions tie
        # "low confidence" to "I could not write a self-contained commands[]
        # recipe" (empty commands[] + confidence=low is the documented
        # fallback). That's a meaningful signal on the branches that actually
        # execute commands[] (`runner_stage.COMMAND_EXECUTING_BRANCHES`), and
        # not connected to anything on `runner_stage.SCRATCH_BRANCHES`, which
        # build their reproduction sequence from `surface`, never from
        # `commands[]`. Gating scratch-branch hypotheses on it silenced
        # `#2639` -- the project's one end-to-end-confirmed reproduction --
        # in 2 of 3 corpus-v4 runs, for a reason unconnected to whether the
        # reproduction would work. Every in-scope, confidence=low, k8s/lxd
        # hypothesis recorded anywhere in this project's history that routes
        # to a scratch branch is a real bug (spike-step-5/
        # confidence-gate-2639/RESULT.md); the confidence-gated false-keep
        # rate this exemption is measured against is 0/0, not merely 0-of-
        # something-untried.
        branch = runner_stage.choose_branch(hypothesis)
        if (
            not calibration_mode
            and hypothesis.confidence == "low"
            and branch not in runner_stage.SCRATCH_BRANCHES
        ):
            return PipelineResult(issue.number, "extraction", None, "confidence=low, staying silent", None)

        # Per-issue scratch dir. Every issue used to share one workdir, so a
        # 40-issue batch ran `uv init` in an already-initialised project for
        # all but the first (exit 2, "Project is already initialized"), which
        # the classifier scored as `partial` -- a comment claiming partial
        # reproduction off a setup collision, for issues whose test then
        # passed. It also put every synthesised test file in one pytest
        # rootdir, so unrelated issues' tests shared a collection.
        issue_dir = self.work_dir / f"issue-{issue.number}"
        issue_dir.mkdir(parents=True, exist_ok=True)
        context = {"issue_number": issue.number, "repo": issue.repo, "workdir": str(issue_dir)}

        # Skip-when-stale gate first (Approach §4): cheapest possible
        # drop, before spending any time on surface inference or a
        # scratch-charm render for a hypothesis that's going nowhere.
        stale, stale_reason = runner_stage.is_stale(hypothesis, issue, self.runner, context)
        if stale:
            return PipelineResult(issue.number, "runner:skipped_stale", Outcome.SKIPPED_STALE, stale_reason, None)

        surface: SurfaceInference | None = None
        if branch == "none":
            # Approach §3/§4 delta (spike-step-5/2045/RESULT.md §1): a
            # `substrate: none` hypothesis with no runnable pytest
            # invocation gets a synthesised test file instead of staying
            # un-runnable.
            hypothesis = runner_stage.write_synthesized_test_file_if_needed(hypothesis, issue, self.llm, context)
        # Both scratch-charm branches need a rendered charm before the
        # runner seam can pack/deploy it. `lxd-scratch` joined `k8s-scratch`
        # here when `seams/runner.py` grew `_run_lxd_scratch` -- before that,
        # a `substrate: lxd` hypothesis never reached this far (the runner
        # seam raised `ValueError: unknown branch` first). Surface inference
        # and `scaffold.render_charm()` are substrate-agnostic (render.py
        # picks a machine- vs k8s-shaped charm from whether
        # `pebble_service.container` is set), so nothing else here changes.
        if branch in ("k8s-scratch", "lxd-scratch"):
            try:
                surface = self.surface_inferrer.infer(issue, hypothesis)
            except SurfaceInferenceInvalid as exc:
                return PipelineResult(issue.number, "surface_inference", None, f"invalid surface inference: {exc}", None)
            # `expected_signal` is decided by the charm template, not by the
            # model: `scaffold.expected_signal_for()` reads render.py's own
            # table. A guessed string can only match by luck, and the first
            # successful k8s stimulus (2026-08-18) was scored against an
            # invented one while the charm emitted something else entirely.
            # Applied before the gates below so they see the real value.
            surface = dataclasses.replace(surface, expected_signal=expected_signal_for(surface))
            # Before rendering or packing anything: a scratch branch with no
            # stimulus deploys a charm, pokes nothing, and produces a verdict
            # about an experiment that never happened. Checked here rather
            # than after the run, so it costs an LLM call instead of a
            # substrate bootstrap.
            has_stimulus, no_stimulus_reason = runner_stage.check_stimulus(branch, surface)
            if not has_stimulus:
                return PipelineResult(
                    issue.number,
                    "runner:no_stimulus",
                    Outcome.UNRUNNABLE_NO_STIMULUS,
                    no_stimulus_reason,
                    None,
                    surface=surface,
                    branch=branch,
                    hypothesis=hypothesis,
                )
            charm_dir = self.work_dir / f"charm-{issue.number}"
            try:
                render_charm(surface, charm_dir, source_issue=issue.number)
            except ScaffoldError as exc:
                return PipelineResult(issue.number, "scaffold", None, f"render failed: {exc}", None)
            context["charm_dir"] = str(charm_dir)

        # Approach §4 pre-run gate (spike-step-5/live-llm/RESULT.md Finding
        # 2): commands[] that aren't shell exit 127 on contact with one, and
        # the classifier reads that as evidence of a reproduction. Checked
        # after synthesis, which can repair the sequence.
        runnable, unrunnable_reason = runner_stage.check_runnable(hypothesis, branch)
        if not runnable:
            return PipelineResult(
                issue.number,
                "runner:unrunnable_commands",
                Outcome.UNRUNNABLE_COMMANDS_NOT_SHELL,
                unrunnable_reason,
                None,
            )

        run_result: RunResult = self.runner.run(branch=branch, hypothesis=hypothesis, surface=surface, context=context)
        outcome, class_reason = classifier.classify(hypothesis, surface, run_result)
        comment = self.composer.compose(
            hypothesis, issue, run_result, outcome, class_reason, run_id=run_id, timestamp=timestamp
        )
        return PipelineResult(
            issue.number,
            "classifier",
            outcome,
            class_reason,
            comment,
            run_result=run_result,
            branch=branch,
            hypothesis=hypothesis,
            surface=surface,
        )


def load_issues(path: Path) -> list[Issue]:
    data = json.loads(Path(path).read_text())
    return [Issue.from_dict(d) for d in data]


def build_pipeline(
    *,
    fixture_mode: bool,
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
    work_dir: Path | None = None,
    fixture_llm: bool = False,
) -> Pipeline:
    """Assemble the two seams.

    `fixture_mode` picks fixtures for both; `--live` picks live for both.
    `fixture_llm` splits them: fixture extraction, real runner. That is the
    combination a substrate measurement needs -- the runner half is the only
    part a real multipass VM exercises, and re-extracting costs an
    OpenRouter call for an answer already recorded in `fixtures/`. It is a
    measurement mode, never production: the extraction it replays is a
    recorded one, so nothing about it says the extractor would produce the
    same hypothesis today.
    """
    if fixture_mode:
        llm: LLMSeam = FixtureLLM(fixtures_dir)
        runner: RunnerSeam = FixtureRunnerSeam(fixtures_dir)
    else:
        llm = FixtureLLM(fixtures_dir) if fixture_llm else LiveOpenRouterLLM()
        runner = SubprocessRunnerSeam()
    return Pipeline(llm, runner, work_dir=work_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--issues", type=Path, required=True, help="gh issue list --json ... dump (see README)")
    parser.add_argument("--live", action="store_true", help="call OpenRouter + shell out to juju/concierge for real")
    parser.add_argument("--out-dir", type=Path, default=Path("add-reproducer-out"))
    parser.add_argument(
        "--calibration-mode",
        action="store_true",
        help=(
            "bypass Approach §3's confidence gate, as spike-step-4 did by hand. "
            "Needed to exercise the runner against a `confidence: low` extraction "
            "(#2639's own is one); never correct for a production run, since the "
            "gate is what keeps low-confidence hypotheses from composing a comment."
        ),
    )
    parser.add_argument(
        "--ignore-in-scope",
        action="store_true",
        help=(
            "run issues the extractor judged out of scope anyway. A MEASUREMENT TOOL, "
            "never a production setting: the scope judgement is what keeps the pipeline "
            "off feature requests and docs tickets. Use it to see what the harness would "
            "say about issues it currently drops -- which is the only way to get a "
            "false-comment sample, since production composes nothing."
        ),
    )
    parser.add_argument(
        "--fixture-llm",
        action="store_true",
        help=(
            "with --live: replay the recorded extraction from fixtures/ instead of "
            "calling OpenRouter, but still shell out to concierge/charmcraft/juju for "
            "real. The mode a substrate/wall-clock measurement wants -- it exercises "
            "the only half a VM can exercise, at zero LLM cost. Never a production "
            "setting: the hypothesis it runs is a recorded one, not one the extractor "
            "produced from the issue in front of it."
        ),
    )
    parser.add_argument(
        "--dump-run",
        action="store_true",
        help=(
            "write the captured RunResult (every command, exit code, stdout, stderr) "
            "to <out-dir>/<issue>-run.json. Without this there is no record of what "
            "a live run actually executed, which is how a failed pack can reach the "
            "classifier disguised as a verdict about the bug."
        ),
    )
    args = parser.parse_args()

    issues = load_issues(args.issues)
    if args.fixture_llm and not args.live:
        parser.error("--fixture-llm only means something with --live (without it both seams are fixtures already)")
    pipeline = build_pipeline(
        fixture_mode=not args.live, work_dir=args.out_dir / "work", fixture_llm=args.fixture_llm
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for issue in issues:
        result = pipeline.run_for_issue(
            issue, calibration_mode=args.calibration_mode, ignore_in_scope=args.ignore_in_scope
        )
        print(f"#{result.issue_number}: stage={result.stage_reached} outcome={result.outcome} reason={result.reason}")
        if result.comment:
            comment_path = args.out_dir / f"{issue.number}.md"
            comment_path.write_text(result.comment)
            print(f"  would-be comment written to {comment_path}")
        if args.dump_run and result.run_result is not None:
            run_path = args.out_dir / f"{issue.number}-run.json"
            run_path.write_text(
                json.dumps(
                    {
                        "branch": result.branch,
                        "hypothesis": asdict(result.hypothesis) if result.hypothesis is not None else None,
                        "surface": asdict(result.surface) if result.surface is not None else None,
                        **asdict(result.run_result),
                    },
                    indent=2,
                    default=str,
                )
            )
            print(f"  run record written to {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
