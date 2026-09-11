# add-reproducer harness

The pipeline: issue in, reproducer comment (sometimes) out. `pipeline.py`
chains the stages and is runnable on its own; `../run.py` is the thin
GitHub Actions wrapper that fetches one issue and delivers the result.

Two boundaries this code cannot exercise in an ordinary checkout - the LLM
and the juju substrate - are behind seams with both a live and a replay
implementation, so the whole pipeline runs offline, against recorded
inputs, with no API key and no controller. That is what the test suite
does.

## Pipeline stages

1. **fetch issues** - `load_issues()` reads a JSON dump of issues (shape
   below). `../run.py` builds a one-issue dump from the REST API instead.
2. **in-scope filter** (`filter_stage.py`) - deterministic, no LLM: drops
   anything outside the target repositories, anything whose issue type or
   labels mark it as not-a-bug, anything whose title reads as a feature
   request, and anything whose body already contains a runnable
   reproducer. It also pre-extracts a linked CI run URL, which is
   regex-findable and so never worth asking a model for.
3. **hypothesis extraction** (`extraction.py`) - one LLM call, schema
   validated: the versions, base, substrate and commands the reporter was
   using, plus what they said they saw. The issue's comments are included
   in the prompt, not just its body, because a report's real scope is
   often set by a reply rather than the opening text.
4. **scope second opinion** (`inscope_second_pass.py`) - a second, narrower
   LLM call on everything the first pass dropped, which recovers reports
   the deterministic filter was too blunt for without widening it.
5. **surface inference** (`surface_inference.py`) - a further LLM call, only
   for hypotheses that need a deployed scratch charm: which relation,
   storage, pebble service or `ops` API the charm has to exercise, and what
   signal counts as the bug showing up. It also synthesises a test file
   when the extracted commands name a pytest target that nothing in the
   sequence actually writes.
6. **scratch-charm render** (`scaffold.py`) - a thin wrapper around the
   parameterised scratch charm's renderer, invoked exactly as that
   renderer documents rather than reimplemented, so its charmcraft and
   juju fixes cannot drift out of sync with a second copy.
7. **reproduction run** (`runner_stage.py` and the runner seam) - in order:

   - a **skip-when-stale gate**, which drops hypotheses that are almost
     certainly about a version or a symbol that no longer exists;
   - **branch selection**: `none` (a plain scratch directory, `uv venv` and
     `pytest`, no controller at all - the cheapest and commonest case),
     `k8s-scratch` / `lxd-scratch` (render, pack and deploy a charm on a
     concierge-provisioned substrate), or `k8s-clone` (clone the repository
     the issue was filed in and run its own tests);
   - the **runnability gate** (`runnability.py`), which drops anything
     whose commands are not executable shell, before the runner sees them.

   That last gate is load-bearing and not obvious. Prose does not fail
   safe: `bash -c 'Run the test that ...'` exits 127, and the classifier
   reads any non-zero last command as a weak reproduction, which composes
   a comment. So a model that answers the "commands" question in English
   would otherwise produce a confident, wrong comment on a maintainer's
   issue. The gate asks one conservative question - would these strings
   survive contact with a shell at all? - and refuses to guess.

   `_run_k8s_scratch` and `_run_lxd_scratch` are both thin wrappers around
   one shared sequence builder, `_scratch_sequence()`; the only difference
   between them is the `concierge prepare` substrate flag and the label on
   the result. The sequence is: prepare → pack → deploy → stimulus →
   diagnostics (`juju status`, `juju debug-log --replay`) → a control run
   whenever an expected signal is known. The stimulus is derived from the
   inferred surface rather than from the extracted commands, because a
   hand-written reproduction often carries placeholders (`<unit>`,
   `<k8s-charm-with-workload>`) that a shell reads as redirects. The
   control run is what makes silence meaningful: without it, "the signal
   never appeared" and "the charm never worked" are the same observation.

   `concierge prepare` runs with the juju channel the hypothesis pins,
   mapped onto a track the snap actually publishes; an unpinned report gets
   an explicit default (`4.0/stable`) rather than whatever concierge
   happens to choose that month, so the substrate cannot move underneath a
   verdict when concierge is upgraded.
8. **outcome classification** (`classifier.py`) - a ladder of rungs, tried
   in order, from the most specific evidence to the least:
   `un-runnable (test selector stale)` → `reproduced` →
   `reproduced (log-only)` → `un-runnable (API-shape mismatch)` →
   `reproduced (positive-signal absent)` (with the control check) →
   `reproduced (weaker)` → `partial` → `did not reproduce`.
9. **comment composition** (`composer.py`) - only some of those outcomes
   compose anything at all; the rest are silent by design. A composed
   comment always opens with the automation disclaimer and ends with an
   idempotency marker, both appended by this module's own code rather than
   asked of the model, so that "have we already commented here?" is a
   reliable check.

`pipeline.py` never posts. It writes what it composed to
`<out-dir>/<issue>.md`; delivering it is `../run.py`'s job, and only when
asked.

## Two pluggable seams

| Seam | Live | Replay |
|---|---|---|
| LLM (`seams/llm.py`) | `LiveOpenRouterLLM` - reads `OPENROUTER_API_KEY`, calls OpenRouter | `FixtureLLM` - replays `fixtures/extractions/<n>.json`, `fixtures/surface/<n>.json`, … |
| Runner (`seams/runner.py`) | `SubprocessRunnerSeam` - shells out to `concierge`, `charmcraft`, `juju` | `FixtureRunnerSeam` - replays `fixtures/runs/<n>.json` |

Everything the test suite executes runs against the replay
implementations. The live ones are what `pipeline.py --live` and the
workflow use.

## Fixture corpus

`fixtures/` holds the recorded corpus the test suite runs against:

- `issues/` - real `canonical/operator` issues (#2327, #2341, #2484,
  #2639, #2107) plus #2304 as a real out-of-scope case and a synthetic
  #9001 that the deterministic filter drops.
- `extractions/` - recorded extraction output for those issues.
- `surface/` - recorded surface inference for the two that need a charm.
- `runs/` - captured command output, used to pin the classifier's rungs.
- `symbols.json` - the lookup table the stale gate's symbol trigger uses.

One caveat is worth stating because it is easy to misread: `#2107` is the
only `substrate: lxd` extraction anywhere in the corpus, and while the
issue and the extraction are real, **its captured run is not**.
`runs/2107.json` is synthetic (no real lxd run has ever been recorded)
and says so inline in the fixture itself. The `lxd-scratch` branch's
command building is pinned directly in `tests/test_seams_runner.py`
instead.

## Running

```shell
uv run pytest              # the whole suite: no key, no juju, no network
```

Replay mode from the CLI, which uses the same seams as the test suite:

```shell
uv run python pipeline.py --issues fixtures/issues-dump.json --out-dir out/
```

`--issues` expects a JSON array shaped like `gh issue list --json
number,title,body,labels,createdAt,state,author`, plus a `repo` field that
`gh` does not emit itself:

```shell
gh issue list --repo canonical/operator --state open --limit 200 \
  --json number,title,body,labels,createdAt,state,author \
  | jq '[.[] | . + {repo: "canonical/operator"}]' > issues-dump.json
```

## Live mode

On a host with concierge and juju available:

```shell
# Key: OPENROUTER_API_KEY, else ~/.config/ai-reproducer.key.
# OPENROUTER_MODEL optional, defaults to deepseek/deepseek-chat.
uv run python pipeline.py --issues issues-dump.json --live --out-dir out/
```

This calls OpenRouter for real and shells out to concierge, charmcraft and
juju for real. It still posts nothing: composed comments land in
`out/<issue>.md`.

`--out-dir` should be absolute. The charm directory built underneath it is
embedded verbatim into `juju deploy` command strings that run with a
different working directory, and juju 4 rejects a local-charm path that is
neither absolute nor `./`-prefixed.

Three flags exist for measurement and are never right for a production
run: `--calibration-mode` (bypasses the confidence gate), `--ignore-in-scope`
(runs issues the model judged out of scope, the only way to sample the
false-comment rate), and `--fixture-llm` (replays a recorded extraction
while still driving real substrate, so the substrate half can be measured
at no LLM cost). Each is the thing that would otherwise keep a bad comment
from being composed, which is why none of them is wired into the workflow.

## Dry-run plan mode

The scratch-charm orchestration is the part of this code that is hardest
to review by reading, and the most expensive to get wrong - a wrong
sequence is discovered three rounds into a k8s bootstrap. `dry_run.py`
emits the exact, fully resolved command sequence a branch would execute,
without executing anything:

```shell
uv run python dry_run.py --issue 2639
```

It is built on the runner seam's `build_plan()`, which shares
`_scratch_sequence()` with the real runner, so a reviewed plan and a real
run cannot silently diverge into two different sequences. Every generated
plan is checked against the runnability gate before it is printed.

## What's still open

**The scratch branches are still thinly exercised on real substrate.**
Every `subprocess.run` in the test suite is mocked; the confidence that
`k8s-scratch` works end to end comes from a small number of real runs on
one issue, not from the suite. Read a dry-run plan before trusting a new
branch shape.

**No comment has ever been posted.** The composer's output has been read
by hand and judged useful on a handful of cases; the false-comment rate in
production is unmeasured, which is why the workflow starts with commenting
switched off.

**One juju channel, not a matrix.** Each run prepares a single channel.
The known version-dependent case in this project's own corpus turns on a
difference between two point releases of the same track, which a
track-level matrix would not see anyway, and a second `concierge prepare`
on an already-prepared runner is unmeasured: it may not even be supported,
and the two legs would collide on the controller and model names concierge
derives from the substrate flag. The verdict a single channel produces
names the juju version it was produced on, so it is a stated claim about
one version rather than a silent claim about all of them.

> **Known limitation — CI-run-only bug reports produce no comment.**
>
> An issue that links a GitHub Actions run and gives no local reproduction
> steps (no commands, no self-contained snippet) is routed to the
> `k8s-clone` runner branch, and is then dropped, silently and by design,
> with `commands[] is empty`. Nothing in this pipeline builds a
> reproduction from a CI run URL alone: it does not fetch CI logs, does not
> infer a test invocation from a failing job, and does not clone a
> repository other than the one the issue was filed in (a linked run often
> belongs to a different repo).
>
> This is a deliberate limitation, not an oversight. Building a
> reproduction from a CI run would mean re-reading CI output the reporter
> has already seen — a triage summariser rather than a reproducer — and the
> alternative, guessing the failing invocation and handing it to a shell,
> is the shape this pipeline's runnability gate exists to stop.
>
> The practical effect: **a bug report whose only evidence is "this CI run
> failed" gets silence.** If you want a reproducer comment on such an
> issue, add local reproduction steps to the issue body. On record
> (2026-09-10) this shape has never occurred on production traffic —
> 0 of 77 open issues across two production-configuration runs — and has
> only ever been produced by two issues in this project's corpora,
> `canonical/operator#1329` and `#2484`.
