# add-reproducer harness

Step-5 harness: the spike pieces from steps 1-4 wired into one runnable
pipeline, implementing the folded design in `../PLAN.md`'s Approach
sections (§3's `ci_run_url`/surface-inference additions, §4's
skip-when-stale gate and third-branch heuristic, §5's classifier rung
ladder).

This harness does **not** run the actual end-to-end dry-run against real
`operator` issues on a live multipass VM (Steps §5's last mile) — that
needs `concierge`/`juju`/`charmcraft`, which don't exist in this sandbox.
What it does do: give you a pipeline you can point at that VM once it's
up, plus a fixture-driven test suite that proves the classifier lands the
four step-4 calibration hypotheses on the right rung, with no LLM key and
no juju required.

## Pipeline stages

`pipeline.py` chains, per `PLAN.md` Approach §1-7:

1. **fetch issues** — `load_issues()` reads a `gh issue list --json ...`
   dump (see below).
2. **in-scope filter** (`filter_stage.py`) — deterministic, ported from
   `spike-step-1/filter.py` / `spike-step-2/filter.py`, plus the
   `ci_run_url` pre-extraction (Approach §3: regex-extractable before any
   LLM call).
3. **hypothesis extraction** (`extraction.py`) — one LLM call via the LLM
   seam, schema-validated, `moving_parts` extended with `symbol_anchor`.
4. **surface inference** (`surface_inference.py`) — a second LLM call,
   only for hypotheses that need a deployed scratch charm (the
   k8s-scratch branch); produces `relation`/`storage`/`pebble_service`/
   `ops_api_surface`/`expected_signal`.
5. **scratch-charm render** (`scaffold.py`) — thin wrapper around
   `../spike-step-3/charm/render.py`, invoked exactly as documented
   rather than reimplemented, so the three charmcraft/juju bugfixes
   landed there (2026-07-22) can't drift out of sync.
6. **reproduction run** (`runner_stage.py` + the runner seam) —
   skip-when-stale gate first (Approach §4, two triggers), then the
   third-branch heuristic picks `none` / `k8s-scratch` / `lxd-scratch` /
   `k8s-clone`, then the **runnability gate** (`runnability.py`) drops
   anything whose `commands[]` isn't executable shell before the runner
   sees it. That gate exists because prose doesn't fail safe: `bash -c 'Run
   the test ...'` exits 127, and the classifier's rung 6 reads any non-zero
   last command as `reproduced (weaker)`, which composes a comment. It runs
   after test-file synthesis (which can repair a sequence) and only for the
   branches that actually shell out to `commands[]` — `k8s-scratch` and
   `lxd-scratch` build their own sequence from the rendered scaffold
   instead. See `../spike-step-5/live-llm/RESULT.md` Finding 2.

   `lxd-scratch` (machine substrate — `sudo concierge prepare -p lxd`
   rather than `-p k8s`) dispatches alongside `k8s-scratch` as of the fix
   recorded in `../PLAN.md`'s Steps §5: before that, `choose_branch()` could
   return `"lxd-scratch"` (the real corpus's only `substrate: lxd`
   extraction, `#2107`, does exactly this) while the runner seam only
   dispatched `none`/`k8s-scratch`/`k8s-clone`, so any substrate:lxd
   hypothesis crashed the pipeline with `ValueError: unknown branch`.

   `seams/runner.py`'s `_run_k8s_scratch` and `_run_lxd_scratch` are both
   thin wrappers around one shared sequence builder, `_scratch_sequence()`
   (the only difference between the two branches is the `concierge prepare`
   substrate flag and the `RunResult.branch` label): prepare → pack → deploy
   → stimulus (derived from `surface.pebble_service`, not `commands[]` —
   `#2639`'s own hand extraction carries unfilled `<k8s-charm-...>`/`<unit>`
   placeholders a shell reads as redirects) → diagnostics (`juju status`,
   `juju debug-log --replay`) → a control run when `surface.expected_signal`
   is set, per `spike-step-4/2639/RESULT.md`'s "silence is only meaningful
   if the control fired" finding. `_run_k8s_scratch` completed this full
   sequence on 2026-07-27 — see the PLAN.md Steps §5 entry for that date;
   before that it only ran `concierge prepare` + `charmcraft pack` and
   never deployed, stimulated, or produced a control. **Still unverified
   against real juju/concierge/charmcraft** for both scratch branches — see
   "What's still open" below.

   `seams/runner.py`'s `build_plan()` builds the exact same sequence, pure
   (no subprocess call), for the dry-run plan mode — see "Dry-run plan
   mode" below.
7. **outcome classification** (`classifier.py`) — the Approach §5 rung
   ladder: `un-runnable (test selector stale)` → `reproduced` →
   `reproduced (log-only)` → `un-runnable (API-shape mismatch)` →
   `reproduced (positive-signal absent)` (with the control-case check) →
   `reproduced (weaker)` → `partial` → `did not reproduce`.
8. **comment composition** (`composer.py`) — minimal stub. Renders the
   would-be comment to a string; the CLI writes it to
   `<out-dir>/<issue>.md` instead of posting anything. `gh issue comment`
   / the idempotency-marker check (Approach §7) is explicitly out of
   scope here — step 5 dry-runs comments, never posts them.

## Two pluggable seams

Both boundaries this pipeline can't safely exercise in a sandbox have a
live implementation and a fixture/replay implementation
(`seams/llm.py`, `seams/runner.py`):

| Seam | Live | Fixture |
|---|---|---|
| LLM (`seams/llm.py`) | `LiveOpenRouterLLM` — reads `OPENROUTER_API_KEY`, calls OpenRouter | `FixtureLLM` — replays `fixtures/extractions/<n>.json` / `fixtures/surface/<n>.json` |
| Runner (`seams/runner.py`) | `SubprocessRunnerSeam` — shells out to `concierge`/`charmcraft`/`juju` | `FixtureRunnerSeam` — replays `fixtures/runs/<n>.json` (captured from `../spike-step-4/<n>/RESULT.md`) |

Everything the test suite actually executes runs against the fixture
implementations. The live implementations exist and are wired into
`pipeline.py --live`, but nothing in this repo's CI or sandbox can reach
OpenRouter or a juju controller, so they're untested here by design.

## Fixture corpus

`fixtures/` holds the step-2/step-4 corpus, trimmed to what the harness
needs:

- `issues/` — 4 real `operator` issues (#2327, #2341, #2484, #2639, the
  full `in_scope: true` set from step 2) plus #2304 (a real
  `in_scope: false` re-drop) and a synthetic #9001 (filter-level DROP,
  `enhancement` label).
- `extractions/` — the step-2 extraction JSON for those 5 issues, with
  `ci_run_url`/`symbol_anchor` filled in per the Approach §3 delta.
- `surface/2639.json` — the only k8s-scratch hypothesis needing a scratch
  charm; derived from
  `../spike-step-3/charm/params/2639-pebble-custom-notice.yaml`.
- `runs/` — captured command output for #2327/#2341/#2484/#2639, derived
  verbatim from `../spike-step-4/<n>/RESULT.md`.
- `symbols.json` — `resolve_symbol()` lookup table for the stale gate's
  second trigger.
- **`#2107`** (`issues/2107.json`, `extractions/2107.json`,
  `surface/2107.json`, `runs/2107.json`) — the real, only `substrate: lxd`
  extraction anywhere in this project's corpus, verbatim from
  `../spike-step-5/corpus-v2/out-inscope-v2-confirm/2107.json`'s
  `live_extraction` field (a real live OpenRouter call against the real
  `canonical/operator#2107`). **The issue and extraction are real; the
  captured run is not.** `runs/2107.json` is synthetic captured output —
  no real lxd/juju/concierge run has ever happened for the `lxd-scratch`
  branch (see the "reproduction run" section above and PLAN.md's Steps §5
  entry) — labelled as such inline in the fixture itself
  (`_synthetic_note`), not just here. The `juju debug-log` command's
  stdout does reuse the real traceback text from the real issue body; what
  is fabricated is that running these commands against a deployed scratch
  charm produced it. `#2107` is also CLOSED with `repo_version: null`, so
  it never actually reaches `choose_branch()`/the runner via the full
  pipeline — Approach §4's stale gate intercepts it first, same as
  #2341/#2327 (see `tests/test_runner_stage.py`). Dispatch and the
  `lxd-scratch` seam's command-building logic are instead pinned directly
  in `tests/test_seams_runner.py`, bypassing the gate the same way
  `tests/test_classifier.py` already does for #2341/#2327's rungs.

## Running

```shell
uv run pytest              # fixture-driven test suite (no key, no juju)
```

CLI, fixture mode (same seams as the test suite):

```shell
uv run python pipeline.py --issues fixtures/issues-dump.json --out-dir out/
```

`--issues` expects a JSON array shaped like `gh issue list --json
number,title,body,labels,createdAt,state,author`, plus a `repo` field
(e.g. `"canonical/operator"`) that `gh issue list` doesn't emit itself —
add it when building the dump, e.g.:

```shell
gh issue list --repo canonical/operator --state open --limit 200 \
  --json number,title,body,labels,createdAt,state,author \
  | jq '[.[] | . + {repo: "canonical/operator"}]' > issues-dump.json
```

## Live mode (the actual Steps §5 dry-run)

On the multipass VM (juju + concierge installed, per `PLAN.md` "Where it
lives"):

```shell
# Key: OPENROUTER_API_KEY, else ~/.config/ai-reproducer.key.
# OPENROUTER_MODEL optional, defaults to deepseek/deepseek-chat.
uv run python pipeline.py --issues issues-dump.json --live --out-dir out/
```

The extraction half of live mode has been exercised for real — seven
`canonical/operator` issues, 7/7 schema-valid, $0.0031, see
`../spike-step-5/live-llm/RESULT.md` for the calibration findings (which
include a `substrate: null` routing bug in `runner_stage.py`'s
`choose_branch`). The runner half — `concierge`/`charmcraft`/`juju` —
still has not run outside fixtures.

This calls OpenRouter for real (extraction + surface inference) and
shells out to `concierge`/`charmcraft`/`juju` for real (`seams/runner.py`
`SubprocessRunnerSeam`). Nothing gets posted anywhere — comments land in
`out/<issue>.md` for a maintainer to read, per Steps §5's "dry-run the
comment past a maintainer rather than actually posting."

`Pipeline.run_for_issue(issue, calibration_mode=True)` bypasses Approach
§3's confidence gate (mirrors what `spike-step-4` did by hand, walking
all four `in_scope: true` extractions regardless of confidence, to
calibrate the runner/classifier). Leave it `False` for a real run.

## Dry-run plan mode

The k8s-scratch/lxd-scratch orchestration above is unexercised code — there
is no juju/k8s/charmcraft/cluster in this sandbox, so it has never run for
real. `dry_run.py` is the piece of this that *can* actually be reviewed
before a human ever touches a real multipass VM: it emits the exact,
fully-resolved command sequence a branch would execute, without executing
anything.

```shell
uv run python dry_run.py --issue 2639
```

Built on `seams/runner.py`'s `build_plan()`, which shares its sequence
builder (`_scratch_sequence()`) with `SubprocessRunnerSeam` itself, so a
reviewed plan and a real run can never silently diverge into two different
sequences. Every generated plan is checked against `runnability.assess()`
before being printed or written — the trap being guarded against is
`#2639`'s own hand extraction, whose `commands` contain literal
`juju deploy <k8s-charm-with-workload>` / `juju ssh --container
<workload-container> <unit>/0` placeholders that a shell reads as
redirects (see "Two pluggable seams" above and `runnability.py`'s module
docstring). `tests/test_dry_run.py` pins that the committed `#2639`
artefact below is what the generator produces today, not a stale
hand-edited snapshot.

The committed artefact, `../spike-step-5/2639-k8s-scratch-dry-run-plan.md`,
is the reviewable plan for `#2639` — the only hypothesis this project has
ever actually reproduced by hand (`spike-step-4/2639/RESULT.md`) — covering
the whole branch including the control case. Regenerate it with:

```shell
uv run python dry_run.py --issue 2639 --out ../spike-step-5/2639-k8s-scratch-dry-run-plan.md
```

## What's still open after this harness

Per `PLAN.md` Steps §5: the actual end-to-end dry-run against live
`operator` issues, on the multipass VM, iterated for two pulses
(false-drop rate, false-comment rate, comment usefulness). This harness
is what that dry-run runs — it hasn't been run for real yet.

The k8s-scratch/lxd-scratch runner branches themselves remain **unexercised
orchestration**, 2026-07-27's completion of `_run_k8s_scratch` included:
every `subprocess.run` call in `tests/test_seams_runner.py` is mocked, and
there is no juju/concierge/charmcraft/cluster in this sandbox to run them
for real. What *is* now reviewable without a VM is the dry-run plan mode
above — a human should read `../spike-step-5/2639-k8s-scratch-dry-run-plan.md`
before the first real multipass walk, not discover a wrong sequence three
rounds into a k8s bootstrap the way the three `render.py` bugs (Steps §3)
were discovered.
