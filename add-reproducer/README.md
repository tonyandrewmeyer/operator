# add-reproducer

The agent behind `.github/workflows/add-reproducer.yaml`. When a bug report
is opened, it decides whether the report is one it can do anything with,
builds a hypothesis about what the reporter was doing, tries that on a real
juju substrate, and (if the attempt reproduces what the reporter described)
posts the reproducer as a comment.

None of this is part of `ops`. Nothing here is imported by the library, the
test suite, or any charm; it is a tool that reads this repository's issues.

```
add-reproducer/
  run.py            the workflow's entry point: fetch one issue, run the
                    pipeline, optionally post the comment it composes
  harness/          the pipeline itself, and its test suite
  spike-step-3/     the parameterised scratch charm the runner deploys
```

`harness/README.md` is the detailed description: the stages, what each one
can and cannot do, how to run the whole thing locally, and what is still
open.

## Trying it by hand

The workflow has a `workflow_dispatch` trigger that takes an issue number,
so a single issue can be put through the pipeline from the Actions tab. It
defaults to **not** posting anything: the run does everything else, and the
comment it would have posted appears in the job summary and in the run's
artifact. Turning `post_comment` on for that one run is how you let it
speak.

`POST_COMMENTS` at the top of the workflow is the same switch for the
automatic `issues.opened` trigger, and starts at `false`.

## What it costs to run

The pipeline calls an LLM (OpenRouter, keyed by `OPENROUTER_API_KEY` in the
`llm` GitHub environment) two to four times per issue that gets past the
deterministic filter, and most issues do not get that far. The expensive
half is the substrate: an issue whose hypothesis needs a deployed charm
spends most of its job on `concierge prepare` and `charmcraft pack`.
Measured end to end on GitHub's own runners, that is roughly five to
twelve minutes.

## The silence is the point

Most issues produce no comment, and that is the designed behaviour rather
than a failure to try. A report is dropped - quietly - when it is not
bug-shaped, when it already carries a reproducer, when the model judges it
out of scope, when the extracted hypothesis is too uncertain to act on,
when the commands it produced are not runnable shell, or when the attempt
ran and did not reproduce anything. A wrong comment on a maintainer's issue
costs more than a missing one, so every gate in the pipeline is built to
fail towards saying nothing.
