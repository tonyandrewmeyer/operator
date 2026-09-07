# add-reproducer wall-clock measurement

This directory is not part of `ops`. It carries the add-reproducer spike
harness so that `.github/workflows/add-reproducer-wallclock.yaml` can take
the one measurement the spike cannot take anywhere else: criterion 7b, the
end-to-end wall-clock on a GHA-hosted `ubuntu-latest` runner.

The spike's cold figures come from dev multipass VMs, and they miss the
20-minute budget: two fresh VMs agreed that `concierge prepare` plus
`charmcraft pack` alone is around 24m45s, before `deploy` is even
requested. `charmcraft pack` is most of it. Whether that holds, improves
or worsens on GitHub's hardware decides whether the design's intended
runtime is viable, and neither a VM nor a container can answer it.

Run the workflow from the Actions tab (`workflow_dispatch`). It installs
concierge and then leaves the harness to run `sudo concierge prepare`
itself -- pre-preparing would warm the step being measured. `--fixture-llm`
replays the recorded extraction for #2639, so no OpenRouter key is needed
and only the substrate half of the pipeline runs.

`summarise-run.py` prints the per-step `elapsed_s` table into the job
summary, in the same shape the VM runs were recorded in, so the two can be
read side by side.
