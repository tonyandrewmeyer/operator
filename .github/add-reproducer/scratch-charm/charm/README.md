# scratch-charm scaffolding

Skeleton generator for step 3 of `add-reproducer` (`../FINDINGS.md`
has the full design writeup). Forked in spirit, not in file content,
from `canonical/charm-ubuntu` (`master`) — reviewed via
`raw.githubusercontent.com`, not cloned locally.

```
render.py params/<name>.yaml examples/<name>/
```

`render.py` reads a `params.yaml` (a proposed extension of the
hypothesis extractor's `moving_parts` schema — see
`../FINDINGS.md` for why the extractor doesn't emit these fields yet)
and generates a minimal `charmcraft.yaml` + `src/charm.py` +
`pyproject.toml`. Structural values (relation endpoint, storage
stanza, pebble container) get baked into `charmcraft.yaml`;
behavioural values (`ops_api_surface`) get baked into `src/charm.py`
as a hardcoded observer registration. Neither is dynamic at runtime —
juju needs the structural shape fixed at pack time, and there's no
benefit to indirection once it is.

`params/` holds three example inputs:

- `2639-pebble-custom-notice.yaml` — the one corpus-driven example
  (operator#2639, the sole step-2 KEEP that actually needs a deployed
  charm rather than a self-contained `ops.testing` script).
- `relation-changed-example.yaml` — structural example, not tied to a
  real issue; demonstrates the relation axis for a *deployed* charm.
- `storage-example.yaml` — speculative, zero corpus evidence; proves
  the storage stanza renders, nothing more.

`examples/` holds the corresponding rendered output (regenerate with
the command above; not hand-edited). Verified: `py_compile` and `ruff
check` clean on all three generated `src/charm.py`; all three
`charmcraft.yaml` round-trip through `yaml.safe_load`. `charmcraft
pack` was not run — not installed in this sandbox.

See `../mapped-hypotheses.md` for how each real step-2 hypothesis maps
(or doesn't) onto this scaffolding.
