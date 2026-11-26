# Scriptlet runner

Runs a unitless Juju charm's Starlark scriptlets for one event, the way the
Juju 4.2 controller does, and reports what happened as JSON: the intents,
the `print`/`debug` output, the steps and allocations used, and a structured
error. It's the core that the `scriptlet_testing` Python package (in the
parent directory) wraps, and the binary its wheels bundle.

The execution path is Juju's own code: `internal/unitless/` and
`internal/domainunitless/` are copies of Juju's
`internal/worker/unitless/{executor,intent,setstatus,scriptlet}.go` and
`domain/unitless/types.go`, with only the import paths of Juju-internal
packages rewritten (to the two small stand-ins in `internal/errors` and
`internal/coreerrors`). `scripts/check-juju-copy.sh /path/to/juju [ref]`
diffs them, and the starform/starlark pins, against a Juju checkout.

## Licence

The runner is licensed under the GNU Affero General Public License v3.0
only (`AGPL-3.0-only`); the full text is in `LICENSE`. Unlike the rest of
this repository, which is Apache-2.0, because the runner contains Juju's
code: `internal/unitless/` and `internal/domainunitless/` are copies of
files from [juju/juju](https://github.com/juju/juju) (AGPLv3), taken at the
commit recorded as `JujuCommit` in `main.go` and named at the top of each
copied file, and `runner version` reports that commit.

The `ops-testing-scriptlets` wheels bundle the compiled runner as a
separate program that the Python package starts and talks to over pipes, so
the wheels are `Apache-2.0 AND AGPL-3.0-only` and carry both licence texts.

## Build and test

```
go build -o runner .
go vet ./... && go test ./...
```

## One-shot mode

```
runner run --charm DIR|FILE.charm --event FILE.json [--pretty]
           [--max-steps N] [--max-allocs N] [--no-measure]
           [--cap-factor N] [--skip-charm-checks]
```

`--event -` reads the event from stdin. The event file is
`{"name": "config_changed", "attrs": {...}}`. One result object is printed
on stdout; the exit status is 0 whenever a result was printed (scriptlet
failures are in the result), and 2 for usage errors.

```
runner load --charm DIR|FILE.charm [--pretty] [--max-steps N] [--max-allocs N]
            [--no-measure] [--cap-factor N] [--skip-charm-checks]
runner version
```

`load` loads the charm (its top-level code and every `init()`) and prints
a load response (below) without running an event. `version` prints the
version object.

## Persistent mode

```
runner serve
```

Newline-delimited JSON: one request object per line on stdin, one response
object per line on stdout, in order, flushed after each. JSON string
escaping means a serialised object never contains a raw newline. Blank lines
are ignored. The process exits on EOF or after answering `shutdown`.

Requests:

```json
{"id": 1, "op": "run", "charm": "/path/to/charm",
 "event": {"name": "config_changed", "attrs": {"port": 8080}},
 "options": {"max_steps": 100000, "max_allocs": 10485760, "measure": true,
             "cap_factor": 10, "skip_charm_checks": false}}
{"id": 2, "op": "load", "charm": "/path/to/charm", "options": {"measure": false}}
{"id": 3, "op": "version"}
{"id": 4, "op": "shutdown"}
```

`op` is `run` (the default), `load`, `version` or `shutdown`; `options` and
all its fields are optional (`load` takes the same options as `run`, and no
`event`); `id` is any JSON value and is echoed back. The charm's sources
are re-read on every request, and the loaded script set is reused when the
sources and options are unchanged (`load.cached` in the result), as Juju
loads once per application and handles every event on the same script set.
So a `load` followed by `run`s with the same options loads once.

An unknown `op` or a line that isn't JSON gets
`{"protocol": 1, "id": ..., "ok": false, "error": {"kind": "invalid_input", "phase": "input", ...}}`.

`load` answers with the `load` and `error` parts of a run result:

```json
{"protocol": 1, "id": 2, "ok": true,
 "load": {"sources": ["charm.star", "lib/checks.star"], "logs": [],
          "steps": {...}, "allocs": {...}, "cached": false},
 "error": null}
```

`load` is null when the charm itself couldn't be read (an `error` in phase
`input` or `charm`); otherwise it's there, failed or not, with the load's
logs and budgets. `error` is as for a run, in phase `charm` or `load`.

`version` and `shutdown` answer with the version object (plus `id` and
`ok`):

```json
{"protocol": 1, "juju_commit": "145ae4877aba52e983dd426bf7a5009d7363fd6d",
 "juju_tag": null, "juju_version": "4.2-beta1",
 "starform": "v0.0.0-20260521120235-5b4249f3efd2",
 "starlark": "v0.0.0-20250127110116-8c33ed2b410b", "go": "go1.26.0"}
```

`juju_commit` is the `juju/juju` commit the copied files came from,
`juju_tag` the release tag at that commit (null while the commit isn't in a
release: no tag contains `145ae48` yet), and `juju_version` the version in
Juju's `core/version/version.go` at the commit, which is what a controller
built from it reports. `check-juju-copy.sh` checks all three.

## Result

```json
{
  "protocol": 1,
  "id": 1,
  "ok": true,
  "event": "config_changed",
  "observed": true,
  "intents": [{"type": "set-status", "args": {"message": "serving on 8080", "status": "active"}}],
  "logs": [{"level": "print", "event": "config_changed", "path": "charm.star", "line": 9,
            "message": "serving on 8080",
            "juju": "scriptlet charm.star:9: serving on 8080", "juju_level": "INFO"}],
  "steps": {"limit": 100000, "used": 116, "limit_hit": false, "measured": true},
  "allocs": {"limit": 10485760, "used": 3416, "limit_hit": false, "measured": true},
  "load": {"sources": ["charm.star", "lib/checks.star"], "logs": [],
           "steps": {...}, "allocs": {...}, "cached": false},
  "error": null
}
```

- `intents` is Juju's `[]Intent` as `{type, args}`. As in Juju, a handler
  that fails returns no intents, even ones appended before the error.
- `logs` are the event's `print` and `debug` calls; `load.logs` are those
  made by top-level code and `init()`. `juju` is the line Juju's log adapter
  writes, at `juju_level`.
- `steps.used` / `allocs.used` are the smallest limits at which the phase
  doesn't fail on that budget, found by bisection (runs are deterministic).
  When the real limit was hit, the search continues up to `cap` (limit x
  `cap_factor`), and `used` is null if even that isn't enough. `measured` is
  false with `--no-measure`. An event nobody observes reports 0.
- `error` is null or:

  ```json
  {"kind": "runtime_error", "phase": "event",
   "message": "key \"port\" not in dict",
   "juju_log": "handling event \"config_changed\": key \"port\" not in dict",
   "backtrace": "Traceback (most recent call last):\n  hook.star:7:39: in on_config_changed\n  hook.star:2:52: in endpoint\nError: key \"port\" not in dict",
   "position": {"path": "hook.star", "line": 2, "col": 52},
   "builtin": ""}
  ```

  `phase` is `input`, `charm`, `load` or `event`. `kind` is one of
  `syntax_error` (parse and resolve errors, including dialect violations
  such as `while`), `runtime_error`, `builtin_argument_error` (a builtin
  rejected its arguments, e.g. `set_status()`), `step_limit`, `alloc_limit`,
  `load_path_error` (invalid file or `load()` path, or a missing file),
  `invalid_event` (an event name starform rejects), `charm_invalid` (what
  Juju would refuse at add-charm: hooks or dispatch beside scriptlets, a
  missing `assumes: [unitless]`, an empty source file), `invalid_input`,
  `interpreter_panic` (a panic inside starform/Starlark, recovered; Juju
  wouldn't recover it) and `internal`. starform names compiled files
  `cached-<hash>.star`; `message`, `backtrace` and `position` map those back
  to load paths, and `juju_log` keeps the text Juju would log.

## Numbers

Event attributes are decoded with `json.Decoder.UseNumber`: a number with no
`.`, `e` or `E` becomes an `int64` (out of range is an `invalid_input`
error), anything else a `float64`. Juju's `valueToStarlark` then gives
`int` and `float` respectively, so `8080` and `8080.0` stay different, as
they would be for any Go value Juju passed.

## Not done

No proposed APIs: only the builtins Juju has. No `.charm` packing.
