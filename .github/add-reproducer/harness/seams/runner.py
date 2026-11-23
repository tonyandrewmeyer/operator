"""The juju/charmcraft reproduction runner seam.

Two implementations:

- `SubprocessRunnerSeam` -- shells out to `concierge`/`charmcraft`/`juju`
  for real. Never exercised against the real tools by the test suite (no
  juju, no concierge, no network) -- the code path live runs use.
  `tests/test_seams_runner.py` does exercise its command-building logic
  with `subprocess.run` mocked, which pins the *decisions* (sequence,
  substitutions, non-zero/timeout handling) without proving the commands
  themselves work against real infrastructure.
- `FixtureRunnerSeam` -- replays captured command output, recorded as
  `fixtures/runs/<n>.json`. What
  the pipeline actually runs against here.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from models import CommandResult, Hypothesis, RunResult, SurfaceInference

# Default command and whole-hypothesis budgets.
PER_COMMAND_TIMEOUT_S = 5 * 60
PER_HYPOTHESIS_TIMEOUT_S = 20 * 60

# How long to wait for the deployed unit to settle before stimulating it.
# Generous, because not waiting gives a verdict about nothing.
UNIT_READY_TIMEOUT = "15m"
UNIT_READY_TIMEOUT_S = 15 * 60
UNIT_READY_POLL_INTERVAL_S = 10
UNIT_READY_POLLS = UNIT_READY_TIMEOUT_S // UNIT_READY_POLL_INTERVAL_S

# Bounded poll for `juju remove-application` to finish -- removal is async and
# `juju deploy` will collide with an application still tearing down.
APP_REMOVAL_POLLS = 60
APP_REMOVAL_POLL_INTERVAL_S = 5

# Bounded poll for the control's own signal to show in `juju status`. An
# immediate read races the hook that sets it, and losing that race turns
# `REPRODUCED_POSITIVE_SIGNAL_ABSENT` into a wrong `DID_NOT_REPRODUCE`. The
# control is the last step, so a generous budget costs nothing.
CONTROL_SIGNAL_TIMEOUT_S = 2 * 60
CONTROL_SIGNAL_POLL_INTERVAL_S = 5
CONTROL_SIGNAL_POLLS = CONTROL_SIGNAL_TIMEOUT_S // CONTROL_SIGNAL_POLL_INTERVAL_S

# Every juju command names its controller and model explicitly, because
# `concierge prepare` switches the current controller and one run's substrate
# would otherwise leak into the next. The names are concierge's defaults; if
# they change, the next command fails loudly.
CONTROLLER_BY_PREPARE_FLAG = {"k8s": "concierge-k8s", "machine": "concierge-lxd"}
CONCIERGE_MODEL = "testing"

# Budgets for steps that a flat 5 minutes cannot cover: a cold `prepare` or
# `pack` can take much longer. A step that embeds its own timeout (`wait`,
# `control`) needs a longer budget here, or the inner timeout is dead code.
STEP_TIMEOUTS_S = {
    # A backstop against a wedged snapd; a store error returns in seconds.
    "install-concierge": 10 * 60,
    "prepare": 25 * 60,
    "pack": 20 * 60,
    "cleanup": APP_REMOVAL_POLLS * APP_REMOVAL_POLL_INTERVAL_S + 60,
    "deploy": 15 * 60,
    "wait": UNIT_READY_TIMEOUT_S + 60,
    # Doubled because each poll iteration also spends two `juju status` calls,
    # so the loop's wall clock runs ahead of its sleep budget.
    "control": 2 * CONTROL_SIGNAL_TIMEOUT_S + 60,
}

# Steps that have to succeed before anything after them means anything:
# `juju status` and `juju debug-log` exit 0 against an empty model, so a
# failed pack, stimulus or clone would otherwise be scored as evidence about
# the bug. `cleanup` is excluded because there is often nothing to remove.
PREREQUISITE_STEPS = ("install-concierge", "prepare", "pack", "deploy", "wait", "stimulus", "clone")

# Installed in the sequence rather than by the workflow, because only the
# scratch branches need concierge and the branch isn't known until extraction
# has run. The guard is needed because `snap install` of an installed snap
# exits non-zero. `which`, not `command -v`, because `runnability.assess()`
# treats `command` as a wrapper word.
INSTALL_CONCIERGE_COMMAND = "which concierge >/dev/null 2>&1 || sudo snap install --classic concierge"

# `juju deploy` refuses a local k8s charm without its `oci-image` resources.
# Juju injects Pebble as pid 1 whatever the image, so any maintained base
# works.
DEFAULT_WORKLOAD_IMAGE = "ubuntu:24.04"

# Pebble socket path by juju snap track. Keyed on the track `prepare` actually
# installs (a `_juju_track()` result), not the raw pin, so the socket and the
# substrate come from the same input: a "3.4" pin installs juju 3.6, which
# uses the new path. Unlisted tracks fall back to `_PEBBLE_SOCKET_LEGACY`.
_PEBBLE_SOCKET_NEW = "/charm/container/pebble.socket"
_PEBBLE_SOCKET_LEGACY = "/var/lib/pebble/default/.pebble.socket"
_PEBBLE_SOCKET_BY_JUJU_TRACK = {
    "4": _PEBBLE_SOCKET_NEW,
    "4.0": _PEBBLE_SOCKET_NEW,
    "3.6": _PEBBLE_SOCKET_NEW,
    "3": _PEBBLE_SOCKET_NEW,  # `3/stable` installs 3.6
}


def _pebble_socket_path(juju_track: str) -> str:
    """Look up `juju_track` (a `_juju_track()` result, e.g. `"3.6"` or
    `"4.0"`) against `_PEBBLE_SOCKET_BY_JUJU_TRACK`, falling back to the
    major alone (so an unlisted `"4.2"` still hits the `"4"` entry) and then
    to `_PEBBLE_SOCKET_LEGACY` for anything unmeasured.

    The major fallback is not dead code even though `_juju_track()` can only
    return a `_KNOWN_JUJU_TRACKS` entry or the default: this is also called
    with a track a caller chose, and `as_user_in_container_command()` takes
    one as a plain argument."""
    if juju_track in _PEBBLE_SOCKET_BY_JUJU_TRACK:
        return _PEBBLE_SOCKET_BY_JUJU_TRACK[juju_track]
    major = juju_track.split(".", 1)[0]
    return _PEBBLE_SOCKET_BY_JUJU_TRACK.get(major, _PEBBLE_SOCKET_LEGACY)


# The variables that decide which virtualenv a `uv` command writes to,
# whatever directory it runs in. Inheriting either (from an activated venv or
# from `tox-uv`) would send the scratch project's `uv add`/`uv sync` into
# someone else's environment.
_UV_ENVIRONMENT_VARS = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")


def scratch_command_env(project_environment: str | None = None) -> dict[str, str]:
    """`os.environ` with uv's environment-selection variables neutralised, and
    `UV_PROJECT_ENVIRONMENT` pinned to `project_environment` when one is
    given.

    Pinning rather than only clearing, because clearing leaves the answer to
    uv's directory walk -- which is right when the scratch project has a root
    of its own and wrong the moment it does not. Naming the environment makes
    the scratch commands hermetic either way."""
    env = dict(os.environ)
    for name in _UV_ENVIRONMENT_VARS:
        env.pop(name, None)
    if project_environment is not None:
        env["UV_PROJECT_ENVIRONMENT"] = project_environment
    return env


def _observed_juju_version() -> str | None:
    """Best-effort `juju version` on this substrate (`RunResult.
    observed_juju_version` -- see that field's docstring for why it exists).

    Called after `concierge prepare` has had a chance to run, so this is the
    concrete client version concierge actually put on PATH, not any
    hypothesis's own `moving_parts.juju_version` pin. `None`, never a raise,
    if `juju` isn't installed or the call fails -- a missing version string
    must not abort a run that otherwise completed; every other seam in this
    module treats infrastructure absence as data, not as a crash."""
    try:
        proc = subprocess.run(["juju", "version"], capture_output=True, timeout=30, text=True)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


# The `ops` version `charmcraft pack` resolved into the scratch charm. It
# comes from PyPI inside charmcraft's build instance, so the pack log line
# (`:: + ops==3.8.2`) is the only record. The lookbehind stops a name that
# ends in `ops` (`charmops==1.2.3`) from matching.
_PACKED_OPS_VERSION_RE = re.compile(r"(?<![\w.\-])ops==(?P<version>[A-Za-z0-9][\w.!+\-]*)")


def _ops_version_in(commands: list[CommandResult]) -> str | None:
    """Last `ops==<version>` in these commands' captured output, or `None`.

    The **last** match wins. uv prints a package's removal before its addition
    (`- ops==3.8.1` then `+ ops==3.8.2`) when it replaces one, so the last
    `ops==` in the log is the one that ended up installed; the first would name
    the version that was thrown away.
    """
    text = "\n".join(f"{c.stdout or ''}\n{c.stderr or ''}" for c in commands)
    matches = _PACKED_OPS_VERSION_RE.findall(text)
    return matches[-1] if matches else None


def _packed_ops_version(commands: list[CommandResult]) -> str | None:
    """Best-effort `ops` version from the `pack` step's captured output
    (`RunResult.observed_ops_version` -- see that field's docstring for why it
    exists).

    Fails soft in every direction `_observed_juju_version()` does, and for the
    same reason: no `pack` step in this branch, no captured output, or nothing
    in the log that parses, all give `None` rather than raising. A version
    string nobody could read must not abort a run that otherwise completed.
    """
    pack = next((c for c in commands if c.step == "pack"), None)
    if pack is None:
        return None
    return _ops_version_in([pack])


# Commands whose output can say which `ops` the `none` branch resolved,
# matched on the command string because that branch labels no steps. Only
# installer output counts: a `pytest` failure dump can quote `ops==` too.
_OPS_RESOLVING_COMMAND_RE = re.compile(r"(?:\buv\s+(?:add|sync|pip\s+install)|\bpip3?\s+install)\b")


def _resolved_ops_version(commands: list[CommandResult]) -> str | None:
    """Best-effort `ops` version the `none` branch resolved, read out of its
    own install commands' captured output.

    The `none` branch never packs, but it does resolve an `ops`, from PyPI
    via `uv add 'ops[testing]'`, and prints it:

        + ops==3.8.2

    That line is captured and rendered into the composed comment's
    "Observed output" block, so without this a comment would quote the
    answer a few lines above a versions line that declined to state it.
    Packing is not what matters; resolving is.

    Fails soft in every direction `_packed_ops_version()` does: no install
    command, no captured output, or nothing that parses, all give `None`.
    """
    installs = [c for c in commands if _OPS_RESOLVING_COMMAND_RE.search(c.command or "")]
    if not installs:
        return None
    return _ops_version_in(installs)


class RunnerSeam(Protocol):
    def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
        """False only when `symbol_anchor` (e.g.
        "ops.model.Model.get_relation") is positively *absent* from the
        target repo's checked-out `main` -- the skip-when-stale gate's second
        trigger, and the only thing that trips it.

        True when the symbol resolves, and also true when the implementation
        cannot tell (nothing importable to ask, no interpreter, a probe that
        hung). A `False` here skips the hypothesis outright and composes
        nothing, so it has to mean "this API surface is gone", never "I did
        not look"."""
        ...

    def run(
        self,
        *,
        branch: str,
        hypothesis: Hypothesis,
        surface: SurfaceInference | None,
        context: dict,
    ) -> RunResult:
        """Execute `hypothesis` via the given branch ("none" | "k8s-scratch"
        | "lxd-scratch" | "k8s-clone") and return captured output."""
        ...


def as_user_in_container_command(
    *,
    unit: str,
    container: str | None,
    user: str,
    juju_track: str,
    pebble_command: str,
    model: str | None = None,
) -> str:
    """Build the "run this command as user X inside container Y" primitive.

    Needed for #2639: `pebble notify` has no `--user` flag, so reproducing a
    non-root notice means actually creating the user and executing pebble as
    them. `su -` strips the environment,
    so `PEBBLE_SOCKET` has to be re-exported inside the shell it starts,
    not before.

    `container` is optional (default `None`) for the `lxd-scratch` branch:
    a machine unit has no k8s-style Pebble *container* to target with
    `juju ssh --container`, so omitting `--container` entirely and targeting
    the unit directly is the conservative reading, not an invented one.
    `_run_k8s_scratch`'s (k8s) callers pass a real container name.
    """
    model_flag = f" -m {model}" if model else ""
    socket = _pebble_socket_path(juju_track)
    inner = f"su - {user} -s /bin/sh -c 'PEBBLE_SOCKET={socket} {pebble_command}'"
    ensure_user = f"id -u {user} >/dev/null 2>&1 || useradd -M {user}"
    if container:
        # No `--` on the k8s container path: `juju ssh --container` forwards it into
        # the container's shell, which fails with `Illegal option --`. The machine
        # form keeps it, as the documented separator.
        return f'juju ssh{model_flag} --container {container} {unit} "{ensure_user}; {inner}"'
    return f'juju ssh{model_flag} {unit} -- "{ensure_user}; {inner}"'


_JUJU_VERSION_RE = re.compile(r"^(?P<major>\d+)(?:\.(?P<minor>\d+))?")


def _juju_version_key(hypothesis: Hypothesis, default: str = "4") -> str:
    """Best-effort `"major"` or `"major.minor"` key from
    `moving_parts.juju_version`. Major alone when no minor is present (or
    none parses). Defaults to "4" when the extraction didn't pin a version --
    the common case in practice.

    This is the *pin*, and nothing downstream of `prepare` should use it
    directly: the substrate is whatever `_juju_track()` resolves the pin to,
    which is not the same string whenever the snap has retired the pinned
    track. `_juju_track()` is built on this."""
    match = _JUJU_VERSION_RE.match((hypothesis.moving_parts.juju_version or "").strip())
    if not match:
        return default
    major, minor = match.group("major"), match.group("minor")
    return f"{major}.{minor}" if minor else major


# Juju snap tracks with a populated `stable` risk (check with `snap info
# juju`). `_juju_channel()` only appends `/stable`, so listing a track without
# one, such as `4.1`, would build a channel that doesn't exist and fail
# `prepare`, whereas an unlisted track falls back safely.
_KNOWN_JUJU_TRACKS = ("2.9", "3", "3.6", "4.0")

# The `stable` version each track above carried when checked, so a test can
# assert that every listed track has one.
_KNOWN_JUJU_TRACK_STABLE_VERSIONS = {
    "2.9": "2.9.60",
    "3": "3.6.28",
    "3.6": "3.6.28",
    "4.0": "4.0.14",
}

# Used when the extraction pins nothing. Explicit so that the substrate can't
# move when concierge's own default changes. `RunResult.observed_juju_version`
# puts the version used in the comment, so a mismatch with the reporter's
# juju is disclosed.
_DEFAULT_JUJU_TRACK = "4.0"
_DEFAULT_JUJU_CHANNEL = f"{_DEFAULT_JUJU_TRACK}/stable"


def _juju_channel(hypothesis: Hypothesis, default: str = _DEFAULT_JUJU_CHANNEL) -> str:
    """Snap channel for `concierge prepare --juju-channel`, from
    `moving_parts.juju_version`.

    Without this, the extractor's pin would reach only the pebble-socket
    lookup and never the substrate, so `prepare` would provision whatever
    concierge defaulted to and a version-dependent bug could return
    `DID_NOT_REPRODUCE` purely because the substrate was the wrong juju.

    A pinned `major.minor` wins when the juju snap publishes that track,
    falling back to the bare major track and then to `default` -- so
    `"3.6.27"` selects `3.6/stable`, `"3.4"` (a real juju version, but not
    its own track) selects `3/stable`, and an unparseable or absent pin
    selects `default`. The fallbacks are deliberate: `prepare` against a
    channel the snap has never published fails outright, which costs the whole
    run rather than the precision of one version segment.
    """
    return f"{_juju_track(hypothesis, default.split('/', 1)[0])}/stable"


def _juju_track(hypothesis: Hypothesis, default: str = _DEFAULT_JUJU_TRACK) -> str:
    """The juju snap track `prepare` will actually install for `hypothesis`.

    The one place a pin turns into a substrate. `_juju_channel()` appends
    `/stable` to it, and `_pebble_socket_path()` keys off it, so the socket
    the stimulus talks to is chosen by the same input as the juju it is
    talking to. Otherwise `"3.4"` would provision `3/stable` (juju 3.6) and
    then aim the stimulus at the pre-3.6 legacy socket, a combination that
    fails on every substrate because it describes none of them."""
    match = _JUJU_VERSION_RE.match((hypothesis.moving_parts.juju_version or "").strip())
    if not match:
        return default
    major, minor = match.group("major"), match.group("minor")
    if minor and f"{major}.{minor}" in _KNOWN_JUJU_TRACKS:
        return f"{major}.{minor}"
    if major in _KNOWN_JUJU_TRACKS:
        return major
    return default


def _unit_expr(app: str, target: str) -> str:
    """Command substitution that resolves to the application's actual unit name.

    Written as an inline `$(...)` in the unit *argument* position rather than
    a `UNIT=...` prelude, so every planned command still begins with an
    executable -- `runnability.assess()` reads a leading shell assignment as
    prose, and rightly: "starts with something runnable" is most of what it
    can check.

    A hardcoded `<app>/0` holds only for an application's *first* deploy:
    juju continues an application's unit sequence across
    remove-and-redeploy, so on a substrate that has run this issue before,
    `wait` would sit on `repro-i2639/0` while the live unit was, say,
    `repro-i2639/4` and time out -- reported as an infrastructure failure
    about a charm that had in fact deployed perfectly well. The `cleanup`
    step makes redeploy the normal case rather than an unusual one.

    Each step is its own `bash -c`, so the prelude is repeated per command
    rather than exported once.
    """
    query = (
        'import json,sys;u=json.load(sys.stdin)["applications"]["%s"]["units"];'
        'print(sorted(u,key=lambda n:int(n.split("/")[1]))[0])' % app
    )
    return f"$(juju status -m {target} --format=json | python3 -c '{query}')"


def _application_name(hypothesis: Hypothesis) -> str:
    """Per-issue juju application name for the scratch branches.

    Every hyphen-separated segment of a juju application name must contain at
    least one letter, so the issue number needs the same non-numeric prefix
    `render.py` puts on charm names.
    """
    return f"repro-i{hypothesis.issue_number}"


@dataclass(frozen=True)
class PlannedCommand:
    """One step of a dry-run plan (the exact command sequence a branch would
    execute, without executing it). `step`
    names which part of the branch this belongs to; `command` is the fully
    resolved string a shell would receive."""

    step: str
    command: str


def _wait_command(target: str, app: str, unit: str = "", container: str | None = None) -> str:
    """Poll `juju status` until every unit of `app` is alive and active.

    Not `juju wait-for application ... --query=... --timeout=15m`, which
    reads better and does exactly the right thing -- on juju 3. **juju 4
    does not have the command at all**: on 4.0.14 `juju wait-for` is
    `ERROR unrecognized command`, and `juju help commands` lists nothing
    matching `wait`. Since `_DEFAULT_JUJU_CHANNEL` is `4.0/stable`, that
    would make the *default* path abort at `wait` on every scratch run, one
    step short of ever running a stimulus. No fixture can see that, because
    `FixtureRunnerSeam` never shells out.

    Polling `juju status --format=json` instead is version-portable, so this
    stays one command rather than a juju-major branch: the run's substrate is
    whatever `_juju_channel()` provisioned, and a wait step that works on both
    majors cannot drift out of step with that choice the way a second version
    map would.

    Waits on `workload-status`, not on the agent, because an idle agent is
    reached *before* the install and start hooks run. On the *application*, not on `<app>/0`
    -- the unit index is not predictable once an application has been
    redeployed. Empty `units` counts as not-ready, so a status read taken
    before the unit exists keeps polling rather than passing.

    **Active is not the same as exec-able.** When `container` is given, the
    step then polls `juju ssh --container <c> <unit> true` until it succeeds.
    An active unit whose workload container juju will not exec into yet is a
    real state, and it is the one the very next step walks into: on juju
    4.0 a unit can go active and its stimulus die a fraction of a second
    later on `ERROR container "workload" not running`. Waiting on the
    charm's own status cannot see that -- the charm is up; it is `juju ssh`
    into the *workload* that is not ready -- so the only sound check is the
    operation the stimulus is about to perform. Both polls share the one
    `UNIT_READY_TIMEOUT` budget.
    """
    ready = (
        "import json,sys;"
        'u=(json.load(sys.stdin).get("applications",{}).get("' + app + '",{}).get("units") or {});'
        'sys.exit(0 if u and all(x.get("workload-status",{}).get("current")=="active" '
        'and x.get("life","alive")=="alive" for x in u.values()) else 1)'
    )
    active = (
        f"for _ in $(seq 1 {UNIT_READY_POLLS}); do "
        f"juju status -m {target} {app} --format=json 2>/dev/null | python3 -c '{ready}' && break; "
        f"sleep {UNIT_READY_POLL_INTERVAL_S}; done; "
        f"juju status -m {target} {app} --format=json 2>/dev/null | python3 -c '{ready}' || "
        f"{{ echo 'timed out after {UNIT_READY_TIMEOUT} waiting for {app} units to reach active' >&2; exit 1; }}"
    )
    if not container:
        return active
    return (
        f"{active}; "
        f"for _ in $(seq 1 {UNIT_READY_POLLS}); do "
        f"juju ssh -m {target} --container {container} {unit} true >/dev/null 2>&1 && exit 0; "
        f"sleep {UNIT_READY_POLL_INTERVAL_S}; done; "
        f"echo 'timed out after {UNIT_READY_TIMEOUT} waiting for container {container} to accept exec' >&2; exit 1"
    )


def _control_signal_command(target: str, unit: str, expected_signal: str) -> str:
    """Poll `juju status` until `expected_signal` is visible, print the status
    the classifier will read, and fail loudly if it never turns up.

    Appended to the control's `pebble notify` so the control waits for its own
    effect instead of racing it -- see `CONTROL_SIGNAL_TIMEOUT_S` for the
    measurement that motivated this and for what losing the race costs.

    Same shape as `_wait_command()`, deliberately: a bounded `for _ in
    $(seq 1 N)` loop that breaks on the condition, then a re-check that
    `echo`es to stderr and `exit 1`s when the condition never held. This is
    the module's one polling idiom and the control uses it rather than a
    second one of its own.

    Three things it has to do at once, which is why the tail is not simply the
    loop's own check:

    - **Print the status.** `classifier.classify()`'s rung 5 matches
      `expected_signal` against the control's captured stdout/stderr, and
      `composer._control_output()` renders that same text into the comment as
      the evidence for the claim. A `grep -q` alone would poll correctly and
      hand both of them an empty control.
    - **Fail when it times out.** A control that never fired is not a control,
      and rung 5's whole job is telling "the bug is real" apart from "the
      observer is broken". Exiting 0 on a timed-out poll would report the
      second as the first. `control` is not in `PREREQUISITE_STEPS`, so the
      non-zero exit records itself and stops nothing -- it is the last step in
      the sequence anyway.
    - **Agree with the classifier about what "the signal appeared" means.**
      The poll greps for the same literal string rung 5 substring-matches, so
      a poll that passes and a classifier that matches cannot disagree.
      `grep -F` (fixed string, never a pattern) and `shlex.quote` for the same
      reason: `expected_signal` is free text an LLM wrote, so it is data here
      and not shell or regex.

    The final read is captured once into `$status`, printed, and then grepped
    from that same variable rather than re-read. So `exit 0` and "the printed
    text contains the signal" are one fact and not two, and the classifier's
    exit-code check and its substring check cannot reach opposite conclusions
    about the same control. It also keeps `_unit_expr()`'s inline `$(juju
    status --format=json | python3 ...)` substitution to two occurrences in
    the step instead of three, on a command string that is already long
    enough to be read carefully rather than skimmed.
    """
    signal = shlex.quote(expected_signal)
    read = f"juju status -m {target} {unit}"
    # "The unit status", not "juju status": a test checks that every
    # `;`-separated fragment mentioning `juju ` names the controller and model,
    # and it can't tell an `echo` from a command.
    return (
        f"for _ in $(seq 1 {CONTROL_SIGNAL_POLLS}); do "
        f"{read} 2>/dev/null | grep -qF -- {signal} && break; "
        f"sleep {CONTROL_SIGNAL_POLL_INTERVAL_S}; done; "
        f"status=$({read} 2>&1); printf '%s\\n' \"$status\"; "
        f"printf '%s\\n' \"$status\" | grep -qF -- {signal} || "
        f"{{ echo 'timed out after {CONTROL_SIGNAL_TIMEOUT_S}s waiting for the control signal' {signal} "
        f"'to appear in the unit status' >&2; exit 1; }}"
    )


def _scratch_sequence(
    hypothesis: Hypothesis, surface: SurfaceInference | None, context: dict, *, prepare_flag: str
) -> list[PlannedCommand]:
    """The ordered, fully-resolved command sequence for a scratch-charm
    branch (k8s or lxd): prepare -> pack -> deploy -> stimulus (only when
    `surface.pebble_service` names a user *and* command) -> status ->
    debug-log -> control (only when `surface.expected_signal` is set and
    there's a real stimulus to control against: silence is only meaningful
    if the control fired).

    Pure -- no subprocess calls, no juju/charmcraft/concierge required to
    call this. Both `SubprocessRunnerSeam._run_scratch_branch` (which
    executes each step in turn) and `build_plan()` (the dry-run mode) are
    built on this single sequence, so a reviewed dry-run plan and a real run
    can never silently diverge.

    **Deliberately does not use `hypothesis.commands`.** Same reasoning
    `runner_stage.py`'s `COMMAND_EXECUTING_BRANCHES` comment gives: a
    scratch-charm hypothesis's `commands[]` is written against the
    *reporter's* charm/units (`#2639`'s hand extraction has literal
    `<k8s-charm-...>`/`<unit>` placeholders a shell reads as redirects), not
    the "repro" app this branch deploys. The stimulus is built from
    `surface.pebble_service` instead, which surface inference is expected to
    hand over as concrete strings, never a bracket-shaped placeholder.
    """
    charm_dir = context.get("charm_dir")
    # A per-issue application name, so that on a reused substrate a previous
    # run's unit is never mistaken for this one's.
    app = _application_name(hypothesis)
    target = f"{CONTROLLER_BY_PREPARE_FLAG[prepare_flag]}:{CONCIERGE_MODEL}"
    # Resolved at execution time by `_unit_expr()`: unit indexes change on
    # redeploy.
    unit = _unit_expr(app, target)
    # `render.py` keys machine-vs-k8s and its `<container>-image` resources off
    # this field, so the `--resource` flags key off it too.
    pebble_container = (dict(surface.pebble_service) if surface else {}).get("container")
    resource_args = f" --resource {pebble_container}-image={DEFAULT_WORKLOAD_IMAGE}" if pebble_container else ""
    steps = [
        # The only branches that need concierge install it, immediately before
        # using it. See `INSTALL_CONCIERGE_COMMAND`.
        PlannedCommand("install-concierge", INSTALL_CONCIERGE_COMMAND),
        # `--juju-channel` explicitly, never concierge's own default.
        PlannedCommand(
            "prepare",
            f"sudo concierge prepare --juju-channel {_juju_channel(hypothesis)} -p {prepare_flag}",
        ),
        # `-o` as well as `-p`: `charmcraft pack -p <dir>` writes the `.charm` to the
        # current directory, but deploy globs `<dir>/*.charm`.
        PlannedCommand(
            "pack",
            # `rm -f` first, so a stale artefact can't make the deploy glob match twice.
            f"rm -f {charm_dir}/*.charm && charmcraft pack -p {charm_dir} -o {charm_dir}"
            if charm_dir
            else "charmcraft pack",
        ),
        # Remove a leftover application from an earlier run of the same issue
        # (`juju deploy` has no `--replace`), polling because removal is async.
        # Always exits 0, since there is usually nothing to remove.
        PlannedCommand(
            "cleanup",
            f"juju remove-application -m {target} {app} --destroy-storage --no-prompt >/dev/null 2>&1; "
            f"for _ in $(seq 1 {APP_REMOVAL_POLLS}); do "
            f"juju status -m {target} --format=json 2>/dev/null | grep -q '\"{app}\"' || break; "
            f"sleep {APP_REMOVAL_POLL_INTERVAL_S}; done; true",
        ),
        PlannedCommand(
            "deploy",
            f"juju deploy -m {target} {charm_dir}/*.charm {app}{resource_args}"
            if charm_dir
            else f"juju deploy -m {target} ./*.charm {app}{resource_args}",
        ),
    ]

    # `juju deploy` returns before the unit is up. Wait for workload status
    # `active`, which `render.py` sets from `start`; an idle agent is reached
    # before the install and start hooks run.
    steps.append(PlannedCommand("wait", _wait_command(target, app, unit, pebble_container)))

    pebble = dict(surface.pebble_service) if surface else {}
    stimulus_user = pebble.get("user")
    stimulus_pebble_command = pebble.get("command")
    if stimulus_user and stimulus_pebble_command:
        steps.append(
            PlannedCommand(
                "stimulus",
                as_user_in_container_command(
                    unit=unit,
                    container=pebble.get("container"),
                    user=stimulus_user,
                    juju_track=_juju_track(hypothesis),
                    pebble_command=stimulus_pebble_command,
                    model=target,
                ),
            )
        )

    steps.append(PlannedCommand("status", f"juju status -m {target} {unit}"))
    steps.append(PlannedCommand("debug-log", f"juju debug-log -m {target} --include {unit} --replay"))

    if surface is not None and surface.expected_signal and stimulus_user and stimulus_pebble_command:
        # The control reruns the stimulus as root, which should always produce the
        # signal, to tell a real reproduction from a broken observer. Only built when
        # there is a real stimulus to control against.
        control_command = as_user_in_container_command(
            unit=unit,
            container=pebble.get("container"),
            user="root",
            juju_track=_juju_track(hypothesis),
            pebble_command=stimulus_pebble_command,
            model=target,
        )
        # The status read polls for the signal rather than following the
        # notify immediately -- see `_control_signal_command()` and
        # `CONTROL_SIGNAL_TIMEOUT_S` for the race this closes.
        steps.append(
            PlannedCommand(
                "control",
                f"{control_command}; {_control_signal_command(target, unit, surface.expected_signal)}",
            )
        )

    return steps


def _clone_sequence(hypothesis: Hypothesis, context: dict) -> list[PlannedCommand]:
    """The `k8s-clone` branch's sequence: clone the issue's own repo, then run
    the reporter's remaining commands against the checkout.

    Split out of `_run_k8s_clone`/`build_plan` -- which each built it
    separately -- for the same reason `_scratch_sequence()` exists: a reviewed
    dry-run plan and a real run must not be able to diverge.

    The clone URL comes from `context["repo"]`, the issue's own repository,
    deliberately not from `moving_parts.ci_run_url`. That URL decides only
    *routing*, and `choose_branch()` keys on any GitHub Actions URL in the
    body rather than one belonging to this repo -- `#1329`'s matched URL is a
    run in `canonical/mysql-k8s-operator`.
    """
    repo_url = f"https://github.com/{context.get('repo', 'canonical/operator')}"
    steps = [PlannedCommand("clone", f"git clone {repo_url}")]
    steps += [PlannedCommand(f"run[{i}]", c) for i, c in enumerate(hypothesis.commands[1:], start=1)]
    return steps


def build_plan(
    *, branch: str, hypothesis: Hypothesis, surface: SurfaceInference | None, context: dict
) -> list[PlannedCommand]:
    """Dry-run plan mode: the exact, fully
    resolved command sequence `SubprocessRunnerSeam.run()` would execute for
    `branch`, without executing anything. No juju/k8s/charmcraft/subprocess
    involved -- pure string construction, so it's reachable on a box with
    none of those installed. See `harness/dry_run.py` for the CLI wrapper.
    """
    if branch == "none":
        return [PlannedCommand(f"run[{i}]", c) for i, c in enumerate(hypothesis.commands)]
    if branch == "k8s-scratch":
        return _scratch_sequence(hypothesis, surface, context, prepare_flag="k8s")
    if branch == "lxd-scratch":
        # Must match `_run_lxd_scratch`'s flag exactly -- concierge has no
        # `lxd` preset. These two call sites are the one place the dry-run
        # plan and the real run can still diverge, so a test pins them.
        return _scratch_sequence(hypothesis, surface, context, prepare_flag="machine")
    if branch == "k8s-clone":
        return _clone_sequence(hypothesis, context)
    raise ValueError(f"unknown branch {branch!r}")


class SubprocessRunnerSeam:
    """Real implementation. The test suite has no juju/concierge/network, so
    the commands it runs are never exercised against real infrastructure
    there -- see `tests/test_seams_runner.py` for what *is* covered
    (command sequencing with `subprocess.run` mocked)."""

    # `resolve_symbol()`'s probe exit codes. "Gone" and "nothing here to ask"
    # are different facts, and only the first is evidence.
    _SYMBOL_RESOLVED = 0
    _SYMBOL_ABSENT = 1
    _SYMBOL_UNKNOWABLE = 2

    def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
        # `symbol_anchor` mixes module and attribute segments with no marker, so
        # import the longest importable prefix and getattr() the rest.
        #
        # Only "a prefix imports but the rest is missing" counts as absent. "Nothing
        # imports" is the normal case in a venv without `ops`, and reading it as
        # absence would skip every anchored hypothesis as stale. Both `except` clauses
        # are broad because an uncaught exception exits 1, the "absent" code.
        script = (
            "import importlib\n"
            f"parts = {symbol_anchor!r}.split('.')\n"
            f"status = {self._SYMBOL_UNKNOWABLE}\n"
            "for i in range(len(parts), 0, -1):\n"
            "    try:\n"
            "        obj = importlib.import_module('.'.join(parts[:i]))\n"
            "    except Exception:\n"
            "        continue\n"
            "    try:\n"
            "        for part in parts[i:]:\n"
            "            obj = getattr(obj, part)\n"
            f"        status = {self._SYMBOL_RESOLVED}\n"
            "    except AttributeError:\n"
            f"        status = {self._SYMBOL_ABSENT}\n"
            "    except Exception:\n"
            f"        status = {self._SYMBOL_UNKNOWABLE}\n"
            "    break\n"
            "raise SystemExit(status)\n"
        )
        try:
            result = subprocess.run(
                ["python3", "-c", script], capture_output=True, timeout=30, cwd=context.get("cwd")
            )
        except (OSError, subprocess.TimeoutExpired):
            # No usable interpreter, or one that hung: "cannot tell", so don't skip.
            return True
        return result.returncode != self._SYMBOL_ABSENT

    def run(
        self,
        *,
        branch: str,
        hypothesis: Hypothesis,
        surface: SurfaceInference | None,
        context: dict,
    ) -> RunResult:
        if branch == "none":
            return self._run_none(hypothesis, context)
        if branch == "k8s-scratch":
            return self._run_k8s_scratch(hypothesis, surface, context)
        if branch == "lxd-scratch":
            return self._run_lxd_scratch(hypothesis, surface, context)
        if branch == "k8s-clone":
            return self._run_k8s_clone(hypothesis, context)
        # Only reachable if `choose_branch()` grows a branch this doesn't handle,
        # which `test_every_choose_branch_output_is_dispatchable` catches.
        raise ValueError(f"unknown branch {branch!r}")

    def _run_none(self, hypothesis: Hypothesis, context: dict) -> RunResult:
        workdir = context.get("workdir", ".")
        # uv picks its environment from `VIRTUAL_ENV`/`UV_PROJECT_ENVIRONMENT` before
        # the working directory, and the harness's own `uv run` may have set both.
        # See `scratch_command_env()`.
        env = scratch_command_env(context.get("uv_project_environment"))
        commands = []
        for command in hypothesis.commands:
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    ["bash", "-c", command],
                    cwd=workdir,
                    capture_output=True,
                    timeout=PER_COMMAND_TIMEOUT_S,
                    text=True,
                    env=env,
                )
                commands.append(
                    CommandResult(
                        command=command,
                        exit_code=proc.returncode,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        elapsed_s=round(time.monotonic() - started, 3),
                    )
                )
            except subprocess.TimeoutExpired:
                commands.append(
                    CommandResult(
                        command=command,
                        exit_code=124,
                        stderr="timed out",
                        elapsed_s=round(time.monotonic() - started, 3),
                    )
                )
        return RunResult(
            hypothesis_number=hypothesis.issue_number,
            branch="none",
            commands=commands,
            # No `observed_juju_version`: this branch never touches juju. It does
            # resolve an `ops`, the one the test imports.
            observed_ops_version=_resolved_ops_version(commands),
        )

    def _run_k8s_scratch(
        self, hypothesis: Hypothesis, surface: SurfaceInference | None, context: dict
    ) -> RunResult:
        """The k8s-substrate scratch-charm branch.

        Full sequence, modelled on reproducing #2639 by hand on real k8s:
        `concierge prepare -p k8s` -> `charmcraft pack` -> `juju
        deploy` -> stimulus (create the pebble-notify user if missing, run
        the notify command as them inside the workload container) ->
        diagnostics (`juju status`, `juju debug-log --replay`) -> a control
        run as root when `surface.expected_signal` is set, so silence in the
        main run can be told apart from a broken observer (the classifier's
        *reproduced (positive-signal-absent)* rung needs both halves).
        `context["charm_dir"]` is the directory `scaffold.render_charm()`
        produced.

        Sequence-building is shared with `_run_lxd_scratch` via
        `_scratch_sequence()` -- the only branch-specific bit is the
        `concierge prepare` substrate flag and the `RunResult.branch` label.
        See `_scratch_sequence()`'s docstring for why `hypothesis.commands`
        is never used here (the reporter's own commands carry unfilled
        `<k8s-charm-...>`/`<unit>` placeholders that only make sense against
        the *reporter's* deployment, not this branch's "repro" app).

        Not exercised by the test suite beyond mocked subprocess calls (no
        juju/concierge/charmcraft/network there) -- see harness/README.md's
        "What's still open" and `build_plan()` / `harness/dry_run.py` for
        the dry-run mode that *can* be reviewed offline.
        """
        return self._run_scratch_branch(hypothesis, surface, context, prepare_flag="k8s", branch="k8s-scratch")

    def _run_lxd_scratch(
        self, hypothesis: Hypothesis, surface: SurfaceInference | None, context: dict
    ) -> RunResult:
        """Machine-substrate counterpart of `_run_k8s_scratch`: an LXD
        instance for `substrate: lxd` (the corpus's real #2107 extraction is
        one).

        Shares its command sequence with `_run_k8s_scratch` via
        `_scratch_sequence()` (concierge prepare -> charmcraft pack -> juju
        deploy -> stimulus -> diagnostics -> control); see that function's
        docstring for the conservative-reading calls made for a machine
        substrate (no k8s-style Pebble *container*, so
        `container=None` throughout; no stimulus invented when `surface`
        carries nothing to hang one on).

        `context["charm_dir"]` must already hold a rendered scratch charm
        (`scaffold.render_charm()`, called by `pipeline.py` for both
        scratch branches) -- this seam only packs and deploys it.

        Not exercised by the test suite beyond mocked subprocess calls (no
        juju/concierge/charmcraft/network there) -- see harness/README.md
        for what is and isn't verified.
        """
        # `-p machine`: concierge has no `lxd` preset.
        return self._run_scratch_branch(hypothesis, surface, context, prepare_flag="machine", branch="lxd-scratch")

    def _run_scratch_branch(
        self,
        hypothesis: Hypothesis,
        surface: SurfaceInference | None,
        context: dict,
        *,
        prepare_flag: str,
        branch: str,
    ) -> RunResult:
        """Execute `_scratch_sequence()`'s plan for real, one command at a
        time, capturing exit code/stdout/stderr and turning a timeout into a
        `CommandResult` (exit 124) rather than letting it raise -- same
        convention `_run_none`/`_run_k8s_clone` use. The `control` step (if
        the sequence has one) is split out into `RunResult.control` rather
        than `RunResult.commands`, matching what `classifier.py` expects.

        Stops at the first failed `PREREQUISITE_STEPS` command and records
        what it abandoned, rather than running `juju deploy` against a charm
        that never packed and then reading `juju status`'s exit 0 on an
        empty model as though it said something about the bug."""
        sequence = _scratch_sequence(hypothesis, surface, context, prepare_flag=prepare_flag)
        commands, control, aborted_at_step, skipped_steps = self._execute_sequence(sequence)
        return RunResult(
            hypothesis_number=hypothesis.issue_number,
            branch=branch,
            commands=commands,
            control=control,
            aborted_at_step=aborted_at_step,
            skipped_steps=skipped_steps,
            # Only the scratch branches provision juju and pack a charm, so only they
            # record these versions.
            observed_juju_version=_observed_juju_version(),
            observed_ops_version=_packed_ops_version(commands),
        )

    @staticmethod
    def _execute_sequence(
        sequence: list[PlannedCommand], *, cwd: str | None = None
    ) -> tuple[list[CommandResult], CommandResult | None, str | None, list[str]]:
        """Run a planned sequence one command at a time, stopping at the first
        failed `PREREQUISITE_STEPS` command and recording what was abandoned.

        Shared by every branch that runs a planned sequence. Keeping one loop
        is what stops the branches drifting apart: a separate loop with no
        `step`, no `aborted_at_step`, and every command run regardless of
        exit code is exactly the defect this prevents.
        """
        commands: list[CommandResult] = []
        control: CommandResult | None = None
        aborted_at_step: str | None = None
        skipped_steps: list[str] = []
        for index, planned in enumerate(sequence):
            timeout = STEP_TIMEOUTS_S.get(planned.step, PER_COMMAND_TIMEOUT_S)
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    ["bash", "-c", planned.command], capture_output=True, timeout=timeout, text=True, cwd=cwd
                )
                result = CommandResult(
                    command=planned.command,
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    step=planned.step,
                    elapsed_s=round(time.monotonic() - started, 3),
                )
            except subprocess.TimeoutExpired:
                result = CommandResult(
                    command=planned.command,
                    exit_code=124,
                    stderr=f"timed out after {timeout}s",
                    step=planned.step,
                    elapsed_s=round(time.monotonic() - started, 3),
                )
            if planned.step == "control":
                control = result
            else:
                commands.append(result)
            if result.exit_code != 0 and planned.step in PREREQUISITE_STEPS:
                aborted_at_step = planned.step
                skipped_steps = [p.step for p in sequence[index + 1 :]]
                break
        return commands, control, aborted_at_step, skipped_steps

    def _run_k8s_clone(self, hypothesis: Hypothesis, context: dict) -> RunResult:
        """Clone the issue's repo, then run the reporter's remaining commands
        against the checkout.

        Built on `_clone_sequence()` and `_execute_sequence()`, so this
        branch gets what the scratch branches get: named steps, abort at the
        first failed prerequisite, and a record of what was skipped. Without
        them, a failed `git clone` would leave the rest running in an empty
        directory and the classifier -- which needs `aborted_at_step` to
        reach rung 0 -- would score their exit codes as evidence about the
        bug.
        """
        sequence = _clone_sequence(hypothesis, context)
        commands, _control, aborted_at_step, skipped_steps = self._execute_sequence(
            sequence, cwd=context.get("workdir", ".")
        )
        return RunResult(
            hypothesis_number=hypothesis.issue_number,
            branch="k8s-clone",
            commands=commands,
            aborted_at_step=aborted_at_step,
            skipped_steps=skipped_steps,
        )


class FixtureRunnerSeam:
    """Replays captured command output from `fixtures/runs/<n>.json`."""

    def __init__(self, fixtures_dir: Path, symbols_path: Path | None = None):
        self.fixtures_dir = Path(fixtures_dir)
        self._symbols = {}
        symbols_path = symbols_path or (self.fixtures_dir / "symbols.json")
        if symbols_path.exists():
            self._symbols = json.loads(symbols_path.read_text())

    def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
        # Absent from the fixture table => assume it resolves cleanly
        # (the common case); only hypotheses that specifically calibrate
        # the "symbol doesn't exist on main" trigger get an explicit entry.
        return self._symbols.get(symbol_anchor, True)

    def run(
        self,
        *,
        branch: str,
        hypothesis: Hypothesis,
        surface: SurfaceInference | None,
        context: dict,
    ) -> RunResult:
        path = self.fixtures_dir / "runs" / f"{hypothesis.issue_number}.json"
        if not path.exists():
            raise FileNotFoundError(f"no run fixture recorded for issue #{hypothesis.issue_number} ({path})")
        return RunResult.from_dict(json.loads(path.read_text()))
