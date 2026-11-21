"""The juju/charmcraft reproduction runner seam.

Two implementations:

- `SubprocessRunnerSeam` — shells out to `concierge`/`charmcraft`/`juju`
  for real. Never exercised against the real tools by the test suite (no
  juju, no concierge, no network) — the code path live runs use.
  `tests/test_seams_runner.py` does exercise its command-building logic
  with `subprocess.run` mocked, which pins the *decisions* (sequence,
  substitutions, non-zero/timeout handling) without proving the commands
  themselves work against real infrastructure.
- `FixtureRunnerSeam` — replays captured command output, recorded as
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

# How long to wait for the deployed unit's agent to settle before
# stimulating it. Generous: on a contended box the pod pull plus agent
# install is minutes, and the cost of waiting is wall-clock while the cost
# of not waiting is a run that reports a verdict about nothing.
UNIT_READY_TIMEOUT = "15m"
UNIT_READY_TIMEOUT_S = 15 * 60
UNIT_READY_POLL_INTERVAL_S = 10
UNIT_READY_POLLS = UNIT_READY_TIMEOUT_S // UNIT_READY_POLL_INTERVAL_S

# Bounded poll for `juju remove-application` to finish -- removal is async and
# `juju deploy` will collide with an application still tearing down.
APP_REMOVAL_POLLS = 60
APP_REMOVAL_POLL_INTERVAL_S = 5

# Bounded poll for the control's own signal to become visible in `juju status`.
#
# The control is a `pebble notify` and a `juju status` read, and it is the
# one step in the sequence that needs the cluster to catch up. Joined by a
# bare `;`, the two race: on a real k8s run, notify-to-visible was under a
# second (the stimulus's `juju ssh` returned, the charm logged the surface
# firing, the hook completed and `juju status` showed the new message within
# about half a second), and an immediate read lands in the same window,
# which is a race rather than a margin.
#
# Losing it is rare but expensive: rung 5 reads the control only when
# `expected_signal` is *absent* from the main run. The only path where the
# control is load-bearing is
# `REPRODUCED_POSITIVE_SIGNAL_ABSENT`, and a read that lands too early there
# turns a comment-worthy outcome into a silent `DID_NOT_REPRODUCE` whose
# stated reason -- "no control confirmed the observer works at all" -- is the
# exact opposite of what happened.
#
# Two minutes against a sub-second latency, which is the posture the other
# polls in this module already take (`UNIT_READY_TIMEOUT` is 15m against a
# unit that typically goes active in about 20s): the control is the last step
# in the sequence, so the cost of waiting is wall-clock nothing else is
# queued behind, and the cost of not waiting is a wrong verdict.
CONTROL_SIGNAL_TIMEOUT_S = 2 * 60
CONTROL_SIGNAL_POLL_INTERVAL_S = 5
CONTROL_SIGNAL_POLLS = CONTROL_SIGNAL_TIMEOUT_S // CONTROL_SIGNAL_POLL_INTERVAL_S

# Every juju command in a scratch sequence names its controller and model
# explicitly. Otherwise each runs against whatever `juju switch` last pointed
# at -- and `concierge prepare` *is* a `juju switch`: a run whose
# extraction came back `substrate: lxd` bootstraps `concierge-lxd` and makes it
# current, so the *next* run, however clearly `substrate: k8s`, silently packs
# a k8s charm and deploys it to LXD. One run's branch choice changing the
# substrate of the next is not a race; it is the ordinary case once a corpus
# contains both substrates, and extraction is unstable enough to produce both
# for one issue.
#
# The names are concierge's defaults, which is an assumption about a tool this
# sequence already shells out to by name. Being wrong here fails loudly on the
# next command; not naming them does not fail at all.
CONTROLLER_BY_PREPARE_FLAG = {"k8s": "concierge-k8s", "machine": "concierge-lxd"}
CONCIERGE_MODEL = "testing"

# A flat 5 minutes is fine for a `juju status` and hopeless for the two
# steps that build things. Cold, `concierge prepare -p k8s` can take over
# 13 minutes, and `charmcraft pack` can blow a 5-minute budget outright
# (exit 124) fetching its LXD build base. Both would time out on every cold
# GHA runner. Values are deliberately
# generous -- the cost of waiting is wall-clock, the cost of timing out is
# a run that reports a verdict about a bug it never tested.
#
# `wait` has to be here: the command it runs is `juju wait-for ...
# --timeout=15m`, and the flat 5-minute default would kill the subprocess at
# 5, so juju's own timeout could never be reached and a unit that was merely
# slow would come back as `'wait' failed (timed out): timed out after 300s`.
# The budget here has to be *longer* than the timeout embedded in the
# command, or the inner one is dead code.
#
# `cleanup` polls for an asynchronous `juju remove-application` to finish, so
# it needs the same order of budget as the removal itself.
STEP_TIMEOUTS_S = {
    # `snap install` typically takes 14-21s on a GHA runner, with the
    # occasional outlier near a minute. Ten minutes is a backstop against a
    # wedged snapd, not a budget: a store error returns in seconds.
    "install-concierge": 10 * 60,
    "prepare": 25 * 60,
    "pack": 20 * 60,
    "cleanup": APP_REMOVAL_POLLS * APP_REMOVAL_POLL_INTERVAL_S + 60,
    "deploy": 15 * 60,
    "wait": UNIT_READY_TIMEOUT_S + 60,
    # `control` polls for its own signal, so the rule `wait` is listed here
    # for applies to it too: this budget has to outlast the timeout embedded
    # in the command, or the inner one is dead code and a slow-but-working
    # control comes back as `timed out after 300s` instead. Doubled rather
    # than given a flat margin because each iteration of the poll spends two
    # `juju status` calls of its own -- one resolving the unit name, one
    # reading it -- so the loop's real wall clock runs ahead of the sleep
    # budget it is written to, and by a factor rather than a constant.
    "control": 2 * CONTROL_SIGNAL_TIMEOUT_S + 60,
}

# Steps that have to succeed before anything after them means anything.
# `juju status` and `juju debug-log` both exit 0 against an empty model,
# so without this the sequence sails past a failed pack looking healthy.
#
# `stimulus` is on this list for the same reason the others are, though it
# is not "infrastructure" in the ordinary sense: if the one command that
# provokes the bug did not run, the diagnostics that follow describe a
# charm nothing was done to: a failed stimulus would otherwise still be
# classified as `did_not_reproduce`, a verdict drawn from an experiment that
# never happened.
# `clone` is here for the same reason: without it, a failed `git clone` --
# network, rate limit, a renamed or private repo -- leaves every later
# command running in an empty directory, and the classifier reads their
# exit codes as evidence about the bug (rung 6 scores a non-zero last
# command as `reproduced (weaker)`, which composes a comment).
#
# `install-concierge` is here because `prepare` is: a run whose concierge
# never installed has nothing to prepare with, and the point of putting the
# install in the sequence at all is that a failed one is then recorded as an
# abandoned prerequisite -- `classifier.py` reads that as
# `INFRASTRUCTURE_FAILED`, which composes nothing -- instead of failing a
# workflow step before any harness code runs, as a snap store error would.
#
# `cleanup` is deliberately *not* here: removing an application that isn't
# deployed is the normal case, not a failure.
PREREQUISITE_STEPS = ("install-concierge", "prepare", "pack", "deploy", "wait", "stimulus", "clone")

# Installed here, immediately before the first command that needs it, rather
# than unconditionally by the workflow before the pipeline runs.
#
# Concierge is used by exactly one thing: `sudo concierge prepare`, the first
# step of `_scratch_sequence()`. The `none` and `k8s-clone` branches never
# touch it, and neither does a run that stops at the filter or the in-scope
# gate -- which is most of them. Well under a tenth of runs reach a scratch
# branch, and installing up front would cost every run 14-21s and fail the
# whole job on a snap store error.
#
# Why here and not as a conditional workflow step: gating a workflow step on
# "will a scratch branch run?" means knowing the substrate, which is the
# extraction's output, so the workflow would have to run extraction, stop,
# install, and then resume -- either paying for the extraction twice or
# growing a resume mode. Worse, the extraction is not deterministic (the
# substrate it picks can change between runs), so the second run could
# choose a branch the first did not install for. Putting it in the sequence
# costs nothing, keeps one pipeline invocation, and the dependency ends up
# stated where it exists.
#
# Idempotent by the guard, not by snap's own behaviour: a VM or a developer
# box that already has concierge spends nothing here, and `snap install` on an
# installed snap exits non-zero ("snap ... is already installed"), which as a
# prerequisite step would abort the run.
#
# `which`, not `command -v`, because `dry_run.generate_plan()` runs the whole
# plan through `runnability.assess()` and that gate's `_first_token()` treats
# `command` as a wrapper word, lands on `-v`, and calls the line prose. `which`
# is in its `RUNNER_TOOLS` allowlist and means the same thing here.
INSTALL_CONCIERGE_COMMAND = "which concierge >/dev/null 2>&1 || sudo snap install --classic concierge"

# A k8s scratch charm declares an `oci-image` resource per container (see
# scratch-charm/charm/render.py), and `juju deploy` refuses a local charm
# whose OCI resources aren't supplied: "ERROR local charm missing OCI
# images for: workload-image". The scratch charm doesn't care what the
# image *is* -- juju injects pebble as pid 1 and the stimulus runs
# `pebble notify` inside it -- so any maintained base works.
DEFAULT_WORKLOAD_IMAGE = "ubuntu:24.04"

# juju-version-dependent pebble socket path, for the "run this command as
# user X inside container Y" primitive.
#
# Keyed on the snap **track** `prepare` actually provisions -- a
# `_juju_track()` result -- not on the hypothesis's raw pin. The two disagree
# wherever the juju snap has retired a track: a hypothesis pinning
# `juju_version: "3.4"` selects channel `3/stable`, which installs **juju
# 3.6**, while a pin-keyed lookup would read `"3.4"`, miss the `"3.6"` entry,
# fall through to `_PEBBLE_SOCKET_LEGACY` and aim the stimulus at a socket
# that is not on that substrate. On juju 3.6 the new path records the
# notice, and the legacy path fails with `cannot communicate with server:
# ... socket "/var/lib/pebble/default/.pebble.socket" not found`. One lookup
# off the track keeps the socket and the substrate decided by the same
# input, so they cannot drift apart.
#
# The 3.0-3.5 band is not merely unmeasured, it is **unreachable**: the juju
# snap publishes no 3.0-3.5 track any more (`snap info juju`: `3/stable` is
# 3.6), so no pin in that band can put a pre-3.6 juju on the
# substrate. Below 3.6 the only reachable track is `2.9`.
#
# A track not listed here falls through to `_PEBBLE_SOCKET_LEGACY` in
# `_pebble_socket_path()`, which is a default, not a confirmed data point.
_PEBBLE_SOCKET_NEW = "/charm/container/pebble.socket"
_PEBBLE_SOCKET_LEGACY = "/var/lib/pebble/default/.pebble.socket"
_PEBBLE_SOCKET_BY_JUJU_TRACK = {
    "4": _PEBBLE_SOCKET_NEW,  # seen on 4.0.5, 4.0.12
    "4.0": _PEBBLE_SOCKET_NEW,  # seen on 4.0.14
    "3.6": _PEBBLE_SOCKET_NEW,  # seen on 3.6.27
    "3": _PEBBLE_SOCKET_NEW,  # `3/stable` is 3.6 (seen on 3.6.28)
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


# The two environment variables that decide which virtualenv a `uv` command
# writes to, regardless of which directory it is run in.
#
# `VIRTUAL_ENV` is set by any activated venv, and `UV_PROJECT_ENVIRONMENT`
# by, among others, `tox-uv` -- which points it at the tox environment for
# the whole test run. Inheriting either sends the scratch project's `uv
# add`/`uv sync` into somebody else's environment, which is the same
# contamination `runner_stage.prepare_scratch_project()` exists to stop,
# arriving by a route no amount of workspace layout can close. For example,
# with `UV_PROJECT_ENVIRONMENT` inherited, a scratch sync empties this
# repository's own `.tox/unit` down to the scratch project's four packages
# and `ops`'s tests stop being able to import `websocket`.
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


# The `ops` version `charmcraft pack` actually resolved into the scratch
# charm, read out of the pack step's own log.
#
# The scratch charm's `pyproject.toml` pins `ops~=3.8`, and `charmcraft pack`
# resolves that from PyPI inside its own managed LXD instance, so the `ops`
# in the checked-out tree is never under test. Which `ops` was packed
# survives only as one line of charmcraft's stderr --
#
#     :: + ops==3.8.2
#
# -- which nothing else reads. `_DEFAULT_JUJU_CHANNEL`'s comment states the
# principle for juju: "a reproduction on a version the reporter did not
# mention is disclosed rather than hidden", honoured by
# `RunResult.observed_juju_version`. This is the equivalent for the library
# the issue is actually *about*.
#
# Requiring the literal `ops==` is already enough to ignore the
# `ops-scenario==8.8.0` and `ops-testing==...` lines that sit next to it in
# the same log. The preceding `(?<![\w.-])` covers the case that requirement
# does not: a package whose *name ends in* `ops` (`charmops==1.2.3`) has
# `ops==` as a substring, and without the lookbehind its version would be
# reported as the packed `ops`.
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


# The commands that can put an `ops` on the `none` branch's PATH, matched
# against the command *string* because that branch labels no steps at all
# (`CommandResult.step` is `None` on every one of them -- they are the
# reporter's or the extractor's own commands, not a sequence this seam
# planned). This is the `none`-branch analogue of `_packed_ops_version()`'s
# `step == "pack"` key: both name the one command in the run whose output
# says which `ops` was resolved.
#
# Deliberately not "scan every command's output". A `pytest` failure dump can
# quote a requirements line or a traceback frame containing `ops==`, and
# reading that as the resolved version would disclose a number nothing
# installed. Only an installer's own output is evidence of what is importable.
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
        target repo's checked-out `main` — the skip-when-stale gate's second
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
        # No `--` on the k8s container path. `juju ssh --container c unit --
        # "cmd"` does not consume the separator the way the machine form
        # does -- it forwards it into the container's shell, which dies with
        # `sh: 0: Illegal option --` before running anything. Against a live
        # k8s unit, the identical command with the `--` removed runs and
        # records the notice. The machine form keeps
        # `--`, where it is the documented separator and plain ssh consumes
        # it; that path is untested here, so it is deliberately left alone.
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


# Snap tracks the juju snap actually publishes **at the `stable` risk**, so a
# pin can only ever select a channel that exists. `_juju_channel()` appends
# `/stable` and nothing else, so a track is only usable here if it has a
# populated `stable`; listing one that does not is worse than not listing it,
# because an unlisted track falls back safely and a listed one is built into
# a channel that 404s.
#
# Source: the snap store itself, `curl -sH 'Snap-Device-Series: 16'
# https://api.snapcraft.io/v2/snaps/info/juju` (`default-track: 3`). Tracks
# with a populated `stable`,
# in aggregate and on `amd64` specifically: `2.9` (2.9.60), `3` (3.6.28),
# `3.6` (3.6.28), `4` (4.0.14) and `4.0` (4.0.14). `4.1` publishes `beta`
# (4.1-beta1) and `edge` (4.1-beta3) only; `4.2` and `latest` publish `edge`
# only.
#
# `4.1` is deliberately absent: it has no `stable`, so listing it would turn
# every `4.1*` pin into `--juju-channel 4.1/stable`, `prepare` would exit
# non-zero on `commands[0]`, and the run would end at rung 0
# (`INFRASTRUCTURE_FAILED`, not in `COMMENT_OUTCOMES`): a whole runner slot
# spent, and silence. Leaving it out lets the fallback handle it, as it does
# for `4.2` and `latest`. `canonical/operator`'s own CI does drive concierge
# with `4.1/edge` (`.github/workflows/integration.yaml`), which can make the
# track look well-attested -- but `edge` is not the risk this constant feeds.
#
# Same discipline as `_PEBBLE_SOCKET_BY_JUJU_TRACK` above: only entries with
# evidence behind them, and an unlisted track falls back rather than being
# guessed into a `prepare` that fails on a channel the snap has never had.
_KNOWN_JUJU_TRACKS = ("2.9", "3", "3.6", "4.0")

# The tracks above, as the store reported them, paired with the version each
# one's `stable` risk carried. Kept beside the tuple so
# `test_every_known_juju_track_has_a_populated_stable_risk` can assert the two
# agree: the check that `4.1` would fail.
_KNOWN_JUJU_TRACK_STABLE_VERSIONS = {
    "2.9": "2.9.60",
    "3": "3.6.28",
    "3.6": "3.6.28",
    "4.0": "4.0.14",
}

# What `prepare` uses when the extraction pins nothing -- the common case in
# practice.
#
# The point of naming it here is that it is *explicit*. Left to concierge's
# own default, the channel moves when concierge is upgraded (concierge
# 1.7.0's default is `3/stable`), and a verdict produced on 4.0 can flip to
# `DID_NOT_REPRODUCE` on 3.6 with nothing in the run record saying why. A
# constant here cannot move underneath a verdict when concierge is upgraded.
#
# The value is forward-looking rather than a claim about where charms are
# today: `operator` still lists `3/stable` first in its own matrix. An
# unpinned bug report is most usefully answered against the juju the library
# is heading towards, and a reproduction on a juju the reporter did not name
# is still a genuine reproducer -- `RunResult.observed_juju_version` puts the
# version that produced the verdict in the composed comment, so a mismatch is
# disclosed to the maintainer rather than hidden.
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
    # The timeout message says "the unit status", not "juju status":
    # `test_scratch_run_isolation.py` splits a planned command on `;` and
    # asserts every fragment mentioning `juju ` also names the controller and
    # model, and prose in an `echo` is indistinguishable from a command to
    # that check. Nothing subtle is being worked around -- the phrasing just
    # has to stay out of the way of a test that is guarding something else.
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
    # Per-issue application name, not a fixed "repro". With one name in one
    # model and no teardown, the second run to reach `deploy` on a reused
    # substrate would die with `ERROR cannot add application "repro":
    # application already exists` -- and, worse, leave the *previous* run's
    # unit sitting `active` with a signal-bearing status message that the
    # `status` step would happily read as evidence about this run's bug. A
    # fresh GHA runner never reuses a substrate, but a long-lived VM or
    # developer box does, from the second issue onwards.
    #
    # `i` prefix for the same reason `render.py` uses one on charm names:
    # every hyphen-separated segment of a juju application name has to
    # contain a letter, so a bare issue number is not a legal segment.
    app = _application_name(hypothesis)
    target = f"{CONTROLLER_BY_PREPARE_FLAG[prepare_flag]}:{CONCIERGE_MODEL}"
    # Resolved at execution by `_unit_expr()`, never a hardcoded `<app>/0` --
    # see that helper for what the hardcoded index cost.
    unit = _unit_expr(app, target)
    # `render.py` emits one `<container>-image` oci-image resource per
    # container, and keys machine-vs-k8s off exactly this field, so keying
    # the `--resource` flags off it too keeps the two in step. A machine
    # charm has no containers and needs none.
    pebble_container = (dict(surface.pebble_service) if surface else {}).get("container")
    resource_args = f" --resource {pebble_container}-image={DEFAULT_WORKLOAD_IMAGE}" if pebble_container else ""
    steps = [
        # The only branches that need concierge install it, immediately before
        # using it. See `INSTALL_CONCIERGE_COMMAND`.
        PlannedCommand("install-concierge", INSTALL_CONCIERGE_COMMAND),
        # `--juju-channel` explicitly, never concierge's own default -- see
        # `_juju_channel()` for what the implicit default cost.
        PlannedCommand(
            "prepare",
            f"sudo concierge prepare --juju-channel {_juju_channel(hypothesis)} -p {prepare_flag}",
        ),
        # `-o` as well as `-p`: `charmcraft pack -p <dir>` writes the
        # `.charm` into the *current working directory*, not into `<dir>`,
        # while the deploy step below globs `<dir>/*.charm`. Without `-o`,
        # pack exits 0, the artefact lands next to whatever cwd the harness
        # happened to run from, and deploy fails with "no charm was found".
        PlannedCommand(
            "pack",
            # `rm -f` first: the deploy step globs `<dir>/*.charm`, so a
            # stale artefact from an earlier pack into the same work dir
            # makes the glob match twice and juju fails with `ERROR
            # unrecognized args: ["repro"]`. Re-running into an existing
            # out-dir is ordinary during development.
            f"rm -f {charm_dir}/*.charm && charmcraft pack -p {charm_dir} -o {charm_dir}"
            if charm_dir
            else "charmcraft pack",
        ),
        # Idempotence, not infrastructure: re-running the *same* issue on a
        # substrate that still holds its application is ordinary during
        # development, and `juju deploy` has no --replace. Removal is async,
        # so poll for the application to actually disappear rather than
        # racing the deploy. Always exits 0 -- "there was nothing to remove"
        # is the common case, which is why `cleanup` is not a prerequisite
        # step.
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

    # `juju deploy` returns as soon as the deployment is *requested*; the
    # unit is still "allocating / installing agent" for a minute or more
    # afterwards. Without this wait the stimulus lands on a pod that isn't
    # there yet and fails with `ERROR container for unit "repro/0" is not
    # ready yet`. Waiting on `workload-status == active`, not on the agent:
    # an idle agent is reached *before* the install and start hooks run, so
    # an agent-only wait can stimulate a unit still reporting "(start)
    # installing charm software", recording the notice before the observer
    # exists. `render.py` sets ActiveStatus(READY_STATUS) from
    # `start` precisely so this wait has something sound to wait for.
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
        # Same "stimulate as a user known to work, then check status" shape
        # as #2639's k8s control: a control
        # rerun as root, which should always produce the signal, is what
        # distinguishes a real reproduction from a broken observer. Only
        # built when there's a real pebble command to control against --
        # no control is invented from nothing.
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

    #: `resolve_symbol()`'s probe exits with one of these. Three states, not
    #: two, because "the symbol is gone" and "there is nothing here to ask"
    #: are different facts and only the first one is evidence.
    _SYMBOL_RESOLVED = 0
    _SYMBOL_ABSENT = 1
    _SYMBOL_UNKNOWABLE = 2

    def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
        # Try progressively shorter module prefixes (import the longest
        # importable dotted prefix, then getattr() the remaining parts) --
        # `symbol_anchor` mixes module and attribute segments
        # ("ops.model.Model.get_relation") with no marker for where one
        # ends and the other begins.
        #
        # The probe distinguishes "no prefix imports at all" from "a prefix
        # imports and the rest of the path is not on it", and only the second
        # is reported as a symbol that has gone away. On a GHA runner the
        # first is the *normal* case and says nothing about the target repo:
        # the workflow runs this under `uv run` in `.github/add-reproducer/harness`,
        # whose venv holds pyyaml and pytest and no `ops` at all, so
        # `import ops` fails there for every anchor any extraction can carry.
        # Reported as absence, that would turn `runner_stage.is_stale()`'s
        # second trigger into "is `ops` importable here", which is always no
        # -- so every anchor-carrying hypothesis would be skipped as
        # `skipped_stale` before reaching a runner, silently, whether or not
        # its API surface still existed. The gate fires on evidence of
        # absence; absence of evidence is not it.
        #
        # Both `except` clauses are broad on purpose. An unhandled exception
        # in `python3 -c` exits **1**, which is the code that means "gone", so
        # anything this script lets escape is read as a positive finding the
        # script never made. A package that raises on import (a missing
        # transitive dependency, an import-time side effect) says nothing
        # about whether the symbol exists, and neither does a descriptor that
        # raises from `getattr`.
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
            # No usable interpreter, or one that hung: the same "cannot tell"
            # as an unimportable package, and treated the same way. Every
            # other seam in this module treats infrastructure absence as data
            # rather than as a crash (`_observed_juju_version()`), and letting
            # the exception out of a gate whose whole job is to decide whether
            # to skip would be worse.
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
        # Reachable only if `runner_stage.choose_branch()` grows a branch
        # this dispatch doesn't know about -- see
        # `test_seams_runner.test_every_choose_branch_output_is_dispatchable`,
        # which fails loudly the moment that happens, instead of this
        # raising in production.
        raise ValueError(f"unknown branch {branch!r}")

    def _run_none(self, hypothesis: Hypothesis, context: dict) -> RunResult:
        workdir = context.get("workdir", ".")
        # These commands are `uv init`/`uv add`/`uv run`, and uv decides which
        # environment to write to from `VIRTUAL_ENV`/`UV_PROJECT_ENVIRONMENT`
        # before it ever looks at the working directory. The harness runs
        # under `uv run` itself, so both can be set to the harness's own
        # environment when we get here. See `scratch_command_env()`, and
        # `runner_stage.prepare_scratch_project()` for the other half.
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
            # No `observed_juju_version`: this branch never touches a juju
            # substrate, so there is nothing to ask and a host-wide `juju
            # version` would attribute one to a run that didn't want it.
            # `observed_ops_version` is a different matter -- this branch does
            # resolve an `ops`, and it is the one the test actually imported.
            # See `_resolved_ops_version()`.
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
        # `-p machine`, not `-p lxd`: concierge 1.7.0's presets are
        # crafts/dev/k8s/machine/microk8s and there is no `lxd` one, so
        # `-p lxd` fails with `unknown preset 'lxd'`. Mocked subprocess calls
        # happily accept a preset that does not exist, so only a live run
        # can catch this.
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
            # Only the scratch branches provision a juju substrate via
            # `concierge prepare` -- `none` never touches juju at all, and
            # `k8s-clone` only ever runs the reporter's own commands against
            # a checkout, not against anything this seam provisioned, so
            # recording a host-wide `juju version` there would attribute a
            # substrate to a run that didn't ask for one.
            observed_juju_version=_observed_juju_version(),
            # Same scoping as `observed_juju_version` above, for the same
            # reason: only the scratch branches pack a charm, so only they
            # have a pack log to read a version out of. `k8s-clone` runs the
            # reporter's own commands against a checkout and `none` never
            # packs at all, and attributing an `ops` version to either would
            # name a library neither of them resolved.
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
