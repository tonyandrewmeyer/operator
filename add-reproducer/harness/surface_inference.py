"""Surface inference pass (PLAN.md Approach §3's "distinct surface inference
pass", added between extraction and scaffolding).

A second LLM call, run only for hypotheses that need a deployed
scratch-charm (the k8s-scratch branch — Approach §4), producing the
`relation`/`storage`/`pebble_service`/`ops_api_surface` values
`scaffold.render_charm()` consumes. Kept separate from extraction
deliberately (PLAN.md's ADR-style aside in Approach §3): it keeps the
extractor's contract (issue -> moving parts + commands) stable and lets
this pass reason over `commands`/`observed`/`expected` jointly, the way
spike-step-3's by-hand derivation did.
"""

from __future__ import annotations

import ast
import re

from models import Hypothesis, Issue, SurfaceInference, TestFile
from seams.llm import LLMError, LLMSeam

# Steps §3 fix (2026-07-22): a numeric segment directly after a hyphen
# breaks juju's local-charm URL parser ("local:repro-2639-... is not a
# valid charm url"). Reject at the surface-inference boundary so a
# regression can never reach render.py.
_CHARM_NAME_NUMERIC_SEGMENT_RE = re.compile(r"-\d")

# `<workload-container>`, `<unit>` and friends -- the exact shape #2639's
# own hand extraction carried, which a shell reads as a redirect.
# `runnability.py` guards `hypothesis.commands` against this; the stimulus
# is built from `pebble_service.command` instead and had no such guard.
_PLACEHOLDER_RE = re.compile(r"<[^>]+>")

# `pebble notify --user=...` -- a flag pebble does not have.
_PEBBLE_USER_FLAG_RE = re.compile(r"--user[= ]")
# Flags whose value pebble parses as a Go duration. `--user` was an invented
# flag; this is the next variant along -- a *real* flag with a malformed
# value. A live call returned `/charm/bin/pebble exec --timeout=1 ls`, which
# pebble rejects outright:
#   error: invalid argument for flag `--timeout' (expected time.Duration):
#   time: missing unit in duration "1"
# Same end state as the `--user` case: the stimulus never fires and the run
# reaches a rung having done nothing to the charm. Prompt text alone has now
# failed twice in this spot, so this is a validation rule.
# See `spike-step-5/gate-substrate/RESULT.md` §7.
_DURATION_FLAGS = ("timeout",)
_DURATION_FLAG_RE = re.compile(
    r"--(?P<flag>" + "|".join(_DURATION_FLAGS) + r")(?:=|\s+)(?P<value>\S+)"
)
# Go's time.ParseDuration: one or more decimal-number/unit pairs, e.g. "1s",
# "500ms", "1m30s". A bare number is exactly what pebble refuses.
_GO_DURATION_RE = re.compile(r"^(?:\d+(?:\.\d+)?(?:ns|us|\u00b5s|ms|s|m|h))+$")

_SCHEMA_INSTRUCTIONS = """\
Given the issue and its extracted hypothesis below, infer the scaffolding
parameters a generated scratch charm needs, as JSON matching exactly this
shape:

{
  "charm_name": str,
  "relation": {"name": str, "interface": str, "role": "provider"|"requirer"} | {},
  "storage": {"name": str, "type": "filesystem"|"block", "location": str} | {},
  "pebble_service": {"container": str, "service": str, "command": str, "user": str} | {},
  "ops_api_surface": "update-status"|"config-changed"|"relation-changed"|"pebble-custom-notice"|null,
  "expected_signal": str|null
}

charm_name must not contain a digit immediately after a hyphen (breaks
juju's local-charm URL parser -- use a non-numeric prefix like "i2639"
rather than "2639").

expected_signal is the literal log line / unit-status substring the
generated charm emits when the surface fires (see the charm.py template)
-- used by the classifier's positive-signal-absent rung to tell "the
observer never fired" from "nothing was checked".

pebble_service is the *stimulus*: the thing that will actually be done to
the deployed charm to try to trigger the bug. Nothing else in the run
pokes it. `command` is run verbatim inside the workload container, as
`user`, with PEBBLE_SOCKET already exported -- so it must be a complete,
concrete shell command with no placeholders, no angle brackets, and no
prose. Give the absolute path to the pebble binary. Typical shape:

  "container": "workload",
  "service": "notice-source",
  "command": "/charm/bin/pebble notify canonical.com/repro/notice-1 key=value",
  "user": "_daemon_"

Choose `user` to match the condition the issue describes -- if the report
is about a non-root workload, name that exact account (e.g. "_daemon_",
with the underscores), because running the stimulus as root instead is a
different experiment and will not reproduce the bug.

Any flag pebble parses as a duration -- `--timeout` above all -- needs a
unit: `--timeout=30s`, not `--timeout=30`. Pebble rejects a bare number and
the stimulus never runs.

Put the user in the `user` field only. `pebble notify` has **no `--user`
flag**; the runner applies `user` by creating that account and executing
`command` as it. A `--user=...` argument inside `command` is not a no-op --
pebble rejects the unknown flag and the stimulus never runs.

Set `pebble_service` to `{}` only when the bug genuinely involves no
workload container and no Pebble interaction at all -- for instance a
charm that crashes during update-status on a machine substrate. In that
case also leave `expected_signal` null, since there is no stimulus to
produce a signal.

Whenever you set `expected_signal`, `pebble_service.command` and
`pebble_service.user` are both required and must be non-empty: an
expected signal describes what the stimulus should produce, so promising
a signal without a stimulus asks the run to check for the result of an
experiment it never performs.
"""


class SurfaceInferenceInvalid(ValueError):
    pass


def validate(raw: dict) -> None:
    if "charm_name" not in raw or not isinstance(raw["charm_name"], str):
        raise SurfaceInferenceInvalid("missing/invalid charm_name")
    if _CHARM_NAME_NUMERIC_SEGMENT_RE.search(raw["charm_name"]):
        raise SurfaceInferenceInvalid(
            f"charm_name={raw['charm_name']!r} has a digit after a hyphen -- "
            "juju's local-charm URL parser rejects this (Steps §3 fix, do not regress)"
        )
    surface = raw.get("ops_api_surface")
    if surface not in (None, "update-status", "config-changed", "relation-changed", "pebble-custom-notice"):
        raise SurfaceInferenceInvalid(f"ops_api_surface invalid: {surface!r}")

    # The stimulus half. `pebble_service.command` came back **null on 3 of
    # 3** live calls for #2639 (`first-real-substrate/RESULT.md` finding 5)
    # and nothing here looked at it, so the run deployed a charm, poked
    # nothing, and still got a verdict. The fixture
    # (`fixtures/surface/2639.json`) had been hand-filled from step 4's
    # by-hand walk -- its own `_command_note` says so -- which is why no
    # fixture run ever saw the null.
    #
    # Tied to `expected_signal` rather than demanded unconditionally:
    # that field is what puts the classifier on the
    # positive-signal-absent rung, so "signal promised, nothing done to
    # produce it" is exactly the invalid combination. A crash-shaped
    # hypothesis with no expected_signal (#2107: a machine charm that
    # errors on update-status, `pebble_service: {}`) is a legitimate
    # deploy-only experiment and stays valid.
    pebble = raw.get("pebble_service") or {}
    if raw.get("expected_signal"):
        missing = [f for f in ("command", "user") if not (isinstance(pebble.get(f), str) and pebble[f].strip())]
        if missing:
            raise SurfaceInferenceInvalid(
                f"expected_signal is set but pebble_service {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} missing -- the run would check "
                "for a signal it never did anything to produce"
            )
    command = pebble.get("command")
    # `pebble notify` has no `--user` flag -- `as_user_in_container_command`
    # exists precisely because the user has to be applied by *running as*
    # them. A live call invented `pebble notify --user=_daemon_ ...`
    # (2026-08-18); pebble rejects the unknown flag, so the stimulus silently
    # never fires.
    if isinstance(command, str) and _PEBBLE_USER_FLAG_RE.search(command):
        raise SurfaceInferenceInvalid(
            f"pebble_service.command={command!r} passes a --user flag, which "
            "`pebble notify` does not have; the user belongs in pebble_service.user, "
            "which the runner applies by executing the command as that account"
        )
    if isinstance(command, str) and _PLACEHOLDER_RE.search(command):
        raise SurfaceInferenceInvalid(
            f"pebble_service.command={command!r} carries an unfilled placeholder; "
            "a shell reads '<...>' as a redirect (same trap runnability.py guards "
            "for hypothesis.commands)"
        )
    if isinstance(command, str):
        for match in _DURATION_FLAG_RE.finditer(command):
            value = match.group("value")
            if not _GO_DURATION_RE.match(value):
                raise SurfaceInferenceInvalid(
                    f"pebble_service.command={command!r} passes "
                    f"--{match.group('flag')}={value!r}, which pebble parses as a "
                    "duration and will reject -- it needs a unit, e.g. "
                    f"'--{match.group('flag')}=1s'"
                )


class SurfaceInferrer:
    def __init__(self, llm: LLMSeam):
        self.llm = llm

    def infer(self, issue: Issue, hypothesis: Hypothesis) -> SurfaceInference:
        """One call, plus one re-ask if the result fails validation.

        The re-ask exists because this call is measurably unstable: five
        live runs against the *same* issue returned three different
        `charm_name`s, two different containers, two different pebble users
        and a confidence that moved between `low` and `medium`
        (`first-real-substrate/RESULT.md` finding 7). Against that, one bad
        completion shouldn't cost the issue its whole run -- especially now
        that validation rejects the shape the model actually produced 3/3
        times. Bounded at one retry: this is a prompt-comprehension failure,
        not a transient, so a third attempt buys nothing the second didn't.
        `seams/llm.py`'s own retry is transport-level (HTTP status) and
        never re-asks on a schema problem.
        """
        prompt = self._build_prompt(issue, hypothesis)
        raw = self.llm.complete_json(
            purpose="surface_inference", prompt=prompt, context={"issue_number": issue.number}
        )
        try:
            validate(raw)
        except SurfaceInferenceInvalid as first_error:
            raw = self.llm.complete_json(
                purpose="surface_inference",
                prompt=f"{prompt}\nYour previous answer was rejected: {first_error}\nReturn corrected JSON.",
                context={"issue_number": issue.number, "retry_after": str(first_error)},
            )
            validate(raw)
        return SurfaceInference.from_dict(raw)

    @staticmethod
    def _build_prompt(issue: Issue, hypothesis: Hypothesis) -> str:
        return (
            f"{_SCHEMA_INSTRUCTIONS}\n"
            f"Issue #{issue.number}: {issue.title}\n"
            f"Commands: {hypothesis.commands}\n"
            f"Expected: {hypothesis.expected}\n"
            f"Observed: {hypothesis.observed}\n"
        )


# --- substrate:none test-body synthesis -------------------------------
#
# PLAN.md Approach §3/§4 delta (spike-step-5/2045/RESULT.md "PLAN deltas
# surfaced" §1): Approach §4's original "skip straight to a plain `uv venv`
# + `pytest` run" assumed `commands[]` already contained a runnable pytest
# invocation. `#2045` (an `ops[testing]`/Scenario `os.getcwd()` bug) is the
# case that broke that assumption -- the extraction has enough `expected`/
# `observed` text to reason about, but the issue body never gave a
# self-contained repro snippet, so `commands[]` came back empty.
#
# This synthesises a runnable `ops.testing.Context`/`State` scaffold instead
# of leaving the hypothesis un-runnable, and (2026-07-30) does so with an
# LLM call rather than the original fixed template. `spike-step-5/
# composer-live/RESULT.md`'s addendum, Finding 6, ran the original
# deterministic template's real #2045 output through an actual `uv add
# 'ops[testing]' pytest && uv run pytest` and found it **passes** --
# `ctx.run(ctx.on.update_status(), state_in)` followed by a `# TODO` comment
# has no assertion at all, so every `substrate: none` hypothesis routed
# through synthesis was guaranteed to report `did_not_reproduce` regardless
# of whether the bug was real. `TestFileSynthesizer` (below) closes that by
# asking the model to encode the specific `expected`/`observed` claim as a
# real assertion, behind the same `LLMSeam` pattern `Extractor`/
# `SurfaceInferrer`/`Composer` already use. Any failure (no key, transport
# error, invalid JSON, unusable body) falls back to
# `_loud_failing_test_file()`, not to the old always-passing template --
# an unfilled scaffold must fail loudly and be recognisable as such
# (`classifier.py`'s `UNRUNNABLE_SYNTHESIS_INCOMPLETE` rung), never read as
# a clean "did not reproduce".
_PYTEST_TARGET_RE = re.compile(r"pytest\s+(\S+\.py)\b")


def _heredoc_writes(filename: str, commands: list[str]) -> bool:
    """True iff some command actually materialises `filename` (a `cat >
    <filename> << ...` heredoc -- the #2327/#2341 shape) rather than just
    naming it in a `pytest <filename>` invocation."""
    pattern = re.compile(rf"(?:cat|tee)\s*>\s*{re.escape(filename)}\b.*<<", re.DOTALL)
    return any(pattern.search(c) for c in commands)


# 2026-07-30 live measurement (see spike-step-5/synthesis-live/RESULT.md):
# `write_synthesized_test_file_if_needed()`'s setup-command filter used to
# drop any command containing the substring "pytest" -- which also silently
# dropped `uv add 'ops[testing]' pytest` (the corpus-v2-standard modern
# extraction shape, `uv init` -> `uv add` -> heredoc -> `pytest`), stripping
# pytest out of the venv entirely before the appended, synthesized-test
# invocation ever ran. This regex only matches an actual pytest *invocation*
# (bare `pytest ...` or `uv run pytest ...`, optionally chained with
# `&&`/`;`/`|`), not "pytest" appearing as a trailing package name to
# `uv add`/`pip install`.
_PYTEST_INVOCATION_RE = re.compile(r"(?:^|&&|;|\|)\s*(?:uv\s+run\s+)?pytest\b")


def is_pytest_invocation(command: str) -> bool:
    """True iff `command` actually invokes pytest, as opposed to merely
    mentioning the word "pytest" (e.g. as a package name). See the
    `_PYTEST_INVOCATION_RE` comment above for the bug this closes."""
    return bool(_PYTEST_INVOCATION_RE.search(command))

# The marker `classifier.py`'s `UNRUNNABLE_SYNTHESIS_INCOMPLETE` rung looks
# for. Public (not `_`-prefixed) because classifier.py imports it directly --
# the two must never drift out of sync with each other.
SYNTHESIS_INCOMPLETE_MARKER = "SYNTHESIS_INCOMPLETE"

_LOUD_FAILING_TEMPLATE = '''\
"""Test-file synthesis fallback -- PLAN.md Approach §3 delta closing
spike-step-5/composer-live/RESULT.md's Finding 6. LLM-driven synthesis
(TestFileSynthesizer) was unavailable or returned an unusable body for this
issue, so this scaffold has no real assertion encoding the reported bug.
Unlike the pre-2026-07-30 deterministic template (which always passed,
silently turning every fallback into a false "did not reproduce"), this
fails loudly on purpose, and classifier.py's UNRUNNABLE_SYNTHESIS_INCOMPLETE
rung recognises the marker in the raised message so it is never misread as
a reproduction.
"""

import ops
from ops import testing  # noqa: F401 -- kept for shape-parity with a real synthesized file


class ProbeCharm(ops.CharmBase):
    pass


META = {{"name": "probe-charm"}}


def test_issue_{issue_number}():
    """Expected: {expected}

    Observed: {observed}
    """
    # Fails immediately, before touching Context/State -- there is no
    # issue-specific stimulus to build here, only the fact that synthesis
    # itself did not succeed. See the module docstring above.
    raise AssertionError({message_literal})
'''


class TestFileSynthesisInvalid(ValueError):
    """Raised when a proposed synthesized test-file body fails validation --
    doesn't parse, doesn't import `ops`, has no assertion, or still carries a
    TODO placeholder. Caught by `TestFileSynthesizer.synthesize()` itself,
    never propagated to a caller."""


def _is_raises_call(node: ast.AST) -> bool:
    """True for a `pytest.raises(...)`/`self.assertRaises(...)`-shaped call --
    the other way (besides a bare `assert`) a test can meaningfully assert
    something rather than passing trivially."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name in ("raises", "assertRaises", "assertRaisesRegex")


def _validate_synthesized_body(body) -> None:
    """The gate PLAN.md's task asked for, analogous to `composer._validate()`'s
    control-mention check: a synthesized test that doesn't parse, doesn't
    import `ops`, or can't possibly fail is worse than no test at all (that's
    exactly Finding 6). Structural, not a real sandbox execution -- consistent
    with how `Extractor`/`SurfaceInferrer`/`Composer` validate LLM output
    elsewhere in this codebase, and cheap enough to run on every synthesis
    call.
    """
    if not isinstance(body, str) or not body.strip():
        raise TestFileSynthesisInvalid(f"body must be a non-empty string, got {body!r}")
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        raise TestFileSynthesisInvalid(f"body is not valid Python: {exc}") from exc
    imports_ops = any(
        (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "ops" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "ops")
        for n in ast.walk(tree)
    )
    if not imports_ops:
        raise TestFileSynthesisInvalid("body does not import the ops package -- would not exercise the real API")
    test_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
    if not test_funcs:
        raise TestFileSynthesisInvalid("body defines no pytest-discoverable test_* function")
    has_assertion = any(
        isinstance(node, ast.Assert) or _is_raises_call(node) for func in test_funcs for node in ast.walk(func)
    )
    if not has_assertion:
        raise TestFileSynthesisInvalid(
            "body's test function has no assert/pytest.raises -- it would pass "
            "trivially regardless of whether the reported bug is real"
        )
    if "TODO" in body:
        raise TestFileSynthesisInvalid("body still contains a TODO placeholder, not a completed assertion")


_TEST_FILE_SCHEMA_INSTRUCTIONS = """\
Given the reproduction hypothesis below, write a self-contained pytest test
file that encodes the REPORTED bug as an executable assertion, as JSON
matching exactly this shape:

{
  "body": str
}

`body` is the full contents of one Python file. It MUST:

- `import ops` and `from ops import testing`;
- define a minimal charm (e.g. `class ProbeCharm(ops.CharmBase): pass`) and a
  `META = {"name": "probe-charm"}` dict, enough to build a
  `testing.Context(ProbeCharm, meta=META)`;
- define exactly one `def test_...():` function, discoverable by pytest;
- inside that function, drive `ctx.run(...)` (or the `with ctx(...) as mgr:`
  form) with a `testing.State()` shaped to exercise the scenario described by
  Expected/Observed below;
- assert the EXPECTED behaviour -- the promise the software is supposed to
  keep -- not the observed (buggy) one. If the bug described below is real,
  this assertion must fail when the file actually runs; if it is not real,
  the assertion passes.

Do not write an assertion that passes regardless of whether the bug exists
(e.g. only checking that the code runs without raising, or a bare `pass`) --
that is worse than no test at all, because it always reports "did not
reproduce" even when the bug is real. Do not leave a TODO or placeholder of
any kind; write the real assertion now, from the Expected/Observed text
given. If a symbol anchor is given, exercise that API surface directly where
practical.

Worked example of the correct `ops.testing.Context` API shape -- dispatch an
event with `ctx.run(event, state)`, NOT `ctx(state, event_name)`. There is no
`ctx.charm_dir` -- the charm root a running hook sees is only observable
*from inside the charm* (`self.charm_dir`, a `pathlib.Path`, on the charm
instance itself), so capture whatever you need to check (cwd, a config
value, a raised exception) inside an event handler into a module-level dict
or similar, and assert on it afterwards, in the test function, once
`ctx.run(...)` has returned:

```python
import os

import ops
from ops import testing

captured = {}


class ProbeCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured["cwd"] = os.getcwd()
        captured["charm_dir"] = str(self.charm_dir)


META = {"name": "probe-charm"}


def test_example():
    ctx = testing.Context(ProbeCharm, meta=META)
    ctx.run(ctx.on.start(), testing.State())
    assert captured["cwd"] == captured["charm_dir"]
```

Use this exact shape (`ctx.run(ctx.on.<event>(), state)`; capture inside a
handler into a dict; assert on the dict afterwards) for the real assertion
below -- do not invent a different calling convention or a `Context`
attribute that isn't shown here.
"""


class TestFileSynthesizer:
    """Approach §3's test-file synthesis, LLM-driven (2026-07-30, closing
    spike-step-5/composer-live/RESULT.md's Finding 6). Same seam pattern as
    `Extractor`/`SurfaceInferrer`/`composer.Composer`: any failure (no key,
    transport error, invalid JSON, a body that fails `_validate_synthesized_
    body`) falls back to `_loud_failing_test_file()` rather than raising or
    silently shipping an always-passing stub.
    """

    def __init__(self, llm: LLMSeam):
        self.llm = llm

    def synthesize(self, issue: Issue, hypothesis: Hypothesis) -> TestFile:
        path = f"test_issue_{hypothesis.issue_number}_repro.py"
        try:
            raw = self.llm.complete_json(
                purpose="test_file_synthesis",
                prompt=self._build_prompt(issue, hypothesis),
                context={"issue_number": issue.number},
            )
            if not isinstance(raw, dict) or "body" not in raw:
                raise TestFileSynthesisInvalid(f"missing 'body' key, got {raw!r}")
            body = raw["body"]
            _validate_synthesized_body(body)
        except (LLMError, TestFileSynthesisInvalid) as exc:
            return _loud_failing_test_file(hypothesis, path, reason=str(exc))
        return TestFile(path=path, body=body)

    @staticmethod
    def _build_prompt(issue: Issue, hypothesis: Hypothesis) -> str:
        anchor = (
            f"\nSymbol anchor: {hypothesis.moving_parts.symbol_anchor}"
            if hypothesis.moving_parts.symbol_anchor
            else ""
        )
        return (
            f"{_TEST_FILE_SCHEMA_INSTRUCTIONS}\n"
            f"Issue #{issue.number}: {issue.title}\n"
            f"Expected: {hypothesis.expected}\n"
            f"Observed: {hypothesis.observed}"
            f"{anchor}\n"
        )


def _loud_failing_test_file(hypothesis: Hypothesis, path: str, *, reason: str) -> TestFile:
    """`TestFileSynthesizer`'s fallback -- see the module-level comment above
    `SYNTHESIS_INCOMPLETE_MARKER` for why this fails loudly instead of
    silently passing. `repr()`'d into the template as one Python string
    literal so an arbitrary `reason` (attacker-free here, but still free text
    from an exception message) can never break the generated file's syntax.
    """
    message = (
        f"{SYNTHESIS_INCOMPLETE_MARKER}: LLM-driven test-file synthesis did not produce a "
        f"usable assertion for this issue ({reason}). This scaffold cannot confirm or refute "
        "the reported bug -- see PLAN.md Approach §3."
    )
    body = _LOUD_FAILING_TEMPLATE.format(
        issue_number=hypothesis.issue_number,
        expected=_docstring_safe(hypothesis.expected),
        observed=_docstring_safe(hypothesis.observed),
        message_literal=repr(message),
    )
    return TestFile(path=path, body=body)


def needs_test_file(hypothesis: Hypothesis) -> bool:
    """True iff `hypothesis` is `substrate: none` and `commands[]` has no
    *runnable* pytest invocation -- the trigger for the delta above.

    Two shapes both count as "not runnable", both real (spike-step-5/2045's
    actual extraction is the second one, discovered only once real data was
    available -- an earlier version of this function only checked for the
    first):

    - no `pytest` invocation anywhere in `commands[]` at all;
    - a `pytest <file>.py` invocation *is* present, but nothing in
      `commands[]` actually materialises `<file>.py` (spike-step-5/2045's
      extraction has `cd ... && uv run pytest test_cwd.py -v` alongside a
      `# write test_cwd.py: ...` *comment*, not a real heredoc -- the LLM
      named the file it wanted without being able to write Python into a
      `commands[]` string list).
    """
    if hypothesis.moving_parts.substrate != "none":
        return False
    match = _PYTEST_TARGET_RE.search("\n".join(hypothesis.commands))
    if match is None:
        return True
    return not _heredoc_writes(match.group(1), hypothesis.commands)


def _docstring_safe(text: str) -> str:
    return (text or "").replace('"""', "'''")
