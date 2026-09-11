"""PLAN.md Approach §3/§4 delta (spike-step-5/2045/RESULT.md "PLAN deltas
surfaced" §1): substrate:none extractions with no runnable pytest
invocation in commands[] get a synthesised test file instead of staying
un-runnable.

2026-07-30: synthesis itself became LLM-driven, behind the same seam
`Extractor`/`SurfaceInferrer`/`Composer` already use, closing
spike-step-5/composer-live/RESULT.md's Finding 6 -- the previous
deterministic template had no real assertion (`ctx.run(...)` followed by a
`# TODO`) and so could never fail regardless of whether the reported bug was
real. `TestFileSynthesizer` now asks the model for a body encoding the
specific `expected`/`observed` claim, validates it structurally, and falls
back to a loud-failing stub (never the old always-passing one) on any
failure.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from models import Hypothesis, Issue
from runner_stage import write_synthesized_test_file_if_needed
from seams.llm import FixtureLLM, LLMError
from seams.runner import FixtureRunnerSeam
from surface_inference import (
    SYNTHESIS_INCOMPLETE_MARKER,
    TestFileSynthesisInvalid,
    TestFileSynthesizer,
    _validate_synthesized_body,
    is_pytest_invocation,
    needs_test_file,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _hypothesis(*, issue_number=2045, **overrides) -> Hypothesis:
    raw = {
        "in_scope": True,
        "moving_parts": {"substrate": "none"},
        "commands": [],
        "expected": "os.getcwd() during a Scenario-executed hook returns the charm root.",
        "observed": "os.getcwd() returns the repository root, not the charm root.",
        "confidence": "medium",
    }
    raw.update(overrides)
    return Hypothesis.from_dict(issue_number, raw)


def _issue(*, number=2045, **overrides) -> Issue:
    raw = {
        "number": number,
        "title": "os.getcwd() is not the charm root in ops[testing]",
        "body": "b",
        "labels": [],
        "state": "OPEN",
        "createdAt": "",
        "author": "a",
        "repo": "canonical/operator",
    }
    raw.update(overrides)
    return Issue.from_dict(raw)


class _StaticLLM:
    """Returns a canned dict (or raises) regardless of prompt content --
    same shape as the stub used in test_composer.py."""

    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls = 0
        self.last_prompt: str | None = None

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        self.calls += 1
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


# --- needs_test_file: unchanged by the synthesis-content fix ---------------


def test_needs_test_file_for_2045_shaped_input_no_commands():
    # #2045's real extraction shape: substrate=none, no self-contained
    # repro snippet in commands[] (the issue body describes the bug but
    # gives no runnable pytest invocation).
    assert needs_test_file(_hypothesis(commands=[]))


def test_needs_test_file_false_when_pytest_already_runnable():
    # #2327/#2341's shape: substrate=none but commands[] already ends in
    # a runnable `uv run pytest ...` invocation -- synthesis must not fire
    # and silently clobber a real repro snippet.
    hyp = _hypothesis(
        commands=[
            "uv venv",
            "uv pip install ops",
            "cat > repro_test.py << 'EOF'\nimport ops\nEOF",
            "uv run pytest repro_test.py -v",
        ]
    )
    assert not needs_test_file(hyp)


def test_needs_test_file_false_for_non_none_substrate():
    hyp = _hypothesis(moving_parts={"substrate": "k8s"}, commands=[])
    assert not needs_test_file(hyp)


def test_needs_test_file_true_when_pytest_target_never_written():
    # #2045's REAL extraction shape (fixtures/extractions/2045.json):
    # commands[] is non-empty and even names a pytest target, but the file
    # is only ever mentioned in a `#`-comment, never actually created by a
    # heredoc -- discovered only once this session got real spike-step-5
    # data; an earlier version of needs_test_file() only checked for an
    # empty commands[] and would have missed this.
    hyp = _hypothesis(
        commands=[
            "uv init /tmp/repro-2045-cwd",
            "cd /tmp/repro-2045-cwd && uv add 'ops[testing]'",
            "# write test_cwd.py: minimal Scenario test that captures os.getcwd() and self.charm_dir",
            "cd /tmp/repro-2045-cwd && uv run pytest test_cwd.py -v",
        ]
    )
    assert needs_test_file(hyp)


# --- is_pytest_invocation: the setup-command filter's precision fix --------


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest test_cwd.py -v",
        "pytest -v",
        "cd /tmp/x && uv run pytest test_cwd.py -v",
    ],
)
def test_is_pytest_invocation_true_for_real_invocations(command):
    assert is_pytest_invocation(command)


@pytest.mark.parametrize(
    "command",
    [
        "uv add 'ops[testing]' pytest",
        "cd /tmp/repro-2045-cwd && uv add pytest",
        "uv init --bare .",
        "uv pip install pytest ops",
    ],
)
def test_is_pytest_invocation_false_for_setup_commands_mentioning_pytest(command):
    assert not is_pytest_invocation(command)


# --- _validate_synthesized_body: the Finding-6 gate -------------------------


def test_validate_accepts_a_real_assertion_body():
    body = "import ops\nfrom ops import testing\n\n\ndef test_x():\n    assert 1 == 2, 'boom'\n"
    _validate_synthesized_body(body)  # must not raise


def test_validate_accepts_pytest_raises_as_the_assertion():
    body = "import ops\nimport pytest\n\n\ndef test_x():\n    with pytest.raises(KeyError):\n        {}['x']\n"
    _validate_synthesized_body(body)


def test_validate_rejects_non_string_body():
    with pytest.raises(TestFileSynthesisInvalid, match="non-empty string"):
        _validate_synthesized_body(None)


def test_validate_rejects_empty_body():
    with pytest.raises(TestFileSynthesisInvalid, match="non-empty string"):
        _validate_synthesized_body("   ")


def test_validate_rejects_syntax_error():
    with pytest.raises(TestFileSynthesisInvalid, match="not valid Python"):
        _validate_synthesized_body("def test_x(:\n    pass\n")


def test_validate_rejects_body_that_never_imports_ops():
    body = "def test_x():\n    assert 1 == 2\n"
    with pytest.raises(TestFileSynthesisInvalid, match="does not import"):
        _validate_synthesized_body(body)


def test_validate_rejects_body_with_no_test_function():
    body = "import ops\n\n\ndef helper():\n    assert 1 == 2\n"
    with pytest.raises(TestFileSynthesisInvalid, match="no pytest-discoverable"):
        _validate_synthesized_body(body)


def test_validate_rejects_body_with_no_assertion():
    # The exact Finding 6 shape: imports ops, defines a test, runs something,
    # but never actually asserts anything -- would pass regardless of
    # whether the reported bug is real.
    body = "import ops\nfrom ops import testing\n\n\ndef test_x():\n    ctx = testing.Context\n"
    with pytest.raises(TestFileSynthesisInvalid, match="no assert"):
        _validate_synthesized_body(body)


def test_validate_rejects_todo_placeholder_even_with_an_assertion():
    body = "import ops\n\n\ndef test_x():\n    assert True  # TODO: fill in the real check\n"
    with pytest.raises(TestFileSynthesisInvalid, match="TODO"):
        _validate_synthesized_body(body)


# --- TestFileSynthesizer: happy path -----------------------------------------


def test_synthesizer_uses_llm_body_when_valid():
    body = "import ops\nfrom ops import testing\n\n\ndef test_x():\n    assert False, 'bug is real'\n"
    llm = _StaticLLM(response={"body": body})
    test_file = TestFileSynthesizer(llm).synthesize(_issue(), _hypothesis())
    assert test_file.body == body
    assert test_file.path == "test_issue_2045_repro.py"
    assert llm.calls == 1


def test_synthesizer_prompt_carries_expected_observed_and_symbol_anchor():
    llm = _StaticLLM(response={"body": "import ops\n\n\ndef test_x():\n    assert False\n"})
    TestFileSynthesizer(llm).synthesize(
        _issue(),
        _hypothesis(moving_parts={"substrate": "none", "symbol_anchor": "ops.testing._runtime"}),
    )
    assert "ops.testing._runtime" in llm.last_prompt
    assert "charm root" in llm.last_prompt  # from _hypothesis()'s default expected/observed text


def test_fixture_llm_replays_recorded_2045_synthesis():
    # fixtures/test_file_synthesis/2045.json -- a real-derived, assertion-
    # bearing body (see that fixture's own _provenance note). Symmetry check
    # with Extractor/SurfaceInferrer/Composer: a recording present means no
    # fallback.
    llm = FixtureLLM(FIXTURES)
    test_file = TestFileSynthesizer(llm).synthesize(_issue(), _hypothesis())
    assert "charm_dir" in test_file.body
    assert SYNTHESIS_INCOMPLETE_MARKER not in test_file.body
    _validate_synthesized_body(test_file.body)  # the recorded fixture itself must be valid


# --- TestFileSynthesizer: fallback -------------------------------------------


def test_synthesizer_falls_back_on_llm_error():
    llm = _StaticLLM(error=LLMError("boom"))
    test_file = TestFileSynthesizer(llm).synthesize(_issue(), _hypothesis())
    assert SYNTHESIS_INCOMPLETE_MARKER in test_file.body
    assert test_file.path == "test_issue_2045_repro.py"


def test_fixture_llm_no_recording_falls_back():
    llm = FixtureLLM(FIXTURES)  # no fixtures/test_file_synthesis/9999.json
    test_file = TestFileSynthesizer(llm).synthesize(_issue(number=9999), _hypothesis(issue_number=9999))
    assert SYNTHESIS_INCOMPLETE_MARKER in test_file.body
    assert test_file.path == "test_issue_9999_repro.py"


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"body": ""},
        {"body": "   "},
        {"body": 12345},
        {"wrong_key": "x"},
        {"body": "def test_x(:\n"},  # syntax error
        {"body": "def test_x():\n    assert False\n"},  # no ops import
        {"body": "import ops\n\n\ndef test_x():\n    pass\n"},  # no assertion -- Finding 6's exact shape
        {"body": "import ops\n\n\ndef test_x():\n    assert True  # TODO\n"},
        "not even a dict",
        None,
    ],
)
def test_synthesizer_falls_back_on_malformed_or_unusable_response(response):
    llm = _StaticLLM(response=response)
    test_file = TestFileSynthesizer(llm).synthesize(_issue(), _hypothesis())
    assert SYNTHESIS_INCOMPLETE_MARKER in test_file.body


def test_fallback_body_is_syntactically_valid_and_fails_loudly_when_run():
    # The fallback must itself be real, executable Python that raises --
    # "fails loudly", not merely un-runnable at the Python level the way a
    # syntax error would be. `ops` isn't a dependency of this harness's own
    # test env (it only ever gets installed inside the scratch `uv`
    # project the real runner builds), so a bare-bones stub module stands
    # in for it here -- the fallback raises before ever touching
    # `ops.testing.Context`/`State`, so nothing beyond `ops.CharmBase`
    # existing as a class needs to work.
    llm = _StaticLLM(error=LLMError("no key configured"))
    test_file = TestFileSynthesizer(llm).synthesize(_issue(), _hypothesis())

    fake_ops = types.ModuleType("ops")
    fake_ops.CharmBase = type("CharmBase", (), {})
    fake_ops.testing = types.ModuleType("ops.testing")
    previous = {name: sys.modules.get(name) for name in ("ops", "ops.testing")}
    sys.modules["ops"] = fake_ops
    sys.modules["ops.testing"] = fake_ops.testing
    try:
        namespace: dict = {}
        exec(compile(test_file.body, "<synthesized>", "exec"), namespace)
        with pytest.raises(AssertionError, match=SYNTHESIS_INCOMPLETE_MARKER):
            namespace["test_issue_2045"]()
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_fallback_embeds_the_failure_reason_safely():
    # A reason/body containing quotes and braces must not break the
    # generated file's syntax -- repr()-embedded, not raw-interpolated.
    llm = _StaticLLM(response={"body": 'contains "quotes" and {braces} and is not valid python(('})
    test_file = TestFileSynthesizer(llm).synthesize(_issue(), _hypothesis())
    compile(test_file.body, "<synthesized>", "exec")  # must not raise
    assert SYNTHESIS_INCOMPLETE_MARKER in test_file.body


# --- write_synthesized_test_file_if_needed: wiring ---------------------------


def test_write_synthesized_test_file_if_needed_writes_file_and_appends_command(tmp_path):
    hyp = _hypothesis(commands=[])
    context = {"workdir": str(tmp_path)}
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), FixtureLLM(FIXTURES), context)
    written = tmp_path / "test_issue_2045_repro.py"
    assert written.exists()
    assert "charm_dir" in written.read_text()
    assert updated.commands[-1] == "uv run pytest test_issue_2045_repro.py -v"
    # Original hypothesis object is untouched (dataclasses.replace, not mutation).
    assert hyp.commands == []
    assert hyp.synthesized_test_file is None


def test_write_synthesized_test_file_if_needed_carries_test_file_onto_hypothesis(tmp_path):
    # spike-step-5/composer-live/RESULT.md Finding 4: the composer needs the
    # synthesized file's body, not just its path in commands[]. Synthesis
    # must attach the actual `TestFile` object to the returned hypothesis,
    # not just write it to disk and forget it.
    hyp = _hypothesis(commands=[])
    context = {"workdir": str(tmp_path)}
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), FixtureLLM(FIXTURES), context)
    assert updated.synthesized_test_file is not None
    assert updated.synthesized_test_file.path == "test_issue_2045_repro.py"
    assert updated.synthesized_test_file.body == (tmp_path / "test_issue_2045_repro.py").read_text()


def test_write_synthesized_test_file_if_needed_noop_when_not_needed_never_calls_llm(tmp_path):
    hyp = _hypothesis(
        commands=["uv venv", "cat > repro_test.py << 'EOF'\nimport ops\nEOF", "uv run pytest repro_test.py -v"]
    )
    llm = _StaticLLM(error=RuntimeError("must not be called when synthesis isn't needed"))
    context = {"workdir": str(tmp_path)}
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), llm, context)
    assert updated is hyp
    assert updated.synthesized_test_file is None
    assert list(tmp_path.iterdir()) == []
    assert llm.calls == 0


def test_write_synthesized_test_file_if_needed_keeps_uv_add_pytest_setup_line(tmp_path):
    # spike-step-5/synthesis-live/RESULT.md: a real live end-to-end run found
    # the old `"pytest" not in c` filter dropped `uv add 'ops[testing]'
    # pytest` -- a legitimate setup command, not the broken invocation this
    # filter exists to remove -- silently stripping pytest out of the venv.
    # `uv add 'ops[testing]' pytest` is the corpus-v2-standard modern
    # extraction shape (uv init -> uv add -> heredoc -> pytest) and must
    # survive this filter.
    hyp = _hypothesis(commands=["uv init --bare .", "uv add 'ops[testing]' pytest"])
    context = {"workdir": str(tmp_path)}
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), FixtureLLM(FIXTURES), context)
    assert updated.commands == [
        "uv init --bare .",
        "uv add 'ops[testing]' pytest",
        "uv run pytest test_issue_2045_repro.py -v",
    ]


def test_write_synthesized_test_file_if_needed_drops_broken_pytest_and_comment(tmp_path):
    # The real #2045 shape: the broken `pytest test_cwd.py` invocation and
    # the `#`-comment placeholder must both be dropped, genuine setup
    # commands (uv init/uv add) kept, and the real pytest invocation
    # appended once.
    hyp = _hypothesis(
        commands=[
            "uv init /tmp/repro-2045-cwd",
            "cd /tmp/repro-2045-cwd && uv add 'ops[testing]'",
            "# write test_cwd.py: minimal Scenario test",
            "cd /tmp/repro-2045-cwd && uv run pytest test_cwd.py -v",
        ]
    )
    context = {"workdir": str(tmp_path)}
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), FixtureLLM(FIXTURES), context)
    assert updated.commands == [
        "uv init /tmp/repro-2045-cwd",
        "cd /tmp/repro-2045-cwd && uv add 'ops[testing]'",
        "uv run pytest test_issue_2045_repro.py -v",
    ]


def test_write_synthesized_test_file_if_needed_falls_back_when_llm_unusable(tmp_path):
    hyp = _hypothesis(commands=[])
    llm = _StaticLLM(error=LLMError("no key"))
    context = {"workdir": str(tmp_path)}
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), llm, context)
    written = (tmp_path / "test_issue_2045_repro.py").read_text()
    assert SYNTHESIS_INCOMPLETE_MARKER in written
    assert updated.synthesized_test_file.body == written


def test_run_hypothesis_2045_writes_file_before_dispatching_to_seam(tmp_path):
    # End-to-end through run_hypothesis(): #2045's real extraction shape
    # (fixtures/extractions/2045.json, fixtures/issues/2045.json,
    # fixtures/runs/2045.json -- see spike-step-5/2045/RESULT.md, the real
    # 2026-07-24 hand-walk this delta is drawn from) has commands[] that
    # names a pytest target but never writes it (see
    # test_needs_test_file_true_when_pytest_target_never_written above),
    # so the "none" branch must still synthesise and write a test file
    # before FixtureRunnerSeam.run() replays the recorded output. Confirms
    # the wiring, not just the pure function in isolation.
    import runner_stage

    issue = Issue.from_dict(json.loads((FIXTURES / "issues" / "2045.json").read_text()))
    hyp = Hypothesis.from_dict(2045, json.loads((FIXTURES / "extractions" / "2045.json").read_text()))
    assert needs_test_file(hyp)  # the delta this test exists to cover
    context = {"workdir": str(tmp_path)}
    result = runner_stage.run_hypothesis(hyp, issue, None, FixtureRunnerSeam(FIXTURES), FixtureLLM(FIXTURES), context)
    assert not result.skipped_stale
    assert result.branch == "none"
    assert (tmp_path / "test_issue_2045_repro.py").exists()
    assert result.run_result.hypothesis_number == 2045


def test_synthesis_builds_an_environment_when_the_extraction_supplied_none(tmp_path):
    """A live extraction can return `commands: []` (the real 2026-08-18
    #2639 case). This stage assumed the `uv init` -> `uv add` prologue was
    already there -- true of the corpus-v2 shape, not of that one. Without
    it, `uv run pytest` finds no project in the scratch dir, walks up to the
    harness's own pyproject, and the synthesised test dies with
    ModuleNotFoundError: No module named 'ops'."""
    updated = write_synthesized_test_file_if_needed(
        _hypothesis(commands=[]), _issue(), FixtureLLM(FIXTURES), {"workdir": str(tmp_path)}
    )
    assert "uv init --bare" in updated.commands
    assert any("uv add" in c and "ops[testing]" in c for c in updated.commands)
    # Setup must precede the test run.
    assert updated.commands.index("uv init --bare") < len(updated.commands) - 1
    assert updated.commands[-1].startswith("uv run pytest")


def test_synthesis_does_not_duplicate_an_environment_the_extraction_already_built(tmp_path):
    """The corpus-v2 shape already carries its own prologue -- adding a
    second one would be noise, and `uv init` in an initialised project
    fails."""
    hyp = _hypothesis(commands=["uv init", "uv add 'ops[testing]' pytest"])
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), FixtureLLM(FIXTURES), {"workdir": str(tmp_path)})
    assert updated.commands.count("uv init --bare") == 0
    assert updated.commands[:2] == ["uv init", "uv add 'ops[testing]' pytest"]


def test_synthesis_treats_a_bare_uv_venv_as_an_environment(tmp_path):
    hyp = _hypothesis(commands=["uv venv"])
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), FixtureLLM(FIXTURES), {"workdir": str(tmp_path)})
    assert "uv init --bare" not in updated.commands


def test_synthesis_does_not_mistake_a_cd_for_an_environment(tmp_path):
    hyp = _hypothesis(commands=["cd /tmp/scratch"])
    updated = write_synthesized_test_file_if_needed(hyp, _issue(), FixtureLLM(FIXTURES), {"workdir": str(tmp_path)})
    assert "uv init --bare" in updated.commands
