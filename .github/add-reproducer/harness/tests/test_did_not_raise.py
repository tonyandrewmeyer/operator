"""A model-written test whose own `pytest.raises(...)` did not hold
(`spike-step-5/static-retry/RESULT.md` §25, §26).

Live on `#2709` (§25, run 5-7), a test wrapped `model.get_relation('db', 1)`,
which never reads the databag, in `pytest.raises(ops.RelationNotFoundError)`.
It failed "DID NOT RAISE" on ops 3.8.3 and on the fix, and rung 6 read that as
`reproduced_weaker`, which composes "The bug reproduced." Rung 1d grades it no
stronger than silent, on the 3.8.3 run, whatever the fix's run says. Every
other rung is unchanged.
"""

from __future__ import annotations

from classifier import classify
from models import COMMENT_OUTCOMES, CommandResult, Hypothesis, Outcome, RunResult

_TEST = """\
import ops
import pytest


def test_gone_relation(model):
    with pytest.raises(ops.RelationNotFoundError):
        model.get_relation('db', 1)
"""

_HEREDOC = f"cat > test_gone.py << 'EOF'\n{_TEST}EOF"

# What pytest 9 printed for run 5-7 on ops 3.8.3 and on `main` (§25's runs
# file), trimmed to the failure.
_DID_NOT_RAISE = """\
F                                                                        [100%]
=================================== FAILURES ===================================
________________ test_permission_denied_mapped_to_relation_not_found ________________

    def test_permission_denied_mapped_to_relation_not_found(tmp_path, monkeypatch):
>       with pytest.raises(ops.RelationNotFoundError):
E       Failed: DID NOT RAISE RelationNotFoundError

test_relation_permission_denied.py:35: Failed
=========================== short test summary info ============================
FAILED test_relation_permission_denied.py::test_permission_denied_mapped_to_relation_not_found - Failed: DID NOT RAISE RelationNotFoundError
1 failed in 0.12s
"""

# Older pytest prints the class.
_DID_NOT_RAISE_OLD_PYTEST = """\
>       with pytest.raises(ops.ModelError):
E       Failed: DID NOT RAISE <class 'ops.model.ModelError'>
1 failed in 0.12s
"""

# §24's 4-0 repair on ops 3.8.3: the read raised, the catch-and-assert
# compared it with `{}`.
_ASSERTION = """\
>       assert data == {}
E       AssertionError: assert ModelError('ERROR permission denied\\n') == {}
1 failed in 0.20s
"""

_PASSED = "1 passed in 0.10s\n"


def _hypothesis(commands=None, observed: str = "") -> Hypothesis:
    return Hypothesis.from_dict(
        2709,
        {
            "in_scope": True,
            "moving_parts": {"substrate": "none"},
            "commands": commands or ["uv init --bare .", "uv add 'ops[testing]==3.8.3' pytest", _HEREDOC, "uv run pytest test_gone.py"],
            "expected": "",
            "observed": observed,
            "confidence": "high",
        },
    )


def _run(stdout: str, exit_code: int = 1, commands=None) -> RunResult:
    commands = commands or ["uv init --bare .", "uv add 'ops[testing]==3.8.3' pytest", _HEREDOC, "uv run pytest test_gone.py"]
    return RunResult(
        hypothesis_number=2709,
        branch="none",
        commands=[CommandResult(command=c, exit_code=0) for c in commands[:-1]]
        + [CommandResult(command=commands[-1], exit_code=exit_code, stdout=stdout)],
    )


def test_did_not_raise_on_both_versions_is_silent():
    """Run 5-7 as it was: "DID NOT RAISE" on 3.8.3 and on the fix."""
    old, reason = classify(_hypothesis(), None, _run(_DID_NOT_RAISE))
    fix, _ = classify(_hypothesis(), None, _run(_DID_NOT_RAISE))
    assert old is Outcome.UNRUNNABLE_SYNTHESIS_INVALID
    assert old not in COMMENT_OUTCOMES
    assert fix not in COMMENT_OUTCOMES
    assert "DID NOT RAISE RelationNotFoundError" in reason
    assert "pytest.raises" in reason


def test_did_not_raise_on_old_and_a_pass_on_fix_is_still_silent():
    """The expectation holding on the fix does not make its not holding on
    3.8.3 the reported symptom."""
    old, _ = classify(_hypothesis(), None, _run(_DID_NOT_RAISE))
    fix, _ = classify(_hypothesis(), None, _run(_PASSED, exit_code=0))
    assert old is Outcome.UNRUNNABLE_SYNTHESIS_INVALID
    assert old not in COMMENT_OUTCOMES
    assert fix is Outcome.DID_NOT_REPRODUCE


def test_older_pytest_wording_is_silent_too():
    outcome, reason = classify(_hypothesis(), None, _run(_DID_NOT_RAISE_OLD_PYTEST))
    assert outcome is Outcome.UNRUNNABLE_SYNTHESIS_INVALID
    assert "<class 'ops.model.ModelError'>" in reason


def test_an_assertion_failure_on_old_and_a_pass_on_fix_is_unchanged():
    old, _ = classify(_hypothesis(), None, _run(_ASSERTION))
    fix, _ = classify(_hypothesis(), None, _run(_PASSED, exit_code=0))
    assert old is Outcome.REPRODUCED_WEAKER
    assert old in COMMENT_OUTCOMES
    assert fix is Outcome.DID_NOT_REPRODUCE


def test_did_not_raise_beside_an_assertion_failure_keeps_the_assertion():
    """Two tests in the file, one failing an assertion: that one is still
    evidence, so the rung leaves the run to the rungs below."""
    outcome, _ = classify(_hypothesis(), None, _run(_DID_NOT_RAISE + _ASSERTION))
    assert outcome is Outcome.REPRODUCED_WEAKER


def test_a_quoted_snippet_in_the_printed_source_does_not_lift_it():
    """pytest prints the test's source around the failure, so a snippet the
    reporter quoted can appear there; rung 1d is checked before rung 2."""
    hypothesis = _hypothesis(observed='it should raise "ops.RelationNotFoundError" instead')
    outcome, _ = classify(hypothesis, None, _run(_DID_NOT_RAISE))
    assert outcome is Outcome.UNRUNNABLE_SYNTHESIS_INVALID


def test_a_reporter_supplied_test_is_untouched():
    """No heredoc and no synthesised file: nothing a model wrote, so a missing
    exception may be the reporter's symptom and the rung does not fire."""
    commands = ["uv init --bare .", "uv add 'ops[testing]==3.8.3' pytest", "uv run pytest tests/test_repro.py"]
    outcome, _ = classify(_hypothesis(commands), None, _run(_DID_NOT_RAISE, commands=commands))
    assert outcome is Outcome.REPRODUCED_WEAKER
