"""Surface inference emits pebble command lines pebble then rejects.

`first-real-substrate/RESULT.md` Finding 13 caught an *invented* flag
(`pebble notify --user=...`) and added a prompt paragraph plus a validation
rule. A live run on 2026-08-21 produced the next variant along -- a real flag
with a malformed value:

    /charm/bin/pebble exec --timeout=1 ls
    error: invalid argument for flag `--timeout' (expected time.Duration):
    time: missing unit in duration "1"

Same end state as the `--user` case: the stimulus never fires, and the run
reaches a rung having done nothing to the charm. Prompt text alone had already
failed once in this spot, which is why this is a validation rule.
`spike-step-5/gate-substrate/RESULT.md` §7.
"""

from __future__ import annotations

import pytest
from surface_inference import SurfaceInferenceInvalid, validate


def _raw(command: str) -> dict:
    return {
        "charm_name": "repro-i1329-timeout",
        "relation": {},
        "storage": {},
        "pebble_service": {"container": "mysql", "service": "ls-exec", "command": command, "user": "_daemon_"},
        "ops_api_surface": "update-status",
        "expected_signal": "observed notice:",
    }


@pytest.mark.parametrize("value", ["1", "30", "0", "abc", "1sec", "30 s"])
def test_a_bare_or_malformed_duration_is_rejected(value):
    with pytest.raises(SurfaceInferenceInvalid) as excinfo:
        validate(_raw(f"/charm/bin/pebble exec --timeout={value} ls"))

    assert "--timeout" in str(excinfo.value)
    assert "unit" in str(excinfo.value)


@pytest.mark.parametrize("value", ["1s", "500ms", "1m30s", "0.5s", "2h"])
def test_a_well_formed_go_duration_is_accepted(value):
    validate(_raw(f"/charm/bin/pebble exec --timeout={value} ls"))


def test_space_separated_form_is_checked_too():
    with pytest.raises(SurfaceInferenceInvalid):
        validate(_raw("/charm/bin/pebble exec --timeout 1 ls"))


def test_the_live_2026_08_21_command_is_rejected():
    """Verbatim from `~/run/f1329-6`'s surface inference."""
    with pytest.raises(SurfaceInferenceInvalid):
        validate(_raw("/charm/bin/pebble exec --timeout=1 ls"))


def test_commands_without_duration_flags_are_untouched():
    validate(_raw("/charm/bin/pebble notify canonical.com/repro/notice-1 key=value"))


def test_the_prompt_says_durations_need_a_unit():
    """The rule and the instruction have to move together -- a validation
    failure the model was never told about just costs a retry."""
    import surface_inference

    assert "--timeout=30s" in surface_inference._SCHEMA_INSTRUCTIONS
