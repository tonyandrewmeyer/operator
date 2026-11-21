"""Which heredoc in `commands[]` is the test file
(`spike-step-5/static-retry/RESULT.md` §27, §28).

Live on `#2709` (§27, run 6-4's second answer), an extraction wrote its fake
hook tools as `cat > bin/relation-ids << 'EOF'` heredocs before the test
file, and `embedded_test_file()`, which took the first heredoc, handed the
static check a shell script. The check failed closed, but the real test was
never checked, and rungs 1a, 1c and 1d read the shell script as the
model-written file. `embedded_test_file()` now prefers the heredoc whose path
is a `.py` file a later command passes to pytest, and falls back to the
first heredoc when there is none.
"""

from __future__ import annotations

from classifier import classify
from models import CommandResult, Hypothesis, Outcome, RunResult, embedded_test_file, replace_embedded_test_body
from static_test_check import check_commands, runs_under_pytest

_FAKE = "cat > bin/relation-ids << 'EOF'\n#!/bin/sh\necho '[\"db:1\"]'\nEOF"
_FAKE_2 = "cat > bin/relation-get << 'EOF'\n#!/bin/sh\necho 'ERROR permission denied' >&2\nexit 1\nEOF"
_TEST_BODY = "def test_x():\n    assert True"
_TEST = f"cat > test_x.py << 'PYEOF'\n{_TEST_BODY}\nPYEOF"
_RUN = "uv run pytest test_x.py -v"


def test_fakes_before_the_test_pick_the_test():
    """6-4's shape: the shell fakes come first."""
    commands = ["uv init --bare .", "mkdir -p bin", _FAKE, "chmod +x bin/relation-ids", _FAKE_2, _TEST, _RUN]
    picked = embedded_test_file(commands)
    assert picked is not None
    assert (picked.path, picked.body) == ("test_x.py", _TEST_BODY)
    assert runs_under_pytest(commands)


def test_test_before_the_fakes_picks_the_test():
    commands = [_TEST, _FAKE, _FAKE_2, _RUN]
    assert embedded_test_file(commands).path == "test_x.py"


def test_a_single_heredoc_is_picked():
    assert embedded_test_file([_TEST, _RUN]).path == "test_x.py"
    # Not run under pytest: still the only heredoc, as before.
    assert embedded_test_file([_TEST, "uv run python test_x.py"]).path == "test_x.py"


def test_no_py_heredoc_falls_back_to_the_first_and_fails_closed():
    commands = [_FAKE, _FAKE_2, "uv run pytest"]
    picked = embedded_test_file(commands)
    assert picked.path == "bin/relation-ids"
    result = check_commands(commands)
    assert result is not None
    assert not result.passed


def test_of_two_py_heredocs_the_one_pytest_runs_is_picked():
    helper = "cat > charm.py << 'PYEOF'\nimport ops\nPYEOF"
    assert embedded_test_file([helper, _TEST, _RUN]).path == "test_x.py"
    # Neither is run under pytest: the first, as before.
    assert embedded_test_file([helper, _TEST, "uv run python test_x.py"]).path == "charm.py"


def test_a_py_heredoc_pytest_runs_only_before_it_is_not_preferred():
    """Same reading as `runs_under_pytest()`: the pytest command has to come
    after the heredoc."""
    helper = "cat > charm.py << 'PYEOF'\nimport ops\nPYEOF"
    assert embedded_test_file([helper, _RUN, _TEST]).path == "charm.py"


def test_no_heredoc_is_still_nothing_to_check():
    assert embedded_test_file(["uv init --bare .", _RUN]) is None
    assert check_commands(["uv init --bare .", _RUN]) is None


def test_two_heredocs_in_one_command():
    commands = [f"{_FAKE}\n{_TEST}", _RUN]
    assert embedded_test_file(commands).path == "test_x.py"


def test_replace_embedded_test_body_replaces_the_picked_heredoc():
    """The missing imports `Extractor._check()` adds go into the file that was
    checked, not into the first fake."""
    commands = [_FAKE, _TEST, _RUN]
    replaced = replace_embedded_test_body(commands, "import os\n" + _TEST_BODY)
    assert replaced[0] == _FAKE
    assert embedded_test_file(replaced).body == "import os\n" + _TEST_BODY


def test_the_check_reads_the_test_not_the_fake():
    commands = [_FAKE, _TEST, _RUN]
    result = check_commands(commands)
    assert result.path == "test_x.py"
    assert result.passed


_COLLECTION_ERROR = """\
collecting ... collected 0 items / 1 error
==================================== ERRORS ====================================
_________________________ ERROR collecting test_x.py _________________________
E   NameError: name 'model' is not defined
"""


def test_rung_1a_names_the_test_behind_fakes():
    """Rung 1a matches "ERROR collecting <name>" against the model-written
    file; with the fakes first it used to look for `relation-ids` and leave
    the run to rung 1's stale selector."""
    commands = ["uv init --bare .", _FAKE, _TEST, _RUN]
    hypothesis = Hypothesis.from_dict(
        2709,
        {
            "in_scope": True,
            "moving_parts": {"substrate": "none"},
            "commands": commands,
            "expected": "",
            "observed": "",
            "confidence": "high",
        },
    )
    run = RunResult(
        hypothesis_number=2709,
        branch="none",
        commands=[CommandResult(command=c, exit_code=0) for c in commands[:-1]]
        + [CommandResult(command=commands[-1], exit_code=2, stdout=_COLLECTION_ERROR)],
    )
    outcome, reason = classify(hypothesis, None, run)
    assert outcome is Outcome.UNRUNNABLE_SYNTHESIS_INVALID
    assert "test_x.py" in reason
