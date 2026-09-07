"""The split-heredoc `commands[]` shape (`spike-step-5/criterion-1-live/
RESULT.md` §5.1, `spike-step-5/gate-accounting/RESULT.md`).

A live extraction sometimes opens a heredoc (`cat > file.py << 'PYEOF'`) and
puts every line of the body -- and the closing terminator -- in its own
`commands[]` element, instead of the single element the `#2327`/`#2341`
extractions carry. `surface_inference.needs_test_file()` joins `commands[]`
with newlines and reads it as a script (sees the heredoc, declines to
synthesise a replacement test file); `runnability.assess()` and
`seams/runner.py`'s `_run_none` (one `bash -c` subprocess per element) both
read the same list as independent commands, and reject/can't execute the
split body. `Hypothesis.from_dict`'s `_join_split_heredocs()` re-joins the
opener through its terminator into one element at construction time, so
every consumer downstream reads the same thing.

`#1685` and `#1552` are the real, recorded live extractions this shape was
found on -- `spike-step-5/corpus-v4/out/twopass-run5/1685.json` and
`out/twopass-run6/1552.json`, not invented examples.
"""

import json
from pathlib import Path

import runnability
from models import Hypothesis, _join_split_heredocs
from runner_stage import needs_test_file

CORPUS_V4_OUT = Path(__file__).resolve().parent.parent.parent / "spike-step-5" / "corpus-v4" / "out"


def _real_commands(run_dir: str, issue_number: int) -> list[str]:
    record = json.loads((CORPUS_V4_OUT / run_dir / f"{issue_number}.json").read_text())
    return record["live_extraction"]["commands"]


def test_1685_heredoc_is_split_across_32_raw_elements():
    """Pin the raw shape this whole test module exists to fix, so a corpus
    re-run that stops reproducing it is visible rather than silently making
    the rest of this file test nothing."""
    commands = _real_commands("twopass-run5", 1685)
    assert len(commands) == 32
    assert commands[3] == "cat > tests/unit/test_charm.py << 'PYEOF'"
    assert commands[-2] == "PYEOF"


def test_1685_split_heredoc_is_rejoined_and_becomes_runnable():
    raw = json.loads((CORPUS_V4_OUT / "twopass-run5" / "1685.json").read_text())
    hyp = Hypothesis.from_dict(raw["issue"], raw["live_extraction"])
    # 32 raw elements collapse to 5: init, add, mkdir, the whole heredoc as
    # one element, the pytest invocation.
    assert len(hyp.commands) == 5
    assert hyp.commands[3].startswith("cat > tests/unit/test_charm.py << 'PYEOF'\nimport ops")
    assert hyp.commands[3].rstrip().endswith("PYEOF")
    # The two consumers now agree: the file is recognised as materialised
    # (no synthesis needed) and the joined heredoc is shell-shaped.
    assert needs_test_file(hyp) is False
    report = runnability.assess(hyp.commands)
    assert report.runnable, report.reason


def test_1552_split_heredoc_is_rejoined_and_becomes_runnable():
    raw = json.loads((CORPUS_V4_OUT / "twopass-run6" / "1552.json").read_text())
    hyp = Hypothesis.from_dict(raw["issue"], raw["live_extraction"])
    assert len(hyp.commands) == 4
    assert needs_test_file(hyp) is False
    report = runnability.assess(hyp.commands)
    assert report.runnable, report.reason


def test_unsplit_heredoc_is_left_alone():
    """The `#2341` shape (`fixtures/extractions/2341.json`): the heredoc
    already carries its full body and terminator in one `commands[]`
    element. `_join_split_heredocs` must not touch it -- there is nothing to
    join, and re-processing an already-complete heredoc must not duplicate
    or corrupt it."""
    commands = [
        "uv venv",
        "uv pip install ops",
        "cat > repro_test.py << 'EOF'\nimport ops\ndef test_x():\n    pass\nEOF",
        "uv run pytest repro_test.py -v",
    ]
    assert _join_split_heredocs(commands) == commands


def test_split_heredoc_with_no_terminator_is_left_alone():
    """If the terminator never appears (a truncated extraction, or a
    genuinely unrelated `<<` in a shell redirect), joining would silently
    swallow the rest of `commands[]` into one element. Safer to leave the
    opener as its own (broken, and correctly gated) element than to guess."""
    commands = ["cat > f.py << 'PYEOF'", "import ops", "uv run pytest f.py -v"]
    assert _join_split_heredocs(commands) == commands


def test_join_is_idempotent():
    """Running the join twice (e.g. if a caller re-wraps an already-`Hypothesis`-
    constructed commands list) must not re-split or duplicate anything."""
    commands = _real_commands("twopass-run5", 1685)
    once = _join_split_heredocs(commands)
    twice = _join_split_heredocs(once)
    assert once == twice


def test_multiple_split_heredocs_in_one_commands_list_both_rejoin():
    commands = [
        "uv init --bare .",
        "cat > a.py << 'AEOF'",
        "import ops",
        "AEOF",
        "cat > b.py << 'BEOF'",
        "import ops",
        "BEOF",
        "uv run pytest a.py b.py -v",
    ]
    joined = _join_split_heredocs(commands)
    assert joined == [
        "uv init --bare .",
        "cat > a.py << 'AEOF'\nimport ops\nAEOF",
        "cat > b.py << 'BEOF'\nimport ops\nBEOF",
        "uv run pytest a.py b.py -v",
    ]
