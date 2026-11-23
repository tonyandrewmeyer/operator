"""Live comments-vs-no-comments A/B, pinned against the recorded run.

The recorded JSON comes from real `LiveOpenRouterLLM` calls on real
`canonical/operator` issue+comment data. These tests assert against that recorded JSON, not a
live model: no network, no key, deterministic -- same pattern as
`test_live_calibration.py`.
"""

import json
from pathlib import Path

import pytest

_COMMENTS_AB = Path(__file__).parent.parent.parent / "spike-step-5" / "comments" / "out"


@pytest.fixture(autouse=True)
def _require_run():
    if not _COMMENTS_AB.is_dir():  # pragma: no cover - recorded run not always present
        pytest.skip(f"comments A/B run not available at {_COMMENTS_AB}")


def _load(issue: int) -> dict:
    return json.loads((_COMMENTS_AB / f"{issue}.json").read_text())


def test_827_repo_version_only_extracted_from_comments():
    """The clean, dramatic case: the 2024 follow-up comment is
    the only place `1.5.5` appears anywhere in the issue -- the 2022 body
    never mentions a post-1.x version. Without it, this hypothesis would
    carry `repo_version: null` on a CLOSED issue, hitting the
    skip-when-stale gate and never reaching a runner at all."""
    record = _load(827)
    without = record["without_comments"]["extraction"]["moving_parts"]
    with_ = record["with_comments"]["extraction"]["moving_parts"]
    assert without["repo_version"] is None
    assert with_["repo_version"] == "1.5.5"


def test_827_with_comments_targets_the_real_ops_api():
    """Without comments, the extraction demonstrates a bare Python `typing.
    Mapping` quirk that never touches `ops` at all. With comments, it
    targets `ops.model.ConfigData` -- the actual class the 2024 follow-up
    names."""
    record = _load(827)
    without_commands = " ".join(record["without_comments"]["extraction"]["commands"])
    with_commands = " ".join(record["with_comments"]["extraction"]["commands"])
    assert "ops" not in without_commands.lower() or "typing" in without_commands.lower()
    assert "configdata" in with_commands.lower()


def test_2185_in_scope_unchanged_at_this_sample():
    """Recorded honestly, not massaged: at n=1, under the current
    in_scope criteria, the body's own design-question language already
    lands #2185 on in_scope=false with no comments needed. Caveat: #2185 is
    independently known to be unstable across repeats -- this just pins
    what this particular recorded sample actually said."""
    record = _load(2185)
    assert record["without_comments"]["extraction"]["in_scope"] is False
    assert record["with_comments"]["extraction"]["in_scope"] is False


def test_2639_in_scope_false_both_variants_flagged_not_fixed():
    """Tangential finding, pinned so it isn't silently lost: this live
    sample calls the project's only-ever-reproduced flagship case out-of-scope
    even without comments, which the committed hand/fixture extraction
    (in_scope=true) disagrees with. This is a possible in_scope false-drop
    worth a repeated-sample follow-up (see `test_2639_in_scope_repeats.py`) --
    not fixed here, and not a regression in harness code (the deterministic
    filter's DROP_LABELS still protects #2639 upstream of this LLM
    second-opinion stage)."""
    record = _load(2639)
    assert record["without_comments"]["extraction"]["in_scope"] is False
    assert record["with_comments"]["extraction"]["in_scope"] is False


def test_all_six_calls_returned_schema_valid_json():
    for issue in (2185, 2639, 827):
        record = _load(issue)
        assert record["without_comments"]["status"] == "ok"
        assert record["with_comments"]["status"] == "ok"
