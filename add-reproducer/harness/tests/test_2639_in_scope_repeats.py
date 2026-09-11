"""#2639 `in_scope` false-drop measurement, pinned against the recorded runs.

`spike-step-5/composer-live/RESULT.md`'s 2026-07-30 addendum flagged (n=1,
explicitly unmeasured) that a live A/B sample called `#2639` -- the project's
only ever-successfully-reproduced case -- `in_scope: false` under the current
tightened `in_scope` prompt, even without comments. `spike-step-5/2639-in-scope/
run_2639_repeats.py` repeats the `in_scope` classification 8 times per variant
(real comments / `comments=[]`) against the real issue capture
(`spike-step-5/comments/2639.json`) to check whether that was a real,
repeatable defect or n=1 noise.

These tests assert against the recorded JSON in `spike-step-5/2639-in-scope/
out/`, not a live model: no network, no key, deterministic -- same pattern as
`test_live_calibration.py` / `test_comments_live_ab.py`.

This measurement does NOT retune `extraction.py`'s `in_scope` prompt --
diagnosing and reporting the rate is the deliverable; a fix has its own
precision/recall consequences across the whole 60-issue corpus and needs its
own measured pass (see `spike-step-5/composer-live/RESULT.md`'s newest
addendum).
"""

import json
from pathlib import Path

import pytest

_RUN_DIR = Path(__file__).parent.parent.parent / "spike-step-5" / "2639-in-scope" / "out"


@pytest.fixture(autouse=True)
def _require_run():
    if not _RUN_DIR.is_dir():  # pragma: no cover - spike output not always present
        pytest.skip(f"2639 in_scope repeats run not available at {_RUN_DIR}")


def _load_runs() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(_RUN_DIR.glob("run_*.json"), key=lambda p: int(p.stem.split("_")[1]))]


def _load_summary() -> dict:
    return json.loads((_RUN_DIR / "summary.json").read_text())


def test_eight_repeats_recorded_for_each_variant():
    runs = _load_runs()
    assert len(runs) == 8
    for r in runs:
        assert r["without_comments"]["status"] == "ok"
        assert r["with_comments"]["status"] == "ok"


def test_2639_in_scope_false_drop_rate_is_high_without_comments():
    # The headline finding: this is not the earlier n=1 sample's noise --
    # 6/8 repeats drop the project's only-ever-reproduced flagship case when
    # comments aren't read at all.
    summary = _load_summary()
    assert summary["without_comments"]["n"] == 8
    assert summary["without_comments"]["dropped"] == 6
    assert summary["without_comments"]["false_drop_rate"] == pytest.approx(0.75)


def test_2639_in_scope_false_drop_rate_drops_but_does_not_vanish_with_comments():
    # Reading the real comment thread substantially reduces the false-drop
    # rate (implicitly: comments add context the tightened in_scope prompt
    # otherwise reads as "no concrete broken promise") but does not close it
    # -- 2/8 repeats still drop it even with the full real thread.
    summary = _load_summary()
    assert summary["with_comments"]["n"] == 8
    assert summary["with_comments"]["dropped"] == 2
    assert summary["with_comments"]["false_drop_rate"] == pytest.approx(0.25)


def test_2639_is_dropped_at_least_once_in_every_variant():
    # The robust form of the finding: not a one-off in either variant.
    runs = _load_runs()
    assert any(r["without_comments"]["in_scope"] is False for r in runs)
    assert any(r["with_comments"]["in_scope"] is False for r in runs)


def test_2639_confidence_is_never_high_across_any_repeat():
    # Every recorded repeat (both variants, dropped or kept) landed on
    # low/medium confidence, never high -- worth carrying alongside the
    # false-drop rate itself: even the runs that correctly keep #2639
    # in_scope don't do so with strong confidence.
    runs = _load_runs()
    confidences = {r["without_comments"]["confidence"] for r in runs} | {r["with_comments"]["confidence"] for r in runs}
    assert confidences <= {"low", "medium"}
