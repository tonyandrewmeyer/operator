"""`in_scope` prompt retune attempt, pinned against the recorded runs --
a negative result.

`spike-step-5/2639-in-scope/RESULT.md` measured a real, repeatable `#2639`
`in_scope` false-drop under the tightened prompt (75% without comments, 25%
with, recorded the day before this pass) and named this task explicitly:
retune the prompt, and prove the retune costs nothing on the 60-issue corpus
(`spike-step-5/corpus-v2/`), where the last refinement holds a precision >=
100% / recall >= 92% floor. Full write-up:
`spike-step-5/inscope-retune/RESULT.md`.

Three structurally different, principled prompt revisions were tried (v1:
broadened what counts as a "promise"; v2: same framing, narrowed promise
sources and added explicit anti-loosening language; v3: minimal edit to the
ambiguity tie-break paragraph only, the task's own named hypothesis). All
three measurably improved `#2639`/`#2045`'s false-drop rate, and all three
failed the corpus-v2 precision/recall floor on at least one of two repeat
runs. Per the task's explicit exit condition, no prompt change shipped:
`harness/extraction.py`'s `in_scope` prompt is byte-identical to what it was
before this task ran. These tests pin the negative result -- what was tried,
what it cost -- against `spike-step-5/inscope-retune/out/`, not a live
model: no network, no key, deterministic -- same pattern as
`test_2639_in_scope_repeats.py` / `test_comments_live_ab.py` /
`test_live_calibration.py`.
"""

import json
from pathlib import Path

import pytest

_OUT = Path(__file__).parent.parent.parent / "spike-step-5" / "inscope-retune" / "out"
_EXTRACTION_PY = Path(__file__).parent.parent / "extraction.py"


@pytest.fixture(autouse=True)
def _require_run():
    if not _OUT.is_dir():  # pragma: no cover - spike output not always present
        pytest.skip(f"inscope-retune run not available at {_OUT}")


def _summary(subdir: str) -> dict:
    path = _OUT / subdir / "summary.json"
    if not path.is_file():
        pytest.skip(f"{path} not recorded")
    return json.loads(path.read_text())


def _corpus_run(subdir: str) -> dict[int, dict]:
    run_dir = _OUT / subdir
    if not run_dir.is_dir():
        pytest.skip(f"{run_dir} not recorded")
    out = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.stem == "summary":
            continue
        record = json.loads(path.read_text())
        if record.get("status") == "ok":
            out[int(path.stem)] = record
    return out


def _score_against_labels(run: dict[int, dict]) -> dict:
    labels_path = Path(__file__).parent.parent.parent / "spike-step-5" / "corpus-v2" / "hand-labels.json"
    labels = json.loads(labels_path.read_text())["labels"]
    tp = fp = fn = 0
    false_keeps, false_drops = [], []
    for key, entry in labels.items():
        number = int(key)
        record = run.get(number)
        if record is None:
            continue
        predicted = record["live_extraction"]["in_scope"]
        actual = entry["label"]
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
            false_keeps.append(number)
        elif not predicted and actual:
            fn += 1
            false_drops.append(number)
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_keeps": sorted(false_keeps),
        "false_drops": sorted(false_drops),
    }


# --- no prompt change shipped ----------------------------------------------


def test_extraction_prompt_unchanged_after_negative_result():
    # The task's explicit exit condition: if the retune can't hold the
    # corpus floor, do not land a prompt change. All three attempts failed
    # it (see below), so extraction.py's in_scope prompt is reverted to
    # exactly what it was before this task -- these phrases (present before
    # and never removed) and the *absence* of every attempt's added text
    # together pin that.
    text = _EXTRACTION_PY.read_text()
    assert "But note what that tie-break is *for*" in text
    assert (
        "An issue that plainly states\nsomething failed is not ambiguous, and this rule is not a reason to drop\nit."
        in text
    )
    # None of the three attempts' additions survived.
    assert "Current behavior" not in text
    assert "Current-behavior" not in text
    assert "concrete non-ambiguous case" not in text.lower()


# --- baseline (today, current/unmodified prompt) ---------------------------


def test_2639_baseline_today_recorded_at_n8_both_variants():
    summary = _summary("2639-baseline")
    assert summary["without_comments"]["n"] == 8
    assert summary["with_comments"]["n"] == 8


def test_2639_baseline_today_drifted_from_yesterdays_recorded_rate():
    # Worth pinning on its own: the unmodified prompt's false-drop rate
    # moved from 75%/25% (recorded the day before, `../2639-in-scope/out/`)
    # to 25%/25% measured today, same prompt, same issue, same n. The model
    # is not seeded -- this is real day-to-day drift, not a regression, and
    # it means any single-run comparison against yesterday's fixture would
    # have been comparing against stale noise rather than a stable target.
    summary = _summary("2639-baseline")
    assert summary["without_comments"]["dropped"] == 2
    assert summary["with_comments"]["dropped"] == 2


def test_2045_baseline_today_recorded_at_n8():
    summary = _summary("2045-baseline")
    assert summary["n"] == 8
    assert summary["dropped"] == 1  # 12.5%, the anecdote's 1/3 in measured form


# --- v1: broadened promise sources -- fixed #2639/#2045, broke precision --


def test_v1_fixed_2639_and_2045_at_n8():
    s2639 = _summary("2639-retuned-v1")
    assert s2639["without_comments"]["dropped"] == 0
    assert s2639["with_comments"]["dropped"] == 0
    s2045 = _summary("2045-retuned-v1")
    assert s2045["dropped"] == 0


def test_v1_reintroduced_1460_as_a_false_keep_in_both_corpus_runs():
    # #1460 ("should actions leak deprecation warnings?") is exactly the
    # false keep the original in_scope refinement fixed. v1's broadened
    # promise-source list (adding "design, or stated recommended usage")
    # let the model read the reporter's own design opinion as a promise --
    # reproduced identically in two independent 60-issue runs, so this is a
    # systematic regression, not sampling noise.
    for subdir in ("corpus-run1-v1", "corpus-run2-v1"):
        score = _score_against_labels(_corpus_run(subdir))
        assert 1460 in score["false_keeps"]
        assert score["precision"] < 1.0
        assert score["recall"] < 0.92


# --- v2: narrowed promise sources -- made both worse ------------------------


def test_v2_made_2639_and_2045_worse_than_baseline():
    s2639 = _summary("2639-retuned-v2")
    baseline = _summary("2639-baseline")
    assert s2639["without_comments"]["dropped"] > baseline["without_comments"]["dropped"]
    assert s2639["with_comments"]["dropped"] > baseline["with_comments"]["dropped"]
    s2045 = _summary("2045-retuned-v2")
    baseline_2045 = _summary("2045-baseline")
    assert s2045["dropped"] > baseline_2045["dropped"]


def test_v2_still_below_corpus_floor():
    score = _score_against_labels(_corpus_run("corpus-run1-v2"))
    assert score["precision"] < 1.0
    assert score["recall"] < 0.92


# --- v3: minimal tie-break edit -- improved but still below floor ----------


def test_v3_improved_2639_and_2045_vs_baseline():
    s2639 = _summary("2639-retuned-v3")
    baseline = _summary("2639-baseline")
    assert s2639["without_comments"]["dropped"] <= baseline["without_comments"]["dropped"]
    assert s2639["with_comments"]["dropped"] <= baseline["with_comments"]["dropped"]
    s2045 = _summary("2045-retuned-v3")
    assert s2045["dropped"] == 0


def test_v3_never_holds_both_floors_in_the_same_run():
    # run1: 11/12 recall (91.7%, just under the 92% floor) and 92%
    # precision (new false keep #1398). run2: precision hits the floor
    # (100%) but recall doesn't (83%, two false drops). Neither run clears
    # both floors simultaneously -- the measured basis for calling this a
    # failed retune rather than landing a prompt that merely "looks fine
    # on one run".
    run1 = _score_against_labels(_corpus_run("corpus-run1-v3"))
    run2 = _score_against_labels(_corpus_run("corpus-run2-v3"))
    assert run1["recall"] < 0.92
    assert run1["precision"] < 1.0
    assert run2["precision"] == pytest.approx(1.0)
    assert run2["recall"] < 0.92
