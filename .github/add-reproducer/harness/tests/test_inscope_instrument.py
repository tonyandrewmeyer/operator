"""Pins the measured results behind `inscope_second_pass.py`'s ship decision
(`spike-step-5/inscope-instrument/RESULT.md`) against the recorded runs --
not a live model, no network, no key. Same pattern as
`test_inscope_retune.py`: these tests read `spike-step-5/inscope-instrument/
out/*` fixtures and skip (not fail) if that directory isn't present.
"""

import json
from pathlib import Path

import pytest

_OUT = Path(__file__).parent.parent.parent / "spike-step-5" / "inscope-instrument" / "out"
_LABELS_PATH = Path(__file__).parent.parent.parent / "spike-step-5" / "corpus-v2" / "hand-labels.json"


@pytest.fixture(autouse=True)
def _require_run():
    if not _OUT.is_dir():  # pragma: no cover - spike output not always present
        pytest.skip(f"inscope-instrument run not available at {_OUT}")


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
    labels = json.loads(_LABELS_PATH.read_text())["labels"]
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


def _repeats_summary(subdir: str, number: int) -> dict:
    path = _OUT / subdir / f"{number}.json"
    if not path.is_file():
        pytest.skip(f"{path} not recorded")
    return json.loads(path.read_text())


# --- step 1: the rationale-instrumented single-pass corpus run -------------


def test_rationale_instrumented_corpus_run_precision_holds_recall_drops():
    # Adding the rationale field alone (no decision-logic change) still held
    # precision at 100% but landed at 75% recall (9/12) -- within this
    # project's established day-to-day noise band, and the run that
    # motivated broadening the second-pass gate to "every drop" rather than
    # "drops with a populated observed field" (see #1790/#2327 below).
    score = _score_against_labels(_corpus_run("corpus-instrumented"))
    assert score["precision"] == pytest.approx(1.0)
    assert score["false_keeps"] == []
    assert 1088 in score["false_drops"]


def test_rationale_run_shows_1790_and_2327_dropped_with_empty_observed():
    # The finding that changed the second-pass gate design: these two real
    # bugs were dropped with observed=="" in the rationale-instrumented run,
    # so a gate keyed on a populated `observed` field would have missed
    # them (see inscope_second_pass.py's module docstring).
    run = _corpus_run("corpus-instrumented")
    for number in (1790, 2327):
        record = run[number]
        assert record["live_extraction"]["in_scope"] is False
        assert record["live_extraction"]["observed"] in (None, "")


# --- step 3: the two-pass corpus floor, two independent runs ---------------


@pytest.mark.parametrize("subdir", ["corpus-twopass-run1", "corpus-twopass-run2"])
def test_twopass_corpus_run_precision_is_perfect(subdir):
    score = _score_against_labels(_corpus_run(subdir))
    assert score["precision"] == pytest.approx(1.0)
    assert score["false_keeps"] == []


@pytest.mark.parametrize("subdir", ["corpus-twopass-run1", "corpus-twopass-run2"])
def test_twopass_corpus_run_recall_ties_the_established_baseline(subdir):
    # 11/12 == 0.91666..., the same fraction spike-step-5/corpus-v2/RESULT.md
    # reports as "recall 92%" for its own out-inscope-v2 baseline run (see
    # this task's RESULT.md for the full discussion of the rounding vs.
    # strict->=0.92 nuance). Both runs land on this exact fraction, with the
    # same sole false drop (#1088) that predates this module.
    score = _score_against_labels(_corpus_run(subdir))
    assert score["recall"] == pytest.approx(11 / 12)
    assert score["false_drops"] == [1088]


def test_twopass_recovers_1790_and_2327_that_the_single_pass_dropped():
    # Both were false drops (empty observed) in the rationale-instrumented
    # single-pass run above; the second pass recovers both, in both runs.
    for subdir in ("corpus-twopass-run1", "corpus-twopass-run2"):
        run = _corpus_run(subdir)
        for number in (1790, 2327):
            assert run[number]["live_extraction"]["in_scope"] is True


def test_twopass_corpus_runs_are_identical_on_in_scope():
    # Zero variance across two independent runs -- far more stable than any
    # of the three abandoned single-prompt rewordings in ../inscope-retune/,
    # which each disagreed with themselves in different ways run to run.
    run1 = _corpus_run("corpus-twopass-run1")
    run2 = _corpus_run("corpus-twopass-run2")
    shared = sorted(set(run1) & set(run2))
    assert len(shared) == 60
    diffs = [n for n in shared if run1[n]["live_extraction"]["in_scope"] != run2[n]["live_extraction"]["in_scope"]]
    assert diffs == []


# --- flagship #2639 / #2045, two-pass vs. same-day single-pass baseline ----


def test_flagship_2639_false_drop_rate_improves_but_is_not_fixed():
    baseline = _repeats_summary("repeats", 2639)  # single-pass, same day
    twopass = _repeats_summary("flagship-twopass", 2639)
    assert baseline["false_drop_rate"] == pytest.approx(0.75)
    assert twopass["false_drop_rate"] == pytest.approx(0.625)
    # Explicitly not a fix: still dropped more often than kept.
    assert twopass["false_drop_rate"] > 0.5


def test_flagship_2045_stays_low_false_drop_with_two_pass():
    twopass = _repeats_summary("flagship-twopass", 2045)
    assert twopass["dropped"] == 0
