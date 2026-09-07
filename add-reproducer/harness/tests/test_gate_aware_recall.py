"""Pins `spike-step-5/gate-aware-recall/RESULT.md`'s finding: raw `in_scope`
recall (92-100% across every recorded run) overstates what a maintainer
would actually see, because `pipeline.py`'s confidence gate silences any
in-scope hypothesis with `confidence: low` before a comment is ever
composed. `corpus-v2/analyse.py.score()` now reports both numbers; these
tests pin the effective figure against the already-recorded `out/`
directories, not a live run. No network, no key.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPIKE_STEP_5 = Path(__file__).parent.parent.parent / "spike-step-5"
_CORPUS_V2 = _SPIKE_STEP_5 / "corpus-v2"
_LABELS_PATH = _CORPUS_V2 / "hand-labels.json"


def _load_analyse():
    spec = importlib.util.spec_from_file_location("analyse", _CORPUS_V2 / "analyse.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyse"] = module
    spec.loader.exec_module(module)
    return module


analyse = _load_analyse()


@pytest.fixture(autouse=True)
def _require_runs():
    if not _LABELS_PATH.is_file():  # pragma: no cover - spike output not always present
        pytest.skip(f"hand-labels not available at {_LABELS_PATH}")


def _score(run_dir: Path) -> dict:
    if not run_dir.is_dir():
        pytest.skip(f"{run_dir} not recorded")
    labels = json.loads(_LABELS_PATH.read_text())["labels"]
    run = analyse.load_run(run_dir)
    return analyse.score(run, labels)


def _record(*, in_scope: bool, confidence: str) -> dict:
    return {
        "live_extraction": {"in_scope": in_scope, "confidence": confidence, "observed": ""},
        "runnability": {"runnable": False},
    }


# --- score() computes both numbers, on a synthetic run ----------------------


def test_effective_recall_excludes_low_confidence_true_positives():
    # Two true positives, one gated by low confidence -> raw recall 2/2,
    # effective recall 1/2.
    labels = {"1": {"label": True}, "2": {"label": True}}
    run = {
        1: _record(in_scope=True, confidence="high"),
        2: _record(in_scope=True, confidence="low"),
    }
    s = analyse.score(run, labels)
    assert s["recall"] == pytest.approx(1.0)
    assert s["effective_recall"] == pytest.approx(0.5)
    assert s["confidence_gated_true_positives"] == [2]


def test_effective_recall_equals_raw_recall_when_no_low_confidence_hits():
    labels = {"1": {"label": True}}
    run = {1: _record(in_scope=True, confidence="high")}
    s = analyse.score(run, labels)
    assert s["recall"] == s["effective_recall"] == pytest.approx(1.0)
    assert s["confidence_gated_true_positives"] == []


def test_confidence_gate_never_touches_false_positives_or_drops():
    # The gate only silences in-scope *true* positives; a low-confidence
    # false keep or an already-dropped false negative isn't "gated" by it
    # in the sense this metric cares about.
    labels = {"1": {"label": False}, "2": {"label": True}}
    run = {
        1: _record(in_scope=True, confidence="low"),
        2: _record(in_scope=False, confidence="high"),
    }
    s = analyse.score(run, labels)
    assert s["confidence_gated_true_positives"] == []
    assert s["fp"] == 1
    assert s["fn"] == 1


# --- against the recorded corpus-v2 runs -------------------------------------


@pytest.mark.parametrize(
    "subdir, expected_raw, expected_effective, expected_gated",
    [
        ("corpus-twopass-run1", 11 / 12, 9 / 12, [1329, 2484]),
        ("corpus-twopass-run2", 11 / 12, 10 / 12, [2484]),
    ],
)
def test_twopass_runs_effective_recall(subdir, expected_raw, expected_effective, expected_gated):
    s = _score(_SPIKE_STEP_5 / "inscope-instrument" / "out" / subdir)
    assert s["recall"] == pytest.approx(expected_raw)
    assert s["effective_recall"] == pytest.approx(expected_effective)
    assert s["confidence_gated_true_positives"] == expected_gated


def test_confirm_run_raw_recall_is_perfect_but_effective_is_not():
    # This is the sharpest instance of the finding: raw recall reads 100%
    # (12/12) -- a maintainer would still see comments for only 10/12,
    # because #1329 and #2484 are both `confidence: low`.
    s = _score(_CORPUS_V2 / "out-inscope-v2-confirm")
    assert s["recall"] == pytest.approx(1.0)
    assert s["effective_recall"] == pytest.approx(10 / 12)
    assert s["confidence_gated_true_positives"] == [1329, 2484]
