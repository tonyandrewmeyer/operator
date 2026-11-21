"""Pins the no-ship decision from `spike-step-5/model-comparison/RESULT.md`:
three candidate OpenRouter models were measured against the incumbent
`deepseek/deepseek-chat`, none beat its corpus-v2 floor, so the default
model is unchanged. Same pattern as `test_inscope_instrument.py`: reads
recorded `out/*` fixtures and skips (not fails) if that directory isn't
present. No network, no key.
"""

import json
from pathlib import Path

import pytest

from seams.llm import DEFAULT_MODEL

_OUT = Path(__file__).parent.parent.parent / "spike-step-5" / "model-comparison" / "out"
_LABELS_PATH = Path(__file__).parent.parent.parent / "spike-step-5" / "corpus-v2" / "hand-labels.json"


def test_default_model_unchanged_by_the_no_ship_decision():
    # The comparison's own conclusion: no candidate beat the incumbent's
    # corpus floor, so the seam's default must still be the incumbent.
    assert DEFAULT_MODEL == "deepseek/deepseek-chat"


@pytest.fixture(autouse=True)
def _require_run():
    if not _OUT.is_dir():  # pragma: no cover - spike output not always present
        pytest.skip(f"model-comparison run not available at {_OUT}")


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
    false_keeps = []
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
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_keeps": sorted(false_keeps),
    }


def test_gpt41_twopass_corpus_misses_the_precision_floor():
    # gpt-4.1 was the closest contender (92% precision / 92% recall in the
    # two-pass condition) but still falls short of the incumbent's 100%
    # precision floor -- pinned here so a future re-run has a concrete bar.
    score = _score_against_labels(_corpus_run("corpus-twopass-gpt-4.1"))
    assert score["precision"] < 1.0
    assert 1460 in score["false_keeps"]


def test_gpt41_1460_false_keep_reads_reporter_opinion_as_a_promise():
    # The same failure mode the original in_scope refinement fixed: an
    # Expected-behaviour sentence about a DeprecationWarning read as a
    # broken promise rather than a design opinion.
    run = _corpus_run("corpus-twopass-gpt-4.1")
    record = run[1460]
    assert record["live_extraction"]["in_scope"] is True


def test_mistral_small_produces_zero_runnable_commands():
    # The categorical disqualifier: every in-scope extraction's commands[]
    # gets flattened (multi-line heredoc content split into separate array
    # entries) into something runnability.assess() rejects.
    run = _corpus_run("corpus-single-mistral-small-3.2")
    in_scope = [r for r in run.values() if r["live_extraction"]["in_scope"]]
    assert in_scope, "expected at least one in-scope extraction to check"
    assert all(not r["runnability"]["runnable"] for r in in_scope)
