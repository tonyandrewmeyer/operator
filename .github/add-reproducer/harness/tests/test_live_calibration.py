"""Live-run calibration, pinned against the recorded runs.

Four real `deepseek/deepseek-chat` runs over the same seven
`canonical/operator` issues are recorded, one per prompt/schema fix. These
tests document how substrate, `in_scope` and confidence agreement moved
across those fixes -- and pin it, so a later prompt change that regresses
one of them fails here.

They assert against recorded JSON, not against a live model: no network, no
key, deterministic.
"""

import json
from pathlib import Path

import pytest

_LIVE_LLM = Path(__file__).parent.parent.parent / "spike-step-5" / "live-llm"
RUNS = {
    "original": _LIVE_LLM / "out-before-substrate-fix",
    "after_substrate_fix": _LIVE_LLM / "out-before-commands-fix",
    "after_commands_fix": _LIVE_LLM / "out-before-inscope-fix",
    "current": _LIVE_LLM / "out",
}
# Three further runs of the same config that produced `out/`, for separating a
# real change from sampling noise. The model is not seeded.
REPEATS = sorted((_LIVE_LLM / "repeats").glob("run*")) if (_LIVE_LLM / "repeats").is_dir() else []

# The hand-derived baseline, from harness/fixtures/extractions/. #2185 has
# no hand extraction, so it is excluded.
HAND = {
    2045: {"in_scope": True, "substrate": "none", "confidence": "medium"},
    2304: {"in_scope": False, "substrate": None, "confidence": "low"},
    2327: {"in_scope": True, "substrate": "none", "confidence": "medium"},
    2341: {"in_scope": True, "substrate": "none", "confidence": "high"},
    2484: {"in_scope": True, "substrate": "k8s", "confidence": "low"},
    2639: {"in_scope": True, "substrate": "k8s", "confidence": "low"},
}


def _load(run_dir: Path) -> dict[int, dict]:
    out = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.stem == "summary":
            continue
        record = json.loads(path.read_text())
        if record.get("status") == "ok":
            out[int(path.stem)] = record["live_extraction"]
    return out


def _agreement(run_dir: Path, field: str) -> int:
    extractions = _load(run_dir)
    hits = 0
    for number, expected in HAND.items():
        extraction = extractions.get(number)
        if extraction is None:
            continue
        actual = (
            extraction["moving_parts"].get("substrate")
            if field == "substrate"
            else extraction.get(field)
        )
        if actual == expected[field]:
            hits += 1
    return hits


@pytest.fixture(autouse=True)
def _require_runs():
    missing = [name for name, path in RUNS.items() if not path.exists()]
    if missing:  # pragma: no cover - recorded runs not always present
        pytest.skip(f"live run(s) not available: {missing}")


@pytest.mark.parametrize(
    "run,expected",
    # Substrate: requiring the field fixed nulls routed to k8s-scratch. The
    # remaining gap is #2304, whose hand value is null because it was dropped.
    [("original", 1), ("after_substrate_fix", 5), ("after_commands_fix", 5), ("current", 5)],
)
def test_substrate_agreement_progression(run, expected):
    assert _agreement(RUNS[run], "substrate") == expected


@pytest.mark.parametrize(
    "run,expected",
    # in_scope: spelling out the criteria fixed #2304, a cleanup request.
    [("original", 5), ("after_substrate_fix", 5), ("after_commands_fix", 5), ("current", 6)],
)
def test_in_scope_agreement_progression(run, expected):
    assert _agreement(RUNS[run], "in_scope") == expected


@pytest.mark.parametrize(
    "run,expected",
    # Confidence: recorded, but not a trend. Repeats of one config vary as
    # much as these do (see `test_confidence_agreement_is_noise_dominated`).
    [("original", 1), ("after_substrate_fix", 1), ("after_commands_fix", 5), ("current", 3)],
)
def test_confidence_agreement_progression(run, expected):
    assert _agreement(RUNS[run], "confidence") == expected


def test_confidence_agreement_is_noise_dominated():
    """Why the numbers above can't be read as a trend.

    `repeats/run{1,2,3}` are three further runs of the same prompt config that
    produced `out/`. If confidence agreement varies this much with nothing
    changed, a one-point difference between configs means nothing.
    """
    if not REPEATS:  # pragma: no cover - repeats not always present
        pytest.skip("repeat runs not available")
    scores = sorted(_agreement(d, "confidence") for d in [RUNS["current"], *REPEATS])
    assert len(scores) == 4
    assert max(scores) - min(scores) >= 2, (
        f"expected confidence agreement to vary across identical runs, got {scores}"
    )


def test_2304_stays_out_of_scope_across_every_repeat():
    """The robust form of the #2304 in_scope result.

    A single run showing `in_scope: false` for #2304 could be luck. All four
    samples of the current config agree, which is the claim worth making.
    """
    if not REPEATS:  # pragma: no cover - repeats not always present
        pytest.skip("repeat runs not available")
    for run_dir in [RUNS["current"], *REPEATS]:
        assert _load(run_dir)[2304]["in_scope"] is False, run_dir.name


def test_in_scope_tightening_can_also_drop_a_real_bug():
    """The cost of the asymmetry, recorded rather than glossed over.

    The in_scope criteria tell the model to drop when genuinely ambiguous,
    which is deliberate (a wrong drop costs a human triage, a wrong keep
    costs maintainer attention). One of the four samples takes that far
    enough to drop #2484, a real bug in the hand baseline. Worth knowing
    before the workflow posts anything: this trades a false-comment risk for an
    occasional false drop, which is the intended direction but not free.
    """
    if not REPEATS:  # pragma: no cover - repeats not always present
        pytest.skip("repeat runs not available")
    dropped = [
        d.name for d in [RUNS["current"], *REPEATS] if _load(d)[2484]["in_scope"] is False
    ]
    assert dropped, "expected at least one sample to false-drop #2484"


def test_2304_false_keep_is_closed():
    """The #2304 in_scope result, stated as the thing that actually matters.

    #2304 is a maintainer-facing "this check is obsolete, remove it" report:
    nobody hit a failure, and a reproduction would only re-assert what the
    code plainly does. Every run before the in_scope criteria landed kept it,
    and the last one attached `confidence: high` to it.
    """
    assert _load(RUNS["after_commands_fix"])[2304]["in_scope"] is True
    assert _load(RUNS["after_commands_fix"])[2304]["confidence"] == "high"
    current = _load(RUNS["current"])[2304]
    assert current["in_scope"] is False
    assert current["confidence"] == "low"


def test_every_run_returned_schema_valid_json_for_all_seven_issues():
    for name, path in RUNS.items():
        records = [
            json.loads(p.read_text()) for p in path.glob("*.json") if p.stem != "summary"
        ]
        assert len(records) == 7, name
        assert all(r["status"] == "ok" for r in records), name
