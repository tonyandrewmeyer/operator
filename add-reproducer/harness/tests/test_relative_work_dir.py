"""spike-step-5/gha-wallclock-2026-09-07/RESULT.md §3: a relative
`--out-dir` builds a relative `charm_dir`, and `seams/runner.py`'s
`_scratch_sequence()` embeds that verbatim into `juju deploy -m <target>
{charm_dir}/*.charm ...` -- a command that runs with `cwd` set to the
*issue's* directory, not wherever the relative path was anchored, and
which juju 4 additionally refuses outright once reached ("... is
ambiguous. To deploy a local charm or bundle, run `juju deploy ./out/...`").

Both VM sessions that measured this harness passed an absolute `--out-dir`
by chance and so never hit this; the first GHA `workflow_dispatch` run
that used `"$PWD/out"` only avoided it because that expands to an
absolute path too -- nothing in the harness itself normalised one.

These tests exercise the real failing shape end to end (a genuinely
relative `Path` handed to `Pipeline`, reaching a scratch branch's
`context["charm_dir"]`), not a synthetic absolute-vs-relative string
comparison."""

from pathlib import Path

import pytest

import pipeline as pipeline_module
from models import CommandResult, Issue, Outcome, RunResult
from pipeline import Pipeline, build_pipeline


class _StaticLLM:
    def __init__(self, extraction: dict, surface: dict):
        self._extraction = extraction
        self._surface = surface

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        return self._extraction if purpose == "extraction" else self._surface


class _CapturingRunner:
    """Like `test_pipeline_e2e._StaticRunner`, but keeps the `context`
    dict `run_for_issue()` built -- the thing that actually carries the
    bug -- so a test can assert on `charm_dir`/`workdir` directly instead
    of only on the final outcome."""

    def __init__(self, result: RunResult):
        self._result = result
        self.captured_context: dict | None = None

    def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
        return True

    def run(self, **kwargs) -> RunResult:
        self.captured_context = kwargs.get("context")
        return self._result


def _lxd_issue() -> Issue:
    return Issue(
        number=1, title="t", body="b", labels=[], state="OPEN", created_at="", author="a", repo="canonical/operator"
    )


@pytest.fixture(autouse=True)
def _stub_render_charm(monkeypatch):
    # `scaffold.render_charm()` shells out to `uv lock` for real (network
    # required); these tests are about work-dir path handling, not about
    # exercising the real charm render, so stub it to just create the
    # directory `context["charm_dir"]` is later built from.
    def _fake_render_charm(surface, out_dir: Path, *, source_issue: int | None = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    monkeypatch.setattr(pipeline_module, "render_charm", _fake_render_charm)


def test_pipeline_resolves_a_relative_work_dir_to_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = Path("relative-out") / "work"
    assert not relative.is_absolute()

    pipeline = build_pipeline(fixture_mode=True, work_dir=relative)

    assert pipeline.work_dir.is_absolute()
    assert pipeline.work_dir == (tmp_path / "relative-out" / "work").resolve()


def test_relative_work_dir_produces_an_absolute_charm_dir_and_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm = _StaticLLM(
        extraction={
            "in_scope": True,
            "moving_parts": {"substrate": "lxd", "repo_version": "main"},
            "commands": [],
            "expected": "the charm reacts to the reported condition",
            "observed": "it does not",
            "confidence": "high",
        },
        surface={
            "charm_name": "repro-i1-x",
            "pebble_service": {"service": "workload", "user": "ubuntu", "command": "pebble notify x"},
        },
    )
    run_result = RunResult(
        hypothesis_number=1,
        branch="lxd-scratch",
        commands=[
            CommandResult(command="sudo concierge prepare -p lxd", exit_code=0, stdout="ready"),
            CommandResult(command="juju status repro/0", exit_code=0, stdout="Workload: active"),
        ],
    )
    runner = _CapturingRunner(run_result)

    relative = Path("relative-out") / "work"
    assert not relative.is_absolute()
    pipeline = Pipeline(llm, runner, work_dir=relative)

    result = pipeline.run_for_issue(_lxd_issue(), calibration_mode=True)

    assert result.stage_reached == "classifier"
    assert result.outcome == Outcome.DID_NOT_REPRODUCE
    assert runner.captured_context is not None
    assert Path(runner.captured_context["charm_dir"]).is_absolute()
    assert Path(runner.captured_context["workdir"]).is_absolute()


def test_default_cli_out_dir_is_relative_and_pipeline_still_resolves_it(tmp_path, monkeypatch):
    # `pipeline.py main()`'s own `--out-dir` default is
    # `Path("add-reproducer-out")` -- relative -- so an invocation with no
    # `--out-dir` flag at all hits exactly this shape, not only a workflow
    # that passes one explicitly.
    monkeypatch.chdir(tmp_path)
    default_out_dir = Path("add-reproducer-out")
    assert not default_out_dir.is_absolute()

    pipeline = build_pipeline(fixture_mode=True, work_dir=default_out_dir / "work")
    assert pipeline.work_dir.is_absolute()
