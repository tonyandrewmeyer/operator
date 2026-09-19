"""The `none` branch must not run inside whatever Python project happens to
enclose its scratch directory.

`spike-step-5/second-dispatch/RESULT.md` §6.4 saw this in the composed
comment and did not act on it; `third-dispatch/RESULT.md` §7 carried it
forward unobserved for a second round. `fourth-dispatch/RESULT.md` §2
reproduced it: the branch's first command is the extraction's own `uv init
--bare .`, uv walks *up* from the working directory looking for a workspace
root, and the shipped layout (`--out-dir "$PWD/out"` from
`add-reproducer/harness`) hands it the harness itself. Measured consequences,
all three from one run of `#2045`'s real four commands:

  - `add-reproducer/harness/pyproject.toml`, a tracked file, gains a
    `[tool.uv.workspace] members` entry naming the scratch directory;
  - the issue's `ops` is installed into the harness's own `.venv`, so two
    issues in one batch share a dependency resolution -- and the second of
    them, if it pins an older `ops`, fails to install at all ("your
    workspace's requirements are unsatisfiable", exit 1);
  - the issue's pytest runs under the harness's `[tool.pytest.ini_options]`,
    whose `pythonpath = ["."]` puts the harness's own modules on the
    reproducer's `sys.path`.

The fix gives each issue a private workspace root instead of rewriting any
command: `uv init --bare .` stays exactly the string the extraction wrote,
which `classifier.py`'s rung 0c keys on and which the composed comment tells
a reader to paste.

These tests exercise the real directory layout and the real `pyproject.toml`
the harness ships. They do not shell out to uv -- `prepare_scratch_project()`
tolerates its absence by design, and the behaviour that needs a network is
recorded in the RESULT rather than asserted here."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import runner_stage
import seams.runner
from models import CommandResult, Hypothesis, Issue, MovingParts, RunResult
from pipeline import _enclosing_python_project, _scratch_root, build_pipeline
from seams.runner import scratch_command_env

_HARNESS_ROOT = Path(__file__).resolve().parent.parent


def _load_issue(number: int) -> Issue:
    import json

    return Issue.from_dict(json.loads((_HARNESS_ROOT / "fixtures" / "issues" / f"{number}.json").read_text()))


@pytest.fixture(autouse=True)
def _no_uv_sync(monkeypatch):
    """`prepare_scratch_project()`'s sync needs a network and a package
    index. Stub it out: these tests are about the layout it produces."""
    calls: list[Path] = []
    real = runner_stage.subprocess.run

    def fake(args, **kwargs):
        if args[:2] == ["uv", "sync"]:
            calls.append(Path(kwargs["cwd"]))
            raise FileNotFoundError("uv: command not found")
        return real(args, **kwargs)

    monkeypatch.setattr(runner_stage.subprocess, "run", fake)
    return calls


def test_the_shipped_harness_is_a_python_project_and_that_is_the_trap():
    """Not a tautology: it is the precondition the whole defect rests on, and
    it is a property of a file that can change. `out/` under this directory
    is where `run.py --out-dir "$PWD/out"` puts the scratch tree."""
    assert (_HARNESS_ROOT / "pyproject.toml").is_file()
    assert _enclosing_python_project(_HARNESS_ROOT / "out" / "work") == _HARNESS_ROOT


def test_the_harness_pyproject_still_injects_its_own_directory_into_pytest():
    """`pythonpath = ["."]` is the third consequence above, and the one a
    reader of the composed comment cannot see. If this line ever goes, the
    isolation is still wanted for the other two -- but the RESULT's claim
    about `sys.path` would need re-measuring."""
    assert 'pythonpath = ["."]' in (_HARNESS_ROOT / "pyproject.toml").read_text()


def test_a_scratch_root_inside_a_project_is_relocated_out_of_it(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'enclosing'\nversion = '0'\n")
    work_dir = tmp_path / "out" / "work"
    work_dir.mkdir(parents=True)

    root = _scratch_root(work_dir)

    assert root != work_dir
    assert _enclosing_python_project(root) is None


def test_a_scratch_root_outside_every_project_is_left_alone(tmp_path):
    """Relocation is a repair, not a policy -- a caller who already passed a
    clean directory keeps it, so the run record stays where they put it."""
    work_dir = tmp_path / "out" / "work"
    work_dir.mkdir(parents=True)
    assert _enclosing_python_project(work_dir) is None

    assert _scratch_root(work_dir) == work_dir


def test_a_pipeline_pointed_into_a_project_puts_its_scratch_tree_elsewhere(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'enclosing'\nversion = '0'\n")

    pipeline = build_pipeline(fixture_mode=True, work_dir=tmp_path / "out" / "work")

    assert pipeline._scratch_root is None, "resolved eagerly; a pipeline that never runs should make no directory"
    assert _enclosing_python_project(pipeline.scratch_root) is None
    # `work_dir` itself is unchanged: the run record still lands where the
    # caller asked for it, and only the directory commands execute in moves.
    assert pipeline.work_dir == (tmp_path / "out" / "work").resolve()


def test_prepare_scratch_project_makes_the_issue_dir_a_workspace_root(tmp_path):
    workdir = runner_stage.prepare_scratch_project(tmp_path / "issue-2045")

    # The commands run one level *below* the root, so uv's upward walk stops
    # at a project this run owns instead of one it found.
    assert workdir == tmp_path / "issue-2045" / runner_stage.SCRATCH_WORK_SUBDIR
    assert workdir.is_dir()
    assert _enclosing_python_project(workdir) == tmp_path / "issue-2045"


def test_the_private_root_supplies_the_pytest_the_extraction_does_not(tmp_path):
    """The part of the fix that is not obvious, and the one that would have
    broken the dominant extraction shape if it had been left out.

    `#2045`'s four commands install `ops[testing]` and nothing else -- no
    pytest anywhere in them. Under the old layout they worked only because
    the harness's own dev group had already put pytest in the venv uv chose,
    so the contamination was load-bearing. `fourth-dispatch/RESULT.md` §2.3
    measures what isolating without replacing it does: `ModuleNotFoundError:
    No module named 'ops'` at collection, exit 2, a reproduction turned into
    a broken run."""
    runner_stage.prepare_scratch_project(tmp_path / "issue-2045")

    pyproject = (tmp_path / "issue-2045" / "pyproject.toml").read_text()
    assert "pytest" in pyproject
    # One dev dependency and no more: this replaces what the harness was
    # supplying by accident, it is not a place to put a standard environment.
    assert 'dev = ["pytest"]' in pyproject
    assert "dependencies = []" in pyproject


def test_prepare_scratch_project_syncs_the_root_before_the_commands_run(_no_uv_sync, tmp_path):
    """Sync order is load-bearing: `uv add` from a workspace member syncs
    that member only, so a pytest that is not already in the environment when
    the extraction's own install runs never arrives."""
    runner_stage.prepare_scratch_project(tmp_path / "issue-2045")

    assert _no_uv_sync == [tmp_path / "issue-2045"]


def test_prepare_scratch_project_survives_having_no_uv(_no_uv_sync, tmp_path):
    """The autouse fixture raises `FileNotFoundError` for every `uv sync`, so
    this whole module has been running the no-uv path throughout. Asserted
    once, explicitly: a scratch setup that cannot complete must leave a
    usable working directory and let the run go on to say something true
    about what happened."""
    workdir = runner_stage.prepare_scratch_project(tmp_path / "issue-77")

    assert workdir.is_dir()
    assert (tmp_path / "issue-77" / "pyproject.toml").is_file()


def test_only_the_none_branch_gets_a_private_root(tmp_path, monkeypatch):
    """Scoped on purpose, and exercised through the real branch routing
    rather than asserted from the code. `k8s-clone` clones a repository into
    its scratch directory, and giving that checkout an enclosing workspace
    root is a change this project has no live run to check.

    `#2045` is `substrate: none`; `#2484` carries a `ci_run_url` with no
    self-contained snippet and so routes to `k8s-clone`."""
    prepared: list[Path] = []
    real = runner_stage.prepare_scratch_project

    def spy(issue_dir: Path) -> Path:
        prepared.append(issue_dir)
        return real(issue_dir)

    monkeypatch.setattr(runner_stage, "prepare_scratch_project", spy)
    fixtures = _HARNESS_ROOT / "fixtures"

    pipeline = build_pipeline(fixture_mode=True, fixtures_dir=fixtures, work_dir=tmp_path / "work")
    pipeline.run_for_issue(_load_issue(2484), calibration_mode=True)
    assert prepared == []

    pipeline.run_for_issue(_load_issue(2045), calibration_mode=True)
    assert [d.name for d in prepared] == ["issue-2045"]


def test_the_none_branch_runs_one_level_below_its_private_root(tmp_path):
    """What the runner seam is actually handed. The working directory the
    extraction's `uv init --bare .` executes in has to be *inside* the
    private root, or uv walks past it and finds whatever encloses the scratch
    tree -- which is the whole defect."""
    captured: dict = {}

    class _CapturingRunner:
        def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
            return True

        def run(self, **kwargs) -> RunResult:
            captured.update(kwargs["context"])
            return RunResult(
                hypothesis_number=2045,
                branch="none",
                commands=[CommandResult(command="true", exit_code=0)],
            )

    fixtures = _HARNESS_ROOT / "fixtures"
    pipeline = build_pipeline(fixture_mode=True, fixtures_dir=fixtures, work_dir=tmp_path / "work")
    pipeline.runner = _CapturingRunner()

    pipeline.run_for_issue(_load_issue(2045), calibration_mode=True)

    workdir = Path(captured["workdir"])
    assert workdir.name == runner_stage.SCRATCH_WORK_SUBDIR
    assert workdir.parent.name == "issue-2045"
    assert _enclosing_python_project(workdir) == workdir.parent


# --- uv's environment-selection variables ---
#
# `fourth-dispatch/RESULT.md` §2.4. Giving the scratch project a workspace
# root of its own closes uv's *directory* walk, and nothing else: `uv` picks
# its target environment from `VIRTUAL_ENV`/`UV_PROJECT_ENVIRONMENT` before it
# looks at any directory at all. The harness runs under `uv run`, so both can
# be set to the harness's own environment by the time a scratch command runs.
#
# This was not reasoned out in advance -- the first version of the fix
# inherited them, and `tox-uv` (which sets `UV_PROJECT_ENVIRONMENT` for the
# whole test run) sent a scratch sync into this repository's `.tox/unit`,
# emptying it down to the scratch project's four packages. 66 of `ops`'s own
# tests then failed to import `websocket`.


def test_scratch_commands_do_not_inherit_uvs_environment_selection(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/somebody/elses/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/home/user/operator/.tox/unit")

    env = scratch_command_env()

    assert "VIRTUAL_ENV" not in env
    assert "UV_PROJECT_ENVIRONMENT" not in env


def test_scratch_commands_can_pin_the_environment_they_write_to(monkeypatch):
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/home/user/operator/.tox/unit")

    env = scratch_command_env("/scratch/issue-2045/.venv")

    assert env["UV_PROJECT_ENVIRONMENT"] == "/scratch/issue-2045/.venv"
    assert "VIRTUAL_ENV" not in env


def test_the_rest_of_the_environment_survives(monkeypatch):
    """Only uv's two selection variables go. A scratch command still needs
    `PATH`, and a `HOME` for uv's own cache.

    Asserted as a subset rather than an exact difference because the suite
    itself may be running under something that sets one or both of them --
    `tox-uv` does, which is how this whole class of defect was found."""
    monkeypatch.setenv("VIRTUAL_ENV", "/somebody/elses/.venv")

    env = scratch_command_env()

    assert env["PATH"] == os.environ["PATH"]
    assert set(os.environ) - set(env) <= {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    assert "VIRTUAL_ENV" in set(os.environ) - set(env)


def test_the_none_branch_runs_its_commands_in_the_pinned_environment(monkeypatch, tmp_path):
    """End to end through the seam: the environment handed to `bash -c` is
    the scratch project's, whatever the harness was started with."""
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/home/user/operator/.tox/unit")
    captured: list[dict] = []

    def fake(args, **kwargs):
        captured.append(kwargs.get("env") or {})
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(seams.runner.subprocess, "run", fake)
    hyp = Hypothesis(
        issue_number=2045,
        in_scope=True,
        moving_parts=MovingParts(substrate="none"),
        commands=["uv add 'ops[testing]'"],
        expected="e",
        observed="o",
        confidence="medium",
    )

    seams.runner.SubprocessRunnerSeam().run(
        branch="none",
        hypothesis=hyp,
        surface=None,
        context={"workdir": str(tmp_path), "uv_project_environment": str(tmp_path / ".venv")},
    )

    assert captured
    assert captured[0]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")


def test_the_pipeline_tells_the_seam_which_environment_to_use(tmp_path):
    captured: dict = {}

    class _CapturingRunner:
        def resolve_symbol(self, symbol_anchor: str, context: dict) -> bool:
            return True

        def run(self, **kwargs) -> RunResult:
            captured.update(kwargs["context"])
            return RunResult(
                hypothesis_number=2045,
                branch="none",
                commands=[CommandResult(command="true", exit_code=0)],
            )

    pipeline = build_pipeline(
        fixture_mode=True, fixtures_dir=_HARNESS_ROOT / "fixtures", work_dir=tmp_path / "work"
    )
    pipeline.runner = _CapturingRunner()

    pipeline.run_for_issue(_load_issue(2045), calibration_mode=True)

    workdir = Path(captured["workdir"])
    assert captured["uv_project_environment"] == str(workdir.parent / ".venv")
