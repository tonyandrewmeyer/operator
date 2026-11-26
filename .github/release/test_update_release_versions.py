# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Tests for update-release-versions.py, which rewrites the version strings.

`tox -e unit` runs this, but only because the unit environment names the file:
pytest's default `norecursedirs` skips dotted directories, so it never finds
anything under `.github/` on its own. A new test file here needs adding to that
list in `tox.ini`. To run this one on its own:

    uv run --group unit pytest .github/release/test_update_release_versions.py

The script has a hyphen in its name, following the other scripts here, which
means it cannot be imported by name; `load` below does it by path.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import types

import pytest

HERE = pathlib.Path(__file__).parent


def load(name: str) -> types.ModuleType:
    """Import one of this directory's scripts by file name."""
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE / f'{name}.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


update = load('update-release-versions')


# A tree small enough to assert on and shaped like the real one, so that the
# post-release fan-out can be checked file by file. The version strings are
# the ones the 3.8.2 release left behind.
TREE = {
    'ops/version.py': (
        '"""Package version."""\n\nfrom __future__ import annotations\n\n'
        "version: str = '3.8.2'\n"
    ),
    'pyproject.toml': (
        '[project]\nname = "ops"\ndependencies = [\n'
        '    "ops-scenario==8.8.2",\n    "ops-tracing==3.8.2",\n]\n'
    ),
    'testing/src/scenario/_version.py': (
        '"""Package version."""\n\nfrom __future__ import annotations\n\n'
        "version: str = '8.8.2'\n"
    ),
    'testing/pyproject.toml': (
        '[project]\nname = "ops-scenario"\ndynamic = ["version"]\ndependencies = [\n'
        '    "ops==3.8.2",\n]\n'
    ),
    'tracing/pyproject.toml': (
        '[project]\nname = "ops-tracing"\nversion = "3.8.2"\ndependencies = [\n'
        '    "ops==3.8.2",\n]\n'
    ),
    'scriptlets/pyproject.toml': (
        '[project]\nname = "ops-testing-scriptlets"\nversion = "0.1.2"\ndependencies = []\n'
    ),
    'scriptlets/runner/main.go': 'package main\n\nconst JujuTag = "v4.2.0"\n',
    'docs/explanation/versions.md': (
        '| Version | Status | Released | EOL |\n'
        '| Ops 2.23 (LTS) | Active | 2023-01-25 | 2038-01-25 |\n'
        '| Ops 3.8 | Active | 2026-08-31 | 2027-08-31 |\n'
    ),
}


@pytest.fixture
def tree(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Write the small tree out and chdir into it, as the workflow does."""
    for name, content in TREE.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.chdir(tmp_path)
    # The real pair starts at a later ops than the one in this tree.
    monkeypatch.setattr(update, 'SCRIPTLETS_BASE', ((3, 8), (0, 1)))
    return tmp_path


@pytest.fixture
def lts_tree(tree: pathlib.Path) -> pathlib.Path:
    """The same tree, at the versions a 2.23 LTS release left behind."""
    # The branch is from before ops-testing-scriptlets existed.
    shutil.rmtree(tree / 'scriptlets')
    for name in TREE:
        if name.endswith('.md') or name.startswith('scriptlets/'):
            continue
        path = tree / name
        path.write_text(path.read_text().replace('3.8.2', '2.23.5').replace('8.8.2', '7.23.5'))
    return tree


class TestPostRelease:
    """update-release-versions.py in its post-release shape."""

    def test_it_writes_every_version_file_and_leaves_the_doc(self, tree: pathlib.Path):
        """The six files it rewrites, and the one it does not."""
        assert update.main(['--version', '3.9.0.dev0', '--post-release']) == 0

        assert "version: str = '3.9.0.dev0'" in (tree / 'ops/version.py').read_text()
        scenario = (tree / 'testing/src/scenario/_version.py').read_text()
        assert "version: str = '8.9.0.dev0'" in scenario

        root = (tree / 'pyproject.toml').read_text()
        assert 'ops-scenario==8.9.0.dev0' in root
        assert 'ops-tracing==3.9.0.dev0' in root

        testing = (tree / 'testing/pyproject.toml').read_text()
        assert 'dynamic = ["version"]' in testing
        assert 'ops==3.9.0.dev0' in testing

        tracing = (tree / 'tracing/pyproject.toml').read_text()
        assert 'version = "3.9.0.dev0"' in tracing
        assert 'ops==3.9.0.dev0' in tracing

        scriptlets = (tree / 'scriptlets/pyproject.toml').read_text()
        assert 'version = "0.2.0.dev0"' in scriptlets

        # The one difference from a release: the tool-versions table records
        # releases, and a bump back to a development version is not one.
        assert (tree / 'docs/explanation/versions.md').read_text() == (
            TREE['docs/explanation/versions.md']
        )

    def test_the_maintenance_shape(self, lts_tree: pathlib.Path):
        """The LTS branch's patch bump, which has no scriptlets package."""
        assert update.main(['--version', '2.23.6.dev0', '--post-release']) == 0
        assert "version: str = '2.23.6.dev0'" in (lts_tree / 'ops/version.py').read_text()
        scenario = (lts_tree / 'testing/src/scenario/_version.py').read_text()
        assert "version: str = '7.23.6.dev0'" in scenario
        assert 'ops==2.23.6.dev0' in (lts_tree / 'testing/pyproject.toml').read_text()
        assert not (lts_tree / 'scriptlets').exists()
        assert (lts_tree / 'docs/explanation/versions.md').read_text() == (
            TREE['docs/explanation/versions.md']
        )

    def test_the_scenario_version_carries_the_dev_suffix(self):
        """ops-scenario is five majors ahead, suffix and all."""
        assert update.scenario_version('3.9.0.dev0') == '8.9.0.dev0'
        assert update.scenario_version('3.4.0.dev0') == '8.4.0.dev0'

    def test_a_tree_already_at_the_version_is_an_error(self, tree: pathlib.Path):
        """The workflow checks the branch first, so this is the backstop."""
        assert update.main(['--version', '3.9.0.dev0', '--post-release']) == 0
        assert update.main(['--version', '3.9.0.dev0', '--post-release']) == 1

    def test_a_pyproject_already_at_the_version_is_an_error(
        self, tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ):
        """Each pyproject is checked on its own, not only ops/version.py."""
        path = tree / 'tracing/pyproject.toml'
        path.write_text(path.read_text().replace('3.8.2', '3.9.0.dev0'))
        assert update.main(['--version', '3.9.0.dev0', '--post-release']) == 1
        assert 'Nothing changed in tracing/pyproject.toml' in capsys.readouterr().err

    def test_the_files_the_workflow_commits_are_the_files_it_writes(self, tree: pathlib.Path):
        """Nothing else in the tree moves, so `git add -u` commits only these."""
        before = {path: path.read_text() for path in sorted(tree.rglob('*')) if path.is_file()}
        assert update.main(['--version', '3.9.0.dev0', '--post-release']) == 0
        changed = {
            str(path.relative_to(tree))
            for path, content in before.items()
            if path.read_text() != content
        }
        # uv.lock is the seventh, and the workflow writes it in its own step.
        assert changed == {
            'ops/version.py',
            'testing/src/scenario/_version.py',
            'pyproject.toml',
            'testing/pyproject.toml',
            'tracing/pyproject.toml',
            'scriptlets/pyproject.toml',
        }


class TestRelease:
    """update-release-versions.py in its release shape."""

    def test_a_release_run_on_the_same_tree_does_write_the_doc(self, tree: pathlib.Path):
        """The other half of the pair: the skipping is the flag's doing."""
        assert update.main(['--version', '3.9.0', '--date', '2026-09-30']) == 0
        doc = (tree / 'docs/explanation/versions.md').read_text()
        assert '| Ops 3.9 | Active | 2026-09-30 | 2027-09-30 |' in doc

    def test_a_release_leaves_an_lts_row_alone(self, lts_tree: pathlib.Path):
        """An LTS row's dates are a support commitment, not a year from now."""
        assert update.main(['--version', '2.23.6', '--date', '2026-09-30']) == 0
        assert "version: str = '2.23.6'" in (lts_tree / 'ops/version.py').read_text()
        assert (lts_tree / 'docs/explanation/versions.md').read_text() == (
            TREE['docs/explanation/versions.md']
        )

    def test_a_release_keeps_a_label_other_than_lts(self, tree: pathlib.Path):
        """The label survives the rewrite; only the minor and the dates move."""
        doc = tree / 'docs/explanation/versions.md'
        doc.write_text(doc.read_text().replace('| Ops 3.8 |', '| Ops 3.8 (Beta) |'))
        assert update.main(['--version', '3.9.0', '--date', '2026-09-30']) == 0
        assert '| Ops 3.9 (Beta) | Active | 2026-09-30 | 2027-09-30 |' in doc.read_text()

    def test_a_second_release_on_the_same_day_is_not_an_error(self, tree: pathlib.Path):
        """The row is already current, which is fine rather than suspicious."""
        doc = tree / 'docs/explanation/versions.md'
        assert update.main(['--version', '3.8.3', '--date', '2026-08-31']) == 0
        assert doc.read_text() == TREE['docs/explanation/versions.md']

    def test_only_the_projects_own_version_line_moves(self, tree: pathlib.Path):
        """A key that only ends in `version` keeps its value."""
        path = tree / 'tracing/pyproject.toml'
        path.write_text(path.read_text() + '\n[tool.example]\nminimum_version = "3.8.2"\n')
        assert update.main(['--version', '3.9.0', '--date', '2026-09-30']) == 0
        content = path.read_text()
        assert '\nversion = "3.9.0"\n' in content
        assert 'minimum_version = "3.8.2"' in content

    def test_a_branch_without_the_versions_doc_still_releases(self, tree: pathlib.Path):
        """The older maintenance branches have no tool-versions table."""
        doc = tree / 'docs/explanation/versions.md'
        doc.unlink()
        assert update.main(['--version', '3.9.0', '--date', '2026-09-30']) == 0
        assert "version: str = '3.9.0'" in (tree / 'ops/version.py').read_text()
        assert not doc.exists()

    def test_a_leap_day_release_ends_support_on_1_march(self, tree: pathlib.Path):
        """29 February has no anniversary in the next year."""
        assert update.main(['--version', '3.9.0', '--date', '2028-02-29']) == 0
        doc = (tree / 'docs/explanation/versions.md').read_text()
        assert '| Ops 3.9 | Active | 2028-02-29 | 2029-03-01 |' in doc


class TestScriptlets:
    """The ops-testing-scriptlets version, and what stops its release."""

    @pytest.mark.parametrize(
        'ops_version,expected',
        [
            ('3.8.0', '0.1.0'),
            ('3.8.2', '0.1.2'),
            ('3.9.0', '0.2.0'),
            ('3.21.1', '0.14.1'),
            ('3.9.0rc1', '0.2.0rc1'),
            ('3.9.0.dev0', '0.2.0.dev0'),
        ],
    )
    def test_the_version_follows_ops(self, tree: pathlib.Path, ops_version: str, expected: str):
        """A minor for each ops minor, with the patch and suffix carried across."""
        assert update.scriptlets_version(ops_version) == expected

    def test_a_base_of_1_0(self, monkeypatch: pytest.MonkeyPatch):
        """What changing the pair for 1.0 does."""
        monkeypatch.setattr(update, 'SCRIPTLETS_BASE', ((3, 20), (1, 0)))
        assert update.scriptlets_version('3.20.0') == '1.0.0'
        assert update.scriptlets_version('3.22.3') == '1.2.3'

    @pytest.mark.parametrize('version', ['3.7.9', '4.0.0'])
    def test_an_ops_version_with_no_scriptlets_version(
        self, tree: pathlib.Path, capsys: pytest.CaptureFixture[str], version: str
    ):
        """Nothing is written, and the message says what to change."""
        assert update.main(['--version', version]) == 1
        assert 'update SCRIPTLETS_BASE' in capsys.readouterr().err
        for name, content in TREE.items():
            assert (tree / name).read_text() == content

    def test_a_release_needs_a_juju_tag(
        self, tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ):
        """A runner that isn't from a Juju release stops the release, before any write."""
        runner = tree / 'scriptlets/runner/main.go'
        runner.write_text(runner.read_text().replace('"v4.2.0"', '""'))
        assert update.main(['--version', '3.9.0']) == 1
        assert 'not built from a Juju release' in capsys.readouterr().err
        assert "version: str = '3.8.2'" in (tree / 'ops/version.py').read_text()

    def test_a_post_release_bump_does_not(self, tree: pathlib.Path):
        """The bump after a release goes ahead whatever the runner is built from."""
        runner = tree / 'scriptlets/runner/main.go'
        runner.write_text(runner.read_text().replace('"v4.2.0"', '""'))
        assert update.main(['--version', '3.9.0.dev0', '--post-release']) == 0

    def test_a_release_updates_the_version(self, tree: pathlib.Path):
        """With a Juju tag, the version is written."""
        assert update.main(['--version', '3.9.0', '--date', '2026-09-30']) == 0
        assert 'version = "0.2.0"' in (tree / 'scriptlets/pyproject.toml').read_text()

    def test_the_juju_tag_is_where_the_script_looks(self):
        """The real runner still has the line that the script reads."""
        root = HERE.parent.parent
        assert update.JUJU_TAG.search((root / update.JUJU_TAG_FILE).read_text())
