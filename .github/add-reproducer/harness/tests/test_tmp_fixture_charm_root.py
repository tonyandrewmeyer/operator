"""A value compared with pytest's `tmp_path` or `tmpdir` in a test that never
hands the fixture to the charm.

On `#2045`, an answer re-asked and told not to read `ctx._charm_dir`
compared the cwd a `start` handler captured with `str(tmp_path)`. The
`Context` was built without `charm_root=`, so the charm root was a separate
temporary directory, and the assertion failed on 3.8.3, `main` and the fix
branch alike: a false comment. Five hand-written probes are the
calibration cases.
"""

from __future__ import annotations

import pytest

from static_test_check import check

_CHARM = '''\
import os

import ops
from ops import testing

captured = {}

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['cwd'] = os.getcwd()
'''

# That `#2045` answer, as saved.
_SAVED = _CHARM + '''\




def test_cwd_is_charm_dir(tmp_path):
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == str(tmp_path)
'''


def _reasons(body: str) -> list[str]:
    return [r for r in check(body).reasons if "pytest's own temporary directory" in r]


def test_prbar2045_8_as_saved_is_rejected():
    result = check(_SAVED)
    assert not result.passed
    assert result.reasons == [
        (
            "line 22: the test compares with `str(tmp_path)`, but `tmp_path` is pytest's own temporary "
        "directory for this test and nothing hands it to the charm: line 20's `Context(...)` is not "
        "given it as `charm_root=`, so the charm root is a separate temporary directory and the "
        "comparison comes out the same whether or not the bug is real; pass the fixture as "
        "`Context(..., charm_root=tmp_path)` if the charm root should be that directory, or read the "
        "charm's directory in a handler as `self.framework.charm_dir` and compare with that"
        )
    ]


def test_the_tmpdir_probe_is_rejected():
    body = _SAVED.replace("(tmp_path)", "(tmpdir)")
    assert check(body).reasons == [
        r.replace("tmp_path", "tmpdir") for r in check(_SAVED).reasons
    ]


@pytest.mark.parametrize(
    "body",
    [
        # The same fixture given as the charm root is a fair comparison.
        _SAVED.replace(
            "meta={'name': 'my-charm'})", "meta={'name': 'my-charm'}, charm_root=tmp_path)"
        ),
        # The cwd before `ctx.run`, no fixture at all.
        _SAVED.replace(
            "def test_cwd_is_charm_dir(tmp_path):\n",
            "def test_cwd_is_charm_dir():\n    import pathlib\n    before = pathlib.Path.cwd()\n",
        )
        .replace("str(tmp_path)", "str(before)")
        .replace("==", "!="),
        # The charm directory captured in the handler; `tmp_path` is taken
        # and never used.
        _SAVED.replace(
            "captured['cwd'] = os.getcwd()",
            "captured['cwd'] = os.getcwd()\n        captured['charm_dir'] = str(self.framework.charm_dir)",
        ).replace("str(tmp_path)", "captured['charm_dir']"),
    ],
    ids=["charm_root=tmp_path", "Path.cwd-before-run", "charm_dir-from-handler"],
)
def test_the_accepted_probes_pass(body):
    assert check(body).passed


@pytest.mark.parametrize(
    "read",
    [
        "tmp_path",
        "str(tmp_path)",
        "Path(tmp_path)",
        "pathlib.Path(tmp_path)",
        "os.fspath(tmp_path)",
        "tmp_path.resolve()",
        "str(tmp_path.resolve())",
        "str(tmp_path / 'charm')",
    ],
)
@pytest.mark.parametrize("op", ["==", "!="])
def test_every_wrapped_comparison_is_rejected(read, op):
    body = _SAVED.replace("import os\n", "import os\nimport pathlib\nfrom pathlib import Path\n").replace(
        "== str(tmp_path)", f"{op} {read}"
    )
    assert len(_reasons(body)) == 1


@pytest.mark.parametrize(
    "assertion",
    ["str(tmp_path) in captured['cwd']", "captured['cwd'] not in str(tmp_path)", "str(tmp_path) == captured['cwd']"],
)
def test_membership_and_either_side_are_rejected(assertion):
    body = _SAVED.replace("captured['cwd'] == str(tmp_path)", assertion)
    assert len(_reasons(body)) == 1


def test_charm_root_given_something_else_is_still_rejected():
    body = _SAVED.replace(
        "meta={'name': 'my-charm'})", "meta={'name': 'my-charm'}, charm_root='/tmp/elsewhere')"
    )
    assert len(_reasons(body)) == 1


def test_two_contexts_are_both_named():
    body = _SAVED.replace(
        "    ctx.run(",
        "    other = testing.Context(MyCharm, meta={'name': 'my-charm'})\n    ctx.run(",
    )
    (reason,) = _reasons(body)
    assert "the `Context(...)`s on lines 20, 21 are" in reason


@pytest.mark.parametrize(
    "body",
    [
        # Given as the charm root through something derived from it.
        _SAVED.replace(
            "meta={'name': 'my-charm'})",
            "meta={'name': 'my-charm'}, charm_root=tmp_path / 'charm')",
        ).replace("str(tmp_path)", "str(tmp_path / 'charm')"),
        _SAVED.replace(
            "    ctx = testing",
            "    root = tmp_path / 'charm'\n    root.mkdir()\n    ctx = testing",
        ).replace("meta={'name': 'my-charm'})", "meta={'name': 'my-charm'}, charm_root=root)"),
        _SAVED.replace(
            "meta={'name': 'my-charm'})", "meta={'name': 'my-charm'}, charm_root=str(tmp_path))"
        ),
        # Through `**`, built from the fixture.
        _SAVED.replace(
            "    ctx = testing",
            "    kwargs = {'charm_root': tmp_path}\n    ctx = testing",
        ).replace("meta={'name': 'my-charm'})", "meta={'name': 'my-charm'}, **kwargs)"),
        # Handed to something else that might give it to the charm.
        _SAVED.replace(
            "def test_cwd_is_charm_dir(tmp_path):\n",
            "def test_cwd_is_charm_dir(tmp_path, monkeypatch):\n    monkeypatch.chdir(tmp_path)\n",
        ),
        _SAVED.replace(
            "testing.State())",
            "testing.State(config={'path': str(tmp_path)}))",
        ),
        # A fixture in the file takes it too, so shares the directory.
        _SAVED.replace(
            "def test_cwd_is_charm_dir(tmp_path):\n",
            "import pytest\n\n@pytest.fixture(autouse=True)\ndef cd(tmp_path, monkeypatch):\n"
            "    monkeypatch.chdir(tmp_path)\n\ndef test_cwd_is_charm_dir(tmp_path):\n",
        ),
        # No `Context` built in the test: it may come from a fixture.
        _SAVED.replace("def test_cwd_is_charm_dir(tmp_path):\n", "def test_cwd_is_charm_dir(tmp_path, ctx):\n")
        .replace("    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})\n", ""),
        # Rebound in the test.
        _SAVED.replace("    ctx = testing", "    tmp_path = '/tmp/x'\n    ctx = testing"),
        # Not a test function.
        _SAVED.replace("def test_cwd_is_charm_dir", "def check_cwd"),
        # `is`, `<` and the like are not the shape.
        _SAVED.replace("== str(tmp_path)", "is not None and tmp_path.exists()"),
    ],
    ids=[
        "charm_root=tmp_path/charm",
        "charm_root=derived-name",
        "charm_root=str(tmp_path)",
        "charm_root-through-kwargs",
        "monkeypatch.chdir",
        "config",
        "shared-with-fixture",
        "no-context-in-test",
        "rebound",
        "not-a-test",
        "not-a-comparison",
    ],
)
def test_uses_that_might_reach_the_charm_pass(body):
    assert _reasons(body) == []


def test_tmp_path_used_only_to_write_files_passes():
    body = _CHARM.replace(
        "captured['cwd'] = os.getcwd()",
        "captured['data'] = (self.framework.charm_dir / 'data.txt').read_text()",
    ) + '''

def test_reads_a_file(tmp_path):
    (tmp_path / 'data.txt').write_text('hello')
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'}, charm_root=tmp_path)
    ctx.run(ctx.on.start(), testing.State())
    assert captured['data'] == 'hello'


def test_writes_a_scratch_file(tmp_path):
    scratch = tmp_path / 'scratch.txt'
    scratch.write_text('x')
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert scratch.read_text() == 'x'
'''
    assert _reasons(body) == []
