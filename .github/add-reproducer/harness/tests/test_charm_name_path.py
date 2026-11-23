"""A value matched against the charm's name as a path component, in a test
whose `Context(...)` has no `charm_root=`.

On `#2045`, two answers checked the cwd a `start` handler captured with
`.endswith('/my-charm')`. Without `charm_root=`, `ops.testing` makes the charm
root a `tempfile` directory that is never named after the charm, so the
assertion failed on 3.8.3, `main` and the fix branch alike: two false
comments. Five hand-written probes are the calibration cases.
"""

from __future__ import annotations

import pytest

from static_test_check import check

_MARK = "matches a path against the charm's name"

# One of those `#2045` answers, as saved.
_SAVED_5 = '''\
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


def test_charm_dir_is_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'].endswith('/my-charm')
'''

# The other `#2045` answer, as saved.
_SAVED_1 = '''\
import os
import pytest
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        global captured
        captured = os.getcwd()

captured = None

def test_current_directory_is_charm_root():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured.endswith('/my-charm')
'''

_ASSERT = "assert captured['cwd'].endswith('/my-charm')"


def _reasons(body: str) -> list[str]:
    return [r for r in check(body).reasons if _MARK in r]


def _with_assert(new: str) -> str:
    assert _ASSERT in _SAVED_5
    return _SAVED_5.replace(_ASSERT, new)


def test_2045b_5_as_saved_is_rejected():
    result = check(_SAVED_5)
    assert not result.passed
    assert result.reasons == [
        "line 21: `captured['cwd'].endswith('/my-charm')` matches a path against the charm's name "
        "`my-charm`, but line 19's `Context(...)` is not given `charm_root=`, so the charm root is a "
        "temporary directory that `ops.testing` makes and is not named after the charm, and the match "
        "comes out the same whether or not the bug is real; read the charm's directory in a handler as "
        "`self.framework.charm_dir` and compare with that (or pass a `Context(..., charm_root=...)` "
        "named `my-charm` if the test needs that name)"
    ]


def test_2045b_1_as_saved_is_rejected():
    result = check(_SAVED_1)
    assert not result.passed
    assert len(result.reasons) == 1
    assert result.reasons[0].startswith("line 20: `captured.endswith('/my-charm')` matches a path")


@pytest.mark.parametrize(
    "assertion",
    [
        # The hand-written probes.
        "assert os.path.basename(captured['cwd']) == 'my-charm'",
        "assert 'my-charm' in captured['cwd']",
        # Neighbours.
        "assert captured['cwd'].endswith('my-charm')",
        "assert captured['cwd'].endswith(('/my-charm', '/my-charm/'))",
        "assert 'my-charm' == os.path.basename(captured['cwd'])",
        "assert pathlib.Path(captured['cwd']).name == 'my-charm'",
        "assert Path(captured['cwd']).name != 'my-charm'",
        "assert captured['cwd'].split('/')[-1] == 'my-charm'",
        "assert '/my-charm' in captured['cwd']",
        "assert 'my-charm' not in captured['cwd']",
        "assert '/my-charm' in captured['value']",
        # A guess at a longer charm-root path.
        "assert captured['cwd'].endswith('/charms/my-charm')",
        "assert '/my-charm/src' in captured['cwd']",
    ],
)
def test_name_as_path_component_is_rejected(assertion: str):
    assert len(_reasons(_with_assert(assertion))) == 1


def test_name_from_module_level_meta_is_rejected():
    body = _SAVED_5.replace(
        "captured = {}\n", "captured = {}\nMETA = {'name': 'demo', 'containers': {}}\n"
    ).replace("meta={'name': 'my-charm'}", "meta=META").replace("'/my-charm'", "'/demo'")
    assert len(_reasons(body)) == 1


def test_name_from_yaml_meta_is_rejected():
    body = _SAVED_5.replace("import os\n", "import os\nimport yaml\n").replace(
        "meta={'name': 'my-charm'}", "meta=yaml.safe_load('name: my-charm\\n')"
    )
    assert len(_reasons(body)) == 1


def test_positional_meta_is_read():
    body = _SAVED_5.replace("meta={'name': 'my-charm'}", "{'name': 'my-charm'}")
    assert len(_reasons(body)) == 1


def test_charm_root_named_after_the_charm_passes():
    # The charm root is named after the charm, so the check is valid.
    body = _SAVED_5.replace(
        "def test_charm_dir_is_cwd():\n",
        "def test_charm_dir_is_cwd(tmp_path):\n    (tmp_path / 'my-charm').mkdir()\n",
    ).replace(
        "meta={'name': 'my-charm'})", "meta={'name': 'my-charm'}, charm_root=tmp_path / 'my-charm')"
    )
    assert check(body).passed


def test_charm_dir_from_handler_passes():
    # The cwd compared with the charm directory read in the handler.
    body = _SAVED_5.replace(
        "captured['cwd'] = os.getcwd()",
        "captured['cwd'] = os.getcwd()\n        captured['charm_dir'] = str(self.framework.charm_dir)",
    ).replace(_ASSERT, "assert captured['cwd'] == captured['charm_dir']")
    assert check(body).passed


def test_double_star_context_is_left_alone():
    body = _SAVED_5.replace(
        "    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})",
        "    kwargs = {}\n    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'}, **kwargs)",
    )
    assert _reasons(body) == []


def test_one_context_with_charm_root_leaves_the_test_alone():
    body = _SAVED_5.replace(
        "    ctx.run(ctx.on.start(), testing.State())\n",
        "    ctx.run(ctx.on.start(), testing.State())\n"
        "    other = testing.Context(MyCharm, meta={'name': 'my-charm'}, charm_root='/tmp/my-charm')\n",
    )
    assert _reasons(body) == []


@pytest.mark.parametrize(
    "assertion",
    [
        # The name compared as a name, not a path.
        "assert captured['app'] == 'my-charm'",
        "assert captured['app'].endswith('my-charm')",
        "assert 'my-charm' in captured['message']",
        "assert 'my-charm' in ['my-charm', 'other']",
        "assert 'my-charm' in captured['apps']",
        # A different name.
        "assert captured['cwd'].endswith('/other-charm')",
        "assert os.path.basename(captured['cwd']) == 'charm'",
        # A longer path that only contains the name, or the name inside a component.
        "assert captured['cwd'].endswith('/my-charm/src')",
        "assert captured['cwd'].endswith('/not-my-charm')",
        "assert os.path.basename(captured['cwd']) == 'charms/my-charm'",
        # `.name` on something that is not a Path.
        "assert captured['unit'].name == 'my-charm'",
    ],
)
def test_name_not_as_path_is_left_alone(assertion: str):
    assert _reasons(_with_assert(assertion)) == []


def test_app_name_in_handler_is_left_alone():
    body = _SAVED_5.replace(
        "captured['cwd'] = os.getcwd()",
        "captured['ok'] = self.app.name == 'my-charm' and self.meta.name == 'my-charm'",
    ).replace(_ASSERT, "assert captured['ok']")
    assert _reasons(body) == []


def test_no_context_in_the_test_is_left_alone():
    body = _SAVED_5.replace(
        "def test_charm_dir_is_cwd():\n    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})\n",
        "@pytest.fixture\ndef ctx():\n    return testing.Context(MyCharm, meta={'name': 'my-charm'})\n\n\n"
        "def test_charm_dir_is_cwd(ctx):\n",
    ).replace("import os\n", "import os\n\nimport pytest\n")
    assert _reasons(body) == []


def test_meta_without_a_literal_name_is_left_alone():
    body = _SAVED_5.replace("meta={'name': 'my-charm'}", "meta=load_meta()").replace(
        "captured = {}\n", "captured = {}\n\ndef load_meta():\n    return {'name': 'my-charm'}\n"
    )
    assert _reasons(body) == []


def test_non_test_function_is_left_alone():
    body = _SAVED_5.replace("def test_charm_dir_is_cwd():", "def helper():") + "\n\ndef test_x():\n    helper()\n"
    assert _reasons(body) == []
