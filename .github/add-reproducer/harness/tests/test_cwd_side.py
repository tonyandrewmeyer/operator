"""A claim about the working directory whose cwd side is not the cwd.

The other cwd rules each catch a shape of false `#2045` final answer, and
`_cwd_compared_with_unsourced_value()` checks the side a cwd is compared
*with*. A final can get past that when the side presented as the cwd is not
one: the handler below stores `str(ops.__path__[0])` under the key `'cwd'`,
so the file reads no working directory at all and fails on 3.8.3, `main` and
the fix branch alike. `_cwd_side_not_a_cwd_read()` closes that from the cwd side: a value
presented as the working directory (by its name, or by the test's name when
the other side is the handler's charm directory) must come from a cwd read
made during the run.
"""

from __future__ import annotations

import ast

import pytest
from test_cwd_handler_sourced import (
    _FALSE_A08,
    _FALSE_B01,
    _FALSE_B05,
    _FALSE_C04,
)
from test_cwd_handler_sourced import (
    _VALID as _VALID_38_TO_40,
)

from static_test_check import (
    _charm_name_as_path_without_charm_root,
    _cwd_compared_with_unsourced_value,
    _cwd_side_not_a_cwd_read,
    _tmp_fixture_without_charm_root,
    check,
)

_MARK = "as the working directory, but nothing in the file reads the working directory"


def _rule(source: str) -> list[str]:
    return _cwd_side_not_a_cwd_read(ast.parse(source))


# `captured['cwd'] = str(ops.__path__[0])`; a saved `#2045` final, a false comment.
_FALSE_D04 = '''\
import ops
from ops import testing

captured = {}

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['charm_dir'] = str(self.charm_dir)
        captured['cwd'] = str(ops.__path__[0])

def test_charm_dir_matches_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['charm_dir'] == captured['cwd']
'''

# A valid `#2045` final.
_VALID_D00 = '''\
import os
import ops
from ops import testing

captured = {}

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured["cwd"] = os.getcwd()
        captured["charm_dir"] = str(self.framework.charm_dir)

def test_charm_root_is_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured["cwd"] == captured["charm_dir"]
'''

# A valid `#2045` final.
_VALID_D01 = '''\
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
        captured['charm_dir'] = str(self.framework.charm_dir)

def test_charm_dir_is_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir'], f"Expected {captured['charm_dir']}, got {captured['cwd']}"
'''

# A valid `#2045` final.
_VALID_D02 = '''\
import os
import pytest
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    pass

meta = {'name': 'my-charm'}

@pytest.fixture
def ctx():
    return testing.Context(MyCharm, meta=meta)

def test_cwd_is_charm_root(ctx):
    with ctx(ctx.on.start(), testing.State()) as mgr:
        assert os.getcwd() == str(mgr.charm.charm_dir)
'''

# A valid `#2045` final.
_VALID_D06 = '''\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    pass

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.update_status(), testing.State()) as mgr:
        assert os.getcwd() == str(mgr.charm.charm_dir)
'''

# A valid `#2045` final.
_VALID_D07 = '''\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['cwd'] = os.getcwd()
        captured['charm_dir'] = str(self.charm_dir)

captured = {}

def test_getcwd_is_charm_root():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir']
'''

# A valid `#2045` final.
_VALID_D10 = '''\
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
        captured['charm_dir'] = str(self.charm_dir)

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir']
'''

# A valid `#2045` final.
_VALID_D11 = '''\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        global captured
        captured['cwd'] = os.getcwd()
        captured['charm_dir'] = str(self.framework.charm_dir)

captured = {}


def test_cwd_is_charm_root():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir'], f"Expected {captured['charm_dir']}, got {captured['cwd']}"
'''

# A valid `#2045` final.
_VALID_D12 = '''\
import os
import pytest
import ops
from ops import testing

captured = {}

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['charm_dir'] = str(self.framework.charm_dir)
        captured['cwd'] = os.getcwd()

def test_charm_dir_is_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['charm_dir'] == captured['cwd']
'''

# A valid `#2045` final.
_VALID_D13 = '''\
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
        captured['charm_dir'] = str(self.framework.charm_dir)


def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir']
'''

# A valid `#2045` final.
_VALID_D14 = '''\
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
        captured['charm_dir'] = str(self.framework.charm_dir)

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir'], f"Expected {captured['charm_dir']}, got {captured['cwd']}"
'''

_VALID_41 = [
    _VALID_D00,
    _VALID_D01,
    _VALID_D02,
    _VALID_D06,
    _VALID_D07,
    _VALID_D10,
    _VALID_D11,
    _VALID_D12,
    _VALID_D13,
    _VALID_D14,
]


def _which(source: str) -> str:
    """The rule that rejects `source` in `check()`."""
    tree = ast.parse(source)
    reasons = check(source).reasons
    for name, rule in (
        ("tmp_fixture", _tmp_fixture_without_charm_root),
        ("charm_name", _charm_name_as_path_without_charm_root),
        ("cwd_other_side", _cwd_compared_with_unsourced_value),
        ("cwd_side", _cwd_side_not_a_cwd_read),
    ):
        found = rule(tree)
        if found and all(r in reasons for r in found):
            return name
    return "none"


@pytest.mark.parametrize(
    ("source", "rejected_by"),
    [
        pytest.param(_FALSE_A08, "tmp_fixture", id="38:a08"),
        pytest.param(_FALSE_B01, "charm_name", id="39:b01"),
        pytest.param(_FALSE_B05, "charm_name", id="39:b05"),
        pytest.param(_FALSE_C04, "cwd_other_side", id="40:c04"),
        pytest.param(_FALSE_D04, "cwd_side", id="41:d04"),
    ],
)
def test_five_false_finals_rejected(source: str, rejected_by: str):
    assert not check(source).passed
    assert _which(source) == rejected_by


def test_41_false_final_under_this_rule():
    # Only this rule sees it: the file reads no cwd, so the other-side rule
    # (which finds the cwd by its value) has nothing to check.
    assert _cwd_compared_with_unsourced_value(ast.parse(_FALSE_D04)) == []
    (reason,) = _rule(_FALSE_D04)
    assert check(_FALSE_D04).reasons == [reason]
    assert reason.startswith(
        "line 18: `captured['charm_dir'] == captured['cwd']` presents `captured['cwd']` as the working directory"
    )
    assert "read the working directory with `os.getcwd()` in a handler" in reason
    assert "compare it with `self.charm_dir`" in reason


@pytest.mark.parametrize("source", [_FALSE_A08, _FALSE_B01, _FALSE_B05, _FALSE_C04])
def test_earlier_false_finals_read_the_cwd(source: str):
    # Their cwd side is a real `os.getcwd()`; the guess is on the other side,
    # which older rules judge.
    assert _rule(source) == []


@pytest.mark.parametrize(
    "source",
    _VALID_38_TO_40 + _VALID_41,
    ids=[f"38-40:{i}" for i in range(len(_VALID_38_TO_40))]
    + ["d00", "d01", "d02", "d06", "d07", "d10", "d11", "d12", "d13", "d14"],
)
def test_valid_finals_accepted(source: str):
    assert _rule(source) == []
    assert check(source).passed


_HOLDER = """\
import os
import pathlib
from pathlib import Path

import ops
from ops import testing

captured = {{}}

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['charm_dir'] = str(self.charm_dir)
        captured['cwd'] = {cwd}

def test_charm_dir_matches_cwd():
    ctx = testing.Context(MyCharm, meta={{'name': 'my-charm'}})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['charm_dir'] == captured['cwd']
"""


@pytest.mark.parametrize(
    "cwd",
    [
        "str(ops.__path__[0])",
        "os.path.dirname(ops.__file__)",
        "os.path.dirname(os.path.abspath(__file__))",
        "str(Path(__file__).parent)",
        "os.environ.get('PWD', '')",
        "os.environ['PWD']",
        "'/path/to/charm'",
        "str(self.charm_dir)",
        "str(self.framework.charm_dir)",
    ],
)
def test_non_cwd_sources_rejected(cwd: str):
    source = _HOLDER.format(cwd=cwd)
    (reason,) = _rule(source)
    assert _MARK in reason
    assert not check(source).passed


@pytest.mark.parametrize(
    "cwd",
    [
        "os.getcwd()",
        "os.getcwdb().decode()",
        "str(Path.cwd())",
        "str(pathlib.Path.cwd())",
        "str(Path('.').resolve())",
        "str(Path().resolve())",
        "os.path.abspath('.')",
        "os.path.realpath('.')",
        "str(Path(os.getcwd()).resolve())",
    ],
)
def test_cwd_reads_accepted(cwd: str):
    source = _HOLDER.format(cwd=cwd)
    assert _rule(source) == []
    assert check(source).passed


_SHAPES = {
    # Captured on the instance, read through `mgr.charm`.
    "instance": """\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        self.cwd = {cwd}

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={{'name': 'my-charm'}})
    with ctx(ctx.on.start(), testing.State()) as mgr:
        mgr.run()
        assert mgr.charm.cwd == str(mgr.charm.charm_dir)
""",
    # A module-level `global`.
    "global": """\
import os
import ops
from ops import testing

cwd_seen = None

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        global cwd_seen
        cwd_seen = {cwd}
        self.unit.status = ops.ActiveStatus(str(self.charm_dir))

def test_cwd_seen():
    ctx = testing.Context(MyCharm, meta={{'name': 'my-charm'}})
    state = ctx.run(ctx.on.start(), testing.State())
    assert cwd_seen == state.unit_status.message
""",
    # A closure's local, read through a local in the test.
    "closure": """\
import os
import ops
from ops import testing

def test_cwd_matches():
    seen = {{}}

    class MyCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            framework.observe(self.on.start, self._on_start)

        def _on_start(self, event):
            seen.update({{'dir': str(self.charm_dir), 'cwd': {cwd}}})

    ctx = testing.Context(MyCharm, meta={{'name': 'my-charm'}})
    ctx.run(ctx.on.start(), testing.State())
    actual = seen['cwd']
    assert actual == seen['dir']
""",
    # Read inside `with ctx(...)`, in the test itself.
    "with-ctx": """\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    pass

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={{'name': 'my-charm'}})
    with ctx(ctx.on.start(), testing.State()) as mgr:
        current_dir = {cwd}
        assert current_dir == str(mgr.charm.charm_dir)
""",
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_shapes_with_a_guessed_cwd_rejected(shape: str):
    source = _SHAPES[shape].format(cwd="str(ops.__path__[0])")
    assert any(_MARK in r for r in _rule(source))


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_shapes_with_a_cwd_read_accepted(shape: str):
    source = _SHAPES[shape].format(cwd="os.getcwd()")
    assert _rule(source) == []


def test_unlabelled_side_in_a_test_named_for_the_cwd():
    # The test's name presents the side compared with the handler's charm
    # directory as the cwd, whatever its key is called.
    source = _HOLDER.replace("captured['cwd']", "captured['where']").format(cwd="str(ops.__path__[0])")
    assert any(_MARK in r for r in _rule(source))
    source = _HOLDER.replace("captured['cwd']", "captured['where']").format(cwd="os.getcwd()")
    assert _rule(source) == []


def test_cwd_from_a_helper_or_property_accepted():
    helper = _HOLDER.format(cwd="current()") + "\n\ndef current():\n    return os.getcwd()\n"
    assert _rule(helper) == []
    prop = """\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    @property
    def cwd(self):
        return os.getcwd()

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.start(), testing.State()) as mgr:
        assert mgr.charm.cwd == str(mgr.charm.charm_dir)
"""
    assert _rule(prop) == []
    assert _rule(prop.replace("os.getcwd()", "os.path.dirname(ops.__file__)"))


def test_expected_cwd_on_the_other_side_left_to_the_other_rule():
    # One side reads the cwd, so it is the claim's cwd; `expected_cwd` is
    # what it is compared with, which the other-side rule judges.
    source = """\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        self.cwd = os.getcwd()

def test_cwd_is_charm_root():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.start(), testing.State()) as mgr:
        mgr.run()
        expected_cwd = '/path/to/charm'
        assert mgr.charm.cwd == expected_cwd
"""
    assert _rule(source) == []
    assert _cwd_compared_with_unsourced_value(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    [
        # A fixture argument: the file does not say where it comes from.
        """\
def test_cwd(cwd, charm_dir):
    assert cwd == charm_dir
""",
        # The file `chdir`s to it.
        """\
import os

def test_cwd(tmp_path):
    expected_cwd = str(tmp_path)
    os.chdir(expected_cwd)
    assert expected_cwd == os.path.realpath(expected_cwd)
""",
        # A comparison with `None`.
        _HOLDER.format(cwd="str(ops.__path__[0])").replace(
            "assert captured['charm_dir'] == captured['cwd']", "assert captured['cwd'] is not None"
        ),
        # A charm root the test chose.
        """\
import ops
from ops import testing

captured = {}

def test_cwd(tmp_path):
    ctx = testing.Context(ops.CharmBase, meta={'name': 'x'}, charm_root=tmp_path)
    ctx.run(ctx.on.start(), testing.State())
    assert captured.get('cwd') == str(ctx.charm_root)
""",
    ],
)
def test_unknown_or_chosen_sources_left_alone(source: str):
    assert _rule(source) == []


@pytest.mark.parametrize(
    "source",
    [
        # No cwd anywhere: a config value.
        """\
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.config_changed, self._on_config_changed)

    def _on_config_changed(self, event):
        self.unit.status = ops.ActiveStatus(str(self.config['name']))

def test_status():
    ctx = testing.Context(MyCharm, meta={'name': 'x'}, config={'options': {'name': {'type': 'string'}}})
    state = ctx.run(ctx.on.config_changed(), testing.State(config={'name': 'a'}))
    assert state.unit_status == testing.ActiveStatus('a')
""",
        # The charm directory compared with a file path, in a test not about the cwd.
        """\
import os
import ops
from ops import testing

captured = {}

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['charm_dir'] = str(self.charm_dir)
        captured['meta'] = os.path.join(str(self.charm_dir), 'metadata.yaml')

def test_metadata_path():
    ctx = testing.Context(MyCharm, meta={'name': 'x'})
    ctx.run(ctx.on.start(), testing.State())
    assert os.path.dirname(captured['meta']) == captured['charm_dir']
""",
        # A Pebble exec working directory: not this process's cwd, but a
        # literal the test gives and reads back.
        """\
from ops import testing

def test_exec_working_dir():
    container = testing.Container('c', can_connect=True)
    working_dir = '/srv'
    assert container.name == 'c'
    assert working_dir != container.name
""",
    ],
)
def test_tests_without_a_cwd_claim_unaffected(source: str):
    assert _rule(source) == []
