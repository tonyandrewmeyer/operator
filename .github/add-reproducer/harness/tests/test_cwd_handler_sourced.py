"""A claim about the working directory whose other side does not come from
the handler.

Four saved answers produced false comments on `#2045` by comparing the cwd with
a guess: pytest's `tmp_path`, a path ending in the charm's name, and the
literal `'/path/to/charm'` (in a plain class with no `ops`). Each failed on
3.8.3, `main` and the fix branch alike. All 25 valid `#2045` finals saved
alongside them compared the cwd with the charm directory read in a handler
(`self.framework.charm_dir` / `self.charm_dir`) or with `mgr.charm...charm_dir`
inside `with ctx(...)`. Hand-written probes are calibration cases too.
"""

from __future__ import annotations

import ast

import pytest

from static_test_check import _cwd_compared_with_unsourced_value, check

_MARK = "compares the working directory with"


def _rule(source: str) -> list[str]:
    return _cwd_compared_with_unsourced_value(ast.parse(source))

# Compared with `str(tmp_path)`; saved final, a false comment.
_FALSE_A08 = '''\
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




def test_cwd_is_charm_dir(tmp_path):
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == str(tmp_path)
'''
# `captured.endswith('/my-charm')`; saved final, a false comment.
_FALSE_B01 = '''\
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
# `captured['cwd'].endswith('/my-charm')`; saved final, a false comment.
_FALSE_B05 = '''\
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
# Plain class, `== '/path/to/charm'`; saved final, a false comment.
_FALSE_C04 = '''\
import os

class MyCharm:
    def __init__(self):
        self.charm_dir = os.getcwd()

def test_cwd_is_charm_dir():
    charm = MyCharm()
    assert str(charm.charm_dir) == '/path/to/charm'
'''
# A valid `#2045` final.
_VALID_A01 = '''\
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
        captured['charm_dir'] = self.framework.charm_dir

def test_cwd_matches_charm_dir():
    global captured
    captured = {}
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == str(captured['charm_dir'])
'''
# A valid `#2045` final.
_VALID_A04 = '''\
import os
import pytest
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['cwd'] = os.getcwd()
        captured['charm_dir'] = str(self.framework.charm_dir)

captured = {}

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir']
'''
# A valid `#2045` final.
_VALID_A05 = '''\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        global captured
        captured = {'cwd': os.getcwd(), 'charm_dir': str(self.charm_dir)}

def test_charm_cwd_matches_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir'], \\
        f"Expected cwd ({captured['cwd']}) to match charm_dir ({captured['charm_dir']})"
'''
# A valid `#2045` final.
_VALID_A07 = '''\
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

def test_charm_dir_matches_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir']
'''
# A valid `#2045` final.
_VALID_B00 = '''\
import os
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    pass

def test_charm_dir_equals_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.update_status(), testing.State()) as mgr:
        assert str(mgr.charm.charm_dir) == os.getcwd()
'''
# A valid `#2045` final.
_VALID_B03 = '''\
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


def test_cwd_is_charm_root():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir']
'''
# A valid `#2045` final.
_VALID_B09 = '''\
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
        captured['charm_dir'] = self.framework.charm_dir

def test_cwd_is_charm_dir():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert str(captured['cwd']) == str(captured['charm_dir'])
'''
# A valid `#2045` final.
_VALID_B10 = '''\
import os
import pytest
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    pass

def test_getcwd_is_charm_root():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.start(), testing.State()) as mgr:
        assert os.getcwd() == str(mgr.charm.framework.charm_dir)
'''
# A valid `#2045` final.
_VALID_B13 = '''\
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


def test_cwd_is_charm_root(tmp_path):
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'}, charm_root=tmp_path)
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == captured['charm_dir']
'''
# A valid `#2045` final.
_VALID_C05 = '''\
import os
import pytest
import ops
from ops import testing

class MyCharm(ops.CharmBase):
    pass

def test_cwd_is_charm_root():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    with ctx(ctx.on.update_status(), testing.State()) as mgr:
        assert os.getcwd() == str(mgr.charm.framework.charm_dir)
'''
# A valid `#2045` final.
_VALID_C08 = '''\
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
        captured['cwd'] = os.getcwd()

def test_charm_dir_matches_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['charm_dir'] == captured['cwd']
'''
# A valid `#2045` final.
_VALID_C13 = '''\
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
        captured['charm_dir'] = str(self.charm_dir)
        captured['cwd'] = os.getcwd()

def test_charm_dir_matches_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['charm_dir'] == captured['cwd']
'''
# A valid `#2045` final.
_VALID_C14 = '''\
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
        captured['charm_dir'] = self.framework.charm_dir

def test_charm_dir_matches_cwd():
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'})
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'] == str(captured['charm_dir'])
'''
# A valid `#2045` final.
_VALID_C15 = '''\
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
PROBE_39_5_AS_SAVED = '''\
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
PROBE_39_5_BASENAME_EQUALS = '''\
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
    assert os.path.basename(captured['cwd']) == 'my-charm'
'''
PROBE_39_5_NAME_IN_CWD = '''\
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
    assert 'my-charm' in captured['cwd']
'''
PROBE_39_5_CHARM_ROOT_NAMED_MY_CHARM = '''\
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


def test_charm_dir_is_cwd(tmp_path):
    (tmp_path / 'my-charm').mkdir()
    ctx = testing.Context(MyCharm, meta={'name': 'my-charm'}, charm_root=tmp_path / 'my-charm')
    ctx.run(ctx.on.start(), testing.State())
    assert captured['cwd'].endswith('/my-charm')
'''
PROBE_39_5_CHARM_DIR_FROM_HANDLER = '''\
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
    assert captured['cwd'] == captured['charm_dir']
'''
PROBE_40_4_AS_SAVED = '''\
import os

class MyCharm:
    def __init__(self):
        self.charm_dir = os.getcwd()

def test_cwd_is_charm_dir():
    charm = MyCharm()
    assert str(charm.charm_dir) == '/path/to/charm'
'''
PROBE_40_4_PLAIN_CLASS_COMPARES_WITH_GETCWD = '''\
import os

class MyCharm:
    def __init__(self):
        self.charm_dir = os.getcwd()

def test_cwd_is_charm_dir():
    charm = MyCharm()
    assert str(charm.charm_dir) == os.getcwd()
'''
PROBE_40_OPS_CONTEXT_LITERAL_PATH = '''\
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
    assert captured['cwd'] == '/path/to/charm'
'''
PROBE_40_OPS_CONTEXT_LITERAL_VAR_LIB = '''\
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
    assert captured['cwd'] == '/var/lib/juju/agents/unit-my-charm-0/charm'
'''
PROBE_40_OPS_CONTEXT_CHARM_DIR_FROM_HANDLER = '''\
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

_FALSE = [_FALSE_A08, _FALSE_B01, _FALSE_B05, _FALSE_C04]
_VALID = [
    _VALID_A01,
    _VALID_A04,
    _VALID_A05,
    _VALID_A07,
    _VALID_B00,
    _VALID_B03,
    _VALID_B09,
    _VALID_B10,
    _VALID_B13,
    _VALID_C05,
    _VALID_C08,
    _VALID_C13,
    _VALID_C14,
    _VALID_C15,
]


@pytest.mark.parametrize(
    "source",
    _FALSE,
    ids=['a08', 'b01', 'b05', 'c04'],
)
def test_false_finals_rejected(source: str):
    # The rule alone rejects each; in `check()` it is a backstop, so for
    # the `tmp_path` and charm-name finals the narrower rules speak instead.
    reasons = _rule(source)
    assert reasons
    assert all(_MARK in r and "self.framework.charm_dir" in r for r in reasons)
    assert not check(source).passed


def test_the_backstop_speaks_when_nothing_else_does():
    assert check(_FALSE_C04).reasons == _rule(_FALSE_C04)
    assert not any(_MARK in r for r in check(_FALSE_B05).reasons)


@pytest.mark.parametrize(
    "source",
    _VALID,
    ids=['a01', 'a04', 'a05', 'a07', 'b00', 'b03', 'b09', 'b10', 'b13', 'c05', 'c08', 'c13', 'c14', 'c15'],
)
def test_valid_finals_accepted(source: str):
    assert _rule(source) == []
    assert check(source).passed


# (probe, whether this rule rejects it). Each matches the probe's
# expectation: the probes that fail on the fix branch too (false comments)
# are rejected; the charm_root, handler-sourced and cwd-vs-cwd ones are not.
_PROBES = [
    pytest.param(PROBE_39_5_AS_SAVED, True, id='39:5-as-saved'),
    pytest.param(PROBE_39_5_BASENAME_EQUALS, True, id='39:5-basename-equals'),
    pytest.param(PROBE_39_5_NAME_IN_CWD, True, id='39:5-name-in-cwd'),
    pytest.param(PROBE_39_5_CHARM_ROOT_NAMED_MY_CHARM, False, id='39:5-charm_root-named-my-charm'),
    pytest.param(PROBE_39_5_CHARM_DIR_FROM_HANDLER, False, id='39:5-charm_dir-from-handler'),
    pytest.param(PROBE_40_4_AS_SAVED, True, id='40:4-as-saved'),
    pytest.param(PROBE_40_4_PLAIN_CLASS_COMPARES_WITH_GETCWD, False, id='40:4-plain-class-compares-with-getcwd'),
    pytest.param(PROBE_40_OPS_CONTEXT_LITERAL_PATH, True, id='40:ops-context-literal-path'),
    pytest.param(PROBE_40_OPS_CONTEXT_LITERAL_VAR_LIB, True, id='40:ops-context-literal-var-lib'),
    pytest.param(PROBE_40_OPS_CONTEXT_CHARM_DIR_FROM_HANDLER, False, id='40:ops-context-charm_dir-from-handler'),
]


@pytest.mark.parametrize(("source", "rejected"), _PROBES)
def test_hand_probes(source: str, rejected: bool):
    assert bool(_rule(source)) is rejected


def test_reason_text():
    (reason,) = _rule(_FALSE_C04)
    assert reason.startswith("line 9: `str(charm.charm_dir) == '/path/to/charm'` compares the working directory")
    assert "`'/path/to/charm'`" in reason
    assert "read the charm's directory in a handler as `self.framework.charm_dir`" in reason


_CHARM = """\
import os
from pathlib import Path

import ops
from ops import testing

captured = {{}}


class MyCharm(ops.CharmBase):
    def __init__(self, framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)

    def _on_start(self, event):
        captured['cwd'] = {cwd}
        captured['charm_dir'] = str(self.framework.charm_dir)


def test_cwd(tmp_path):
    ctx = testing.Context(MyCharm, meta={{'name': 'my-charm'}}{extra})
    ctx.run(ctx.on.start(), testing.State())
    assert {claim}
"""


def _charm(claim: str, cwd: str = "os.getcwd()", extra: str = "") -> str:
    return _CHARM.format(claim=claim, cwd=cwd, extra=extra)


@pytest.mark.parametrize(
    "claim",
    [
        "captured['cwd'] == str(tmp_path)",
        "captured['cwd'] == '/var/lib/juju/agents/unit-my-charm-0/charm'",
        "captured['cwd'] != '/'",
        "captured['cwd'].startswith('/tmp')",
        "os.path.basename(captured['cwd']) == 'charm'",
        "captured['cwd'] == str(ctx.charm_spec.charm_dir)",
        "captured['cwd'] == os.environ['HOME']",
        "os.path.samefile(captured['cwd'], tmp_path)",
        "Path(captured['cwd']) == tmp_path / 'my-charm'",
    ],
)
def test_unsourced_other_side_rejected(claim: str):
    assert _rule(_charm(claim))


@pytest.mark.parametrize(
    "cwd",
    ["os.getcwd()", "str(Path.cwd())", "str(Path('.').resolve())", "os.path.abspath('.')", "str(Path().absolute())"],
)
def test_cwd_sources(cwd: str):
    assert _rule(_charm("captured['cwd'] == '/charm'", cwd=cwd))
    assert not _rule(_charm("captured['cwd'] == captured['charm_dir']", cwd=cwd))


@pytest.mark.parametrize(
    "claim",
    [
        "captured['cwd'] == captured['charm_dir']",
        "captured['charm_dir'] == captured['cwd']",
        "Path(captured['cwd']) == Path(captured['charm_dir'])",
        "os.path.samefile(captured['cwd'], captured['charm_dir'])",
        "captured['cwd'] != captured['charm_dir']",
        # The cwd compared with the cwd is not a charm-directory claim.
        "captured['cwd'] == os.getcwd()",
        "captured['cwd'] is not None",
        "captured['cwd'] == str(ctx.charm_root)",
    ],
)
def test_handler_sourced_or_cwd_accepted(claim: str):
    assert _rule(_charm(claim)) == []


def test_charm_root_leaves_the_test_alone():
    source = _charm("captured['cwd'] == str(tmp_path)", extra=", charm_root=tmp_path")
    assert _rule(source) == []


def test_chdir_target_left_alone():
    source = _charm("captured['cwd'] == str(tmp_path)").replace(
        "    ctx = testing", "    os.chdir(str(tmp_path))\n    ctx = testing"
    )
    assert _rule(source) == []


def test_charm_dir_on_instance_and_closure():
    source = """\
import os
import ops
from ops import testing


def test_cwd():
    seen = {}

    class MyCharm(ops.CharmBase):
        def __init__(self, framework):
            super().__init__(framework)
            self.cwd_at_init = os.getcwd()
            framework.observe(self.on.start, self._on_start)

        def _on_start(self, event):
            seen.update({'cwd': os.getcwd(), 'dir': self.charm_dir})

    ctx = testing.Context(MyCharm, meta={'name': 'c'})
    with ctx(ctx.on.start(), testing.State()) as mgr:
        mgr.run()
        assert mgr.charm.cwd_at_init == str(mgr.charm.framework.charm_dir)
    assert seen['cwd'] == str(seen['dir'])
    assert seen['cwd'] == '/somewhere'
"""
    (reason,) = _rule(source)
    assert reason.startswith("line 23: `seen['cwd'] == '/somewhere'`")


def test_harness_charm_dir_accepted():
    source = """\
import os
import ops
from ops.testing import Harness


class MyCharm(ops.CharmBase):
    pass


def test_cwd():
    harness = Harness(MyCharm, meta='name: c')
    harness.begin()
    assert os.getcwd() == str(harness.charm.charm_dir)
"""
    assert _rule(source) == []


@pytest.mark.parametrize(
    "source",
    [
        # Reads the cwd but compares nothing with it.
        "import os\n\ndef test_x():\n    assert os.path.exists(os.path.join(os.getcwd(), 'x'))\n",
        # Compares paths, none of them the cwd.
        "from pathlib import Path\n\ndef test_x(tmp_path):\n    assert Path(str(tmp_path)) == tmp_path\n",
        # A literal in a non-test function.
        "import os\n\ndef helper():\n    return os.getcwd() == '/x'\n\ndef test_x():\n    helper()\n",
        # A status message.
        "def test_x():\n    message = 'ok'\n    assert message == 'ok'\n",
    ],
)
def test_non_cwd_tests_unaffected(source: str):
    assert _rule(source) == []
