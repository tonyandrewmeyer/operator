"""`.framework` read off an `ops.Model` (`spike-step-5/static-retry/RESULT.md` §29).

Live on `#2709`, three fake-hook-tool answers drove a charm by hand with
`MyCharm(model.framework)` or `model.framework.on.start.emit()`. `ops.Model`
has no `framework`, so each died on the `AttributeError` on ops 3.8.3 and
`main` alike. These tests check by running that the read raises, and cover
the rule.
"""

from __future__ import annotations

import ops
import pytest

from static_test_check import check

from test_hook_tool_rules import _OUTPUT, _run, _shape

_NO_FRAMEWORK = "`ops.Model` has no `framework`"
_TOOLS = {t: _OUTPUT[t] for t in ("relation-ids", "relation-list", "relation-get", "is-leader")}

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")


def _framework_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if _NO_FRAMEWORK in r]


def test_ops_model_has_no_framework():
    assert not hasattr(ops.Model, "framework")


def test_reading_it_raises_attribute_error(tmp_path, monkeypatch):
    with pytest.raises(AttributeError, match="framework"):
        _run(_shape("model.framework.on.start.emit()", dict(_TOOLS)), tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "access",
    [
        "model.framework.on.start.emit()",
        "charm = ops.CharmBase(model.framework)",
        "framework = model.framework",
        "m = model\nm.framework",
    ],
)
def test_a_framework_read_off_the_model_is_rejected(access):
    reasons = _framework_reasons(_shape(access, dict(_TOOLS)))
    assert len(reasons) == 1
    assert "`_ModelBackend` with fake hook tools" in reasons[0]


def test_a_framework_read_off_an_inline_model_is_rejected():
    source = "import ops\n\n\ndef test_x():\n    ops.Model(ops.CharmMeta(), None).framework\n"
    assert len(_framework_reasons(source)) == 1


def test_the_rule_follows_names_per_test_function():
    source = (
        "import ops\n\n\n"
        "def test_a():\n    model = ops.Model(ops.CharmMeta(), None)\n    model.framework\n\n\n"
        "def test_b():\n    model = ops.Model(ops.CharmMeta(), None)\n    model.framework\n"
    )
    assert [r.split(":")[0] for r in _framework_reasons(source)] == ["line 6", "line 11"]


@pytest.mark.parametrize(
    "source",
    [
        # A charm's own `self.framework` is real.
        "import ops\n\n\nclass C(ops.CharmBase):\n    def __init__(self, framework):\n"
        "        super().__init__(framework)\n        self.framework.observe(self.on.start, print)\n\n\n"
        "def test_x():\n    pass\n",
        # `self.model` inside a charm class is not followed.
        "import ops\n\n\nclass C(ops.CharmBase):\n    def f(self):\n        return self.model.framework\n\n\n"
        "def test_x():\n    pass\n",
        # A name that is not certainly an `ops.Model`.
        "import ops\n\n\ndef test_x(model):\n    model.framework\n",
        "import ops\n\n\ndef test_x():\n    model = ops.Model(ops.CharmMeta(), None)\n"
        "    model = object()\n    model.framework\n",
    ],
)
def test_other_framework_reads_pass(source):
    assert _framework_reasons(source) == []


def test_the_read_through_the_model_passes(tmp_path, monkeypatch):
    access = (
        "relation = model.get_relation('db', 2)\n"
        "try:\n    data = dict(relation.data[relation.app])\nexcept ops.ModelError as e:\n    data = e\n"
        "assert data == {'k': 'v'}"
    )
    source = _shape(access, dict(_TOOLS))
    assert check(source).passed, check(source).reasons
    _run(source, tmp_path, monkeypatch)
