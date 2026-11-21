"""An `ops.Relation(...)` built by hand (`spike-step-5/static-retry/RESULT.md` §24).

Live on `#2709` (§23), run 4-0 built `ops.Relation('relation', 1, model)`, a
`TypeError` on ops 3.8.3 and `main` alike, and passed the check, so the
catch-and-assert after it never ran. These tests check by running that such
calls raise and that what the reason suggests instead works, and cover the
rule, which checks the call against the installed signature the way
`_testing_names()` checks `ops.testing` keywords.
"""

from __future__ import annotations

import inspect

import ops
import pytest

from static_test_check import check

from test_hook_tool_rules import _OUTPUT, _run, _shape

_NOT_BUILT = "`ops.Relation` is not built by hand"
_TOOLS = {t: _OUTPUT[t] for t in ("relation-ids", "relation-list", "relation-get", "is-leader")}

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")

_REJECTED = [
    "ops.Relation('db', 2, model)",
    "relation = ops.Relation('db', 2, model)",
    "ops.model.Relation('db', 2)",
    "ops.Relation(relation_name='db', relation_id=2, model=model)",
    "ops.Relation('db', 2, False, model.unit, backend, model._cache, True, None, None)",
    "if os.environ.get('X'):\n    ops.Relation('db', 2, model)",
]


def _hand_built_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if _NOT_BUILT in r]


# -- by running ----------------------------------------------------------------


@pytest.mark.parametrize("access", [a for a in _REJECTED if not a.startswith("if")])
def test_a_call_that_does_not_bind_raises_type_error(access, tmp_path, monkeypatch):
    with pytest.raises(TypeError):
        _run(_shape(access, dict(_TOOLS)), tmp_path, monkeypatch)


def test_what_the_reason_suggests_runs(tmp_path, monkeypatch):
    source = _shape("relation = model.get_relation('db', 2)\nassert dict(relation.data[relation.app]) == {}", dict(_TOOLS, **{"relation-get": "echo '{}'"}))
    assert check(source).passed, check(source).reasons
    _run(source, tmp_path, monkeypatch)


def test_the_signature_is_read_from_the_installed_ops():
    params = list(inspect.signature(ops.model.Relation).parameters)
    assert params[:6] == ["relation_name", "relation_id", "is_peer", "our_unit", "backend", "cache"]
    assert ops.Relation is ops.model.Relation


# -- the rule ------------------------------------------------------------------


@pytest.mark.parametrize("access", _REJECTED)
def test_a_hand_built_relation_that_does_not_bind_is_rejected(access):
    (reason,) = _hand_built_reasons(_shape(access, dict(_TOOLS)))
    assert "does not match the installed `ops.Relation(relation_name, relation_id, is_peer, our_unit, backend, cache, active=True)`" in reason
    assert "so it raises `TypeError` before anything is read, with or without the bug" in reason
    assert "get the relation from the model instead, `model.get_relation('db', 2)`" in reason


@pytest.mark.parametrize(
    "imports",
    ["from ops import Relation", "from ops.model import Relation", "from ops.model import Relation as R"],
)
def test_imported_names_resolve(imports):
    name = "R" if imports.endswith(" R") else "Relation"
    source = imports + "\n" + _shape(f"{name}('db', 2, model)", dict(_TOOLS))
    assert _hand_built_reasons(source)


def test_the_reason_says_what_is_missing():
    (reason,) = _hand_built_reasons(_shape("ops.Relation('db', 2, model)", dict(_TOOLS)))
    assert "(it is missing `our_unit`, `backend`, and `cache`)" in reason


def test_the_reason_names_unfaked_tools_in_a_fake_hook_tool_test():
    fakes = {"relation-get": _OUTPUT["relation-get"]}
    (reason,) = _hand_built_reasons(_shape("ops.Relation('relation', 1, model)", fakes))
    assert "`model.get_relation('relation', 1)`, with `relation` declared in the `CharmMeta` YAML under `requires`" in reason
    assert "and the test does not fake `relation-ids` and `relation-list`, so fake them too" in reason
    assert "like `[\"relation:1\"]`" in reason
    assert "fake it once, with a script that prints a unit list" in reason


def test_without_fake_hook_tools_the_reason_does_not_mention_them():
    source = "import ops\n\n\ndef test_it():\n    relation = ops.Relation('db', 2)\n    assert relation\n"
    (reason,) = _hand_built_reasons(source)
    assert "hook tool" not in reason


@pytest.mark.parametrize(
    "access",
    [
        # Binds.
        "ops.Relation('db', 2, False, model.unit, backend, model._cache)",
        "ops.Relation(relation_name='db', relation_id=2, is_peer=False, our_unit=model.unit, backend=backend, cache=model._cache)",
        # Cannot be checked.
        "ops.Relation(*ARGS)",
        "ops.Relation('db', **KWARGS)",
        # A different class.
        "ops.testing.Relation('db')",
        "testing.Relation('db')",
    ],
)
def test_a_call_it_cannot_be_sure_of_passes(access):
    source = "from ops import testing\nimport ops.testing\nARGS = ()\nKWARGS = {}\n" + _shape(access, dict(_TOOLS))
    assert not _hand_built_reasons(source)


def test_a_name_bound_twice_passes():
    source = "from ops import Relation\n" + _shape("Relation('db', 2)", dict(_TOOLS)) + "\n\ndef Relation(*args):\n    return args\n"
    assert not _hand_built_reasons(source)


def test_an_import_star_turns_the_rule_off():
    source = "from ops.model import *\n" + _shape("ops.Relation('db', 2, model)", dict(_TOOLS))
    assert not _hand_built_reasons(source)
