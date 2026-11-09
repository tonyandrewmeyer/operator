"""Two fake-hook-tool rules from `#2709`'s live answers
(`spike-step-5/static-retry/RESULT.md` §30):

- a `CharmMeta` built from a literal the installed ops cannot build (8-7:
  `requires:\\n  db:\\n`, an endpoint with no mapping);
- a databag read of a relation access written inline, inside
  `pytest.raises(RelationNotFoundError)` (8-1), which `_databag_not_read()`
  only caught for a named relation.
"""

from __future__ import annotations

import ops
import pytest

from static_test_check import check

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")

_HEAD = '''\
import os

import pytest

import ops
from ops.model import _ModelBackend


def fake_hook_tool(bin_dir, name, script):
    path = bin_dir / name
    path.write_text('#!/bin/sh\\n' + script + '\\n')
    path.chmod(0o755)


def test_gone(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('JUJU_VERSION', '3.6.27')
    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'["db:1"]\\'')
    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'["provider/0"]\\'')
    fake_hook_tool(bin_dir, 'relation-get', 'echo "ERROR permission denied" >&2; exit 1')
    fake_hook_tool(bin_dir, 'is-leader', 'echo true')
'''

_GOOD_YAML = "'name: myapp\\nrequires:\\n  db:\\n    interface: db\\n'"
_BAD_YAML = "'name: myapp\\nrequires:\\n  db:\\n'"

_READ = '''\
    relation = model.get_relation('db', 1)
    try:
        data = dict(relation.data[relation.app])
    except ops.ModelError as e:
        data = e
    assert data == {}
'''


def _test(meta_line: str, body: str = _READ) -> str:
    return f"{_HEAD}{meta_line}    model = ops.Model(meta, _ModelBackend('myapp/0'))\n{body}"


def _meta_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if "with the installed ops" in r]


# --- an unbuildable literal meta ---------------------------------------------


def test_the_bad_yaml_raises_with_the_installed_ops():
    with pytest.raises(TypeError):
        ops.CharmMeta.from_yaml("name: myapp\nrequires:\n  db:\n")


def test_the_good_yaml_passes():
    assert check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n")).passed


def test_an_endpoint_with_no_mapping_is_rejected():
    result = check(_test(f"    meta = ops.CharmMeta.from_yaml({_BAD_YAML})\n"))
    assert not result.passed
    assert result.reasons == [
        "line 24: `ops.CharmMeta.from_yaml(...)` raises `TypeError: 'NoneType' object is not "
        "subscriptable` with the installed ops, so the test fails there with or without the bug; "
        "every endpoint under `requires`, `provides` or `peers` needs a mapping with its "
        "`interface:`, and `db` (under `requires`) has none: write it as `db:` with an indented "
        "`interface: db` line under it (`requires:\\n  db:\\n    interface: db\\n` in the YAML string)"
    ]


def test_inline_in_the_model_call_is_rejected():
    """8-7's own shape: the meta built inside the `ops.Model(...)` call."""
    source = _HEAD + (
        f"    model = ops.Model(ops.CharmMeta.from_yaml({_BAD_YAML}), _ModelBackend('myapp/0'))\n" + _READ
    )
    assert len(_meta_reasons(source)) == 1


def test_a_name_assigned_once_to_the_yaml_is_followed():
    source = _test(f"    raw = {_BAD_YAML}\n    meta = ops.CharmMeta.from_yaml(raw)\n")
    assert len(_meta_reasons(source)) == 1


def test_a_dict_literal_is_rejected():
    source = _test("    meta = ops.CharmMeta({'name': 'myapp', 'requires': {'db': None}})\n")
    reasons = _meta_reasons(source)
    assert len(reasons) == 1
    assert "`db` (under `requires`) has none" in reasons[0]


def test_an_interface_string_in_place_of_a_mapping_is_rejected():
    source = _test("    meta = ops.CharmMeta.from_yaml('name: myapp\\nrequires:\\n  db: db\\n')\n")
    reasons = _meta_reasons(source)
    assert len(reasons) == 1
    assert "`TypeError: string indices must be integers" in reasons[0]
    assert "`db` (under `requires`) has none" in reasons[0]


def test_yaml_that_does_not_parse_gets_the_general_advice():
    source = _test("    meta = ops.CharmMeta.from_yaml('name: [\\n')\n")
    reasons = _meta_reasons(source)
    assert len(reasons) == 1
    assert reasons[0].endswith("build the meta from YAML (or a dict) the installed ops accepts, as in the example")


@pytest.mark.parametrize(
    "wrapped",
    [
        f"    with pytest.raises(TypeError):\n        meta = ops.CharmMeta.from_yaml({_BAD_YAML})\n"
        f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n",
        f"    try:\n        meta = ops.CharmMeta.from_yaml({_BAD_YAML})\n    except Exception:\n"
        f"        meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n",
        f"    if os.environ.get('X'):\n        meta = ops.CharmMeta.from_yaml({_BAD_YAML})\n"
        f"    else:\n        meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n",
    ],
)
def test_a_build_that_may_not_run_or_is_caught_passes(wrapped):
    assert _meta_reasons(_test(wrapped)) == []


def test_a_helper_function_is_not_followed():
    source = _HEAD.replace(
        "def test_gone(",
        f"def make_meta():\n    return ops.CharmMeta.from_yaml({_BAD_YAML})\n\n\ndef test_gone(",
    ) + f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n    model = ops.Model(meta, _ModelBackend('myapp/0'))\n" + _READ
    assert _meta_reasons(source) == []


def test_a_yaml_that_is_not_a_literal_passes():
    source = _test("    meta = ops.CharmMeta.from_yaml(open('metadata.yaml').read())\n")
    assert _meta_reasons(source) == []


def test_off_without_fake_hook_tools():
    """A bug in how ops parses metadata is reproduced by exactly this call."""
    source = (
        "import ops\n\n\ndef test_meta():\n"
        f"    meta = ops.CharmMeta.from_yaml({_BAD_YAML})\n"
        "    assert 'db' in meta.requires\n"
    )
    assert check(source).passed


# --- a nested lookup inside pytest.raises(RelationNotFoundError) -------------

_NESTED = '''\
    with pytest.raises(ops.RelationNotFoundError):
        dict(model.get_relation('db', 1).data[model.get_relation('db', 1).app])
'''


def test_the_nested_read_in_raises_is_rejected():
    result = check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", _NESTED))
    assert result.reasons == [
        "line 27: `dict(model.get_relation('db', 1).data[model.get_relation('db', 1).app])` cannot raise "
        "`RelationNotFoundError`: `RelationDataContent._load()` catches it and returns `{}`, in every "
        "version of ops, so a relation that is gone reads as `{}`, and the "
        "`pytest.raises(ops.RelationNotFoundError)` around it fails with \"DID NOT RAISE\" whether or "
        "not the bug is there; assert that the databag reads as `{}` instead, outside the "
        "`pytest.raises`: the test's `relation-get` fake exits 1, so on a version of ops with the bug "
        "the read raises `ops.ModelError`, so catch that and assert on the result: "
        "`try: result = dict(model.get_relation('db', 1).data[model.get_relation('db', 1).app])` / "
        "`except ops.ModelError as e: result = e` / `assert result == {}`"
    ]


def test_the_named_form_gives_the_same_reason_shape():
    named = (
        "    relation = model.get_relation('db', 1)\n"
        "    with pytest.raises(ops.RelationNotFoundError):\n"
        "        dict(relation.data[relation.app])\n"
    )
    reasons = check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", named)).reasons
    assert any(
        r.startswith("line 28: `dict(relation.data[relation.app])` cannot raise `RelationNotFoundError`")
        for r in reasons
    )


def test_the_nested_read_through_relations_is_rejected():
    body = (
        "    with pytest.raises(ops.RelationNotFoundError):\n"
        "        assert model.relations['db'][0].data[model.relations['db'][0].app] == {}\n"
    )
    reasons = check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", body)).reasons
    assert any("cannot raise `RelationNotFoundError`" in r for r in reasons)


def test_the_nested_read_with_an_access_that_may_raise_passes_this_rule():
    """`relation-ids` exiting non-zero: the access itself may raise a
    `ModelError`, so the block is not certain to fail on the read."""
    source = _test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", _NESTED).replace(
        "'echo \\'[\"db:1\"]\\''", "'exit 1'"
    )
    assert "exit 1" in source
    assert not any("cannot raise `RelationNotFoundError`" in r for r in check(source).reasons)


@pytest.mark.parametrize(
    "access",
    [
        "dict(model.get_relation('db', 2).data[model.get_relation('db', 2).app])",
        "dict(model.get_relation('db', 2).data[model.get_relation('db', 2).unit])",
    ],
)
def test_a_relation_relation_ids_does_not_list_passes_this_rule(access):
    """`relation-ids` prints `db:1`: on `main` a relation that is not listed
    can raise `RelationNotFoundError` (`test_databag_not_read.py`)."""
    body = f"    with pytest.raises(ops.RelationNotFoundError):\n        {access}\n"
    reasons = check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", body)).reasons
    assert not any("cannot raise `RelationNotFoundError`" in r for r in reasons)


def test_the_nested_read_of_this_apps_own_databag_passes_this_rule():
    """Narrow by choice: only the inline access's own `.app` / `.unit`."""
    body = "    with pytest.raises(ops.RelationNotFoundError):\n        dict(model.get_relation('db', 1).data[model.app])\n"
    reasons = check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", body)).reasons
    assert not any("cannot raise `RelationNotFoundError`" in r for r in reasons)


def test_the_nested_read_under_model_error_is_left_to_other_rules():
    body = _NESTED.replace("ops.RelationNotFoundError", "ops.ModelError")
    reasons = check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", body)).reasons
    assert not any("cannot raise `RelationNotFoundError`" in r for r in reasons)


def test_the_repair_the_reason_asks_for_passes():
    body = (
        "    try:\n"
        "        result = dict(model.get_relation('db', 1).data[model.get_relation('db', 1).app])\n"
        "    except ops.ModelError as e:\n"
        "        result = e\n"
        "    assert result == {}\n"
    )
    assert check(_test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", body)).passed


def test_the_nested_read_runs_no_differently(tmp_path, monkeypatch):
    """By running: the nested read raises `ModelError` (not
    `RelationNotFoundError`) on a version with the bug, so the
    `pytest.raises` cannot pass there, and on one without it the read is
    `{}`, so it cannot pass there either."""
    namespace: dict = {}
    source = _test(f"    meta = ops.CharmMeta.from_yaml({_GOOD_YAML})\n", _NESTED)
    exec(compile(source, "t.py", "exec"), namespace)
    with pytest.raises(BaseException) as info:
        namespace["test_gone"](tmp_path, monkeypatch)
    assert not isinstance(info.value, ops.RelationNotFoundError)
