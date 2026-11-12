"""`CharmMeta.from_yaml()` given a non-string literal, from `#2709`'s live
answers (`spike-step-5/static-retry/RESULT.md` §33): 10-5's final answer and
10-2's first answer passed a dict, which raises `TypeError` on ops 3.8.3 and
`main` before the test reaches the bug.
"""

from __future__ import annotations

import ops
import pytest

from static_test_check import check

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")

_HEAD = '''\
import os
import ops
from ops.model import _ModelBackend


def fake_hook_tool(bin_dir, name, script):
    path = bin_dir / name
    path.write_text('#!/bin/sh\\n' + script + '\\n')
    path.chmod(0o755)


def test_relation_get_permission_denied(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('JUJU_VERSION', '3.6.0')
    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'["db:1"]\\'')
    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'["provider/0"]\\'')
    fake_hook_tool(bin_dir, 'relation-get', 'echo >&2 \\'ERROR permission denied\\'; exit 1')
    fake_hook_tool(bin_dir, 'is-leader', 'echo true')
'''

# 10-5's tail, as the live answer wrote it.
_TAIL = '''\
    try:
        rel = model.get_relation('db', 1)
        rel_data = rel.data[model.app]
    except ops.ModelError as e:
        rel_data = e
    assert rel_data == {}, 'Dead relations should return empty dict'
'''

_DICT = "{'name': 'myapp', 'requires': {'db': {'interface': 'db'}}}"
_YAML = "'name: myapp\\nrequires:\\n  db:\\n    interface: db\\n'"


def _named(value: str) -> str:
    """10-5's shape: the meta assigned once to a name, then passed."""
    return f"{_HEAD}    meta = {value}\n    model = ops.Model(ops.CharmMeta.from_yaml(meta), _ModelBackend('myapp/0'))\n{_TAIL}"


def _inline(value: str) -> str:
    return f"{_HEAD}    model = ops.Model(ops.CharmMeta.from_yaml({value}), _ModelBackend('myapp/0'))\n{_TAIL}"


def _rule(source: str) -> list[str]:
    return [r for r in check(source).reasons if "takes the metadata as a YAML string" in r]


def _run(source: str, tmp_path, monkeypatch):
    namespace: dict = {}
    exec(compile(source, "t.py", "exec"), namespace)
    namespace["test_relation_get_permission_denied"](tmp_path, monkeypatch)


# --- rejected ------------------------------------------------------------------


def test_10_5_is_rejected():
    reasons = _rule(_named(_DICT))
    assert reasons == [
        "line 22: `ops.CharmMeta.from_yaml(...)` is given `meta`, a dict (line 21), but it takes the "
        "metadata as a YAML string (or a stream): with the installed ops, a dict raises `TypeError: a "
        "string or stream input is required` there, so the test fails before it reaches the bug, with "
        "or without it; pass the YAML as a string, as in the example (`ops.CharmMeta.from_yaml('name: "
        "myapp\\nrequires:\\n  db:\\n    interface: db\\n')`); here that is `'name: myapp\\nrequires:\\n"
        "  db:\\n    interface: db\\n'`"
    ]
    assert not check(_named(_DICT)).passed


def test_the_inline_dict_is_rejected():
    reasons = _rule(_inline(_DICT))
    assert len(reasons) == 1
    assert reasons[0].startswith("line 21: `ops.CharmMeta.from_yaml(...)` is given a dict, but")


@pytest.mark.parametrize("source", [_named(_DICT), _inline(_DICT)], ids=["named", "inline"])
def test_the_rejected_shapes_raise_type_error(source, tmp_path, monkeypatch):
    with pytest.raises(TypeError, match="a string or stream input is required"):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "value, kind",
    [
        ("['name: myapp']", "a list"),
        ("{'name: myapp'}", "a set"),
        ("('name: myapp',)", "a tuple"),
        ("{}", "a dict"),
        ("1", "a number"),
        ("-1.5", "a number"),
        ("{'name': os.environ.get('APP', 'myapp')}", "a dict"),
    ],
)
def test_other_non_string_literals_are_rejected(value, kind):
    reasons = _rule(_inline(value))
    assert len(reasons) == 1
    assert f"is given {kind}, but" in reasons[0]


def test_a_module_level_name_is_followed():
    source = _inline("META").replace("import os\n", f"import os\n\nMETA = {_DICT}\n", 1)
    assert len(_rule(source)) == 1


def test_the_keyword_form_is_rejected():
    assert len(_rule(_inline(f"metadata={_DICT}"))) == 1


def test_charm_meta_imported_from_ops_charm_is_rejected():
    source = _inline(_DICT).replace("ops.CharmMeta.from_yaml", "CharmMeta.from_yaml")
    source = source.replace("import ops\n", "import ops\nfrom ops.charm import CharmMeta\n", 1)
    assert len(_rule(source)) == 1
    assert len(_rule(_inline(_DICT).replace("ops.CharmMeta", "ops.charm.CharmMeta"))) == 1


# --- the same tests with a YAML string -----------------------------------------


@pytest.mark.parametrize("source", [_named(_YAML), _inline(_YAML)], ids=["named", "inline"])
def test_a_yaml_string_passes_this_rule(source):
    assert _rule(source) == []


@pytest.mark.parametrize("source", [_named(_YAML), _inline(_YAML)], ids=["named", "inline"])
def test_a_yaml_string_runs_past_from_yaml(source, tmp_path, monkeypatch):
    """Past `from_yaml` it is §33's repaired 10-5: the lookup is read after
    the `try`, so on 3.8.3 the `relation-get` failure escapes as a
    `ModelError`; on a version without the bug it passes. Either way, no
    `TypeError`."""
    try:
        _run(source, tmp_path, monkeypatch)
    except TypeError:  # pragma: no cover - the failure this rule is about
        pytest.fail("from_yaml raised TypeError on a YAML string")
    except (ops.ModelError, AssertionError):
        pass


# --- left alone ------------------------------------------------------------------


def test_a_name_assigned_more_than_once_is_left_alone():
    source = _named(_DICT).replace(
        f"    meta = {_DICT}\n", f"    meta = {_DICT}\n    meta = {_YAML}\n", 1
    )
    assert _rule(source) == []


def test_read_text_is_left_alone():
    source = _inline("(tmp_path / 'metadata.yaml').read_text()")
    assert _rule(source) == []


@pytest.mark.parametrize(
    "value",
    [
        "open('metadata.yaml')",
        "f'name: {\"myapp\"}\\n'",
        "yaml.safe_dump({'name': 'myapp'})",
        "b'name: myapp\\n'",
    ],
)
def test_calls_streams_and_strings_are_left_alone(value):
    assert _rule(_inline(value)) == []


def test_a_name_assigned_from_a_call_is_left_alone():
    assert _rule(_named("dict(name='myapp')")) == []


def test_a_name_assigned_in_a_nested_block_is_left_alone():
    source = _named(_DICT).replace(f"    meta = {_DICT}\n", f"    if True:\n        meta = {_DICT}\n", 1)
    assert _rule(source) == []


@pytest.mark.parametrize(
    "wrapped",
    [
        f"    with pytest.raises(TypeError):\n        ops.CharmMeta.from_yaml({_DICT})\n",
        f"    try:\n        ops.CharmMeta.from_yaml({_DICT})\n    except Exception:\n        pass\n",
        f"    if os.environ.get('X'):\n        ops.CharmMeta.from_yaml({_DICT})\n",
    ],
)
def test_a_call_that_may_not_run_or_is_caught_is_left_alone(wrapped):
    source = f"import os\nimport ops\nimport pytest\n\n\ndef test_meta():\n{wrapped}"
    assert _rule(source) == []


def test_a_helper_function_is_not_followed():
    source = (
        f"import ops\n\n\ndef make_meta():\n    return ops.CharmMeta.from_yaml({_DICT})\n\n\n"
        "def test_meta():\n    assert make_meta().name == 'myapp'\n"
    )
    assert _rule(source) == []


def test_another_from_yaml_is_left_alone():
    source = f"import ops\n\n\ndef test_meta():\n    ops.CharmMeta.from_dict({_DICT})\n    Thing.from_yaml({_DICT})\n"
    assert _rule(source) == []
