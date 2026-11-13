"""A databag looked up inside a `try` and first read after it
(`spike-step-5/static-retry/RESULT.md` §33's 10-5, §34, §35).

Live on `#2709`, 10-5 looked the relation and its databag up inside a
`try` whose handler catches `ops.ModelError`, and compared the databag
after the `try`. `RelationDataContent` loads lazily, so `relation-get`
runs at the comparison, outside the `try`, and on ops 3.8.3 the
`ModelError` escapes. The shape here has §34's YAML-string repair applied,
so the `from_yaml` rule does not reject it first.
"""

from __future__ import annotations

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


def test_gone(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('JUJU_VERSION', '3.6.0')
    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'["db:1"]\\'')
    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'["provider/0"]\\'')
    fake_hook_tool(bin_dir, 'relation-get', 'echo >&2 \\'ERROR permission denied\\'; exit 1')
    fake_hook_tool(bin_dir, 'is-leader', 'echo true')
    meta = 'name: myapp\\nrequires:\\n  db:\\n    interface: db\\n'
    model = ops.Model(ops.CharmMeta.from_yaml(meta), _ModelBackend('myapp/0'))
'''

# 10-5 as the live answer wrote it, with the YAML-string repair.
_TEN_FIVE = _HEAD + '''\
    try:
        rel = model.get_relation('db', 1)
        rel_data = rel.data[model.app]
    except ops.ModelError as e:
        rel_data = e
    assert rel_data == {}, 'Dead relations should return empty dict'
'''

_CORRECTED = _TEN_FIVE.replace("rel_data = rel.data[model.app]", "rel_data = dict(rel.data[model.app])")


def _run(source: str, tmp_path, monkeypatch) -> BaseException | None:
    namespace: dict = {}
    exec(compile(source, "t.py", "exec"), namespace)
    try:
        namespace["test_gone"](tmp_path, monkeypatch)
    except BaseException as e:  # noqa: BLE001
        return e
    return None


def _changed(source: str, old: str, new: str) -> str:
    assert old in source, old
    return source.replace(old, new)


def _after_try(source: str) -> list[str]:
    return [r for r in check(source).reasons if "is the first read of" in r]


def test_10_5_is_rejected_and_crashes(tmp_path, monkeypatch):
    reasons = _after_try(_TEN_FIVE)
    assert len(reasons) == 1, check(_TEN_FIVE).reasons
    reason = reasons[0]
    assert reason.startswith(
        "line 28: `rel_data == {}` is the first read of `rel_data`, which line 25 "
        "(`rel_data = rel.data[model.app]`) assigned inside the `try`"
    )
    assert "`RelationDataContent` loads lazily" in reason
    assert "outside the `try`" in reason
    assert 'exits 1, printing "ERROR permission denied" to stderr' in reason
    assert "`try: rel_data = dict(rel.data[model.app])`" in reason
    assert "`except ops.ModelError as e: rel_data = e`" in reason
    # It is the only reason: nothing else in the check rejects 10-5.
    assert check(_TEN_FIVE).reasons == reasons
    error = _run(_TEN_FIVE, tmp_path, monkeypatch)
    assert isinstance(error, __import__("ops").ModelError)
    assert "permission denied" in str(error)


def test_the_corrected_shape_passes_and_runs(tmp_path, monkeypatch):
    result = check(_CORRECTED)
    assert result.passed, result.reasons
    error = _run(_CORRECTED, tmp_path, monkeypatch)
    # On ops 3.8.3 (the bug), the handler catches the error and the test
    # fails its assertion; with the fix it passes.
    assert error is None or isinstance(error, AssertionError)
    if __import__("ops").__version__ == "3.8.3":
        assert isinstance(error, AssertionError)
        assert "Dead relations should return empty dict" in str(error)


def test_a_relation_assigned_before_the_try_is_rejected(tmp_path, monkeypatch):
    source = _HEAD + '''\
    rel = model.get_relation('db', 1)
    try:
        data = rel.data[rel.app]
    except (ops.RelationNotFoundError, ops.ModelError):
        data = {}
    assert dict(data) == {}
'''
    assert len(_after_try(source)) == 1, check(source).reasons
    assert isinstance(_run(source, tmp_path, monkeypatch), __import__("ops").ModelError)


def test_no_handler_binding_is_still_rejected():
    source = _changed(_TEN_FIVE, "        rel_data = e\n", "        pass\n")
    assert len(_after_try(source)) == 1


# --- near misses: left alone -----------------------------------------------


def test_a_read_inside_the_try_as_well_is_left_alone():
    source = _changed(_TEN_FIVE, 
        "        rel_data = rel.data[model.app]\n",
        "        rel_data = rel.data[model.app]\n        len(rel_data)\n",
    )
    assert not _after_try(source)


def test_a_lookup_outside_any_try_is_not_this_rule():
    source = _HEAD + '''\
    rel = model.get_relation('db', 1)
    rel_data = rel.data[model.app]
    assert rel_data == {}
'''
    assert not _after_try(source)


def test_a_name_assigned_again_after_the_try_is_left_alone():
    source = _changed(_TEN_FIVE, 
        "    assert rel_data ==", "    rel_data = rel_data\n    assert rel_data =="
    )
    assert not _after_try(source)


def test_a_try_with_else_or_finally_is_left_alone():
    source = _changed(_TEN_FIVE, "        rel_data = e\n", "        rel_data = e\n    finally:\n        pass\n")
    assert not _after_try(source)


def test_a_handler_that_cannot_catch_model_error_is_left_alone():
    source = _changed(_TEN_FIVE, "except ops.ModelError as e", "except KeyError as e")
    assert not _after_try(source)


def test_another_statement_in_the_try_is_left_alone():
    source = _changed(_TEN_FIVE, "        rel = model", "        print('x')\n        rel = model")
    assert not _after_try(source)


def test_a_relation_get_that_succeeds_is_left_alone():
    source = _changed(_TEN_FIVE, "'echo >&2 \\'ERROR permission denied\\'; exit 1'", "'echo {}'")
    assert not _after_try(source)


def test_a_not_found_error_is_left_alone():
    source = _changed(_TEN_FIVE, "ERROR permission denied", "ERROR relation not found")
    assert not _after_try(source)


def test_a_read_under_a_branch_is_left_alone():
    source = _changed(_TEN_FIVE, 
        "    assert rel_data == {}, 'Dead relations should return empty dict'\n",
        "    if os.environ.get('X'):\n        assert rel_data == {}\n",
    )
    assert not _after_try(source)


def test_an_assertion_before_the_read_is_left_alone():
    source = _changed(_TEN_FIVE, "    assert rel_data ==", "    assert model is not None\n    assert rel_data ==")
    assert not _after_try(source)


def test_a_relation_from_an_unfaked_relation_list_is_left_alone():
    source = _changed(_TEN_FIVE, """    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'["provider/0"]\\'')\n""", "")
    assert not _after_try(source)


def test_a_string_key_is_not_this_rule():
    source = _changed(_TEN_FIVE, "rel.data[model.app]", "rel.data[other]")
    assert not _after_try(source)


def test_a_non_reading_use_after_the_try_is_left_alone():
    source = _changed(_TEN_FIVE, "    assert rel_data == {}, 'Dead relations should return empty dict'\n", "    assert rel_data is not None\n")
    assert not _after_try(source)
