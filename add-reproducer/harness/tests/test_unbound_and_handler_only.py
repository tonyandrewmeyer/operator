"""Two narrow widenings from §32's live answers on `#2709`
(`spike-step-5/static-retry/RESULT.md` §32, §33):

- 9-4: an unbound name after a `try` whose body looks the relation up and
  assigns the unread databag lookup to a name (`relation =
  model.get_relation('db', 1)`; `data = relation.data[relation.app]`), so
  the `except ops.ModelError as e: result = e` never runs and `assert
  isinstance(result, ...)` raises `UnboundLocalError` on every version;
- 9-5: an unread lookup in a `try` whose only assertion is in a handler,
  so no handler runs and the test passes on every version.
"""

from __future__ import annotations

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
    monkeypatch.setenv('JUJU_VERSION', '3.6.0')
    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'["db:1"]\\'')
    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'["provider/0"]\\'')
    fake_hook_tool(bin_dir, 'relation-get', 'echo "ERROR permission denied" >&2; exit 1')
    fake_hook_tool(bin_dir, 'is-leader', 'echo true')
    meta = ops.CharmMeta.from_yaml('name: myapp\\nrequires:\\n  db:\\n    interface: db\\n')
    model = ops.Model(meta, _ModelBackend('myapp/0'))
'''


def _run(source: str, tmp_path, monkeypatch) -> BaseException | None:
    namespace: dict = {}
    exec(compile(source, "t.py", "exec"), namespace)
    try:
        namespace["test_gone"](tmp_path, monkeypatch)
    except BaseException as e:  # noqa: BLE001
        return e
    return None


def _unbound(source: str) -> list[str]:
    return [r for r in check(source).reasons if "`UnboundLocalError`" in r]


def _handler_only(source: str) -> list[str]:
    return [r for r in check(source).reasons if "only assertions are in" in r]


# --- 9-4: unbound after `try`, relation and lookup inside it ----------------

_NINE_FOUR = _HEAD + '''\
    try:
        relation = model.get_relation('db', 1)
        data = relation.data[relation.app]
    except ops.ModelError as e:
        result = e
    assert isinstance(result, ops.RelationNotFoundError)
'''


def test_9_4_is_rejected_and_crashes(tmp_path, monkeypatch):
    reasons = _unbound(_NINE_FOUR)
    assert len(reasons) == 1
    reason = reasons[0]
    assert reason.startswith("line 28: `data = relation.data[relation.app]` only looks the databag up, and `data`")
    assert "`result` is never bound" in reason
    assert "`try: result = dict(relation.data[relation.app])`" in reason
    error = _run(_NINE_FOUR, tmp_path, monkeypatch)
    assert isinstance(error, UnboundLocalError)


def test_9_4_as_a_bare_statement_is_rejected(tmp_path, monkeypatch):
    source = _NINE_FOUR.replace("data = relation.data", "relation.data")
    assert _unbound(source)
    assert isinstance(_run(source, tmp_path, monkeypatch), UnboundLocalError)


def test_9_4_corrected_passes_and_runs(tmp_path, monkeypatch):
    source = _HEAD + '''\
    try:
        relation = model.get_relation('db', 1)
        result = dict(relation.data[relation.app])
    except ops.ModelError as e:
        result = e
    assert result == {}
'''
    assert check(source).passed, check(source).reasons
    error = _run(source, tmp_path, monkeypatch)
    assert error is None or isinstance(error, AssertionError)


def test_a_lookup_read_later_in_the_try_is_not_this_rule(tmp_path, monkeypatch):
    source = _NINE_FOUR.replace(
        "        data = relation.data[relation.app]\n",
        "        data = relation.data[relation.app]\n        result = dict(data)\n",
    )
    assert not _unbound(source)
    assert not isinstance(_run(source, tmp_path, monkeypatch), UnboundLocalError)


def test_a_relation_assigned_twice_is_left_alone():
    source = _NINE_FOUR.replace(
        "    try:\n", "    relation = None\n    try:\n"
    )
    assert not _unbound(source)


def test_a_relation_from_an_unfaked_relation_list_is_left_alone():
    source = _NINE_FOUR.replace("""    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'["provider/0"]\\'')\n""", "")
    assert not _unbound(source)


def test_a_relation_from_another_call_is_left_alone():
    source = _NINE_FOUR.replace("relation = model.get_relation('db', 1)", "relation = make_relation(model)")
    assert not _unbound(source)


# --- 9-5: unread lookup, the only assertion in a handler --------------------

_NINE_FIVE = _HEAD + '''\
    relation = model.get_relation('db', 1)
    try:
        relation.data[relation.app]
    except ops.RelationNotFoundError:
        pass
    except ops.ModelError as e:
        assert False, f'Expected RelationNotFoundError, got {type(e).__name__}'
'''


def test_9_5_is_rejected_and_passes_without_information(tmp_path, monkeypatch):
    reasons = _handler_only(_NINE_FIVE)
    assert len(reasons) == 1
    reason = reasons[0]
    assert reason.startswith("line 28: `relation.data[relation.app]` only looks the databag up and never reads it")
    assert "`except ops.RelationNotFoundError` and `except ops.ModelError` after it never run" in reason
    assert "passes whether or not the bug is there" in reason
    assert "`try: result = dict(relation.data[relation.app])`" in reason
    assert _run(_NINE_FIVE, tmp_path, monkeypatch) is None


def test_9_5_with_one_handler_is_rejected(tmp_path, monkeypatch):
    source = _HEAD + '''\
    relation = model.get_relation('db', 1)
    try:
        relation.data[relation.app]
    except ops.ModelError:
        pytest.fail('raised')
'''
    reasons = _handler_only(source)
    assert len(reasons) == 1 and "never runs, and the test's only assertions are in that handler" in reasons[0]
    assert _run(source, tmp_path, monkeypatch) is None


def test_9_5_corrected_passes_and_runs(tmp_path, monkeypatch):
    source = _HEAD + '''\
    relation = model.get_relation('db', 1)
    try:
        result = dict(relation.data[relation.app])
    except ops.ModelError as e:
        result = e
    assert result == {}
'''
    assert check(source).passed, check(source).reasons
    error = _run(source, tmp_path, monkeypatch)
    assert error is None or isinstance(error, AssertionError)


def test_an_assertion_after_the_try_is_not_this_rule():
    source = _NINE_FIVE + "    assert relation is not None\n"
    assert not _handler_only(source)


def test_handlers_with_no_assertion_are_not_this_rule():
    source = _NINE_FIVE.replace(
        "        assert False, f'Expected RelationNotFoundError, got {type(e).__name__}'\n", "        pass\n"
    )
    assert not _handler_only(source)


def test_a_broad_handler_is_left_alone():
    source = _NINE_FIVE.replace("except ops.ModelError as e:", "except Exception as e:")
    assert not _handler_only(source)


def test_an_else_is_left_alone():
    source = _NINE_FIVE + "    else:\n        pass\n"
    assert not _handler_only(source)


def test_a_read_in_the_try_is_not_this_rule():
    source = _NINE_FIVE.replace("        relation.data[relation.app]\n", "        dict(relation.data[relation.app])\n")
    assert not _handler_only(source)
