"""A fake hook tool whose script is not valid shell, and an unread lookup in
a `try` with `assert isinstance(N, X)` after it
(`spike-step-5/static-retry/RESULT.md` §35's 11-7, §36).

Live on `#2709`, 11-7 had both faults, and each alone makes it fail on
every version of ops: its `relation-ids` and `relation-list` fakes wrote
`echo '['"db:2'"]'`, which `sh` cannot parse, and, with that fixed, it
looked the databag up inside the `try` without reading it and then asserted
that the name was a `RelationNotFoundError`. It failed the same way on
3.8.3 and `main`, and the pipeline would have commented.
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

def test_permission_denied_raises_relation_not_found(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('JUJU_VERSION', '3.6.0')
    fake_hook_tool(bin_dir, 'is-leader', 'echo true')
'''

_BROKEN_FAKES = '''\
    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'[\\'\\"db:2\\'\\"]\\'')
    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'[\\'\\"provider/0\\'\\"]\\'')
'''

_FIXED_FAKES = '''\
    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'["db:2"]\\'')
    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'["provider/0"]\\'')
'''

_TAIL = '''\
    fake_hook_tool(bin_dir, 'relation-get', 'echo \\'ERROR permission denied\\' >&2; exit 1')
    model = ops.Model(ops.CharmMeta.from_yaml('name: myapp\\nrequires:\\n  db:\\n    interface: db'), _ModelBackend('myapp/0'))
    try:
        relation = model.get_relation('db', 2)
        relation_data = relation.data[relation.app]
    except ops.ModelError as e:
        relation_data = e
    assert isinstance(relation_data, ops.RelationNotFoundError)
'''

# 11-7's final answer as the live run wrote it
# (`live/retry-hooktool11-2709.json` run 7).
_ELEVEN_SEVEN = _HEAD + _BROKEN_FAKES + _TAIL
_QUOTING_FIXED = _HEAD + _FIXED_FAKES + _TAIL
# And with 9-1's read-and-assert as well.
_BOTH_FIXED = _QUOTING_FIXED.replace(
    "relation_data = relation.data[relation.app]", "relation_data = dict(relation.data[relation.app])"
).replace(
    "assert isinstance(relation_data, ops.RelationNotFoundError)", "assert relation_data == {}"
)

_TEST = "test_permission_denied_raises_relation_not_found"


def _run(source: str, tmp_path, monkeypatch) -> BaseException | None:
    """Run the test on the installed ops, with what `relation_data` is as
    the message of its last `assert` (an `exec`'d assert has no rewriting):
    the exception, or the name of its type (`repr()` of a databag would
    load it, and run `relation-get`)."""
    lines = source.rstrip("\n").split("\n")
    assert lines[-1].startswith("    assert ")
    lines[-1] += (
        ", repr(relation_data) if isinstance(relation_data, BaseException) else type(relation_data).__name__"
    )
    namespace: dict = {}
    exec(compile("\n".join(lines) + "\n", "t.py", "exec"), namespace)
    try:
        namespace[_TEST](tmp_path, monkeypatch)
    except BaseException as e:  # noqa: BLE001
        return e
    return None


def _changed(source: str, old: str, new: str) -> str:
    assert old in source, old
    return source.replace(old, new)


def _shell(source: str) -> list[str]:
    return [r for r in check(source).reasons if "is not valid shell" in r]


def _isinstance(source: str) -> list[str]:
    return [r for r in check(source).reasons if "can never pass" in r]


# --- 11-7 -------------------------------------------------------------------


def test_eleven_seven_as_written_is_rejected_on_its_shell(tmp_path, monkeypatch):
    result = check(_ELEVEN_SEVEN)
    assert not result.passed
    shell = _shell(_ELEVEN_SEVEN)
    assert shell == [
        "line 17: the `relation-ids` fake's script, `echo '['\"db:2'\"]'`, is not valid shell: `sh` exits 2 "
        "with `Syntax error: Unterminated quoted string` whenever `relation-ids` runs, on every version of "
        "ops, so `model.get_relation('db', 2)` (line 22) fails on that `relation-ids` with a plain "
        "`ops.ModelError` before the code under test runs, and the test cannot tell the bug from a broken "
        "fake; quote the script so it parses, with the JSON in single quotes and nothing else, for example "
        "`echo '[\"db:2\"]'`"
    ]
    # As written, `get_relation()` raises inside the `try`, so the lookup
    # is not certain to raise nothing, and rule 2 stays out of it.
    assert result.reasons == shell
    error = _run(_ELEVEN_SEVEN, tmp_path, monkeypatch)
    assert isinstance(error, AssertionError)
    assert "Unterminated quoted string" in str(error)


def test_eleven_seven_with_the_quoting_fixed_is_rejected_on_its_isinstance(tmp_path, monkeypatch):
    result = check(_QUOTING_FIXED)
    assert not _shell(_QUOTING_FIXED)
    reasons = _isinstance(_QUOTING_FIXED)
    assert result.reasons == reasons
    assert len(reasons) == 1
    reason = reasons[0]
    assert reason.startswith(
        "line 26: `assert isinstance(relation_data, ops.RelationNotFoundError)` can never pass: line 23 "
        "(`relation_data = relation.data[relation.app]`) only looks the databag up"
    )
    assert "`RelationDataContent` loads lazily" in reason
    assert "never an instance of `ops.RelationNotFoundError`" in reason
    assert "read the databag with `dict(...)` inside the `try`" in reason
    assert "`try: relation_data = dict(relation.data[relation.app])`" in reason
    assert "`except ops.ModelError as e: relation_data = e`" in reason
    assert "`assert relation_data == {}`" in reason
    error = _run(_QUOTING_FIXED, tmp_path, monkeypatch)
    assert isinstance(error, AssertionError)
    assert "RelationDataContent" in str(error)


def test_eleven_seven_with_both_fixed_passes_and_reproduces(tmp_path, monkeypatch):
    result = check(_BOTH_FIXED)
    assert result.passed, result.reasons
    error = _run(_BOTH_FIXED, tmp_path, monkeypatch)
    assert error is None or isinstance(error, AssertionError)
    if ops.__version__ == "3.8.3":
        assert isinstance(error, AssertionError)
        assert "permission denied" in str(error)


# --- rule 1: what is valid shell, or out of scope ---------------------------


@pytest.mark.parametrize(
    "script",
    [
        "echo \\'[\"db:2\"]\\'",
        "exit 1",
        "echo \\'ERROR permission denied\\' >&2\\nexit 1",
        "echo \\'[\"db:2\"]\\'\\necho done >&2\\nexit 0",
        "true",
    ],
)
def test_a_valid_script_is_left_alone(script):
    source = _changed(
        _QUOTING_FIXED,
        "fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'[\"db:2\"]\\'')",
        f"fake_hook_tool(bin_dir, 'relation-ids', '{script}')",
    )
    assert not _shell(source), check(source).reasons


@pytest.mark.parametrize(
    "script",
    [
        "f'echo \\'[\\'\"db:{2}\\'\"]\\''",
        "'echo \\'[\\'' + '\"db:2\\'\"]\\''",
        "str('echo \\'[\\'\"db:2\\'\"]\\'')",
    ],
)
def test_a_non_literal_script_is_left_alone(script):
    source = _changed(
        _ELEVEN_SEVEN,
        "fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'[\\'\\\"db:2\\'\\\"]\\'')",
        f"fake_hook_tool(bin_dir, 'relation-ids', {script})",
    )
    assert not _shell(source)


def test_a_name_assigned_once_to_the_literal_is_rejected():
    source = _changed(
        _ELEVEN_SEVEN,
        "    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'[\\'\\\"db:2\\'\\\"]\\'')\n",
        "    ids = 'echo \\'[\\'\\\"db:2\\'\\\"]\\''\n    fake_hook_tool(bin_dir, 'relation-ids', ids)\n",
    )
    assert len(_shell(source)) == 1


def test_a_name_assigned_twice_is_left_alone():
    source = _changed(
        _ELEVEN_SEVEN,
        "    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'[\\'\\\"db:2\\'\\\"]\\'')\n",
        "    ids = 'echo 1'\n    ids = 'echo \\'[\\'\\\"db:2\\'\\\"]\\''\n    fake_hook_tool(bin_dir, 'relation-ids', ids)\n",
    )
    assert not _shell(source)


def test_a_shell_heredoc_fake_is_left_alone():
    source = _changed(
        _ELEVEN_SEVEN,
        "    fake_hook_tool(bin_dir, 'relation-ids', 'echo \\'[\\'\\\"db:2\\'\\\"]\\'')\n",
        "    os.system(\"cat > \" + str(bin_dir / 'relation-ids') + \" <<'EOF'\\n#!/bin/sh\\necho '['\\\"db:2'\\\"]'\\nEOF\")\n",
    )
    assert not _shell(source)


def test_a_fake_that_is_never_run_is_left_alone():
    # Only `is-leader` is broken, and nothing in the test runs it first.
    source = _changed(
        _QUOTING_FIXED,
        "fake_hook_tool(bin_dir, 'is-leader', 'echo true')",
        "fake_hook_tool(bin_dir, 'is-leader', 'echo \\'true')",
    )
    assert not _shell(source)


def test_a_helper_that_does_not_chmod_is_left_alone():
    source = _changed(_ELEVEN_SEVEN, "    path.chmod(0o755)\n", "")
    assert not _shell(source)


def test_a_broken_relation_get_is_left_alone():
    source = _changed(
        _QUOTING_FIXED,
        "'echo \\'ERROR permission denied\\' >&2; exit 1'",
        "'echo \\'ERROR permission denied >&2; exit 1'",
    )
    assert not _shell(source)


# --- rule 2: near misses, left alone ----------------------------------------


def test_a_name_read_in_the_try_is_left_alone():
    source = _changed(
        _QUOTING_FIXED,
        "        relation_data = relation.data[relation.app]\n",
        "        relation_data = relation.data[relation.app]\n        len(relation_data)\n",
    )
    assert not _isinstance(source)


def test_an_isinstance_against_a_non_exception_class_is_left_alone():
    source = _changed(_QUOTING_FIXED, "ops.RelationNotFoundError)", "dict)")
    assert not _isinstance(source)


def test_a_tuple_of_exceptions_is_rejected():
    source = _changed(
        _QUOTING_FIXED, "ops.RelationNotFoundError)", "(ops.RelationNotFoundError, KeyError))"
    )
    assert len(_isinstance(source)) == 1


def test_a_tuple_with_a_non_exception_is_left_alone():
    source = _changed(_QUOTING_FIXED, "ops.RelationNotFoundError)", "(ops.RelationNotFoundError, dict))")
    assert not _isinstance(source)


def test_a_branch_around_the_assert_is_left_alone():
    source = _changed(
        _QUOTING_FIXED,
        "    assert isinstance(relation_data, ops.RelationNotFoundError)\n",
        "    if os.environ.get('X'):\n        assert isinstance(relation_data, ops.RelationNotFoundError)\n",
    )
    assert not _isinstance(source)


def test_an_assert_with_more_than_the_isinstance_is_left_alone():
    source = _changed(
        _QUOTING_FIXED,
        "assert isinstance(relation_data, ops.RelationNotFoundError)",
        "assert isinstance(relation_data, ops.RelationNotFoundError) or relation_data is not None",
    )
    assert not _isinstance(source)


@pytest.mark.parametrize("clause", ["else:\n        pass\n", "finally:\n        pass\n"])
def test_a_try_with_else_or_finally_is_left_alone(clause):
    source = _changed(
        _QUOTING_FIXED,
        "        relation_data = e\n",
        "        relation_data = e\n    " + clause,
    )
    assert not _isinstance(source)


def test_a_lookup_that_may_raise_is_left_alone():
    # `relation-ids` prints nothing ops can parse, so `get_relation()` may raise.
    source = _changed(_QUOTING_FIXED, "'echo \\'[\"db:2\"]\\''", "'echo db'")
    assert not _isinstance(source)


def test_an_undeclared_endpoint_is_left_alone():
    source = _changed(_QUOTING_FIXED, "requires:\\n  db:", "requires:\\n  other:")
    assert not _isinstance(source)


def test_another_read_first_is_left_alone():
    source = _changed(
        _QUOTING_FIXED,
        "    assert isinstance(relation_data, ops.RelationNotFoundError)\n",
        "    print(relation_data)\n    assert isinstance(relation_data, ops.RelationNotFoundError)\n",
    )
    assert not _isinstance(source)


def test_a_name_assigned_again_after_the_try_is_left_alone():
    source = _changed(
        _QUOTING_FIXED,
        "    assert isinstance(relation_data, ops.RelationNotFoundError)\n",
        "    relation_data = ops.RelationNotFoundError()\n    assert isinstance(relation_data, ops.RelationNotFoundError)\n",
    )
    assert not _isinstance(source)


def test_the_models_app_as_the_key_is_rejected():
    source = _changed(_QUOTING_FIXED, "relation.data[relation.app]", "relation.data[model.app]")
    assert len(_isinstance(source)) == 1


def test_a_relation_assigned_before_the_try_is_rejected(tmp_path, monkeypatch):
    source = _changed(
        _QUOTING_FIXED,
        "    try:\n        relation = model.get_relation('db', 2)\n",
        "    relation = model.get_relation('db', 2)\n    try:\n",
    )
    assert len(_isinstance(source)) == 1, check(source).reasons
    error = _run(source, tmp_path, monkeypatch)
    assert isinstance(error, AssertionError)
