"""Two fake-hook-tool rules from §29's control answers, which the substrate
override now runs (`spike-step-5/static-retry/RESULT.md` §31, §32):

- a fake hook tool written with no `#!` line (c-2: `path.write_text(script)`,
  `OSError: [Errno 8] Exec format error: 'relation-ids'`);
- an attribute the installed `CharmMeta` does not have (c-7:
  `CharmMeta.from_yaml_file`; §27's 6-6: `CharmMeta.from_dict`).
"""

from __future__ import annotations

import ops
import pytest

from static_test_check import check

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")


def _head(write: str = "'#!/bin/sh\\n' + script + '\\n'") -> str:
    return f'''\
import os

import pytest

import ops
from ops.model import _ModelBackend


def fake_hook_tool(bin_dir, name, script):
    path = bin_dir / name
    path.write_text({write})
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


_YAML = "'name: myapp\\nrequires:\\n  db:\\n    interface: db\\n'"
_META = f"    meta = ops.CharmMeta.from_yaml({_YAML})\n"
_READ = '''\
    model = ops.Model(meta, _ModelBackend('myapp/0'))
    try:
        relation = model.get_relation('db', 1)
        data = dict(relation.data[relation.app])
    except ops.ModelError as e:
        data = e
    assert data == {}
'''


def _test(write: str | None = None, meta: str = _META, read: str = _READ) -> str:
    head = _head() if write is None else _head(write)
    return head + meta + read


def _run(source: str, tmp_path, monkeypatch) -> BaseException | None:
    namespace: dict = {}
    exec(compile(source, "t.py", "exec"), namespace)
    try:
        namespace["test_gone"](tmp_path, monkeypatch)
    except BaseException as e:  # noqa: BLE001
        return e
    return None


def _shebang_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if "no `#!` line" in r]


def _meta_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if "`CharmMeta` has no" in r]


# --- a fake with no `#!` line -------------------------------------------------


def test_the_example_shape_passes_and_runs(tmp_path, monkeypatch):
    source = _test()
    assert check(source).passed
    error = _run(source, tmp_path, monkeypatch)
    # It reaches the bug: on a version with it, the assertion fails on the
    # `ModelError`; on one without it, the test passes.
    assert error is None or isinstance(error, AssertionError)


def test_a_fake_with_no_shebang_is_rejected(tmp_path, monkeypatch):
    source = _test("script")
    reasons = _shebang_reasons(source)
    assert len(reasons) == 1
    reason = reasons[0]
    assert reason.startswith("line 27: `model.get_relation('db', 1)` runs `relation-ids` first")
    assert "`relation-ids`" in reason and "`relation-get`" in reason
    assert "`#!/bin/sh`" in reason
    assert "Exec format error: 'relation-ids'" in reason
    error = _run(source, tmp_path, monkeypatch)
    assert isinstance(error, OSError) and "Exec format error" in str(error)


def test_a_literal_prefix_with_no_shebang_is_rejected(tmp_path, monkeypatch):
    source = _test("'set -e\\n' + script + '\\n'")
    assert _shebang_reasons(source)
    assert isinstance(_run(source, tmp_path, monkeypatch), OSError)


def test_an_fstring_with_a_shebang_passes():
    assert check(_test("f'#!/bin/sh\\n{script}\\n'")).passed


def test_an_unknown_prefix_is_left_alone():
    # `head + script`: what `head` is, the rule cannot know.
    source = _test("head + script").replace(
        "def fake_hook_tool(bin_dir, name, script):", "def fake_hook_tool(bin_dir, name, script, head='#!/bin/sh\\n'):"
    )
    assert not _shebang_reasons(source)


def test_a_fake_that_is_never_chmodded_is_left_alone():
    source = _test("script").replace("    path.chmod(0o755)\n", "")
    assert not _shebang_reasons(source)


def test_os_chmod_counts():
    source = _test("script").replace("    path.chmod(0o755)\n", "    os.chmod(path, 0o755)\n")
    assert _shebang_reasons(source)


def test_a_shebangless_fake_of_a_tool_not_run_first_is_left_alone():
    # Only `is-leader` lacks a `#!` line, and the read runs `relation-ids` first.
    source = _test().replace(
        "def test_gone(", "def raw_tool(bin_dir, name, script):\n    path = bin_dir / name\n"
        "    path.write_text(script)\n    path.chmod(0o755)\n\n\ndef test_gone("
    ).replace("    fake_hook_tool(bin_dir, 'is-leader', 'echo true')\n", "    raw_tool(bin_dir, 'is-leader', 'echo true')\n")
    assert not _shebang_reasons(source)


def test_an_access_under_except_exception_is_left_alone():
    read = _READ.replace("except ops.ModelError as e:", "except Exception as e:")
    assert not _shebang_reasons(_test("script", read=read))


def test_the_reason_reaches_the_retry_feedback():
    result = check(_test("script"))
    assert not result.passed


# --- an attribute `CharmMeta` does not have --------------------------------------


@pytest.mark.parametrize("attr", ["from_yaml_file", "from_dict"])
def test_the_attributes_do_not_exist(attr):
    assert not hasattr(ops.CharmMeta, attr)


@pytest.mark.parametrize(
    "meta",
    [
        "    meta = ops.CharmMeta.from_yaml_file('metadata.yaml')\n",
        "    meta = ops.CharmMeta.from_dict({'name': 'myapp', 'requires': {'db': {'interface': 'db'}}})\n",
        "    meta = ops.charm.CharmMeta.from_dict({'name': 'myapp'})\n",
    ],
)
def test_a_missing_attribute_is_rejected(meta, tmp_path, monkeypatch):
    source = _test(meta=meta)
    reasons = _meta_reasons(source)
    assert len(reasons) == 1
    assert reasons[0].startswith("line 24: `ops.")
    assert "raises `AttributeError`" in reasons[0]
    assert "`CharmMeta.from_yaml(...)`" in reasons[0]
    error = _run(source, tmp_path, monkeypatch)
    assert isinstance(error, AttributeError)


def test_an_imported_alias_is_rejected():
    source = _test(meta="    meta = Meta.from_dict({'name': 'myapp'})\n").replace(
        "from ops.model import _ModelBackend\n", "from ops.model import _ModelBackend\nfrom ops import CharmMeta as Meta\n"
    )
    reasons = _meta_reasons(source)
    assert len(reasons) == 1 and "`Meta.from_dict`" in reasons[0]


def test_a_name_assigned_to_the_class_is_followed():
    meta = "    M = ops.CharmMeta\n    meta = M.from_dict({'name': 'myapp'})\n"
    assert _meta_reasons(_test(meta=meta))


def test_from_yaml_passes_and_runs(tmp_path, monkeypatch):
    source = _test()
    assert not _meta_reasons(source)
    assert check(source).passed
    error = _run(source, tmp_path, monkeypatch)
    assert error is None or isinstance(error, AssertionError)


def test_existing_attributes_pass():
    meta = (
        f"    meta = ops.CharmMeta.from_yaml({_YAML})\n"
        "    assert ops.CharmMeta.from_yaml is not None\n"
        "    assert ops.CharmMeta.__name__ == 'CharmMeta'\n"
    )
    assert not _meta_reasons(_test(meta=meta))


def test_an_attribute_the_test_sets_is_left_alone():
    meta = (
        "    monkeypatch.setattr(ops.CharmMeta, 'from_dict', classmethod(lambda cls, d: cls(d)), raising=False)\n"
        "    meta = ops.CharmMeta.from_dict({'name': 'myapp', 'requires': {'db': {'interface': 'db'}}})\n"
    )
    assert not _meta_reasons(_test(meta=meta))


def test_a_read_under_try_attributeerror_is_left_alone():
    meta = (
        "    try:\n"
        "        meta = ops.CharmMeta.from_dict({'name': 'myapp'})\n"
        "    except AttributeError:\n"
        f"        meta = ops.CharmMeta.from_yaml({_YAML})\n"
    )
    assert not _meta_reasons(_test(meta=meta))


def test_an_instance_attribute_is_not_this_rule():
    # `meta.foo` is an instance, not the class: other rules' business.
    meta = _META + "    _ = meta.relations\n"
    assert not _meta_reasons(_test(meta=meta))


def test_outside_a_fake_hook_tool_test_is_left_alone():
    source = "import ops\n\n\ndef test_meta():\n    ops.CharmMeta.from_dict({'name': 'x'})\n"
    assert not _meta_reasons(source)
