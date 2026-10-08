"""Static rules for the fake hook tool shape (`spike-step-5/static-retry/RESULT.md` §17).

On `#2709` the extraction took the fake hook tool shape in 8 of 8 live runs,
and every test died before it reached the bug (§16): an endpoint the
`CharmMeta` never declares, `relation-ids` left unfaked, and keywords
`_ModelBackend.relation_get()` does not take. These tests cover the three rules
that reject those shapes, and check the hook tools each model access needs by
running the access on the installed ops with every tool faked, and with each
one left out.

The calibration corpus is the 8 saved extractions under
`fixtures/static_test_check/2709-hooktool.json`.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

import static_test_check
from models import StaticCheckResult
from static_test_check import check, retry_hints

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Six of the eight swap the quotes in the prompt example's
# `f"{bin_dir}{os.pathsep}{os.environ['PATH']}"`, which only parses on
# Python 3.12 and later.
_NESTED_QUOTES = {0, 3, 4, 5, 6, 7}
_UNDECLARED = "which the `CharmMeta` on line"
_NOT_FAKED = "and the test does not fake"
_IS_LEADER = "ops runs `is-leader` as well"

# run -> phrases each expected reason contains (on Python 3.12 and later).
EXPECTED = {
    0: [_UNDECLARED, "does not fake `relation-ids` and `relation-list`", _IS_LEADER],
    1: [_IS_LEADER, "has no `unit_name` argument", "has no `app_name` argument"],
    2: ["does not fake `relation-ids`", "its fake prints `provider`, which is not JSON"],
    3: [_UNDECLARED, "does not fake `relation-ids`", "its fake prints nothing, which is not JSON", _IS_LEADER],
    4: [_UNDECLARED, "does not fake `relation-ids` and `relation-list`"],
    5: [_UNDECLARED, "does not fake `relation-ids` and `relation-list`"],
    6: ["passes 1 where the endpoint name goes"],
    7: [_UNDECLARED, "does not fake `relation-ids` and `relation-list`"],
}


def _corpus():
    for record in json.loads((FIXTURES / "static_test_check" / "2709-hooktool.json").read_text()):
        yield pytest.param(record, id=f"run-{record['run']}")


def test_the_corpus_is_the_8_extractions():
    assert len(list(_corpus())) == 8 == len(EXPECTED)


@pytest.mark.parametrize("record", list(_corpus()))
def test_calibration_verdict(record):
    run = record["run"]
    result = check(record["test_file"]["body"])
    assert not result.passed
    if sys.version_info < (3, 12) and run in _NESTED_QUOTES:
        assert len(result.reasons) == 1 and "does not parse" in result.reasons[0]
        return
    phrases = EXPECTED[run]
    for phrase in phrases:
        assert any(phrase in reason for reason in result.reasons), (phrase, result.reasons)
    for reason in result.reasons:
        assert any(phrase in reason for phrase in phrases), reason


def test_run_6s_reason_says_what_the_call_takes_and_needs():
    (record,) = [p.values[0] for p in _corpus() if p.values[0]["run"] == 6]
    if sys.version_info < (3, 12):
        pytest.skip("run 6 only parses on Python 3.12 and later")
    (reason,) = check(record["test_file"]["body"]).reasons
    assert reason.startswith("line 18: `model.get_relation(1)` passes 1 where the endpoint name goes")
    assert "as in `model.get_relation('<endpoint>', 1)`" in reason
    assert "That call needs the `relation-ids` and `relation-list` hook tools" in reason


# -- the test file the rules see ----------------------------------------------

_SHAPE = '''\
import os

import ops
from ops.model import _ModelBackend


def fake_hook_tool(bin_dir, name, script):
    path = bin_dir / name
    path.write_text("#!/bin/sh\\n" + script + "\\n")
    path.chmod(0o755)


def test_access(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("JUJU_VERSION", "3.6.0")
    fake_hook_tool(bin_dir, "status-get", "exit 1")
{fakes}
    meta = ops.CharmMeta.from_yaml({meta!r})
    backend = _ModelBackend("myapp/0")
    model = ops.Model(meta, backend)
{access}
'''

_META = "name: myapp\nrequires:\n  db:\n    interface: db\npeers:\n  cluster:\n    interface: cluster\n"

# What each fake prints when it is not the one failing.
_OUTPUT = {
    "relation-ids": """echo '["db:2"]'""",
    "relation-list": """echo '["provider/0"]'""",
    "relation-get": """echo '{"k": "v"}'""",
    "is-leader": "echo false",
    "config-get": "echo '{}'",
}
_DENIED = "echo 'ERROR permission denied' >&2; exit 1"


def _fakes(tools: list[str]) -> dict[str, str]:
    """A fake printing something ops accepts for each of `tools`. With only
    `relation-ids` to fake, it prints no IDs, so that nothing else runs."""
    fakes = {tool: _OUTPUT[tool] for tool in tools}
    if "relation-ids" in fakes and "relation-list" not in tools:
        fakes["relation-ids"] = "echo '[]'"
    return fakes


def _shape(access: str, fakes: dict[str, str], meta: str = _META) -> str:
    lines = "\n".join(f"    fake_hook_tool(bin_dir, {tool!r}, {script!r})" for tool, script in fakes.items())
    body = textwrap.indent(textwrap.dedent(access).strip(), "    ")
    return _SHAPE.format(fakes=lines, meta=meta, access=body)


def _run(source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the file's `test_access` in this process, on the installed ops, in
    a directory of its own."""
    namespace: dict = {}
    exec(compile(source, "test_access.py", "exec"), namespace)
    work = tmp_path / str(len(list(tmp_path.iterdir())))
    work.mkdir()
    with monkeypatch.context() as patch:
        namespace["test_access"](work, patch)


# (access, the hook tools it needs, the first one it runs). Read from ops
# 3.8.3's `ops/model.py` (`RelationMapping.__getitem__`, `_get_unique`,
# `Relation.__init__`, `RelationDataContent._load`, and `_ModelBackend`).
ACCESSES = [
    ("model.relations['db']", ["relation-ids"], "relation-ids"),
    ("model.get_relation('db')", ["relation-ids"], "relation-ids"),
    ("model.get_relation('db', 2)", ["relation-ids", "relation-list"], "relation-ids"),
    ("model.get_relation('db', 7)", ["relation-ids", "relation-list"], "relation-ids"),
    ("model.get_relation(relation_name='db', relation_id=2)", ["relation-ids", "relation-list"], "relation-ids"),
    ("model.get_relation('cluster', 2)", ["relation-ids", "relation-list"], "relation-ids"),
    ("model.relations['db'][0]", ["relation-ids", "relation-list"], "relation-ids"),
    (
        "relation = model.get_relation('db', 2)\ndict(relation.data[relation.app])",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    (
        "relation = model.get_relation('db')\nrelation.data[relation.app] == {}",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    (
        "relation = model.relations['db'][0]\nlen(relation.data[model.unit])",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    (
        "relation = model.get_relation('db', 2)\ndata = relation.data[model.app]\ndata.get('k')",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    (
        "relation = model.get_relation('db', 2)\n'k' in relation.data[relation.app]",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    (
        "relation = model.get_relation('db', 2)\nrelation.data[relation.app]['k']",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    (
        "relation = model.get_relation('db', 2)\nbool(relation.data[relation.app])",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    (
        "relation = model.get_relation('db', 2)\nlist(relation.data[relation.app])",
        ["relation-ids", "relation-list", "relation-get"],
        "relation-get",
    ),
    ("backend.relation_ids('db')", ["relation-ids"], "relation-ids"),
    ("model._backend.relation_list(2)", ["relation-list"], "relation-list"),
    ("backend.relation_get(2, 'provider/0', False)", ["relation-get"], "relation-get"),
    ("backend.is_leader()", ["is-leader"], "is-leader"),
    ("backend.config_get()", ["config-get"], "config-get"),
]


def _ids(accesses):
    return [a[0].replace("\n", "; ") for a in accesses]


def _tool_reasons(result: StaticCheckResult) -> list[str]:
    return [r for r in result.reasons if _NOT_FAKED in r or "is-leader" in r or "not JSON" in r]


@pytest.mark.parametrize(("access", "tools", "first"), ACCESSES, ids=_ids(ACCESSES))
def test_an_access_with_its_tools_faked_runs_and_passes(access, tools, first, tmp_path, monkeypatch):
    source = _shape(access, _fakes(tools))
    assert check(source).passed, check(source).reasons
    _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("access", "tools", "left_out"),
    [(a, tools, tool) for a, tools, _ in ACCESSES for tool in tools],
    ids=[f"{i}-without-{tool}" for i, (_, tools, _) in zip(_ids(ACCESSES), ACCESSES) for tool in tools],
)
def test_an_access_without_one_of_its_tools_is_rejected_and_dies(access, tools, left_out, tmp_path, monkeypatch):
    source = _shape(access, {t: f for t, f in _fakes(tools).items() if t != left_out})
    reasons = _tool_reasons(check(source))
    assert len(reasons) == 1, reasons
    assert f"does not fake `{left_out}`" in reasons[0]
    assert "so it fails with `FileNotFoundError` before it reaches the bug" in reasons[0]
    with pytest.raises(FileNotFoundError) as raised:
        _run(source, tmp_path, monkeypatch)
    assert raised.value.filename == left_out


_IS_LEADER_RUNS = static_test_check._security_event_runs_is_leader()


@pytest.mark.skipif(not _IS_LEADER_RUNS, reason="the installed ops does not check leadership there")
@pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")
@pytest.mark.parametrize(
    ("access", "tools", "first"),
    [a for a in ACCESSES if a[2] != "is-leader"],
    ids=_ids([a for a in ACCESSES if a[2] != "is-leader"]),
)
def test_a_denied_first_tool_needs_is_leader(access, tools, first, tmp_path, monkeypatch):
    fakes = _fakes(tools)
    fakes[first] = _DENIED
    source = _shape(access, fakes)
    reasons = _tool_reasons(check(source))
    assert len(reasons) == 1, reasons
    assert f'the `{first}` fake fails with "ERROR permission denied"' in reasons[0]
    assert "fake `is-leader` too, printing `true` or `false`" in reasons[0]
    with pytest.raises(FileNotFoundError) as raised:
        _run(source, tmp_path, monkeypatch)
    assert raised.value.filename == "is-leader"

    # With `is-leader` faked the access gets to the bug, and the error it
    # raises is uncaught, which `_uncaught_hook_tool_error()` rejects (§18).
    fakes["is-leader"] = "echo false"
    source = _shape(access, fakes)
    (reason,) = check(source).reasons
    assert "so the installed ops raises `ops.ModelError` there, and nothing around it catches that" in reason
    with pytest.raises(static_test_check._ops().ModelError):
        _run(source, tmp_path, monkeypatch)


def test_relation_list_is_not_first_so_a_denied_one_says_nothing_of_is_leader():
    """`relation-ids` runs before `relation-list`, and a failing
    `relation-ids` fake (not this one) would stop the access first, so the
    leadership check is only certain for the first tool."""
    fakes = dict(_OUTPUT, **{"relation-list": _DENIED})
    del fakes["is-leader"]
    assert check(_shape("model.get_relation('db', 2)", fakes)).passed


# -- what stops rule 2 --------------------------------------------------------


def _without(tool: str, access: str = "model.get_relation('db', 2)") -> str:
    return _shape(access, {t: _OUTPUT[t] for t in ("relation-ids", "relation-list") if t != tool})


def test_a_test_that_fakes_no_hook_tools_is_left_alone():
    source = _without("relation-ids").replace('    fake_hook_tool(bin_dir, "status-get", "exit 1")\n', "")
    source = source.replace("    fake_hook_tool(bin_dir, 'relation-list', 'echo \\'[\"provider/0\"]\\'')\n", "")
    assert static_test_check.faked_hook_tools(source) == []
    assert check(source).passed, check(source).reasons


@pytest.mark.parametrize(
    "wrap",
    [
        "if os.environ.get('X'):\n    {access}",
        "for _ in os.environ:\n    {access}",
        "try:\n    {access}\nexcept Exception:\n    pass",
        "try:\n    {access}\nexcept:\n    pass",
        "try:\n    {access}\nexcept FileNotFoundError:\n    pass",
        "try:\n    {access}\nexcept (ops.ModelError, OSError):\n    pass",
        "try:\n    {access}\nexcept SomethingElse:\n    pass",
        "with something():\n    {access}",
        "with pytest.raises(Exception):\n    {access}",
        "try:\n    os.getcwd()\n    {access}\nexcept ops.ModelError:\n    pass",
        "try:\n    x = os.getcwd() and {access}\nexcept ops.ModelError:\n    pass",
        "x = os.environ.get('X') and {access}",
        "x = [{access} for _ in range(0)]",
        "x = (lambda: {access})",
        "return\n{access}",
        "pytest.skip('no')\n{access}",
    ],
)
def test_an_access_that_might_not_run_or_whose_error_might_be_caught_is_left_alone(wrap):
    source = _without("relation-ids", wrap.format(access="model.get_relation('db', 2)"))
    source = "import pytest\n" + source
    assert not _tool_reasons(check(source)), check(source).reasons


@pytest.mark.parametrize(
    "wrap",
    [
        "try:\n    {access}\nexcept ops.ModelError:\n    pass",
        "try:\n    {access}\nexcept (ops.ModelError, ops.RelationNotFoundError):\n    pass",
        "try:\n    x = {access}\nexcept ops.model.ModelError as e:\n    x = e",
        "with pytest.raises(ops.ModelError):\n    {access}",
        "assert {access} is not None",
        # Certain to run, and nothing swallows the error (§20).
        "for _ in range(1):\n    {access}",
        "with open(os.devnull):\n    {access}",
    ],
)
def test_an_access_whose_error_certainly_escapes_is_checked(wrap):
    source = "import pytest\n" + _without("relation-ids", wrap.format(access="model.get_relation('db', 2)"))
    assert any("does not fake `relation-ids`" in r for r in check(source).reasons), check(source).reasons


def test_a_decorated_test_is_left_alone():
    source = _without("relation-ids").replace(
        "def test_access(", "@pytest.mark.xfail\ndef test_access("
    )
    assert not _tool_reasons(check("import pytest\n" + source))


def test_pytestmark_turns_the_rule_off():
    source = _without("relation-ids") + "\npytestmark = []\n"
    assert not _tool_reasons(check(source))


def test_a_model_assigned_twice_is_not_tracked():
    source = _without("relation-ids", "model = ops.Model(meta, backend)\nmodel.get_relation('db', 2)")
    assert not _tool_reasons(check(source))


def test_an_access_before_the_model_is_built_is_not_checked():
    source = _without("relation-ids", "")
    source = source.replace(
        '    meta = ops.CharmMeta', "    model.get_relation('db', 2) if False else None\n    meta = ops.CharmMeta"
    )
    assert not _tool_reasons(check(source))


def test_a_model_over_something_other_than_a_model_backend_is_not_tracked():
    source = _without("relation-ids").replace("ops.Model(meta, backend)", "ops.Model(meta, object())")
    assert not _tool_reasons(check(source))


def test_a_databag_read_is_only_tracked_from_a_top_level_assignment():
    fakes = {t: _OUTPUT[t] for t in ("relation-ids", "relation-list")}
    access = """\
        try:
            relation = model.get_relation('db', 2)
        except ops.ModelError:
            relation = None
        dict(relation.data[relation.app])
    """
    assert not _tool_reasons(check(_shape(access, fakes)))


def test_lazy_databag_views_are_not_reads():
    fakes = {t: _OUTPUT[t] for t in ("relation-ids", "relation-list")}
    access = "relation = model.get_relation('db', 2)\nrelation.data[relation.app].keys()"
    assert check(_shape(access, fakes)).passed


def test_comparing_a_databag_with_something_not_a_mapping_is_not_a_read():
    fakes = {t: _OUTPUT[t] for t in ("relation-ids", "relation-list")}
    access = "relation = model.get_relation('db', 2)\nrelation.data[relation.app] == 'x'"
    assert check(_shape(access, fakes)).passed


def test_is_leader_is_not_required_after_something_that_could_fail_first():
    """With the fix, the leadership check is gone, so a test that fails on an
    earlier assertion on 3.8.3 could pass with the fix without `is-leader`."""
    fakes = dict(_OUTPUT, **{"relation-get": _DENIED})
    del fakes["is-leader"]
    access = "relation = model.get_relation('db', 2)\nassert relation.app.name == 'provider'\ndict(relation.data[relation.app])"
    assert check(_shape(access, fakes)).passed


def test_is_leader_needs_a_fake_ops_certainly_sees_failing():
    fakes = dict(_OUTPUT, **{"relation-get": "echo 'ERROR permission denied'; exit 1"})
    del fakes["is-leader"]
    access = "relation = model.get_relation('db', 2)\ndict(relation.data[relation.app])"
    # To stdout, not stderr: no `is-leader`, though the uncaught `ModelError`
    # it raises is rejected (§18).
    assert not [r for r in check(_shape(access, fakes)).reasons if "is-leader" in r]
    fakes["relation-get"] = "echo 'ERROR permission denied' >&2"
    assert check(_shape(access, fakes)).passed  # exits 0
    fakes["relation-get"] = "echo \"ERROR $X denied\" >&2; exit 1"
    assert check(_shape(access, fakes)).passed  # not a literal message
    fakes["relation-get"] = "echo 'ERROR permission denied' >&2; exit 1"
    source = _shape(access, fakes) + "\n\ndef helper(bin_dir):\n    fake_hook_tool(bin_dir, 'relation-get', 'exit 0')\n"
    assert check(source).passed  # faked twice: which one runs is not certain


@pytest.mark.parametrize(
    "helper",
    [
        'path.write_text("#!/bin/bash\\n" + script + "\\n")',
        'path.write_text("#!/bin/sh\\nset -e\\n" + script + "\\n")',
        "path.write_text(script)",
    ],
)
def test_is_leader_needs_a_helper_that_writes_the_script_as_it_is(helper):
    fakes = dict(_OUTPUT, **{"relation-get": _DENIED})
    del fakes["is-leader"]
    access = "relation = model.get_relation('db', 2)\ndict(relation.data[relation.app])"
    source = _shape(access, fakes).replace('path.write_text("#!/bin/sh\\n" + script + "\\n")', helper)
    assert check(source).passed, check(source).reasons


def test_an_f_string_helper_is_recognised():
    fakes = dict(_OUTPUT, **{"relation-get": _DENIED})
    del fakes["is-leader"]
    access = "relation = model.get_relation('db', 2)\ndict(relation.data[relation.app])"
    source = _shape(access, fakes).replace(
        'path.write_text("#!/bin/sh\\n" + script + "\\n")', 'path.write_text(f"#!/bin/sh\\n{script}\\n")'
    )
    assert any(_IS_LEADER in r for r in check(source).reasons)


@pytest.mark.parametrize(
    ("script", "flagged"),
    [
        ("echo provider", True),
        ("echo 'provider'; exit 0", True),
        ("", True),
        ("""echo '["provider/0"]'""", False),
        ("echo '[]'", False),
        ("echo provider; exit 1", False),
        ("cat /dev/null", False),
        ("echo $UNITS", False),
    ],
)
def test_a_fake_that_prints_something_ops_cannot_parse(script, flagged):
    fakes = {"relation-ids": _OUTPUT["relation-ids"], "relation-list": script}
    reasons = check(_shape("model.get_relation('db', 2)", fakes)).reasons
    assert any("which is not JSON" in r for r in reasons) is flagged, reasons


# -- rule 1: endpoints the meta does not declare ------------------------------


def _endpoint_test(meta: str, access: str, meta_expr: str | None = None) -> str:
    source = _shape(access, dict(_OUTPUT), meta=meta)
    if meta_expr is not None:
        source = source.replace(f"ops.CharmMeta.from_yaml({meta!r})", meta_expr)
    return source


def _endpoint_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if "where the endpoint name goes" in r or _UNDECLARED in r]


@pytest.mark.parametrize(
    "access",
    ["model.get_relation('db', 2)", "model.get_relation('db')", "model.relations['db']", "model.get_relation(relation_name='db')"],
)
def test_an_undeclared_endpoint_is_rejected(access):
    reasons = _endpoint_reasons(_endpoint_test("name: myapp\n", access))
    assert len(reasons) == 1
    assert "reads the `db` endpoint, which the `CharmMeta` on line 24 does not declare (it declares no relations)" in reasons[0]
    assert "declare `db` in that YAML under `requires`" in reasons[0]


@pytest.mark.parametrize("section", ["requires", "provides", "peers"])
def test_a_declared_endpoint_passes(section):
    meta = f"name: myapp\n{section}:\n  db:\n    interface: db\n"
    assert not _endpoint_reasons(_endpoint_test(meta, "model.get_relation('db', 2)"))


def test_the_reason_lists_what_is_declared():
    (reason,) = _endpoint_reasons(_endpoint_test(_META, "model.get_relation('database', 2)"))
    assert "(it declares only `cluster`, `db`)" in reason


def test_an_id_where_the_name_goes_names_the_one_endpoint():
    meta = "name: myapp\nrequires:\n  db:\n    interface: db\n"
    (reason,) = _endpoint_reasons(_endpoint_test(meta, "model.get_relation(2)"))
    assert "passes 2 where the endpoint name goes" in reason
    assert "as in `model.get_relation('db', 2)`" in reason
    assert "That call needs" not in reason  # both tools are faked


def test_a_meta_built_from_a_dict_literal_is_read():
    meta_expr = "ops.CharmMeta({'name': 'myapp', 'requires': {'db': {'interface': 'db'}}})"
    assert not _endpoint_reasons(_endpoint_test(_META, "model.get_relation('db', 2)", meta_expr))
    assert _endpoint_reasons(_endpoint_test(_META, "model.get_relation('other', 2)", meta_expr))


@pytest.mark.parametrize(
    "meta_expr",
    [
        "ops.CharmMeta.from_yaml(YAML)",  # not a literal
        "ops.CharmMeta.from_yaml('name: [unterminated')",  # does not parse
        "ops.CharmMeta.from_yaml('- a list')",  # parses, does not build
        "ops.CharmMeta.from_yaml('name: myapp\\n' + 'requires: {}')",  # not one literal
        "ops.CharmMeta({'name': 'myapp', **EXTRA})",
        "make_meta()",
    ],
)
def test_a_meta_it_cannot_read_lets_the_test_through(meta_expr):
    source = "YAML = 'name: myapp\\n'\nEXTRA = {}\n" + _endpoint_test(_META, "model.get_relation('other', 2)", meta_expr)
    assert not _endpoint_reasons(source)


def test_a_meta_built_twice_lets_the_test_through():
    source = _endpoint_test("name: myapp\n", "model.get_relation('db', 2)")
    source += "\n\ndef test_other():\n    ops.CharmMeta.from_yaml('name: other\\nrequires: {db: {interface: db}}')\n"
    assert not _endpoint_reasons(source)


def test_a_model_name_bound_twice_lets_the_test_through():
    source = _endpoint_test("name: myapp\n", "model = ops.Model(meta, backend)\nmodel.get_relation('db', 2)")
    assert not _endpoint_reasons(source)


def test_a_meta_name_bound_twice_lets_the_test_through():
    source = _endpoint_test("name: myapp\n", "meta = None\nmodel.get_relation('db', 2)")
    assert not _endpoint_reasons(source)


@pytest.mark.parametrize(
    "access",
    [
        "if 'db' in model.relations:\n    model.relations['db']",
        "try:\n    model.relations['db']\nexcept KeyError:\n    pass",
        "try:\n    model.relations['db']\nexcept LookupError:\n    pass",
        "with pytest.raises(KeyError):\n    model.relations['db']",
        "model.relations.get('db')",
        "'db' in model.relations",
        "x = model.relations if False else None",
        "model.get_relation(name)",
    ],
)
def test_a_read_that_cannot_raise_or_is_caught_lets_the_test_through(access):
    source = "import pytest\nname = 'db'\n" + _endpoint_test("name: myapp\n", access)
    assert not _endpoint_reasons(source), check(source).reasons


def test_a_read_whose_key_error_escapes_a_model_error_handler_is_rejected():
    access = "try:\n    model.relations['db']\nexcept ops.ModelError:\n    pass"
    assert _endpoint_reasons(_endpoint_test("name: myapp\n", access))


def test_an_import_star_turns_the_endpoint_rule_off():
    source = "from os.path import *\n" + _endpoint_test("name: myapp\n", "model.get_relation('db', 2)")
    assert not _endpoint_reasons(source)


def test_imported_names_resolve():
    source = _endpoint_test("name: myapp\n", "m = Model(CharmMeta.from_yaml('name: x\\n'), _ModelBackend('x/0'))\nm.relations['db']")
    source = source.replace("    meta = ops.CharmMeta.from_yaml('name: myapp\\n')\n", "").replace(
        "    model = ops.Model(meta, backend)\n", ""
    )
    source = "from ops import CharmMeta\nfrom ops.model import Model\n" + source
    assert _endpoint_reasons(source)


# -- rule 3: `_ModelBackend` keywords -----------------------------------------


def _backend_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if "argument; `_ModelBackend." in r]


def test_an_unknown_backend_keyword_is_rejected_with_what_it_takes():
    source = _shape("backend.relation_get(relation_id=2, unit_name='provider/0', is_app=False)", dict(_OUTPUT))
    assert _backend_reasons(source) == [
        "line 27: `backend.relation_get(...)` has no `unit_name` argument; `_ModelBackend.relation_get()` takes "
        "`relation_id`, `member_name`, `is_app`, and keyword-only `relation_name`"
    ]


def test_an_unknown_keyword_through_the_models_backend_is_rejected():
    source = _shape("model._backend.relation_ids(name='db')", dict(_OUTPUT))
    (reason,) = _backend_reasons(source)
    assert "`model._backend.relation_ids(...)` has no `name` argument" in reason
    assert "takes `relation_name`" in reason


@pytest.mark.parametrize(
    "access",
    [
        "backend.relation_get(relation_id=2, member_name='provider/0', is_app=False, relation_name='db')",
        "backend.relation_list(2, relation_name='db')",
        "backend.no_such_method(x=1)",
        "backend.unit_name.split(sep='/')",
    ],
)
def test_keywords_it_takes_or_cannot_check_pass(access):
    assert not _backend_reasons(_shape(access, dict(_OUTPUT)))


def test_a_backend_bound_twice_is_not_checked():
    source = _shape("backend = _ModelBackend('other/0')\nbackend.relation_get(unit_name='x')", dict(_OUTPUT))
    assert not _backend_reasons(source)


def test_the_backend_keyword_rule_is_off_under_ops_testing():
    source = _shape("backend.relation_get(unit_name='x')", dict(_OUTPUT))
    source = "from ops import testing\n" + source + "\n\ndef test_ctx():\n    testing.Context(ops.CharmBase, meta={'name': 'x'})\n"
    assert not _backend_reasons(source)


# -- the re-ask's hint --------------------------------------------------------


def test_the_is_leader_hint_goes_with_a_fake_hook_tool_reason():
    fakes = {"relation-get": _DENIED}
    access = "relation = model.get_relation('db', 2)\nassert relation\ndict(relation.data[relation.app])"
    result = check(_shape(access, fakes))
    hints = retry_hints(result)
    if not _IS_LEADER_RUNS:
        assert hints == []
        return
    assert len(hints) == 1 and hints[0].startswith("When a hook tool fails with an authorisation error")


def test_no_is_leader_hint_when_a_reason_already_says_so():
    fakes = dict(_OUTPUT, **{"relation-get": _DENIED})
    del fakes["is-leader"]
    del fakes["relation-ids"]
    result = check(_shape("relation = model.get_relation('db', 2)\ndict(relation.data[relation.app])", fakes))
    assert any("`is-leader`" in r for r in result.reasons)
    assert retry_hints(result) == []


def test_no_is_leader_hint_for_other_reasons():
    result = StaticCheckResult(path=None, passed=False, reasons=["line 3: the test file has no test function"])
    assert retry_hints(result) == []
