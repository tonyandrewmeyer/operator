"""`relation-list` fakes (`spike-step-5/static-retry/RESULT.md` §24).

Live on `#2709` (§23), the non-JSON `relation-list` reason's aside "if it prints
`[]`, ops runs `relation-list --app` next" led three re-asks to fake `--app`
separately (a file named `relation-list --app`, or a fourth argument to the
fake helper), and two of them to a `relation-list` fake printing `[]`. That
fake answers `--app` too, with `[]` again, and ops fails with `TypeError:
unhashable type: 'list'` on 3.8.3 and `main`. These tests check by running
when that happens and when it does not, cover the rule that rejects it, and
check the reworded reason.
"""

from __future__ import annotations

import pytest

from static_test_check import check

from test_hook_tool_rules import _META, _OUTPUT, _run, _shape

_EMPTY = "echo '[]'"
_TYPE_ERROR = "fails with `TypeError: unhashable type: 'list'`"
_DENIED = "echo 'ERROR permission denied' >&2; exit 1"

pytestmark = pytest.mark.filterwarnings("ignore:JujuLogHandler is not set up")

# Each runs `relation-list` once `relation-ids` prints an ID.
_ACCESSES = [
    "model.get_relation('db', 2)",
    "model.get_relation('db')",
    "model.relations['db']",
    "model.relations['db'][0]",
    "relation = model.get_relation('db', 2)",
    "assert model.get_relation('db', 2) is not None",
]


def _fakes(relation_list: str = _EMPTY, relation_get: str = "echo '{}'") -> dict[str, str]:
    return {
        "relation-ids": _OUTPUT["relation-ids"],
        "relation-list": relation_list,
        "relation-get": relation_get,
        "is-leader": "echo false",
    }


def _with_env(source: str, env: dict[str, str]) -> str:
    """`source` with `env` set before the backend is built."""
    lines = "".join(f"    monkeypatch.setenv({k!r}, {v!r})\n" for k, v in env.items())
    return source.replace("    meta = ops.CharmMeta", lines + "    meta = ops.CharmMeta", 1)


def _empty_list_reasons(source: str) -> list[str]:
    return [r for r in check(source).reasons if _TYPE_ERROR in r]


# -- by running ----------------------------------------------------------------


@pytest.mark.parametrize("access", _ACCESSES)
def test_an_empty_list_for_every_argument_is_a_type_error(access, tmp_path, monkeypatch):
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        _run(_shape(access, _fakes()), tmp_path, monkeypatch)


@pytest.mark.parametrize("access", _ACCESSES)
def test_a_unit_list_for_every_argument_runs(access, tmp_path, monkeypatch):
    """What the reason says to write: one fake that prints a unit list, so ops
    never runs `relation-list --app`."""
    _run(_shape(access, _fakes(_OUTPUT["relation-list"])), tmp_path, monkeypatch)


def test_an_empty_list_on_a_peer_endpoint_runs(tmp_path, monkeypatch):
    """A peer relation's remote app is this charm's own, so ops does not ask."""
    _run(_shape("model.get_relation('cluster', 2)", _fakes()), tmp_path, monkeypatch)


def test_an_empty_list_with_the_remote_app_in_the_environment_runs(tmp_path, monkeypatch):
    source = _with_env(_shape("model.get_relation('db', 2)", _fakes()), {"JUJU_RELATION_ID": "db:2", "JUJU_REMOTE_APP": "provider"})
    _run(source, tmp_path, monkeypatch)


def test_an_empty_list_with_no_ids_runs(tmp_path, monkeypatch):
    fakes = dict(_fakes(), **{"relation-ids": _EMPTY})
    _run(_shape("assert model.get_relation('db') is None", fakes), tmp_path, monkeypatch)


# -- the rule ------------------------------------------------------------------


@pytest.mark.parametrize("access", _ACCESSES)
def test_an_empty_list_for_every_argument_is_rejected(access):
    (reason,) = _empty_list_reasons(_shape(access, _fakes()))
    assert "needs `relation-list`, and its fake prints `[]` whatever its arguments are" in reason
    assert "ops runs `relation-list --app`" in reason


@pytest.mark.parametrize("access", _ACCESSES)
@pytest.mark.parametrize("relation_get", ["echo '{}'", _DENIED])
def test_a_rejected_empty_list_fails_whether_or_not_the_bug_is_there(access, relation_get, tmp_path, monkeypatch):
    source = _shape(access, _fakes(relation_get=relation_get))
    assert _empty_list_reasons(source)
    with pytest.raises(TypeError):
        _run(source, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "access",
    [
        "with pytest.raises(ops.ModelError):\n    model.get_relation('db', 2)",
        "try:\n    model.get_relation('db', 2)\nexcept ops.ModelError:\n    pass",
        "try:\n    model.get_relation('db', 2)\nexcept Exception as e:\n    pytest.fail(str(e))",
    ],
)
def test_the_type_error_escapes_a_model_error_handler_and_is_rejected(access):
    assert _empty_list_reasons("import pytest\n" + _shape(access, _fakes()))


@pytest.mark.parametrize(
    ("access", "fakes", "env"),
    [
        # Peer endpoint: no `--app` (by running, above).
        ("model.get_relation('cluster', 2)", _fakes(), {}),
        # The remote app comes from the environment.
        ("model.get_relation('db', 2)", _fakes(), {"JUJU_RELATION_ID": "db:2", "JUJU_REMOTE_APP": "provider"}),
        ("model.get_relation('db', 2)", _fakes(), {"JUJU_REMOTE_APP": "provider"}),
        # No IDs, so no `relation-list`.
        ("model.get_relation('db')", dict(_fakes(), **{"relation-ids": _EMPTY}), {}),
        # A unit list, or a script that answers by its arguments.
        ("model.get_relation('db', 2)", _fakes(_OUTPUT["relation-list"]), {}),
        ("model.get_relation('db', 2)", _fakes('case "$*" in *--app*) echo \'"provider"\';; *) echo \'[]\';; esac'), {}),
        ("model.get_relation('db', 2)", _fakes("echo $UNITS"), {}),
        # Caught, conditional, or after a skip.
        ("with pytest.raises(TypeError):\n    model.get_relation('db', 2)", _fakes(), {}),
        ("try:\n    model.get_relation('db', 2)\nexcept Exception:\n    pass", _fakes(), {}),
        ("if os.environ.get('X'):\n    model.get_relation('db', 2)", _fakes(), {}),
        ("pytest.skip('later')\nmodel.get_relation('db', 2)", _fakes(), {}),
    ],
)
def test_an_empty_list_it_cannot_be_sure_of_passes(access, fakes, env):
    source = _with_env("import pytest\n" + _shape(access, fakes), env)
    assert not _empty_list_reasons(source)


def test_a_second_relation_list_fake_passes():
    source = _shape("model.get_relation('db', 2)", _fakes())
    source = source.replace(
        "    meta = ops.CharmMeta", f"    fake_hook_tool(bin_dir, 'relation-list', {_OUTPUT['relation-list']!r})\n    meta = ops.CharmMeta", 1
    )
    assert not _empty_list_reasons(source)


@pytest.mark.parametrize(
    "meta",
    [
        "ops.CharmMeta.from_yaml(make_yaml())",  # not a literal
        "ops.CharmMeta.from_yaml('- a list')",  # does not build
    ],
)
def test_a_meta_it_cannot_read_passes(meta):
    source = _shape("model.get_relation('db', 2)", _fakes())
    source = source.replace(f"ops.CharmMeta.from_yaml({_META!r})", meta) + "\n\ndef make_yaml():\n    return ''\n"
    assert meta in source
    assert not _empty_list_reasons(source)


def test_a_model_over_another_meta_passes():
    source = _shape("model.get_relation('db', 2)", _fakes())
    source = source.replace("model = ops.Model(meta, backend)", "model = ops.Model(other_meta(), backend)")
    source += "\n\ndef other_meta():\n    return ops.CharmMeta.from_yaml('name: myapp\\npeers: {db: {interface: db}}')\n"
    assert not _empty_list_reasons(source)


def test_a_meta_through_a_name_assigned_once_is_followed():
    """§23's 4-6 builds its meta from a name assigned a string."""
    source = _shape("model.get_relation('db', 2)", _fakes())
    source = source.replace(f"ops.CharmMeta.from_yaml({_META!r})", "ops.CharmMeta.from_yaml(yaml)")
    source = source.replace("    meta = ops.CharmMeta", f"    yaml = {_META!r}\n    meta = ops.CharmMeta", 1)
    assert _empty_list_reasons(source)



# -- the reason ----------------------------------------------------------------


@pytest.mark.parametrize(
    "relation_list",
    ["echo provider", "", _EMPTY, None],
    ids=["non-JSON", "nothing", "empty list", "unfaked"],
)
def test_every_relation_list_reason_says_one_fake_printing_units(relation_list):
    fakes = _fakes() if relation_list is None else _fakes(relation_list)
    if relation_list is None:
        del fakes["relation-list"]
    reasons = [r for r in check(_shape("model.get_relation('db', 2)", fakes)).reasons if "`relation-list` prints" in r]
    assert len(reasons) == 1, reasons
    (reason,) = reasons
    assert "fake it once, with a script that prints a unit list whatever its arguments are, like `echo '[\"provider/0\"]'`" in reason
    assert "that one fake also answers `relation-list --app`" in reason
    assert "do not fake `--app` separately and do not print `[]`" in reason
    # The old aside, which sent re-asks to a separate `--app` fake.
    assert "has to print the remote application as a JSON string" not in reason


def test_what_the_reason_says_to_write_passes_the_check_and_runs(tmp_path, monkeypatch):
    source = _shape("model.get_relation('db', 2)", _fakes())
    assert _empty_list_reasons(source)
    repaired = _shape("model.get_relation('db', 2)", _fakes("""echo '["provider/0"]'"""))
    assert check(repaired).passed, check(repaired).reasons
    _run(repaired, tmp_path, monkeypatch)
