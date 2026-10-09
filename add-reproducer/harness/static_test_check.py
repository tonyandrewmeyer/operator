"""A cheap, deterministic check on the test file an extraction embeds.

Until this module existed, an extraction whose heredoc test was broken went
all the way to the runner, failed there, and only then was classified
`UNRUNNABLE_SYNTHESIS_INVALID` by rung 1a or 1c and kept silent. On `#2045`,
5 of 8 extractions at `0e6d7e28` were that shape, and so was the one live
dispatch at `7888bb13` (`ctx.charm_dir`, an `AttributeError`): the model
reaches for `testing.Context` attributes that do not exist, even with the
prompt saying plainly that they do not (`spike-step-5/assert-choice/
RESULT.md` §3, §5).

Rung 1c already keeps those silent, so catching them earlier does not by
itself buy a reproduction. What it buys is the chance to ask again:
`extraction.Extractor` re-asks once, with this module's reasons in the
prompt, when the first answer's test fails here.

Everything is read from the AST. The model-written file is parsed, never
imported or run: running it is what the throwaway runner is for. The only
code executed is `ops` itself, to learn which attributes `testing.Context`
and `CharmBase` have, so the allowed sets follow the installed `ops` rather
than a list that goes stale.

Each rule is there because it cannot fire on a test that would work: the
calibration against the 33 saved `#2045` extractions
(`spike-step-5/static-retry/RESULT.md`) rejects none of the five valid ones.
A false rejection costs a reproduction (the retry may come back worse, and is
kept whatever it says), so anything this module cannot be sure of, it lets
through. In particular, a name that is ever bound to something it cannot
type is treated as unknown, not as a match.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import inspect
import json
import re
import shlex
import typing

from models import StaticCheckResult, embedded_test_file

# Attribute calls that run a charm or emit an event. At module level they run
# while pytest is importing the file, so a failure is a broken file rather
# than a failing test.
_RUN_METHODS = frozenset({"run", "run_action", "emit", "begin", "begin_with_initial_hooks"})

# `os.path` functions that return a `str` whatever they are given.
_OS_PATH_STR_FUNCS = frozenset({
    "abspath",
    "basename",
    "dirname",
    "expanduser",
    "join",
    "normpath",
    "realpath",
    "relpath",
})

# `CharmBase.charm_dir` and `Framework.charm_dir` are both `pathlib.Path`.
_PATH_ATTRS = frozenset({"charm_dir"})

# Module-level names Python binds that are not in `builtins`.
_MODULE_DUNDERS = frozenset({
    "__file__",
    "__name__",
    "__doc__",
    "__spec__",
    "__loader__",
    "__package__",
    "__builtins__",
    "__annotations__",
    "__dict__",
})


@functools.cache
def context_attributes() -> frozenset[str] | None:
    """Every attribute an `ops.testing.Context` instance has, or `None` if
    `ops.testing` cannot be imported or a `Context` cannot be built.

    Most of `Context`'s public surface (`on`, `emitted_events`, `juju_log`,
    `charm_root`, ...) is set in `__init__`, so `dir()` on the class is not
    enough: a throwaway instance of a bare `CharmBase` is built and its
    `vars()` added. `None` switches the rule off rather than falling back to
    the class alone, which would reject attributes that exist.
    """
    try:
        import ops
        from ops import testing

        ctx = testing.Context(ops.CharmBase, meta={"name": "static-check"})
    except Exception:
        return None
    try:
        return frozenset(dir(testing.Context)) | frozenset(vars(ctx))
    finally:
        ctx.close()


# The tail of every `_context_attributes` reason, so `retry_hints()` can tell
# which results it applies to.
_NO_SUCH_CONTEXT_ATTRIBUTE = "which `testing.Context` does not have"


def retry_hints(result: StaticCheckResult) -> list[str]:
    """What the re-ask says beyond `result.reasons`, which only name what is
    wrong.

    Told that `ctx.charm_dir` does not exist, the model reaches for another
    attribute that does not exist either (`ctx.mgr`, `ctx.charm`,
    `ctx.framework`): live, the re-ask rescued 1 of 4 such tests
    (`spike-step-5/static-retry/RESULT.md` §7). So when a reason is a missing
    `Context` attribute, the re-ask also lists the attributes there are, and
    says how to read something off the charm instead.
    """
    hints = []
    allowed = context_attributes()
    if allowed is not None and any(reason.endswith(_NO_SUCH_CONTEXT_ATTRIBUTE) for reason in result.reasons):
        public = ", ".join(
            f"`{name}`" for name in sorted(allowed - _UNLISTED_CONTEXT_ATTRIBUTES) if not name.startswith("_")
        )
        hints.append(
            f"The `testing.Context` attributes a test can use are: {public}. None of "
            "them is the charm or anything on it. To compare something only the charm can "
            "see (`self.charm_dir`, `self.framework`, `self.model`, `os.getcwd()` during "
            "the hook), read it in an event handler, store it in a module-level dict, and "
            "assert on the dict after `ctx.run(...)` returns."
        )
    hints += _is_leader_hint(result)
    return hints


def _is_leader_hint(result: StaticCheckResult) -> list[str]:
    """When a reason is about a fake-hook-tool test and none already says to
    fake `is-leader`, the leadership check ops 3.8.3 makes when a hook tool
    fails with an authorisation error.

    `_unfaked_hook_tools()` only rejects a missing `is-leader` fake when
    nothing before the access could fail the test first, so it misses some
    tests that need one. By hand, applying the other reasons to the `#2709`
    tests at §16 still left 2 of 8 dying on `FileNotFoundError: 'is-leader'`
    (`spike-step-5/static-retry/RESULT.md` §17).
    """
    if not _security_event_runs_is_leader():
        return []
    if not any(marker in reason for reason in result.reasons for marker in _FAKE_HOOK_TOOL_REASONS):
        return []
    if any("`is-leader`" in reason for reason in result.reasons):
        return []
    return [
        "When a hook tool fails with an authorisation error on stderr (`permission denied`, "
        "`access denied`, `not the leader` or `cannot write relation settings`), the installed "
        "ops runs `is-leader` before it raises, to log a security event. So a test with a fake "
        "like that needs an `is-leader` fake too, printing `true` or `false`, or it fails with "
        "`FileNotFoundError` before it reaches the bug."
    ]


# Phrases only the fake-hook-tool rules' reasons contain, for `retry_hints()`.
_FAKE_HOOK_TOOL_REASONS = ("before any hook tool runs", "hook tool", "which is not JSON", "where the endpoint name goes")


# Left off `retry_hints()`'s list. `charm_root` reads like the charm's
# directory, but it is only the `Context(charm_root=...)` argument, `None` by
# default, so a test comparing the cwd with it fails whether or not the bug is
# real. Listed, it was taken in 1 of 7 re-asks; listed with a sentence saying
# what it is, in 0 of 7, 2 of 11 and 3 of 13 (`spike-step-5/static-retry/
# RESULT.md` §8, §9, §12, §13). The check still allows it: only the hint
# leaves it out.
_UNLISTED_CONTEXT_ATTRIBUTES = frozenset({"charm_root"})


@functools.cache
def testing_module():
    """`ops.testing`, or `None` if it cannot be imported, which switches off
    the rules that read it."""
    try:
        from ops import testing
    except Exception:
        return None
    return testing


@functools.cache
def _keyword_parameters(name: str) -> frozenset[str] | None:
    """The keywords `ops.testing.<name>(...)` accepts, or `None` when that is
    not knowable: not a class, no signature, or it takes `**kwargs`."""
    testing = testing_module()
    obj = getattr(testing, name, None) if testing is not None else None
    if not isinstance(obj, type):
        return None
    try:
        params = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return frozenset(
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


@functools.cache
def charm_read_only_properties() -> frozenset[str]:
    """`CharmBase` properties with no setter: assigning one on `self` in a
    charm raises `AttributeError` when the charm is constructed."""
    try:
        import ops
    except Exception:
        return frozenset()
    return frozenset(
        name
        for name in dir(ops.CharmBase)
        if isinstance(getattr(ops.CharmBase, name, None), property)
        and getattr(ops.CharmBase, name).fset is None
    )


# What a test file means by these names when it uses one and imports nothing
# that binds it. Every name here is one the extraction prompt's own example
# imports, or the standard library's.
_KNOWN_IMPORTS = {
    "ops": "import ops",
    "testing": "from ops import testing",
    "pytest": "import pytest",
    "os": "import os",
    "pathlib": "import pathlib",
    "Path": "from pathlib import Path",
}


def add_missing_imports(source: str, donor: str | None = None) -> tuple[str, list[str]]:
    """`source` with an import added for each name it uses and never binds,
    where the import is known, and the lines added.

    The re-ask's answer sometimes fixes everything it was told about and
    returns a test with no `import ops` or `from ops import testing` (3 of
    18 re-asks, `spike-step-5/static-retry/RESULT.md` §11). The re-ask is
    bounded at one, so the check catching it lost the test. A name that is
    used and bound nowhere is a `NameError` whatever it was meant to be, so
    adding the import cannot break a test that would have worked.

    The import comes from `donor` (the first answer's test file) when it
    has one binding the name, otherwise from `_KNOWN_IMPORTS`. Names neither
    knows are left for the check to report.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    missing = _undefined_name_lines(tree)
    if not missing:
        return source, []
    donated = _import_lines_by_name(donor)
    added = []
    for name in missing:
        line = donated.get(name) or _KNOWN_IMPORTS.get(name)
        if line is not None and line not in added:
            added.append(line)
    if not added:
        return source, []
    # After a module docstring and any `from __future__` import, which
    # have to come first.
    at = 0
    for node in tree.body:
        is_docstring = (
            node is tree.body[0]
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if is_docstring or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            at = node.end_lineno or at
            continue
        break
    lines = source.splitlines(keepends=True)
    block = "".join(f"{line}\n" for line in added)
    return "".join(lines[:at]) + block + "".join(lines[at:]), added


def _import_lines_by_name(source: str | None) -> dict[str, str]:
    """{name: an import statement binding only that name}, for each
    module-level import in `source`."""
    if not source:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                found.setdefault(name, ast.unparse(ast.Import(names=[alias])))
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                if alias.name == "*":
                    continue
                statement = ast.ImportFrom(module=node.module, names=[alias], level=node.level)
                found.setdefault(alias.asname or alias.name, ast.unparse(statement))
    return found


def check_commands(commands: list[str]) -> StaticCheckResult | None:
    """Check the test file a heredoc in `commands[]` writes, or `None` when
    there is none (nothing to check: the synthesiser may still write one)."""
    test_file = embedded_test_file(commands)
    if test_file is None:
        return None
    return check(test_file.body, path=test_file.path)


def runs_under_pytest(commands: list[str]) -> bool:
    """Whether a command after the test file's heredoc runs pytest on it."""
    test_file = embedded_test_file(commands)
    if test_file is None:
        return False
    name = test_file.path.rsplit("/", 1)[-1]
    after = False
    for command in commands:
        if not after:
            after = f"{test_file.path}" in command and "<<" in command
            continue
        if "pytest" in command and (name in command or not command.split("pytest", 1)[1].strip()):
            return True
    return False


def check(source: str, *, path: str | None = None) -> StaticCheckResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return StaticCheckResult(
            path=path,
            passed=False,
            reasons=[f"the test file does not parse: {exc.msg} (line {exc.lineno})"],
        )
    reasons: list[str] = []
    context_names = _module_context_names(tree)
    reasons += _no_test_function(tree)
    reasons += _module_level_run(tree, context_names)
    reasons += _undefined_names(tree)
    reasons += _context_attributes(tree)
    reasons += _attribute_chains(tree)
    reasons += _testing_names(tree)
    reasons += _unset_charm_root(tree)
    reasons += _charm_assigns_read_only_property(tree)
    reasons += _assert_in_charm(tree)
    reasons += _str_compared_with_path(tree)
    reasons += _capture_read_before_handlers_run(tree)
    reasons += _backend_under_testing(tree)
    reasons += _undeclared_endpoints(tree)
    reasons += _unfaked_hook_tools(tree)
    reasons += _uncaught_hook_tool_error(tree)
    reasons += _databag_not_read(tree)
    reasons += _string_databag_key(tree)
    reasons += _backend_keywords(tree)
    # Several rules can name the same line for the same reason (a nested
    # scope sees its parent's bindings); the model only needs telling once.
    # Then in line order, file-wide reasons first, so the feedback reads
    # top to bottom like the file.
    reasons = sorted(dict.fromkeys(reasons), key=_line_of)
    return StaticCheckResult(path=path, passed=not reasons, reasons=reasons)


# -- helpers ---------------------------------------------------------------


def _is_context_call(node: ast.AST) -> bool:
    """`Context(...)`, `testing.Context(...)`, `ops.testing.Context(...)`,
    or `scenario.Context(...)`, which is the same class."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Context"
    if isinstance(func, ast.Attribute) and func.attr == "Context":
        owner = ast.unparse(func.value)
        return owner in {"testing", "ops.testing", "scenario"}
    return False


def _is_charm_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "CharmBase":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "CharmBase":
            return True
    return False


def _own_nodes(scope: ast.AST):
    """Every node in `scope` that is not inside a nested function, lambda or
    class -- the nodes whose bindings belong to `scope` itself."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _scope_bindings(scope: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound to a `Context` in `scope`, names bound to anything else).

    A name in both is ambiguous and is not tracked."""
    as_context: set[str] = set()
    other: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = scope.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if arg is not None:
                other.add(arg.arg)
    for node in _own_nodes(scope):
        if isinstance(node, ast.Assign):
            target_set = as_context if _is_context_call(node.value) else other
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        target_set.add(name.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            (as_context if node.value is not None and _is_context_call(node.value) else other).add(
                node.target.id
            )
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is None:
                    continue
                target_set = as_context if _is_context_call(item.context_expr) else other
                for name in ast.walk(item.optional_vars):
                    if isinstance(name, ast.Name):
                        target_set.add(name.id)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    other.add(name.id)
        elif isinstance(node, ast.NamedExpr):
            (as_context if _is_context_call(node.value) else other).add(node.target.id)
        elif isinstance(node, (ast.AugAssign,)) and isinstance(node.target, ast.Name):
            other.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                other.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            other.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            other.add(node.name)
    return as_context, other


def _module_context_names(tree: ast.Module) -> set[str]:
    as_context, other = _scope_bindings(tree)
    return as_context - other


def _context_scopes(tree: ast.Module):
    """Yield (scope, names bound to a `Context` visible in it), for the
    module and every function in it. A function sees the module's context
    names unless it binds the same name itself; `global` is ignored, which
    can only make the rule see less."""
    module_names = _module_context_names(tree)
    yield tree, module_names
    stack: list[tuple[ast.AST, set[str]]] = [(tree, module_names)]
    while stack:
        scope, visible = stack.pop()
        for node in _own_nodes(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                own_context, own_other = _scope_bindings(node)
                names = (visible - own_other - own_context) | (own_context - own_other)
                yield node, names
                stack.append((node, names))
            elif isinstance(node, ast.ClassDef):
                # A class body does not leak its names into its methods, so
                # its methods see the enclosing scope's names.
                stack.append((node, visible))


# -- rules -----------------------------------------------------------------


def _no_test_function(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            return []
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test")
                for child in node.body
            ):
                return []
    return [
        "the test file defines no module-level `def test_...():` function, so pytest "
        "collects nothing from it; put the run and the assertion inside one"
    ]


def _module_level_run(tree: ast.Module, context_names: set[str]) -> list[str]:
    reasons = []
    for node in tree.body:
        if isinstance(node, ast.Assert):
            reasons.append(
                f"line {node.lineno}: `assert` at module level runs while pytest imports "
                "the file; move it into the test function"
            )
        elif isinstance(node, (ast.With, ast.AsyncWith)) and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id in context_names
            for item in node.items
        ):
            reasons.append(
                f"line {node.lineno}: `with {ast.unparse(node.items[0].context_expr)}` at module "
                "level runs the charm while pytest imports the file; move it into the test function"
            )
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in _RUN_METHODS
        ):
            reasons.append(
                f"line {node.lineno}: `{ast.unparse(node.value.func)}(...)` at module level runs "
                "while pytest imports the file; move it into the test function"
            )
    return reasons


def _undefined_names(tree: ast.Module) -> list[str]:
    """Names read somewhere and bound nowhere, in any scope. Deliberately
    loose about scope (a binding anywhere counts), so it only fires on a
    name that cannot resolve at all: a missing import is a `NameError` at
    import time, never a reproduction."""
    return [
        f"line {line}: `{name}` is used but never imported or defined"
        for name, line in _undefined_name_lines(tree).items()
    ]


def _undefined_name_lines(tree: ast.Module) -> dict[str, int]:
    """{name: first line it is read on}, for `_undefined_names()`, in line
    order; empty when the file has a `*` import."""
    bound: set[str] = set(dir(builtins)) | _MODULE_DUNDERS
    loads: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loads.setdefault(node.id, node.lineno)
            else:
                bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return {}
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    return {
        name: line
        for name, line in sorted(loads.items(), key=lambda item: item[1])
        if name not in bound
    }


def _context_attributes(tree: ast.Module) -> list[str]:
    allowed = context_attributes()
    if allowed is None:
        return []
    reasons = []
    for scope, names in _context_scopes(tree):
        if not names:
            continue
        for node in _own_nodes(scope):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in names
                and node.attr not in allowed
            ):
                reasons.append(
                    f"line {node.lineno}: the test accesses `{node.value.id}.{node.attr}`, "
                    f"{_NO_SUCH_CONTEXT_ATTRIBUTE}"
                )
    return list(dict.fromkeys(reasons))


# -- attribute chains ------------------------------------------------------
#
# `ctx.charm_spec.charm_dir` passes `_context_attributes()`, which only looks
# one attribute deep: `charm_spec` is a real `Context` attribute, and the
# `AttributeError` is one level down (2 of 14 re-asks,
# `spike-step-5/static-retry/RESULT.md` §14). These helpers follow such a
# chain through the types the installed `ops` declares, and give up, which
# lets the test through, at the first step whose type is not certain.


def _resolve_annotation(annotation: object, namespace: dict, _depth: int = 0) -> type | None:
    """The one class an annotation means, or `None` if it is not exactly one.

    `Optional[X]` and `X | None` are `X` (a `None` fails on any attribute
    anyway). `Any`, other unions, type variables, `type[...]`, `Literal`, a
    string that does not evaluate, and an abstract class or protocol (the
    value may be any concrete class that implements it) are all `None`. A
    generic class is its origin: its attributes do not depend on the type
    arguments.
    """
    import types
    import typing

    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        try:
            annotation = eval(annotation, dict(namespace))  # noqa: S307 - ops's own annotations.
        except Exception:
            return None
        # A quoted annotation under `from __future__ import annotations` is
        # a string of a string.
        if isinstance(annotation, str):
            return _resolve_annotation(annotation, namespace, _depth + 1) if _depth < 2 else None
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return _resolve_annotation(members[0], namespace, _depth) if len(members) == 1 else None
    if origin is typing.Annotated:
        return _resolve_annotation(typing.get_args(annotation)[0], namespace, _depth)
    if origin is not None:
        annotation = origin
    if not isinstance(annotation, type) or annotation is type or annotation is type(None):
        return None
    if inspect.isabstract(annotation) or getattr(annotation, "_is_protocol", False):
        return None
    if annotation.__module__ in {"typing", "collections.abc", "abc"}:
        return None
    return annotation


def _module_namespace(obj: object) -> dict:
    import sys

    module = sys.modules.get(getattr(obj, "__module__", ""), None)
    return vars(module) if module is not None else {}


@functools.cache
def _class_tree(cls: type) -> ast.ClassDef | None:
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(inspect.cleandoc("\n" + source) if source[:1].isspace() else source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return node
    return None


def _self_stores(cls_tree: ast.ClassDef) -> set[str] | None:
    """Attribute names the class's methods set on their first argument, or
    `None` if one sets a name it does not spell out (`setattr(self, name,
    ...)`, `self.__dict__`, `vars(self)`)."""
    stores: set[str] = set()
    for method in ast.walk(cls_tree):
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or not method.args.args:
            continue
        me = method.args.args[0].arg
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                return None
            if isinstance(node, ast.Call):
                func = ast.unparse(node.func)
                if func == "vars":
                    return None
                if func in {"setattr", "object.__setattr__"} or func.endswith(".__setattr__"):
                    if len(node.args) < 2:
                        return None
                    name = node.args[1]
                    if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                        return None
                    stores.add(name.value)
            if (
                isinstance(node, ast.Attribute)
                and not isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == me
            ):
                stores.add(node.attr)
    return stores


def _own_instance_attributes(cls: type) -> frozenset[str] | None:
    """`cls`'s contribution to what its instances can have."""
    import typing

    if cls is object or cls is typing.Generic:
        return frozenset()
    names = set(vars(cls)) | set(vars(cls).get("__annotations__", {}))
    slots = vars(cls).get("__slots__", ())
    names.update([slots] if isinstance(slots, str) else slots)
    if cls.__module__ == "builtins":
        return frozenset(names)
    tree = _class_tree(cls)
    if tree is None:
        return None
    stores = _self_stores(tree)
    if stores is None:
        return None
    return frozenset(names | stores)


def _subclasses(cls: type) -> list[type]:
    seen: list[type] = []
    stack = [cls]
    while stack:
        klass = stack.pop()
        if klass in seen:
            continue
        seen.append(klass)
        try:
            stack.extend(type.__subclasses__(klass))
        except TypeError:
            return []
    return seen


@functools.cache
def instance_attributes(cls: type) -> frozenset[str] | None:
    """Every attribute an instance of `cls` (or of any subclass loaded now)
    can have, or `None` when that is not knowable.

    `dir()` misses dataclass fields with no default and anything set in a
    method, so those come from the class's annotations and its source. Not
    knowable: a `__getattr__` or `__getattribute__` anywhere, a method that
    sets an attribute by a computed name, or a class whose source cannot be
    read that is not a builtin.
    """
    classes = _subclasses(cls)
    if not classes:
        return None
    names: set[str] = set()
    for klass in classes:
        for base in klass.__mro__:
            if base is not object and ("__getattr__" in vars(base) or "__getattribute__" in vars(base)):
                return None
            own = _own_instance_attributes(base)
            if own is None:
                return None
            names |= own
        names |= set(dir(klass))
    return frozenset(names)


def _getter_returns_self_attribute(func: object) -> str | None:
    """`x` when a property's getter is, after its docstring, `return self.x`."""
    func = inspect.unwrap(func)  # type: ignore[arg-type]
    try:
        source = inspect.getsource(func)  # type: ignore[arg-type]
        tree = ast.parse(inspect.cleandoc("\n" + source) if source[:1].isspace() else source)
    except (OSError, TypeError, SyntaxError):
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    function = tree.body[0]
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Attribute)
        and isinstance(body[0].value.value, ast.Name)
        and function.args.args
        and body[0].value.value.id == function.args.args[0].arg
    ):
        return body[0].value.attr
    return None


def _assigned_type(cls: type, name: str) -> type | None:
    """The class of `self.<name>` from what `cls`'s methods assign to it,
    when every assignment agrees: `self.x: T = ...`, `self.x = T(...)`, or
    `self.x = local`, where every binding of `local` in that method is
    annotated with or constructs the same `T` (a parameter counts by its
    annotation)."""
    import typing

    found: set[type | None] = set()
    for base in cls.__mro__:
        # `typing.Generic` assigns nothing to `self`, and from Python 3.12 it
        # is implemented in C, so it has no source to read: without this,
        # every generic class (`testing.Context` among them) was unknowable
        # on 3.12 and later, and the chain rule only worked on 3.11.
        if base is object or base is typing.Generic:
            continue
        tree = _class_tree(base)
        if tree is None:
            if base.__module__ == "builtins":
                continue
            return None
        namespace = _module_namespace(base)
        for method in ast.walk(tree):
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or not method.args.args:
                continue
            me = method.args.args[0].arg
            for node in ast.walk(method):
                if not (
                    isinstance(node, ast.Attribute)
                    and node.attr == name
                    and not isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == me
                ):
                    continue
                statement = _simple_assignment_to(method, node)
                if statement is None:
                    # Unpacking, a loop target, `+=`, `del`: not typed.
                    return None
                if isinstance(statement, ast.AnnAssign):
                    found.add(_resolve_annotation(ast.unparse(statement.annotation), namespace))
                else:
                    found.add(_value_type(statement.value, method, namespace))
    if len(found) != 1:
        return None
    return next(iter(found))


def _simple_assignment_to(scope: ast.AST, target: ast.AST) -> ast.Assign | ast.AnnAssign | None:
    """The statement in `scope` whose one, whole target is `target`."""
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and node.targets[0] is target:
            return node
        if isinstance(node, ast.AnnAssign) and node.target is target and node.value is not None:
            return node
    return None


def _value_type(value: ast.AST, method: ast.FunctionDef | ast.AsyncFunctionDef, namespace: dict) -> type | None:
    if isinstance(value, ast.Call) and isinstance(value.func, (ast.Name, ast.Attribute)):
        try:
            called = eval(ast.unparse(value.func), dict(namespace))  # noqa: S307 - ops's own source.
        except Exception:
            return None
        return _resolve_annotation(called, namespace) if isinstance(called, type) else None
    if not isinstance(value, ast.Name):
        return None
    local = value.id
    kinds: set[type | None] = set()
    args = method.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        if arg.arg == local:
            if arg.annotation is None:
                return None
            kinds.add(_resolve_annotation(ast.unparse(arg.annotation), namespace))
    for vararg in (args.vararg, args.kwarg):
        if vararg is not None and vararg.arg == local:
            return None
    for node in ast.walk(method):
        if not (isinstance(node, ast.Name) and node.id == local and not isinstance(node.ctx, ast.Load)):
            continue
        statement = _simple_assignment_to(method, node)
        if statement is None:
            return None
        if isinstance(statement, ast.AnnAssign):
            kinds.add(_resolve_annotation(ast.unparse(statement.annotation), namespace))
        elif isinstance(statement.value, ast.Call):
            kinds.add(_value_type(statement.value, method, namespace))
        else:
            return None
    if len(kinds) != 1:
        return None
    return next(iter(kinds))


@functools.cache
def attribute_type(cls: type, name: str, _depth: int = 0) -> type | None:
    """The one class `instance_of_cls.<name>` is, from the installed code's
    own declarations, or `None` when it is not certain.

    In order: a property's return annotation (or, with none, a getter that
    is only `return self.x`, followed to `x`); a class-level annotation,
    which includes dataclass fields; what the class's methods assign to
    `self.<name>`. A method, or anything else, is `None`.
    """
    if _depth > 3:
        return None
    for base in cls.__mro__:
        own = vars(base)
        annotations = own.get("__annotations__", {})
        value = own.get(name, inspect.Parameter.empty)
        if isinstance(value, (property, functools.cached_property)):
            getter = value.fget if isinstance(value, property) else value.func
            if getter is None:
                return None
            returns = getattr(inspect.unwrap(getter), "__annotations__", {}).get("return")
            if returns is not None:
                return _resolve_annotation(returns, _module_namespace(inspect.unwrap(getter)))
            followed = _getter_returns_self_attribute(getter)
            if followed is None or followed == name:
                return None
            return attribute_type(cls, followed, _depth + 1)
        if name in annotations:
            return _resolve_annotation(annotations[name], _module_namespace(base))
        if value is not inspect.Parameter.empty:
            return None
    return _assigned_type(cls, name)


@functools.cache
def _chain_root_types() -> dict[str, type] | None:
    """{"context": Context, "manager": what `Context.__call__` returns}, or
    `None` if `ops.testing` cannot be read.

    Not `State`: its `__init__` sets the status fields by a computed name,
    so `instance_attributes()` cannot know what it has and would let every
    chain through anyway.
    """
    testing = testing_module()
    if testing is None:
        return None
    try:
        context = testing.Context
        roots = {"context": context}
        returns = inspect.unwrap(context.__call__).__annotations__.get("return")
        manager = _resolve_annotation(returns, _module_namespace(context)) if returns is not None else None
    except Exception:
        return None
    if manager is not None:
        roots["manager"] = manager
    return roots


def _type_name(cls: type) -> str:
    return cls.__qualname__


def _binding_counts(scope: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = scope.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if arg is not None:
                add(arg.arg)
    for node in _own_nodes(scope):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            # Bound somewhere else too, so never "exactly once".
            for name in node.names:
                counts[name] = counts.get(name, 0) + 2
    return counts


def _chain_roots(tree: ast.Module, roots: dict[str, type]):
    """Yield (scope, {name: (root type, checked from depth)}).

    A `Context`-bound name is checked from the second attribute (the first
    is `_context_attributes()`'s). A `with ctx(...) as mgr:` target is
    checked from the first, only in the scope that binds it, and only when
    nothing else in that scope binds it.
    """
    for scope, context_names in _context_scopes(tree):
        found: dict[str, tuple[type, int]] = {name: (roots["context"], 1) for name in context_names}
        counts = _binding_counts(scope)
        for node in _own_nodes(scope):
            if isinstance(node, (ast.With, ast.AsyncWith)) and "manager" in roots:
                for item in node.items:
                    if (
                        isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id in context_names
                        and isinstance(item.optional_vars, ast.Name)
                        and counts.get(item.optional_vars.id) == 1
                    ):
                        found[item.optional_vars.id] = (roots["manager"], 0)
        if found:
            yield scope, found


def _attribute_chains(tree: ast.Module) -> list[str]:
    """`ctx.a.b`, where `b` is not an attribute of what `ctx.a` is.

    Each step's type comes from the installed `ops` (`attribute_type()`),
    and the chain is only followed while it is a plain attribute chain whose
    every type is certain: a call, a subscript, `Any`, a union, a type
    variable, or a class whose attributes are not knowable stops it, and the
    test is let through. So is any attribute name the file itself assigns
    anywhere, and any class the file subclasses.
    """
    if context_attributes() is None:
        return []
    roots = _chain_root_types()
    if roots is None:
        return []
    assigned = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load)
    }
    subclassed = {
        ast.unparse(base).rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
    }
    reasons = []
    for scope, found in _chain_roots(tree, roots):
        for node in _own_nodes(scope):
            if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)):
                continue
            chain: list[str] = []
            root: ast.AST = node
            while isinstance(root, ast.Attribute):
                chain.insert(0, root.attr)
                root = root.value
            if not isinstance(root, ast.Name) or root.id not in found:
                continue
            current, start = found[root.id]
            for depth, attr in enumerate(chain):
                if depth >= start:
                    if attr in assigned or any(
                        klass.__name__ in subclassed for klass in _subclasses(current)
                    ):
                        break
                    allowed = instance_attributes(current)
                    if allowed is None:
                        break
                    if attr not in allowed:
                        spelled = ".".join([root.id, *chain[: depth + 1]])
                        if depth == 0:
                            reasons.append(
                                f"line {node.lineno}: the test reads `{spelled}`; `{root.id}` is a "
                                f"`{_type_name(current)}`, which has no `{attr}`"
                            )
                        else:
                            reasons.append(
                                f"line {node.lineno}: the test reads `{spelled}`; `{chain[depth - 1]}` "
                                f"is a `{_type_name(current)}`, which has no `{attr}`"
                            )
                        break
                if depth == len(chain) - 1:
                    break
                following = attribute_type(current, attr)
                if following is None:
                    break
                current = following
    return list(dict.fromkeys(reasons))


def _unset_charm_root(tree: ast.Module) -> list[str]:
    """`ctx.charm_root` compared for equality, on a `Context` built without
    `charm_root=`.

    It is the `charm_root=` argument and nothing else, so it is `None`, and
    a test comparing the charm's cwd with it fails whether or not the bug is
    real (`spike-step-5/static-retry/RESULT.md` §8, §12). Only for a name
    bound exactly once in the file, by `Context(...)` with no `charm_root=`
    and no `**` argument, and never assigned an attribute.
    """
    stores: dict[str, int] = {}
    unset: dict[str, int] = {}
    attribute_stores: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stores[node.id] = stores.get(node.id, 0) + 1
        elif isinstance(node, ast.arg):
            stores[node.arg] = stores.get(node.arg, 0) + 1
        elif isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name):
                attribute_stores.add(node.value.id)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_context_call(node.value)
            and all(k.arg not in {"charm_root", None} for k in node.value.keywords)
        ):
            unset[node.targets[0].id] = node.lineno
    names = {n: line for n, line in unset.items() if stores.get(n) == 1 and n not in attribute_stores}
    if not names:
        return []

    def unset_read(operand: ast.AST) -> ast.Attribute | None:
        # `ctx.charm_root`, or it wrapped in `str(...)` / `Path(...)`.
        if isinstance(operand, ast.Call) and len(operand.args) == 1 and not operand.keywords:
            if ast.unparse(operand.func) in {"str", "Path", "pathlib.Path", "os.fspath"}:
                operand = operand.args[0]
        if (
            isinstance(operand, ast.Attribute)
            and operand.attr == "charm_root"
            and isinstance(operand.value, ast.Name)
            and operand.value.id in names
        ):
            return operand
        return None

    reasons = []
    for compare in ast.walk(tree):
        # Only an equality: `ctx.charm_root is None` is a fair thing to assert.
        if not (isinstance(compare, ast.Compare) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in compare.ops)):
            continue
        for operand in [compare.left, *compare.comparators]:
            node = unset_read(operand)
            if node is None:
                continue
            reasons.append(
                f"line {node.lineno}: `{node.value.id}.charm_root` is `None`: it is only the "
                f"`charm_root=` given to `Context(...)`, and line {names[node.value.id]}'s has "
                "none, so comparing with it says nothing about the bug. It is not the directory "
                "the charm runs in; read that in a handler"
            )
    return list(dict.fromkeys(reasons))


def _testing_aliases(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    """(names bound to the `ops.testing` module, {local name: `ops.testing`
    name} for names imported from it).

    Only names that nothing else in the file binds are returned: anything
    rebound, even once, is left alone rather than guessed at. `ops.testing`
    written out in full is handled by the caller, and only when `ops` itself
    is never rebound.
    """
    modules: set[str] = set()
    imported: dict[str, str] = {}
    other: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "ops" and not node.level:
            for alias in node.names:
                (modules if alias.name == "testing" else other).add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "ops.testing" and not node.level:
            for alias in node.names:
                if alias.name != "*":
                    imported[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ops.testing" and alias.asname:
                    modules.add(alias.asname)
                elif alias.name not in {"ops", "ops.testing"}:
                    other.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                other.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            other.add(node.id)
        elif isinstance(node, ast.arg):
            other.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            other.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            other.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            other.update(node.names)
    clashes = other | (modules & set(imported))
    return modules - clashes, {k: v for k, v in imported.items() if k not in clashes}


def _testing_names(tree: ast.Module) -> list[str]:
    """A name `ops.testing` does not export, and a keyword argument a
    `testing` class does not take.

    Both are a guess at the API rather than a test of the bug: on `#2709`, 3
    of 8 extractions wrote `testing.StateRelation`, `Context(relations=...)`
    or `Relation(remote_apps=...)`, and failed with `AttributeError` or
    `TypeError` before reaching an assertion
    (`spike-step-5/static-retry/RESULT.md` §10). The reason lists what does
    exist, since naming only the wrong one sends the model to another guess
    (§7).
    """
    testing = testing_module()
    if testing is None:
        return []
    modules, imported = _testing_aliases(tree)
    ops_rebound = "ops" in _testing_aliases_rebinding(tree)
    reasons = []

    def owner_is_testing(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return value.id in modules
        return not ops_rebound and ast.unparse(value) == "ops.testing"

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "ops.testing" and not node.level:
            for alias in node.names:
                if alias.name != "*" and not hasattr(testing, alias.name):
                    reasons.append(
                        f"line {node.lineno}: `ops.testing` has no `{alias.name}`, so the import "
                        "fails"
                    )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and owner_is_testing(node.value)
            and not hasattr(testing, node.attr)
        ):
            reasons.append(
                f"line {node.lineno}: `{ast.unparse(node)}` does not exist; `ops.testing` has no "
                f"`{node.attr}`"
            )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in imported:
                name = imported[func.id]
            elif isinstance(func, ast.Attribute) and owner_is_testing(func.value):
                name = func.attr
            else:
                continue
            accepted = _keyword_parameters(name)
            if accepted is None:
                continue
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in accepted:
                    listed = ", ".join(f"`{p}`" for p in sorted(accepted) if not p.startswith("_"))
                    reasons.append(
                        f"line {node.lineno}: `{ast.unparse(func)}(...)` has no `{keyword.arg}` "
                        f"argument; the arguments it takes are {listed}"
                    )
    return list(dict.fromkeys(reasons))


def _testing_aliases_rebinding(tree: ast.Module) -> set[str]:
    """Names bound by anything other than `import ops` / `import ops.testing`."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"ops", "ops.testing"} or alias.asname:
                    bound.add((alias.asname or alias.name).split(".")[0])
    return bound


def _charm_assigns_read_only_property(tree: ast.Module) -> list[str]:
    read_only = charm_read_only_properties()
    reasons = []
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and _is_charm_class(cls)):
            continue
        for node in ast.walk(cls):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
                if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                else []
            )
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in read_only
                ):
                    reasons.append(
                        f"line {target.lineno}: the charm assigns `self.{target.attr}`, a read-only "
                        "`CharmBase` property, which raises `AttributeError` before any handler runs"
                    )
    return reasons


def _assert_in_charm(tree: ast.Module) -> list[str]:
    reasons = []
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and _is_charm_class(cls):
            for node in ast.walk(cls):
                if isinstance(node, ast.Assert):
                    reasons.append(
                        f"line {node.lineno}: `assert` inside the charm surfaces as an uncaught "
                        "charm error, not a test failure; capture the value in the handler and "
                        "assert in the test function"
                    )
    return reasons


# A "slot" is where a value can be stored and later read back: a bare name,
# an attribute name on any object (`self.cwd` and `mgr.charm.cwd` share the
# slot `cwd`), or a constant-key subscript of a name (`captured['cwd']`).
def _slot(node: ast.AST) -> tuple | None:
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Attribute):
        return ("attr", node.attr)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
    ):
        return ("item", node.value.id, node.slice.value)
    return None


def _direct_kind(node: ast.AST, path_attrs: frozenset[str]) -> str | None:
    """'str', 'path' or None, from the expression alone."""
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "str"
    if isinstance(node, ast.Attribute) and node.attr in path_attrs:
        return "path"
    if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
        return "str"
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        if func in {"os.getcwd", "str", "os.fspath", "os.getenv", "os.environ.get"}:
            return "str"
        if func.startswith("os.path.") and func.rsplit(".", 1)[1] in _OS_PATH_STR_FUNCS:
            return "str"
        if func in {"Path", "pathlib.Path", "PurePath", "pathlib.PurePath"}:
            return "path"
    return None


def _str_compared_with_path(tree: ast.Module) -> list[str]:
    # `.charm_dir` is a Path only if the file never assigns an attribute of
    # that name itself; one that does may be holding anything.
    assigned_attrs = {
        target.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
    }
    path_attrs = _PATH_ATTRS - assigned_attrs
    kinds: dict[tuple, set] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs = [(target, node.value) for target in node.targets]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        elif isinstance(node, ast.AugAssign):
            pairs = [(node.target, None)]
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            pairs = [(node.target, None)]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            pairs = [(item.optional_vars, None) for item in node.items if item.optional_vars]
        elif isinstance(node, ast.NamedExpr):
            pairs = [(node.target, node.value)]
        elif isinstance(node, ast.arg):
            pairs = [(ast.Name(id=node.arg), None)]
        else:
            continue
        for target, value in pairs:
            slot = _slot(target)
            if slot is None:
                # Unpacking: every name in it is unknown.
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        kinds.setdefault(("name", name.id), set()).add(None)
                continue
            kind = None if value is None else _direct_kind(value, path_attrs)
            kinds.setdefault(slot, set()).add(kind)

    def kind_of(node: ast.AST) -> str | None:
        direct = _direct_kind(node, path_attrs)
        if direct is not None:
            return direct
        slot = _slot(node)
        if slot is None:
            return None
        seen = kinds.get(slot)
        if seen is None or len(seen) != 1:
            return None
        return next(iter(seen))

    reasons = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for op, left, right in zip(node.ops, operands[:-1], operands[1:], strict=True):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            if {kind_of(left), kind_of(right)} == {"str", "path"}:
                reasons.append(
                    f"line {node.lineno}: `{ast.unparse(left)}` and `{ast.unparse(right)}` are a "
                    "`str` and a `pathlib.Path`, which never compare equal, so the assertion fails "
                    "whether or not the bug is real; convert both sides first"
                )
    return reasons


def _capture_read_before_handlers_run(tree: ast.Module) -> list[str]:
    """Inside `with ctx(event, state) as mgr:` the handlers have not run
    until `mgr.run()` or the end of the block, so reading something a handler
    stores, before then, reads nothing (or the previous test's value)."""
    handler_slots: set[tuple] = set()
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and _is_charm_class(cls)):
            continue
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.name == "__init__":
                continue
            declared_global = {
                name for node in ast.walk(method) if isinstance(node, ast.Global) for name in node.names
            }
            for node in ast.walk(method):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                    if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                    else []
                )
                for target in targets:
                    if isinstance(target, ast.Attribute) and ast.unparse(target.value) == "self":
                        handler_slots.add(("attr", target.attr))
                    elif isinstance(target, ast.Subscript):
                        slot = _slot(target)
                        if slot is not None:
                            handler_slots.add(slot)
                    elif isinstance(target, ast.Name) and target.id in declared_global:
                        handler_slots.add(("name", target.id))
    if not handler_slots:
        return []
    reasons = []
    for scope, context_names in _context_scopes(tree):
        if not context_names:
            continue
        for node in _own_nodes(scope):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            managers = [
                item.optional_vars.id
                for item in node.items
                if isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id in context_names
                and isinstance(item.optional_vars, ast.Name)
            ]
            if not managers:
                continue
            for statement in node.body:
                if _calls_manager_run(statement, managers):
                    break
                for read in ast.walk(statement):
                    if not isinstance(getattr(read, "ctx", None), ast.Load):
                        continue
                    slot = _slot(read)
                    if slot in handler_slots:
                        reasons.append(
                            f"line {read.lineno}: `{ast.unparse(read)}` is set by an event handler "
                            f"but read inside `with {ast.unparse(node.items[0].context_expr)} as "
                            f"{managers[0]}:` before the handlers run; use `ctx.run(...)` and assert "
                            "afterwards, or call `mgr.run()` first"
                        )
    return reasons


def _calls_manager_run(statement: ast.AST, managers: list[str]) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in managers
        for node in ast.walk(statement)
    )


def _backend_under_testing(tree: ast.Module) -> list[str]:
    """`._backend` in a file that builds a `testing.Context`.

    `ops.testing` swaps the model backend for one that never runs a hook
    tool, so it cannot return what Juju's hook tools return. On `#2709`
    (Juju answers "permission denied" for a gone relation), a test whose
    charm called `self.model._backend.relation_get(...)` and expected that
    error failed on its own assertion with or without the fix, and rung 6
    called it a reproduction (`spike-step-5/static-retry/RESULT.md` §10). No
    `ops.testing` test has a reason to reach into the backend, so this cannot
    reject one that would work.
    """
    if not any(_is_context_call(node) for node in ast.walk(tree)):
        return []
    return [
        f"line {node.lineno}: `{ast.unparse(node)}` under `ops.testing`, which replaces the "
        "model backend, so no hook tool runs and nothing Juju's hook tools return can "
        "happen; for a bug in how ops handles a hook tool's output, test "
        "`ops.model._ModelBackend` with fake hook tools instead"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_backend"
    ]


# Juju 3.6's hook tools: the executables `ops.model._ModelBackend` runs.
_HOOK_TOOLS = frozenset({
    "action-fail",
    "action-get",
    "action-log",
    "action-set",
    "application-version-set",
    "close-port",
    "config-get",
    "credential-get",
    "goal-state",
    "is-leader",
    "juju-log",
    "juju-reboot",
    "leader-get",
    "leader-set",
    "network-get",
    "open-port",
    "opened-ports",
    "pod-spec-get",
    "pod-spec-set",
    "relation-get",
    "relation-ids",
    "relation-list",
    "relation-model-get",
    "relation-set",
    "resource-get",
    "secret-add",
    "secret-get",
    "secret-grant",
    "secret-ids",
    "secret-info-get",
    "secret-remove",
    "secret-revoke",
    "secret-set",
    "state-delete",
    "state-get",
    "state-set",
    "status-get",
    "status-set",
    "storage-add",
    "storage-get",
    "storage-list",
    "unit-get",
})


def faked_hook_tools(source: str) -> list[str]:
    """The hook tools a test fakes, sorted, or `[]` when it fakes none.

    A test fakes hook tools when it drives `ops.model._ModelBackend` and
    names hook tools as string literals (the files it writes onto `PATH`).
    Whatever those fakes print is the issue's account of what Juju returned,
    not something the run observed, and the composer says so.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _faked_hook_tools(tree)


# -- the fake hook tool shape -------------------------------------------------
#
# A test in the shape the prompt asks for on a hook-tool bug builds
# `ops.Model(meta, _ModelBackend(...))` with fake hook tools on `PATH`. On
# `#2709` all 8 live tests took that shape and none reached the bug
# (`spike-step-5/static-retry/RESULT.md` §16): 5 asked for an endpoint their
# `CharmMeta` never declares, 1 left `relation-ids` unfaked, and 1 passed
# `_ModelBackend.relation_get()` keywords it does not take. The rules below
# reject those shapes. As everywhere in this module, anything they cannot be
# sure of, they let through.


def _plain_import_roots(tree: ast.Module) -> set[str]:
    """Names bound by `import x` / `import x.y` with no `as`."""
    return {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if not alias.asname
    }


def _file_binding_counts(tree: ast.Module) -> dict[str, int] | None:
    """How many times each name is bound anywhere in the file, ignoring scope,
    or `None` when an `import *` makes that unknowable.

    `import ops` and `import ops.model` both bind `ops` to the same module, so
    plain imports count once per root name.
    """
    counts: dict[str, int] = {}

    def add(name: str, n: int = 1) -> None:
        counts[name] = counts.get(name, 0) + n

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            add(node.id)
        elif isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return None
                add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                add(name, 2)
    for root in _plain_import_roots(tree):
        add(root)
    return counts


class _Names:
    """What the file's names certainly are: imports bound exactly once, and
    names assigned exactly once by a plain `name = value` statement."""

    def __init__(self, tree: ast.Module, counts: dict[str, int]):
        self.counts = counts
        self.imports: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.imports[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".")[0]
                        self.imports[root] = root
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        self.imports = {name: dotted for name, dotted in self.imports.items() if counts.get(name) == 1}
        self.assignments: dict[str, ast.Assign] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if counts.get(name) == 1:
                    self.assignments[name] = node

    def dotted(self, node: ast.AST) -> str | None:
        """`ops.model.Model` for `ops.model.Model`, `Model` imported from
        `ops.model`, and so on; `None` for anything not rooted in an import."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.insert(0, node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id not in self.imports:
            return None
        return ".".join([self.imports[node.id], *parts])

    def resolve(self, node: ast.AST) -> object | None:
        """The `ops` object an expression names, or `None`. Builtins count
        too, when the file never binds the name."""
        if isinstance(node, ast.Name) and node.id not in self.counts and node.id not in self.imports:
            return getattr(builtins, node.id, None)
        return _ops_object(self.dotted(node))

    def value_of(self, node: ast.AST) -> ast.AST:
        """`node`, or the value a once-assigned name was assigned."""
        if isinstance(node, ast.Name) and node.id in self.assignments:
            return self.assignments[node.id].value
        return node


@functools.cache
def _ops_object(dotted: str | None) -> object | None:
    """The object `ops.x.y` names in the installed ops. Only `ops` modules
    are imported: a test file's other imports are never run."""
    if dotted is None or not (dotted == "ops" or dotted.startswith("ops.")):
        return None
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        for part in parts[i:]:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None
        return obj
    return None


def _ops():
    try:
        import ops
        import ops.model
    except Exception:
        return None
    return ops


def _same_function(obj: object, target: object) -> bool:
    return getattr(obj, "__func__", obj) is getattr(target, "__func__", target)


def _call_arg(call: ast.Call, index: int, keyword: str) -> ast.AST | None:
    if len(call.args) > index and not any(isinstance(a, ast.Starred) for a in call.args[: index + 1]):
        return call.args[index]
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _meta_endpoints(call: ast.Call, names: _Names) -> frozenset | None:
    """The endpoints a `CharmMeta` built by `call` from literals declares, by
    building it with the installed ops; `None` when `call` is not that, or
    the literal does not build."""
    ops = _ops()
    if ops is None:
        return None
    func = names.resolve(call.func)
    if func is ops.CharmMeta:
        build = ops.CharmMeta
    elif _same_function(func, ops.CharmMeta.from_yaml):
        build = ops.CharmMeta.from_yaml
    else:
        return None
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None
    try:
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if build is not ops.CharmMeta and not all(isinstance(v, str) for v in [*args, *kwargs.values()]):
        return None
    try:
        meta = build(*args, **kwargs)
        return frozenset(meta.relations)
    except Exception:
        return None


def _is_meta_call(node: ast.AST, names: _Names) -> bool:
    ops = _ops()
    if ops is None or not isinstance(node, ast.Call):
        return False
    func = names.resolve(node.func)
    return func is ops.CharmMeta or _same_function(func, ops.CharmMeta.from_yaml)


def _is_backend_call(node: ast.AST, names: _Names) -> bool:
    ops = _ops()
    return (
        ops is not None
        and isinstance(node, ast.Call)
        and names.resolve(node.func) is ops.model._ModelBackend
    )


def _models(names: _Names) -> dict[str, ast.Call]:
    """{name: the `ops.Model(...)` call}, for names
    assigned exactly once, to an `ops.Model` over a `_ModelBackend(...)`
    (built in the call or assigned once to a name)."""
    ops = _ops()
    if ops is None:
        return {}
    found = {}
    for name, assign in names.assignments.items():
        call = assign.value
        if not (isinstance(call, ast.Call) and names.resolve(call.func) is ops.Model):
            continue
        backend = _call_arg(call, 1, "backend")
        if backend is not None and _is_backend_call(names.value_of(backend), names):
            found[name] = call
    return found


def _backends(names: _Names) -> dict[str, ast.Assign]:
    """{name: its assignment}, for names assigned exactly once to
    `_ModelBackend(...)`."""
    return {name: a for name, a in names.assignments.items() if _is_backend_call(a.value, names)}


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _may_catch(handler_type: ast.AST | None, exception: type, names: _Names) -> bool:
    """Whether an `except` (or `pytest.raises`) of `handler_type` could catch
    `exception`. Anything it cannot resolve might."""
    if handler_type is None:
        return True
    if isinstance(handler_type, ast.Tuple):
        return any(_may_catch(elt, exception, names) for elt in handler_type.elts)
    cls = names.resolve(handler_type)
    if not isinstance(cls, type):
        return True
    return issubclass(exception, cls)


def _raises_types(item: ast.withitem, names: _Names) -> ast.AST | None:
    """The exception argument of a `with pytest.raises(E):` item, or `None`
    when the item is anything else."""
    call = item.context_expr
    if (
        isinstance(call, ast.Call)
        and names.dotted(call.func) == "pytest.raises"
        and len(call.args) == 1
        and not isinstance(call.args[0], ast.Starred)
    ):
        return call.args[0]
    return None


def _conditional(node: ast.AST, parent: ast.AST) -> bool:
    """Whether `parent` may evaluate its child `node` zero times."""
    if isinstance(parent, (ast.If, ast.While)):
        return node is not parent.test
    if isinstance(parent, (ast.For, ast.AsyncFor)):
        return node is not parent.iter
    if isinstance(parent, ast.IfExp):
        return node is not parent.test
    if isinstance(parent, ast.BoolOp):
        return node is not parent.values[0]
    if isinstance(parent, ast.Assert):
        return node is parent.msg
    if isinstance(parent, (ast.Lambda, ast.comprehension, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return True
    if isinstance(parent, ast.Match):
        return node is not parent.subject
    if isinstance(parent, (ast.ExceptHandler, ast.match_case)):
        return True
    if isinstance(parent, (ast.Try, ast.TryStar)):
        return node in parent.orelse
    return False


def _escapes(node: ast.AST, parents: dict, exception: type, names: _Names) -> bool:
    """Whether `exception`, raised at `node`, certainly fails the function it
    is in: it propagates out, or into a handler that certainly fails the test
    (`pytest.fail(...)`, `raise`, `assert False`), with nothing conditional on
    the way up. Blocks that certainly run count as unconditional: `if` on a
    constant, `while True`, `for` over a literal that is not empty, and the
    body of a `with` whose managers cannot swallow it (`pytest.raises` of
    something else, `open()`, `contextlib.nullcontext()`). Past a failing
    handler, any `try` or `pytest.raises` further up might catch what it
    raises, so the walk stops there."""
    child = node
    failed = False
    while child in parents:
        parent = parents[child]
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            return True
        if _conditional(child, parent) and not _certainly_runs(child, parent, names):
            return False
        if isinstance(parent, ast.TryStar) and child in parent.body:
            return False
        if isinstance(parent, ast.Try) and child in parent.body and parent.handlers:
            if failed:
                return False
            catching = [h for h in parent.handlers if _may_catch(h.type, exception, names)]
            if not all(_certainly_fails(h.body, names, handler=True) for h in catching):
                return False
            failed = bool(catching)
        if isinstance(parent, (ast.With, ast.AsyncWith)) and child in parent.body:
            for item in parent.items:
                raised = _raises_types(item, names)
                if raised is None:
                    if not _passes_exceptions_on(item, names):
                        return False
                elif failed or _may_catch(raised, exception, names):
                    return False
        child = parent
    return True


def _certainly_runs(node: ast.AST, parent: ast.AST, names: _Names) -> bool:
    """Whether `parent`, which `_conditional()` says may skip `node`, is
    certain to run it at least once anyway: the branch an `if` on a constant
    takes, the body of a `while` on a true constant, or of a `for` over a
    literal that is not empty (`range(n)` with a literal `n > 0` included).
    A loop with a `break` or `continue` in it is not certain."""
    if isinstance(parent, ast.If) and isinstance(parent.test, ast.Constant):
        return node in (parent.body if parent.test.value else parent.orelse)
    if isinstance(parent, (ast.While, ast.For)) and node in parent.body:
        if any(isinstance(n, (ast.Break, ast.Continue)) for s in parent.body for n in ast.walk(s)):
            return False
        if isinstance(parent, ast.While):
            return isinstance(parent.test, ast.Constant) and bool(parent.test.value)
        return _not_empty(parent.iter, names)
    return False


def _not_empty(node: ast.AST, names: _Names) -> bool:
    """A literal iterable that is certainly not empty."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts) and not any(isinstance(e, ast.Starred) for e in node.elts)
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return bool(node.value)
    if (
        isinstance(node, ast.Call)
        and names.resolve(node.func) is range
        and not node.keywords
        and 1 <= len(node.args) <= 2
        and all(isinstance(a, ast.Constant) and type(a.value) is int for a in node.args)
    ):
        start, stop = (0, node.args[0].value) if len(node.args) == 1 else (node.args[0].value, node.args[1].value)
        return stop > start
    return False


def _passes_exceptions_on(item: ast.withitem, names: _Names) -> bool:
    """A `with` item whose context manager never swallows an exception:
    `open(...)` and `contextlib.nullcontext(...)`."""
    call = item.context_expr
    if not isinstance(call, ast.Call):
        return False
    return names.resolve(call.func) is open or names.dotted(call.func) == "contextlib.nullcontext"


def _certainly_fails(body: list[ast.stmt], names: _Names, *, handler: bool = False) -> bool:
    """Whether running `body` certainly fails the test at its first statement,
    with something that is not an `ops.ModelError` (so a `ModelError` handler
    further up cannot swallow it): `pytest.fail(...)`, `assert` on a false
    constant, or `raise` of a builtin or `ops`-resolved exception class that
    is not a `ModelError`. In a handler, a bare `raise` counts too: it raises
    what the handler caught."""
    if not body:
        return False
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Call):
        return names.dotted(first.value.func) == "pytest.fail"
    if isinstance(first, ast.Assert):
        return isinstance(first.test, ast.Constant) and not first.test.value
    if isinstance(first, ast.Raise):
        if first.exc is None:
            return handler
        exc = first.exc.func if isinstance(first.exc, ast.Call) else first.exc
        cls = names.resolve(exc)
        ops = _ops()
        return (
            isinstance(cls, type)
            and issubclass(cls, BaseException)
            and ops is not None
            and not issubclass(cls, ops.ModelError)
        )
    return False


def _undeclared_endpoints(tree: ast.Module) -> list[str]:
    """A relation read by a literal endpoint name its `CharmMeta` does not
    declare.

    `model.get_relation('db', ...)` and `model.relations['db']` raise
    `KeyError: 'db'` when `db` is not under `requires`, `provides` or `peers`,
    before any hook tool runs. On `#2709`, 5 of 8 fake-hook-tool tests copied
    the prompt example's `CharmMeta.from_yaml('name: myapp\\n')` and then asked
    for `db` (`spike-step-5/static-retry/RESULT.md` §16), and one passed the
    relation ID where the name goes (`get_relation(1)`, `KeyError: 1`).

    Only when the file builds exactly one `CharmMeta`, from literals that the
    installed ops accepts (`CharmMeta.from_yaml('...')` or
    `CharmMeta({...})`), the model is a name assigned once to `ops.Model` over
    that meta (inline, or through a name assigned once) and a
    `_ModelBackend`, and a `KeyError` there would certainly escape the test.
    """
    counts = _file_binding_counts(tree)
    if counts is None:
        return []
    names = _Names(tree, counts)
    metas = [node for node in ast.walk(tree) if _is_meta_call(node, names)]
    if len(metas) != 1:
        return []
    meta = metas[0]
    endpoints = _meta_endpoints(meta, names)
    if endpoints is None:
        return []
    models = {
        name: call
        for name, call in _models(names).items()
        if names.value_of(_call_arg(call, 0, "meta")) is meta
    }
    if not models:
        return []
    parents = _parents(tree)
    faked = set(_faked_hook_tools(tree))
    declared = ", ".join(f"`{e}`" for e in sorted(endpoints))
    has = f"it declares only {declared}" if endpoints else "it declares no relations"
    reasons = []
    for node in ast.walk(tree):
        endpoint = _endpoint_read(node, models)
        if endpoint is None:
            continue
        model, key = endpoint
        if not isinstance(key, ast.Constant) or key.value in endpoints:
            continue
        if not _escapes(node, parents, KeyError, names):
            continue
        value = key.value
        spelled = ast.unparse(node)
        if isinstance(value, str):
            reasons.append(
                f"line {node.lineno}: `{spelled}` reads the `{value}` endpoint, which the "
                f"`CharmMeta` on line {meta.lineno} does not declare ({has}), so ops raises "
                f"`KeyError: '{value}'` before any hook tool runs; declare `{value}` in that YAML "
                f"under `requires` (or `provides`, or `peers` for a peer relation), for example "
                f"`requires: {{{value}: {{interface: {value}}}}}`"
            )
        else:
            example = sorted(endpoints)[0] if len(endpoints) == 1 else "<endpoint>"
            if isinstance(node, ast.Call):
                fix = (
                    f"`get_relation()` takes the endpoint name first and the relation ID "
                    f"second, as in `{model}.get_relation('{example}', {value!r})`"
                )
            else:
                fix = f"`{model}.relations` is keyed by endpoint name, as in `{model}.relations['{example}']`"
            if not endpoints:
                fix += ", with the endpoint declared in the `CharmMeta` YAML under `requires` (or `provides`, or `peers`)"
            unfaked = [t for t in ("relation-ids", "relation-list") if t not in faked]
            if isinstance(node, ast.Call) and faked and unfaked:
                what = "; ".join(_tool_output(t, example if example != "<endpoint>" else None, value) for t in unfaked)
                fix += (
                    f". That call needs the `relation-ids` and `relation-list` hook tools, and the test "
                    f"does not fake {_and_list(f'`{t}`' for t in unfaked)}: {what}"
                )
            reasons.append(
                f"line {node.lineno}: `{spelled}` passes {value!r} where the endpoint name goes, "
                f"and the `CharmMeta` on line {meta.lineno} declares no such endpoint ({has}), so "
                f"ops raises `KeyError: {value!r}`; {fix}"
            )
    return list(dict.fromkeys(reasons))


def _endpoint_read(node: ast.AST, models) -> tuple[str, ast.AST] | None:
    """(model name, the endpoint argument) for `model.get_relation(name, ...)`
    and `model.relations[name]`, on a tracked model."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_relation":
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id in models:
            key = _call_arg(node, 0, "relation_name")
            return (owner.id, key) if key is not None else None
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        owner = node.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "relations"
            and isinstance(owner.value, ast.Name)
            and owner.value.id in models
        ):
            return owner.value.id, node.slice
    return None


# What each fake hook tool has to print for ops to get past it, for the
# reasons below.
def _tool_output(tool: str, endpoint: str | None, relation_id: int | None) -> str:
    if tool == "relation-ids":
        example = f"{endpoint or 'db'}:{relation_id if relation_id is not None else 1}"
        return f'`relation-ids` prints the endpoint\'s relation IDs as a JSON list, like `["{example}"]`'
    if tool == "relation-list":
        return (
            '`relation-list` prints the remote units as a JSON list, like `["provider/0"]` (if it '
            "prints `[]`, ops runs `relation-list --app` next, which has to print the remote "
            'application as a JSON string, like `"provider"`)'
        )
    if tool == "relation-get":
        return "`relation-get` prints the databag as a JSON object, like `{}`"
    if tool == "is-leader":
        return "`is-leader` prints `true` or `false`"
    if tool == "config-get":
        return "`config-get` prints the config as a JSON object, like `{}`"
    return f"`{tool}`"


# The hook tools a `_ModelBackend` method runs, every time it is called. Each
# one is checked by running it against the installed ops
# (`tests/test_hook_tool_rules.py`). `relation_remote_app_name` is left out:
# it runs nothing when the test sets `JUJU_RELATION_ID` and `JUJU_REMOTE_APP`.
_BACKEND_METHOD_TOOLS = {
    "relation_ids": ("relation-ids",),
    "relation_list": ("relation-list",),
    "relation_get": ("relation-get",),
    "is_leader": ("is-leader",),
    "config_get": ("config-get",),
}

# What makes Juju's hook tools fail as an authorisation error, which ops
# 3.8.3 reports as a security event, checking leadership first.
_AUTHZ_MESSAGES = ("access denied", "permission denied", "not the leader", "cannot write relation settings")


@functools.cache
def _security_event_runs_is_leader() -> bool:
    """Whether the installed ops runs `is-leader` when a hook tool fails with
    an authorisation error. ops 3.8.3 does (`_check_for_security_event()`
    calls `is_leader()`); the `#2709` fix uses the cached value instead."""
    ops = _ops()
    if ops is None:
        return False
    try:
        source = inspect.getsource(ops.model._ModelBackend._check_for_security_event)
    except (AttributeError, OSError, TypeError):
        return False
    return "self.is_leader()" in source and all(message in source for message in _AUTHZ_MESSAGES)


def _fake_helper(func: ast.FunctionDef) -> tuple[int, int] | None:
    """(index of the tool-name parameter, index of the script parameter) for
    a module-level helper that writes `"#!/bin/sh\\n" + script + "\\n"` to
    `<dir> / name`, as the prompt's example does; `None` for anything else."""
    args = func.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults:
        return None
    params = [a.arg for a in args.args]
    script = name = None
    for node in ast.walk(func):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Name):
            if node.right.id in params:
                name = params.index(node.right.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and len(node.args) == 1
        ):
            parts = _concatenated(node.args[0])
            if (
                parts is not None
                and len(parts) == 3
                and parts[0] == "#!/bin/sh\n"
                and isinstance(parts[1], ast.Name)
                and parts[1].id in params
                and parts[2] == "\n"
            ):
                script = params.index(parts[1].id)
    if script is None or name is None or script == name:
        return None
    return name, script


def _concatenated(node: ast.AST) -> list | None:
    """The pieces of `"a" + x + "b"` or `f"a{x}b"`: strings for literals,
    the node for anything else."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _concatenated(node.left)
        right = _concatenated(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts: list = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue) and value.conversion == -1 and value.format_spec is None:
                parts.append(value.value)
            else:
                return None
        return parts
    if isinstance(node, ast.Name):
        return [node]
    return None


def _fake_scripts(tree: ast.Module) -> dict[str, list[str | None]]:
    """{hook tool: the script of each fake of it}, `None` for a fake whose
    script is not certain. A tool's name appearing anywhere other than as
    the name argument of a recognised helper call counts as an unknown
    fake."""
    helpers = {
        node.name: shape
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and (shape := _fake_helper(node)) is not None
    }
    scripts: dict[str, list[str | None]] = {}
    recognised: set[int] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in helpers):
            continue
        name_at, script_at = helpers[node.func.id]
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args) or len(node.args) <= max(name_at, script_at):
            continue
        tool, script = node.args[name_at], node.args[script_at]
        if not (isinstance(tool, ast.Constant) and tool.value in _HOOK_TOOLS):
            continue
        recognised.add(id(tool))
        certain = isinstance(script, ast.Constant) and isinstance(script.value, str)
        scripts.setdefault(tool.value, []).append(script.value if certain else None)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _HOOK_TOOLS
            and id(node) not in recognised
        ):
            scripts.setdefault(node.value, []).append(None)
    return scripts


def _authz_failure(script: str) -> str | None:
    """The message, when a fake's script is exactly `echo <message> >&2;
    exit <non-zero>` and the message is one ops treats as an authorisation
    failure."""
    commands = [c.strip() for line in script.splitlines() for c in line.split(";") if c.strip()]
    if len(commands) != 2:
        return None
    try:
        echo = shlex.split(commands[0])
        leave = shlex.split(commands[1])
    except ValueError:
        return None
    if len(leave) != 2 or leave[0] != "exit" or not leave[1].isdigit() or int(leave[1]) == 0:
        return None
    if echo.count(">&2") != 1:
        return None
    echo.remove(">&2")
    if not echo or echo[0] != "echo" or len(echo) < 2:
        return None
    words = echo[1:]
    if any(w.startswith("-") or any(c in w for c in "$`|&<>*?[\\") for w in words):
        return None
    message = " ".join(words)
    lowered = message.lower()
    return message if any(m in lowered for m in _AUTHZ_MESSAGES) else None



def _non_json_output(script: str) -> str | None:
    """What a fake prints, when its script is exactly `echo <words>`
    (optionally followed by `exit 0`), or empty, and that is not JSON. Every
    relation hook tool's output goes through `json.loads()` in ops, so such
    a fake fails the test with `JSONDecodeError`."""
    commands = [c.strip() for line in script.splitlines() for c in line.split(";") if c.strip()]
    if commands and commands[-1] == "exit 0":
        commands.pop()
    if not commands:
        printed = ""
    elif len(commands) == 1:
        try:
            words = shlex.split(commands[0])
        except ValueError:
            return None
        if not words or words[0] != "echo" or any(
            w.startswith("-") or any(c in w for c in "$`|&<>*?\\") for w in words[1:]
        ):
            return None
        printed = " ".join(words[1:])
    else:
        return None
    try:
        json.loads(printed)
    except ValueError:
        return printed
    return None

def _test_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Module-level `def test*` that pytest runs as written: no decorators
    and no `pytestmark` anywhere in the file."""
    if any(isinstance(node, ast.Name) and node.id == "pytestmark" for node in ast.walk(tree)):
        return []
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test") and not node.decorator_list
    ]


_STOPS = frozenset({"skip", "xfail", "exit", "importorskip"})


def _stops_the_test(statement: ast.AST) -> bool:
    """A `return`, or a call that ends the test without failing it."""
    for node in ast.walk(statement):
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if name in _STOPS:
                return True
    return False


def _may_fail_an_assertion(statement: ast.AST, functions: dict[str, ast.FunctionDef]) -> bool:
    """Whether `statement` could fail the test on an assertion: an `assert`,
    `pytest.fail` or `pytest.raises`, or a call to a function in the file that
    has one of those or raises."""
    for node in ast.walk(statement):
        if isinstance(node, (ast.Assert, ast.Raise)):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"fail", "raises"}:
                return True
            if isinstance(func, ast.Name) and func.id in functions:
                if any(isinstance(n, (ast.Assert, ast.Raise)) for n in ast.walk(functions[func.id])):
                    return True
    return False


def _relation_access(
    node: ast.AST, models: set[str], ids: int | None = None
) -> tuple[tuple[str, ...], str, int | None] | None:
    """(the hook tools it needs, endpoint, relation ID) for a relation access
    on a tracked model whose tools do not depend on what the fakes print, or
    depend only on how many IDs `relation-ids` prints, when that is `ids`:

    - `model.relations['db']` and `model.get_relation('db')` run
      `relation-ids`, and `relation-list` once for each ID it prints (so
      when `ids` is 1 or more, both);
    - `model.get_relation('db', 2)` runs `relation-ids` and `relation-list`,
      whether or not 2 is among the IDs `relation-ids` prints;
    - `model.relations['db'][0]` runs both too, or raises `IndexError`.
    """
    ids_only = ("relation-ids", "relation-list") if ids else ("relation-ids",)
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        owner = node.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "relations"
            and isinstance(owner.value, ast.Name)
            and owner.value.id in models
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            return ids_only, node.slice.value, None
        inner = _relation_access(owner, models)
        if (
            inner is not None
            and isinstance(owner, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
        ):
            return ("relation-ids", "relation-list"), inner[1], None
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_relation"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in models
        and not any(isinstance(a, ast.Starred) for a in node.args)
        and all(k.arg in {"relation_name", "relation_id"} for k in node.keywords)
        and len(node.args) + len(node.keywords) in (1, 2)
    ):
        endpoint = _call_arg(node, 0, "relation_name")
        relation_id = _call_arg(node, 1, "relation_id")
        if not (isinstance(endpoint, ast.Constant) and isinstance(endpoint.value, str)):
            return None
        if relation_id is None or (isinstance(relation_id, ast.Constant) and relation_id.value is None):
            return ids_only, endpoint.value, None
        if isinstance(relation_id, ast.Constant) and type(relation_id.value) is int:
            return ("relation-ids", "relation-list"), endpoint.value, relation_id.value
    return None


def _databag(node: ast.AST, relations: dict, contents: dict) -> str | None:
    """The relation name, when `node` is `R.data[X]` on a tracked relation, or
    a name assigned once (at the top of the test, earlier) to that."""
    if isinstance(node, ast.Name) and node.id in contents:
        return contents[node.id]
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "data"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id in relations
    ):
        return node.value.value.id
    return None


def _databag_reads(target: ast.AST, relations: dict, contents: dict, names: _Names):
    """Yield (read node, relation name) for each read of a relation databag
    in `target` that certainly loads it, which is when ops runs `relation-get`:
    `dict(bag)`, `len(bag)`, `bool(bag)`, `list(bag)`, `bag == {...}`,
    `bag != {...}`, `key in bag`, `bag[key]` and `bag.get(key)`. Not
    `bag.keys()` and the like, which are lazy views."""

    def mapping_literal(node: ast.AST) -> bool:
        return isinstance(node, ast.Dict) or (isinstance(node, ast.Call) and names.resolve(node.func) is dict)

    for node in ast.walk(target):
        relation = None
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            if names.resolve(node.func) in (dict, len, bool, list):
                relation = _databag(node.args[0], relations, contents)
        if relation is None and isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get":
                relation = _databag(node.func.value, relations, contents)
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            relation = _databag(node.value, relations, contents)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, op, right = node.left, node.ops[0], node.comparators[0]
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if mapping_literal(right):
                    relation = _databag(left, relations, contents)
                elif mapping_literal(left):
                    relation = _databag(right, relations, contents)
            elif isinstance(op, (ast.In, ast.NotIn)):
                relation = _databag(right, relations, contents)
        if relation is not None:
            yield node, relation


def _and_list(items) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


class _Place(typing.NamedTuple):
    """A statement inside a test's top-level statement that certainly runs
    when the test gets to that top-level statement, and how it is wrapped."""

    statement: ast.stmt
    guarded: bool  # inside a `try` or `with pytest.raises(...)`
    strict: bool  # a handler or `pytest.raises` around it catches `ops.ModelError`s and does not fail
    caught: list[ast.AST]  # the types every `try` and `pytest.raises` around it catch
    in_raises: bool  # inside a `with pytest.raises(...)`
    earlier: list[ast.stmt]  # statements inside the top-level one that run before it


def _certain_places(statement: ast.stmt, names: _Names, ops, _place: _Place | None = None):
    """Yield a `_Place` for `statement`, a statement at the top of a test,
    and for each statement inside it that certainly runs when it does, with
    an exception raised there certainly failing the test, unless it is an
    `ops.ModelError` that a handler around it might catch.

    It goes into the body of a `try` whose handlers each either catch only
    `ops.ModelError` subclasses or certainly fail the test (`pytest.fail`,
    `raise`, `assert False`), of a `with` whose managers are `pytest.raises`
    of `ops.ModelError` subclasses, `open()` or `contextlib.nullcontext()`, of
    an `if` on a constant (the branch it takes), of `while True`, and of a
    `for` over a literal that is not empty, with no `break` or `continue`.
    Not into an `else`, `finally` or handler, a `match`, or a nested function,
    class or lambda.

    Below a handler or `pytest.raises` that catches `ModelError`s without
    failing (`strict`), only the first statement of each block: one before
    it could raise a `ModelError` that is swallowed, and then it never runs.
    Otherwise, every statement of the block: one before it either completes
    or fails the test."""
    if _place is None:
        _place = _Place(statement, False, False, [], False, [])
    else:
        _place = _place._replace(statement=statement)

    def model_errors_only(types: list[ast.AST | None]) -> bool:
        for handler_type in types:
            for t in handler_type.elts if isinstance(handler_type, ast.Tuple) else [handler_type]:
                cls = names.resolve(t) if t is not None else None
                if not (isinstance(cls, type) and issubclass(cls, ops.ModelError)):
                    return False
        return True

    yield _place
    place = _place
    if isinstance(statement, ast.Try):
        swallowing = [h.type for h in statement.handlers if not _certainly_fails(h.body, names, handler=True)]
        if not model_errors_only(swallowing):
            return
        place = place._replace(
            guarded=True,
            strict=place.strict or bool(swallowing),
            caught=place.caught + [h.type for h in statement.handlers],
        )
        body = statement.body
    elif isinstance(statement, ast.With):
        types = []
        for item in statement.items:
            raised = _raises_types(item, names)
            if raised is not None:
                types.append(raised)
            elif not _passes_exceptions_on(item, names):
                return
        if types:
            if not model_errors_only(types):
                return
            place = place._replace(guarded=True, strict=True, caught=place.caught + types, in_raises=True)
        body = statement.body
    elif isinstance(statement, (ast.If, ast.While, ast.For)):
        body = statement.body if not isinstance(statement, ast.If) else statement.body + statement.orelse
        body = [s for s in body if _certainly_runs(s, statement, names)]
    else:
        return
    if place.strict:
        body = body[:1]
    for i, child in enumerate(body):
        yield from _certain_places(child, names, ops, place._replace(earlier=place.earlier + body[:i]))


# The statements an access can be in, once `_certain_places()` has walked
# into the compound ones.
_SIMPLE_STATEMENTS = (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Assert, ast.Delete, ast.Pass)


def _unconditional(node: ast.AST, statement: ast.AST) -> bool:
    """Whether evaluating `statement` certainly evaluates `node`."""
    parents = _parents(statement)
    child = node
    while child is not statement and child in parents:
        parent = parents[child]
        if _conditional(child, parent):
            return False
        child = parent
    return True


def _calls_only(statement: ast.AST, allowed: ast.AST) -> bool:
    """Whether every call in `statement` is `allowed` or inside it, so that
    nothing else in the statement runs first and raises a `ModelError` an
    enclosing handler would swallow."""
    inside = {id(n) for n in ast.walk(allowed)}
    return all(id(n) in inside for n in ast.walk(statement) if isinstance(n, ast.Call))


def _called_first(statement: ast.AST, node: ast.AST) -> bool:
    """Whether no call in `statement` runs before `node` is evaluated, so
    that nothing else in the statement raises first (a `ModelError` an
    enclosing handler would swallow, say). Python evaluates an assignment's
    value before its targets, and everything else left to right, in the
    order `ast` lists the children."""

    def visit(current: ast.AST) -> bool | None:
        """`True` once `node` is reached, `False` at a call that runs
        before it, `None` to go on."""
        if current is node:
            return True
        # A call that holds `node` (`dict(<node>)`) runs after it; any other
        # call reached first runs before.
        if isinstance(current, ast.Call) and not any(n is node for n in ast.walk(current)):
            return False
        children = list(ast.iter_child_nodes(current))
        if isinstance(current, ast.Assign):
            children = [current.value, *current.targets]
        elif isinstance(current, ast.AnnAssign):
            children = [c for c in (current.value, current.target) if c is not None]
        for child in children:
            found = visit(child)
            if found is not None:
                return found
        return None

    return visit(statement) is True


def _faked_hook_tools(tree: ast.Module) -> list[str]:
    uses_backend = any(
        (isinstance(node, ast.Name) and node.id == "_ModelBackend")
        or (isinstance(node, ast.Attribute) and node.attr == "_ModelBackend")
        or (isinstance(node, ast.alias) and node.name == "_ModelBackend")
        for node in ast.walk(tree)
    )
    if not uses_backend:
        return []
    return sorted({
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _HOOK_TOOLS
    })


def _unfaked_hook_tools(tree: ast.Module) -> list[str]:
    """A model access whose hook tools a fake-hook-tool test does not fake.

    A hook tool missing from `PATH` fails the test with `FileNotFoundError`
    before it reaches the bug, with or without a fix: on `#2709`, the one
    test of 8 that declared its endpoint faked `relation-get` and
    `relation-list` and died on `relation-ids`
    (`spike-step-5/static-retry/RESULT.md` §16). Which tools an access needs
    comes from ops 3.8.3's `ops/model.py`, and each is checked by running the
    access with every tool faked and with each one left out
    (`tests/test_hook_tool_rules.py`):

    - `model.relations['db']`, `model.get_relation('db')`: `relation-ids`;
    - `model.get_relation('db', 2)`, `model.relations['db'][0]`:
      `relation-ids` and `relation-list`;
    - reading a databag (`_databag_reads()`) of a relation from one of
      those: `relation-get`, and `relation-list` for `get_relation('db')`
      (there is a relation to read only if there was exactly one);
    - a direct `_ModelBackend` call: `_BACKEND_METHOD_TOOLS`;
    - `is-leader`, when the first tool an access runs has a fake that prints
      an authorisation error to stderr and exits non-zero, while the
      installed ops checks leadership to log that as a security event (ops
      3.8.3 does; the `#2709` fix does not). Only when nothing before the
      access could fail the test on an assertion: a test that fails earlier
      on 3.8.3 never gets there, and with the fix it may not need `is-leader`.

    Only in a test that fakes hook tools, in a module-level `def test*` with
    no decorators, on a model (`ops.Model` over `_ModelBackend(...)`) or
    backend assigned once at the top of that function, before the access.
    The access has to be certain to run and its `FileNotFoundError` certain
    to fail the test: unconditional, in a statement `_certain_places()`
    yields (at the top of the function, or inside a `try`, `with`, `if`,
    `for` or `while` that certainly runs it, where any handler either fails
    the test or only catches `ops.ModelError`s), and, below a handler or
    `pytest.raises` that catches `ModelError`s without failing, in the first
    statement of its block with no call in that statement running before it.
    Everything after a `return` or a skip is left alone.

    `get_relation('db')` and `relations['db']` also run `relation-list` once
    `relation-ids` prints any IDs, so when the test's one `relation-ids` fake
    certainly prints some (`_relation_id_count()`), that is checked too
    (§20; by running, the same on ops 3.8.3 and `main`).
    """
    reasons = []
    found = _hook_tool_accesses(tree)
    if found is None:
        return []
    scripts = _fake_scripts(tree)
    faked = set(scripts)

    def authz_failure(tool: str) -> str | None:
        found = scripts.get(tool, [])
        if len(found) == 1 and found[0] is not None and _security_event_runs_is_leader():
            return _authz_failure(found[0])
        return None

    for access in found:
        node, tools, endpoint, relation_id = access.node, access.tools, access.endpoint, access.relation_id
        if access.strict and not _called_first(access.target, node):
            continue
        failure = authz_failure(access.first)
        if failure is not None and "is-leader" not in faked and not access.asserted:
            reasons.append(
                f"line {node.lineno}: the `{access.first}` fake fails with \"{failure}\", so "
                f"for `{ast.unparse(node)}` ops runs `is-leader` as well (it checks "
                "leadership to log the failure as a security event), and the test does "
                "not fake `is-leader`, so it fails with `FileNotFoundError` before it "
                "reaches the bug; fake `is-leader` too, printing `true` or `false`"
            )
        for tool in tools:
            found_scripts = scripts.get(tool, [])
            printed = (
                _non_json_output(found_scripts[0])
                if len(found_scripts) == 1 and found_scripts[0] is not None
                else None
            )
            if printed is not None:
                shown = f"`{printed}`" if printed else "nothing"
                reasons.append(
                    f"line {node.lineno}: `{ast.unparse(node)}` needs `{tool}`, and its "
                    f"fake prints {shown}, which is not JSON, so ops fails with "
                    "`JSONDecodeError` before it reaches the bug; "
                    f"{_tool_output(tool, endpoint, relation_id)}"
                )
        missing = [t for t in tools if t not in faked]
        if missing:
            what = "; ".join(_tool_output(t, endpoint, relation_id) for t in missing)
            reasons.append(
                f"line {node.lineno}: `{ast.unparse(node)}` needs the "
                f"{_and_list(f'`{t}`' for t in tools)} hook tool"
                f"{'s' if len(tools) > 1 else ''}, and the test does not fake "
                f"{_and_list(f'`{t}`' for t in missing)}, so it fails with "
                "`FileNotFoundError` before it reaches the bug; fake "
                f"{'it' if len(missing) == 1 else 'them'} too: {what}"
            )
    return list(dict.fromkeys(reasons))


class _Access(typing.NamedTuple):
    """A model access `_hook_tool_accesses()` is certain runs, if the test
    gets to it."""

    function: ast.FunctionDef
    index: int  # of the top-level statement it is in, in `function.body`
    node: ast.AST
    tools: tuple[str, ...]  # the hook tools it runs
    first: str  # the one it runs first
    endpoint: str | None
    relation_id: int | None
    target: ast.AST  # the statement it is in, inside any `try` / `with` / ...
    guarded: bool  # inside a `try` / `with pytest.raises(...)`
    caught: list[ast.AST]  # what those catch
    asserted: bool  # something before it could fail the test on an assertion
    strict: bool  # see `_Place`
    in_raises: bool  # inside a `with pytest.raises(...)`
    earlier: list[ast.stmt]  # statements inside the top-level one that run before it


class _Scope(typing.NamedTuple):
    """What a test has bound by the top-level statement `_scan_tests()`
    yields with it."""

    names: _Names
    functions: dict[str, ast.FunctionDef]  # the file's module-level functions
    models: set[str]  # tracked models assigned so far
    backends: set[str]  # tracked backends assigned so far
    # {relation name: (endpoint, relation ID, whether reading it needs
    # `relation-list` too)}, {databag name: relation name}
    relations: dict[str, tuple[str, int | None, bool]]
    contents: dict[str, str]
    asserted: bool  # a top-level statement before could fail on an assertion
    ids: int | None  # how many IDs `relation-ids` certainly prints by now


def _scan_tests(tree: ast.Module):
    """Yield (test function, index of the top-level statement, `_Place`,
    `_Scope`) for every statement a fake-hook-tool test certainly runs when
    it gets to the top-level statement it is in (`_certain_places()`), in
    order, stopping at a `return` or a skip. Nothing for a file that is not
    that shape or whose names cannot be known."""
    if not _faked_hook_tools(tree):
        return
    counts = _file_binding_counts(tree)
    ops = _ops()
    if counts is None or ops is None:
        return
    names = _Names(tree, counts)
    models = _models(names)
    backends = _backends(names)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    scripts = _fake_scripts(tree)
    helpers = _fake_helpers(tree)
    for function in _test_functions(tree):
        assigned: set[str] = set()
        relations: dict[str, tuple[str, int | None, bool]] = {}
        contents: dict[str, str] = {}
        asserted = False
        for index, statement in enumerate(function.body):
            if _stops_the_test(statement):
                break
            ids = _certain_fake(scripts, helpers, function.body[:index], "relation-ids", _relation_id_count)
            scope = _Scope(
                names,
                functions,
                {m for m in models if m in assigned},
                {b for b in backends if b in assigned},
                dict(relations),
                dict(contents),
                asserted,
                ids,
            )
            for place in _certain_places(statement, names, ops):
                yield function, index, place, scope
            # What this statement binds, for the statements after it: only
            # plain assignments at the top of the test.
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and counts.get(statement.targets[0].id) == 1
            ):
                name = statement.targets[0].id
                assigned.add(name)
                value = statement.value
                access = _relation_access(value, {m for m in models if m in assigned})
                if access is not None and (isinstance(value, ast.Call) or access[0] == ("relation-ids", "relation-list")):
                    # When `relation-ids` certainly printed IDs, the
                    # assignment ran `relation-list` already.
                    needs_list = access[0] == ("relation-ids",) and not ids
                    relations[name] = (access[1], access[2], needs_list)
                else:
                    relation = _databag(value, relations, {})
                    if relation is not None and isinstance(value, ast.Subscript):
                        contents[name] = relation
            if _may_fail_an_assertion(statement, functions):
                asserted = True


def _hook_tool_accesses(tree: ast.Module) -> list[_Access] | None:
    """The model accesses in a fake-hook-tool test whose hook tools are
    known (see `_unfaked_hook_tools()`), or `None` when the file is not that
    shape or its names cannot be known."""
    if not _faked_hook_tools(tree):
        return None
    counts = _file_binding_counts(tree)
    if counts is None or _ops() is None:
        return None
    found: list[_Access] = []
    for function, index, place, scope in _scan_tests(tree):
        target = place.statement
        if not isinstance(target, _SIMPLE_STATEMENTS):
            continue
        names = scope.names
        accesses = []  # (node, tools, endpoint, relation ID)
        for node in ast.walk(target):
            access = _relation_access(node, scope.models, scope.ids)
            if access is not None:
                accesses.append((node, *access))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                backend = (isinstance(owner, ast.Name) and owner.id in scope.backends) or (
                    isinstance(owner, ast.Attribute)
                    and owner.attr == "_backend"
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id in scope.models
                )
                if backend and node.func.attr in _BACKEND_METHOD_TOOLS:
                    accesses.append((node, _BACKEND_METHOD_TOOLS[node.func.attr], None, None))
        for read, relation in _databag_reads(target, scope.relations, scope.contents, names):
            endpoint, relation_id, needs_list = scope.relations[relation]
            tools = ("relation-list", "relation-get") if needs_list else ("relation-get",)
            accesses.append((read, tools, endpoint, relation_id))
        # `model.relations['db'][0]` holds `model.relations['db']`,
        # and `dict(bag)` holds `bag[...]`: report the outer one.
        nested = {id(n) for a in accesses for n in ast.walk(a[0]) if n is not a[0]}
        asserted = scope.asserted or any(_may_fail_an_assertion(s, scope.functions) for s in place.earlier)
        for node, tools, endpoint, relation_id in accesses:
            if id(node) in nested or not _unconditional(node, target):
                continue
            first = tools[-1] if tools[0] == "relation-list" and "relation-get" in tools else tools[0]
            found.append(
                _Access(
                    function,
                    index,
                    node,
                    tools,
                    first,
                    endpoint,
                    relation_id,
                    target,
                    place.guarded,
                    place.caught,
                    asserted,
                    place.strict,
                    place.in_raises,
                    place.earlier,
                )
            )
    return found


def _databag_not_read(tree: ast.Module) -> list[str]:
    """A relation databag that is looked up and never read, or a read that
    is expected to raise `RelationNotFoundError`, where that certainly fails
    the test whether or not the bug is there.

    Live on `#2709`, two tests were posted as reproductions that fail the
    same way with the fix (`spike-step-5/static-retry/RESULT.md` §19): each
    ends with `relation.data[relation.app]` as a statement of its own, inside
    `pytest.raises(RelationNotFoundError)` or a `try` with
    `except RelationNotFoundError` and an `else` that raises. Two things
    make that certain to fail:

    - `RelationData.__getitem__()` only looks up a `RelationDataContent`,
      which loads lazily, so the statement runs no `relation-get` and raises
      nothing but a failed lookup (checked by running, on ops 3.8.3 and
      `main`: no hook tool runs, even with `relation-get` unfaked).
    - a read that loads (`_databag_reads()`) cannot raise
      `RelationNotFoundError`: `RelationDataContent._load()` catches it and
      returns `{}`, on 3.8.3 and on `main` (checked by running too).

    So the rule rejects, in a fake-hook-tool test, on a statement that
    certainly runs (`_certain_places()`):

    - a `with pytest.raises(X):` with no other manager, or a `try` whose
      handlers are `except X` (any other handler has to certainly fail the
      test) and that certainly fails if its body completes (an `else`, or a
      last statement of its body, that is `pytest.fail(...)`, `raise` or
      `assert False`), where every `X` is an `ops.ModelError` subclass, and
      whose body has only lookups (and `pass`), or, when every `X` is a
      `RelationNotFoundError` subclass, lookups and reads. Such a block can
      only end by failing the test, on every version of ops.
    - a lookup on its own in a test with nothing in it that could fail on an
      assertion, which can never fail the way a reproduction has to.

    A lookup is a statement that is only `R.data[K]`, with `R` a relation
    the test assigned at its top (`_scan_tests()`) or a relation access
    whose fakes certainly succeed (`relation-ids` printing IDs, exactly one
    for `get_relation(name)`, and `relation-list` printing unit names), and
    `K` a name, or `.app` or `.unit` of one or of such an access. A read is a
    statement whose only calls are the reads, on relations the test assigned,
    with keys like those. The rule is off when the file mentions
    `_hook_is_running` (then a read checks leadership, which runs
    `is-leader`).

    The reason says to read the databag and assert on it, and when the
    test's `relation-get` fake certainly fails (a `ModelError` on a version
    with the bug), to catch that and assert, as `_uncaught_hook_tool_error()`
    asks.
    """
    ops = _ops()
    if ops is None or any(isinstance(n, ast.Attribute) and n.attr == "_hook_is_running" for n in ast.walk(tree)):
        return []
    scripts = _fake_scripts(tree)
    helpers = _fake_helpers(tree)
    reasons = []
    for function, index, place, scope in _scan_tests(tree):
        statement = place.statement
        before = function.body[:index]
        if isinstance(statement, (ast.With, ast.Try)):
            reasons += _misread_block(statement, scope, scripts, helpers, before, ops)
        elif (
            isinstance(statement, ast.Expr)
            and _looked_up(statement.value, scope, scripts, helpers, before)
            and not any(_may_fail_an_assertion(s, scope.functions) for s in function.body)
        ):
            reasons.append(
                _not_read_reason(
                    statement.value,
                    "and nothing in the test can fail on an assertion, so it cannot show the bug",
                    _relation_get_fails(scripts, helpers, before),
                    "",
                )
            )
    return list(dict.fromkeys(reasons))


def _looked_up(node: ast.AST, scope: _Scope, scripts, helpers, before) -> bool:
    """Whether `node` is `R.data[K]`, a databag looked up and not read (see
    `_databag_not_read()`)."""
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "data"
    ):
        return False
    relation = node.value.value
    known = (isinstance(relation, ast.Name) and relation.id in scope.relations) or _quiet_relation(
        relation, scope, scripts, helpers, before
    )
    return known and _inert_key(node.slice, scope, scripts, helpers, before)


def _quiet_relation(node: ast.AST, scope: _Scope, scripts, helpers, before) -> bool:
    """A relation access (`get_relation(...)`, `relations[name][i]`) that
    certainly raises no `ops.ModelError`: each hook tool it runs is unfaked
    (`FileNotFoundError`), or faked once with a script that cannot exit
    non-zero and prints something that is not JSON (`JSONDecodeError`) or
    what ops expects (`relation-ids` printing IDs, `relation-list` printing
    unit names), and `get_relation(name)` does not get two IDs
    (`TooManyRelatedAppsError`)."""
    if not isinstance(node, (ast.Call, ast.Subscript)):
        return False
    if isinstance(node, ast.Subscript) and not isinstance(node.value, ast.Subscript):
        return False  # `relations[name]` is a list
    access = _relation_access(node, scope.models, scope.ids)
    if access is None:
        return False

    def state(tool: str, expected) -> str | None:
        if tool not in scripts:
            return "unfaked"
        if _certain_fake(scripts, helpers, before, tool, lambda script: _non_json_output(script) is not None):
            return "not JSON"
        if _certain_fake(scripts, helpers, before, tool, lambda script: expected(script) is not None):
            return "expected"
        return None

    ids = state("relation-ids", _relation_id_count)
    if ids is None:
        return False
    if ids != "expected":
        return True  # it fails there, on something that is not a `ModelError`
    tools, _, relation_id = access
    if isinstance(node, ast.Call) and relation_id is None and scope.ids is not None and scope.ids > 1:
        return False
    return "relation-list" not in tools or state("relation-list", _unit_names) is not None


def _inert_key(node: ast.AST, scope: _Scope, scripts, helpers, before) -> bool:
    """A databag key whose evaluation runs nothing that could raise a
    `ModelError`: a name, or `.app` / `.unit` of a name or of a quiet
    relation access."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute) and node.attr in ("app", "unit"):
        return _inert_key(node.value, scope, scripts, helpers, before) or _quiet_relation(
            node.value, scope, scripts, helpers, before
        )
    return False


def _relation_get_fails(scripts, helpers, before) -> int | None:
    """The exit status of the test's one `relation-get` fake, when it
    certainly fails with something other than `not found`."""
    failure = _certain_fake(scripts, helpers, before, "relation-get", _certain_failure)
    if failure is None or "not found" in failure[1].lower():
        return None
    return failure[0]


def _misread_block(block: ast.With | ast.Try, scope: _Scope, scripts, helpers, before, ops) -> list[str]:
    names = scope.names
    body = block.body
    if isinstance(block, ast.With):
        if len(block.items) != 1:
            return []
        raised = _raises_types(block.items[0], names)
        if raised is None:
            return []
        guards = [raised]
        around = f'the `pytest.raises({ast.unparse(raised)})` around it fails with "DID NOT RAISE" whether or not the bug is there'
        drop = ", outside the `pytest.raises`,"
    else:
        if not block.handlers:
            return []
        guards = [h.type for h in block.handlers if not _certainly_fails(h.body, names, handler=True)]
        if not guards or None in guards:
            return []
        if _certainly_fails(block.orelse, names):
            fails_at = block.orelse[0]
        elif _certainly_fails(body[-1:], names):
            fails_at = body[-1]
        else:
            return []
        excepts = _and_list(f"`except {ast.unparse(t)}`" for t in guards)
        around = (
            f"the {excepts} after it never runs, and the test fails on line {fails_at.lineno} "
            "whether or not the bug is there"
        )
        drop = ", in place of that `try`,"
    classes = []
    for guard in guards:
        for t in guard.elts if isinstance(guard, ast.Tuple) else [guard]:
            cls = names.resolve(t)
            if not (isinstance(cls, type) and issubclass(cls, ops.ModelError)):
                return []
            classes.append(cls)
    not_found = all(issubclass(cls, ops.RelationNotFoundError) for cls in classes)
    statements = body[:-1] if _certainly_fails(body[-1:], names) else body
    lookups, reads = [], []
    for statement in statements:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and _looked_up(statement.value, scope, scripts, helpers, before):
            lookups.append(statement.value)
            continue
        if _inert(statement):
            continue
        found = _only_reads(statement, scope, scripts, helpers, before) if not_found else None
        if not found:
            return []
        reads += found
    fails = _relation_get_fails(scripts, helpers, before)
    not_raised = (
        "; reading it would not raise `RelationNotFoundError` either: `RelationDataContent._load()` "
        "catches that and returns `{}`, in every version of ops, so a relation that is gone reads as `{}`"
        if not_found
        else ""
    )
    reasons = [_not_read_reason(node, f"and {around}", fails, drop, not_raised) for node in lookups]
    for read, bag in reads:
        reasons.append(
            f"line {read.lineno}: `{ast.unparse(read)}` cannot raise `RelationNotFoundError`: "
            "`RelationDataContent._load()` catches it and returns `{}`, in every version of ops, so a "
            f"relation that is gone reads as `{{}}`, and {around}; assert that the databag reads as "
            f"`{{}}` instead{drop.rstrip(',')}: {_read_and_assert(bag, fails)}"
        )
    return reasons


def _only_reads(statement: ast.stmt, scope: _Scope, scripts, helpers, before) -> list[tuple[ast.AST, ast.AST]] | None:
    """[(read, the databag it reads)] when `statement` is an expression, a
    plain assignment or an `assert` whose only calls are databag reads of
    relations the test assigned, with keys `_inert_key()` accepts, and that
    looks up nothing but `.data`, `.app`, `.unit` and `.get`; else `None`."""
    if isinstance(statement, ast.Assign):
        if not all(isinstance(t, ast.Name) for t in statement.targets):
            return None
    elif isinstance(statement, ast.Assert):
        if statement.msg is not None and not isinstance(statement.msg, ast.Constant):
            return None
    elif not isinstance(statement, ast.Expr):
        return None
    names = scope.names
    reads = list(_databag_reads(statement, scope.relations, scope.contents, names))
    if not reads:
        return None
    read_ids = {id(read) for read, _ in reads}
    for node in ast.walk(statement):
        if isinstance(node, ast.Call) and id(node) not in read_ids:
            return None
        if isinstance(node, ast.Attribute) and node.attr not in ("data", "app", "unit", "get"):
            return None
        if isinstance(node, (ast.NamedExpr, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom, ast.Starred)):
            return None
        if isinstance(node, ast.Subscript):
            bag = _databag(node, scope.relations, {})
            if bag is not None and not _inert_key(node.slice, scope, scripts, helpers, before):
                return None
            if bag is None and not (id(node) in read_ids):
                return None
    found = []
    for read, _ in reads:
        if isinstance(read, ast.Call):
            bag = read.func.value if isinstance(read.func, ast.Attribute) else read.args[0]
        elif isinstance(read, ast.Subscript):
            bag = read.value
        else:
            right = isinstance(read.ops[0], (ast.In, ast.NotIn)) or _databag(read.left, scope.relations, scope.contents) is None
            bag = read.comparators[0] if right else read.left
        found.append((read, bag))
    return found


def _inert(statement: ast.stmt) -> bool:
    """An expression, plain assignment or `assert` that calls nothing and
    looks nothing up (`x = 1`, `assert x == {}`), so it cannot raise an
    `ops.ModelError`."""
    if isinstance(statement, ast.Assign):
        if not all(isinstance(t, ast.Name) for t in statement.targets):
            return False
    elif not isinstance(statement, (ast.Expr, ast.Assert)):
        return False
    return not any(
        isinstance(n, (ast.Call, ast.Attribute, ast.Subscript, ast.NamedExpr, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom))
        for n in ast.walk(statement)
    )


def _read_and_assert(bag: ast.AST, fails: int | None) -> str:
    shown = ast.unparse(bag)
    if fails is None:
        return f"`assert dict({shown}) == {{}}`"
    return (
        f"the test's `relation-get` fake exits {fails}, so on a version of ops with the bug the read "
        "raises `ops.ModelError`, so catch that and assert on the result: "
        f"`try: result = dict({shown})` / `except ops.ModelError as e: result = e` / `assert result == {{}}`"
    )


def _not_read_reason(node: ast.AST, around: str, fails: int | None, drop: str, not_raised: str = "") -> str:
    shown = ast.unparse(node)
    gone = "" if not_raised else " (a relation that is gone reads as `{}`)"
    return (
        f"line {node.lineno}: `{shown}` only looks the databag up and never reads it: "
        "`RelationDataContent` loads lazily, so this runs no `relation-get` and cannot raise an "
        f"`ops.ModelError`, with or without the bug, {around}{not_raised}; read the databag instead{drop} and compare "
        f"it with what the issue says it should be{gone}: {_read_and_assert(node, fails)}"
    )


def _string_databag_key(tree: ast.Module) -> list[str]:
    """A relation databag indexed with a string, `relation.data['provider']`,
    in a fake-hook-tool test, where that certainly raises `KeyError`.

    `RelationData` is keyed by `ops.Unit` and `ops.Application` objects, and
    its `__getitem__()` is a plain dict lookup, so any string raises
    `KeyError` at the subscript, before anything is read and without running
    a hook tool (checked by running, on ops 3.8.3 and `main`, with the app's
    name, a unit's name, this charm's own, and the endpoint's). Live on
    `#2709`, two of §19's tests did this (`spike-step-5/static-retry/
    RESULT.md` §19, §20), and the rules there let both through.

    `R` is a relation the test assigned at its top (`_scan_tests()`), from
    `get_relation(name, id)`, `relations[name][i]`, or `get_relation(name)`
    once `relation-ids` certainly prints one ID, so it is a `Relation` and
    not `None` or a list; or a relation access whose fakes cannot raise a
    `ModelError` (`_quiet_relation()`), so a handler that swallows one cannot
    skip the subscript. The statement is one `_certain_places()` is sure
    runs, with every handler and `pytest.raises` around it catching only
    `ModelError` subclasses or failing the test, so the `KeyError` gets out,
    and the subscript is evaluated whenever the statement is, before any
    other call in it when a handler around it could swallow a `ModelError`.
    Only a string literal: a name, an attribute, a call or anything else is
    let through. The rule is off when the file assigns a `.data` attribute or
    calls `setattr` or anything named `patch`.

    The reason says what to index with instead: `R.app` for the remote
    application, `model.get_unit(name)` for a unit, and `model.app` /
    `model.unit` for this charm's own.
    """
    ops = _ops()
    if ops is None:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "data" and isinstance(node.ctx, ast.Store):
            return []
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if name in ("setattr", "patch") or (name == "object" and "patch" in ast.unparse(func)):
                return []
    scripts = _fake_scripts(tree)
    helpers = _fake_helpers(tree)
    reasons = []
    for function, index, place, scope in _scan_tests(tree):
        target = place.statement
        if not isinstance(target, _SIMPLE_STATEMENTS):
            continue
        before = function.body[:index]
        for node in ast.walk(target):
            if not (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "data"
            ):
                continue
            relation = node.value.value
            if isinstance(relation, ast.Name):
                if not _assigned_a_relation(relation.id, function, scope, scripts, helpers):
                    continue
            elif not _quiet_relation(relation, scope, scripts, helpers, before):
                continue
            if not _unconditional(node, target) or (place.strict and not _called_first(target, node)):
                continue
            reasons.append(_string_key_reason(node, relation, scope))
    return list(dict.fromkeys(reasons))


def _assigned_a_relation(name: str, function: ast.FunctionDef, scope: _Scope, scripts, helpers) -> bool:
    """Whether `name` is a relation `_scan_tests()` tracks that is certainly
    an `ops.Relation` once its assignment has run: from `get_relation(name,
    id)`, `relations[name][i]`, or `get_relation(name)` with `relation-ids`
    certainly printing exactly one ID by then (none gives `None`)."""
    if name not in scope.relations:
        return False
    for index, statement in enumerate(function.body):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            value = statement.value
            if isinstance(value, ast.Subscript):
                return isinstance(value.value, ast.Subscript)
            if _call_arg(value, 1, "relation_id") is not None:
                return True
            ids = _certain_fake(scripts, helpers, function.body[:index], "relation-ids", _relation_id_count)
            return ids == 1
    return False


def _string_key_reason(node: ast.Subscript, relation: ast.AST, scope: _Scope) -> str:
    key = node.slice.value
    shown = ast.unparse(relation)
    root = relation
    if isinstance(root, ast.Name):
        assign = scope.names.assignments.get(root.id)
        root = assign.value if assign is not None else root
    while isinstance(root, (ast.Subscript, ast.Attribute, ast.Call)):
        root = root.func if isinstance(root, ast.Call) else root.value
    model = root.id if isinstance(root, ast.Name) and root.id in scope.models else "model"
    own_unit = None
    call = scope.names.assignments.get(model)
    if call is not None:
        backend = _call_arg(call.value, 1, "backend") if isinstance(call.value, ast.Call) else None
        backend = scope.names.value_of(backend) if backend is not None else None
        unit = _call_arg(backend, 0, "unit_name") if isinstance(backend, ast.Call) else None
        if isinstance(unit, ast.Constant) and isinstance(unit.value, str):
            own_unit = unit.value
    remote_app = f"`{shown}.data[{shown}.app]` for the remote application's databag"
    remote_unit = f"`{shown}.data[{model}.get_unit('<unit name>')]` for a remote unit's"
    own = f"`{shown}.data[{model}.app]` or `{shown}.data[{model}.unit]` for this charm's own"
    if key == own_unit:
        use = f"`{shown}.data[{model}.unit]` for this unit's databag"
        others = [remote_app, f"`{shown}.data[{model}.app]` for this application's"]
    elif own_unit is not None and key == own_unit.split("/")[0]:
        use = f"`{shown}.data[{model}.app]` for this application's databag"
        others = [remote_app, f"`{shown}.data[{model}.unit]` for this unit's"]
    elif "/" in key:
        use = f"`{shown}.data[{model}.get_unit({key!r})]` for that unit's databag (the same object as the one in `{shown}.units`)"
        others = [remote_app, own]
    else:
        use = remote_app
        others = [remote_unit, own]
    return (
        f"line {node.lineno}: `{ast.unparse(node)}` indexes the databag with the string `{key!r}`, but "
        "`Relation.data` is keyed by `ops.Application` and `ops.Unit` objects, not by name, so this raises "
        f"`KeyError: {key!r}` at the subscript, before anything is read, with or without the bug; "
        f"index it with the object instead: {use} ({'; '.join(others)})"
    )


def _certain_json(script: str) -> object:
    """What a fake certainly prints, parsed as JSON, when its script is
    exactly `echo '<JSON>'` (single quotes, so the shell leaves it as it is,
    and no backslash, which `sh`'s `echo` would read) or `echo <word>`,
    optionally followed by `exit 0`; `_NOT_JSON` otherwise."""
    commands = [c.strip() for line in script.splitlines() for c in line.split(";") if c.strip()]
    if commands and commands[-1] == "exit 0":
        commands.pop()
    if len(commands) != 1:
        return _NOT_JSON
    quoted = re.fullmatch(r"echo\s+'([^'\\]*)'", commands[0])
    if quoted is not None:
        printed = quoted.group(1)
    else:
        echoed = _echoes(commands[0])
        if echoed is None or echoed[1]:
            return _NOT_JSON
        printed = echoed[0]
    try:
        return json.loads(printed)
    except ValueError:
        return _NOT_JSON


_NOT_JSON = object()


def _relation_id_count(script: str) -> int | None:
    """How many relation IDs a `relation-ids` fake certainly prints, in a
    form ops parses (`["db:1"]`: a JSON list of strings whose part after the
    last `:` is an integer); `None` when that is not certain."""
    printed = _certain_json(script)
    if not isinstance(printed, list) or not all(isinstance(i, str) for i in printed):
        return None
    try:
        for relation_id in printed:
            int(relation_id.split(":")[-1])
    except ValueError:
        return None
    return len(printed)


def _unit_names(script: str) -> bool | None:
    """`True` when a `relation-list` fake certainly prints a JSON list of one
    or more unit names (`["provider/0"]`), so that ops builds the relation
    from them without running `relation-list --app`."""
    printed = _certain_json(script)
    if isinstance(printed, list) and printed and all(isinstance(u, str) and "/" in u for u in printed):
        return True
    return None


def _uncaught_hook_tool_error(tree: ast.Module) -> list[str]:
    """A model access whose first hook tool's fake certainly fails, so ops
    certainly raises `ops.ModelError` there, with nothing around it that
    catches `ops.ModelError`.

    On the buggy version the test then dies on that exception rather than
    failing an assertion, and rung 1c keeps it silent: repaired by hand from
    §17's reasons, 4 of `#2709`'s 8 tests reached the bug and passed with the
    fix, and none was valid. Two read the databag with nothing around it and
    one wrapped the read in `pytest.raises(RelationNotFoundError)`, which a
    plain `ModelError` gets out of (`spike-step-5/static-retry/RESULT.md`
    §17, §18). So this also rejects a `pytest.raises(X)` or `except X`
    around the access when `X` is a strict subclass of `ops.ModelError`.

    What ops raises comes from the installed ops's `_wrap_hookcmd()` (ops
    3.8.3): a hook tool that exits non-zero is `ModelError(stderr)`, except
    `RelationNotFoundError` for a relation tool whose stderr says `relation
    not found` (which a databag read turns into `{}`). Printing `ERROR ...`
    and exiting 0 is not a failure to ops (the output goes to
    `json.loads()`), so only the exit status counts. Each row is checked by
    running it (`tests/test_hook_tool_rules.py`).

    It lets the test through unless all of these hold:

    - the access is one `_hook_tool_accesses()` is certain runs, and nothing
      before it in the test could fail on an assertion (if something can,
      the test may fail there on the buggy version and pass with the fix,
      which is a valid test whatever comes after: so no `assert`, `raise`,
      `pytest.fail` or `pytest.raises`, and no call to a function the file
      defines other than the fake helper), and it is the only call in its
      statement;
    - the first tool it runs is faked exactly once, by a statement at the top
      of the test before the access, with a literal script that is only
      `echo <literal words>` (to stdout or `>&2`) followed by `exit <1-255>`,
      and its stderr does not say `not found`;
    - when that stderr is an authorisation error and the installed ops checks
      leadership for it: the tool is not `is-leader` (which recurses on 3.8.3
      and raises `RecursionError`), and `is-leader` is faked the same way,
      once, before the access, printing `true` or `false`;
    - when the tool is `relation-get` on an application databag, or on one
      the check cannot tell: the test sets `JUJU_VERSION` once, to a literal
      version with application data, at its top before every
      `_ModelBackend(...)` in the file (ops reads it when the backend is
      built, and raises `RuntimeError` instead of running `relation-get`
      without it); a direct `relation_get()` call has to bind to the
      installed signature, with `is_app` a literal `True` or `False`.
    """
    ops = _ops()
    if ops is None or not _hook_tool_failure_raises_model_error():
        return []
    accesses = _hook_tool_accesses(tree)
    if not accesses:
        return []
    names = _Names(tree, _file_binding_counts(tree) or {})
    scripts = _fake_scripts(tree)
    helpers = _fake_helpers(tree)
    reasons = []
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for access in accesses:
        node, tool = access.node, access.first
        if access.asserted or not _calls_only(access.target, node):
            continue
        before = access.function.body[: access.index]
        # `asserted` looks one call deep; here any call into the file's own
        # functions but the fake helper might fail an assertion first.
        if any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defined - set(helpers)
            for statement in before + access.earlier
            for n in ast.walk(statement)
        ):
            continue
        failure = _certain_fake(scripts, helpers, before, tool, _certain_failure)
        if failure is None:
            continue
        code, stderr = failure
        lowered = stderr.lower()
        if "not found" in lowered:
            continue
        if _security_event_runs_is_leader() and any(m in lowered for m in _AUTHZ_MESSAGES):
            if tool == "is-leader":
                continue
            if _certain_fake(scripts, helpers, before, "is-leader", _prints_a_json_bool) is None:
                continue
        if tool == "relation-get" and not _relation_get_runs(access, tree, names, ops):
            continue
        if any(_may_catch(t, ops.ModelError, names) for t in access.caught):
            continue
        reasons.append(_uncaught_reason(access, code, stderr))
    return list(dict.fromkeys(reasons))


def _uncaught_reason(access: _Access, code: int, stderr: str) -> str:
    node, tool = access.node, access.first
    shown = ast.unparse(node)
    printed = f', printing "{stderr}" to stderr' if stderr else ""
    what = f"line {node.lineno}: `{shown}` runs `{tool}` first, and the test's `{tool}` fake exits {code}{printed}, so the installed ops raises `ops.ModelError` there"
    if access.caught:
        wrong = _and_list(f"`{ast.unparse(t)}`" for t in access.caught)
        pytest_raises = access.in_raises
        around = f"the `pytest.raises({ast.unparse(access.caught[0])})`" if pytest_raises else f"the `except` for {wrong}"
        what += f", not {wrong}, and {around} around it does not catch a plain `ModelError`"
    else:
        pytest_raises = False
        what += ", and nothing around it catches that"
    if isinstance(node, ast.Compare):
        # A databag compared with a mapping, or `key in bag`: read the bag.
        on_right = isinstance(node.ops[0], (ast.In, ast.NotIn)) or isinstance(node.left, (ast.Dict, ast.Call))
        suggested = f"dict({ast.unparse(node.comparators[0] if on_right else node.left)})"
    else:
        suggested = shown
    reason = (
        f"{what}: the test dies on that exception instead of failing an assertion, so its run "
        "cannot tell the bug from a broken test; catch `ops.ModelError` around the access and "
        f"assert on what the issue says should happen instead: `try: result = {suggested}` / "
        "`except ops.ModelError as e: result = e` / `assert result == <what the issue says it "
        "should be>`, so the buggy version fails that assertion and the fixed version passes"
    )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _BACKEND_METHOD_TOOLS:
        # Asserting on the backend call's own result can fail with the fix too:
        # ops can catch that layer's exception and return a value a layer up.
        # By hand, `#2709` run 1 asserting `{}` on `relation_get()` failed on
        # 3.8.3 and on `main`, a false reproduction (§18).
        reason += (
            f". `{ast.unparse(node.func)}()` is `_ModelBackend`, a layer below what a charm sees, "
            "and ops can catch an exception there and return a value instead, so a test of the "
            "backend call can fail with the fix as well; when the issue says what a charm should "
            "see, read that through `ops.Model` (a relation's databag, `model.config`, ...) in "
            "the same pattern and assert on it"
        )
    if pytest_raises:
        reason += (
            "; `pytest.raises` is only right when the issue says an exception should be raised, "
            "and then only with the exact class ops raises"
        )
    return reason


@functools.cache
def _hook_tool_failure_raises_model_error() -> bool:
    """Whether the installed ops turns a failed hook tool into
    `ModelError(stderr)`, except a relation tool whose stderr says `relation
    not found` (ops 3.8.3). The `#2709` fix also treats a gone relation's
    "permission denied" as `RelationNotFoundError`, so with it installed the
    rule turns itself off rather than claim the wrong class."""
    ops = _ops()
    if ops is None:
        return False
    try:
        source = inspect.getsource(ops.model._ModelBackend._wrap_hookcmd)
    except (AttributeError, OSError, TypeError):
        return False
    return (
        "raise ModelError(e.stderr) from e" in source
        and "'relation not found' in e.stderr.lower()" in source
        and "_relation_is_gone" not in source
    )


def _fake_helpers(tree: ast.Module) -> dict[str, tuple[int, int]]:
    return {
        node.name: shape
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and (shape := _fake_helper(node)) is not None
    }


def _certain_fake(scripts, helpers, before: list[ast.stmt], tool: str, read):
    """`read(script)` for `tool`'s one fake, when the file fakes it exactly
    once, with a literal script, in one of the `before` statements (a plain
    call to a recognised helper); otherwise `None`."""
    found = scripts.get(tool, [])
    if len(found) != 1 or found[0] is None:
        return None
    for statement in before:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in helpers):
            continue
        name_at, _ = helpers[call.func.id]
        if len(call.args) > name_at and isinstance(call.args[name_at], ast.Constant) and call.args[name_at].value == tool:
            return read(found[0])
    return None


_SHELL_SPECIALS = "$`|&<>*?[\\;(){}'\"#~="


def _echoes(command: str) -> tuple[str, bool] | None:
    """(what it prints, whether to stderr) for `echo <literal words>`,
    optionally with one `>&2` or `1>&2`; `None` for anything else."""
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    to_stderr = False
    for redirect in (">&2", "1>&2"):
        if redirect in words:
            if to_stderr or words.count(redirect) != 1:
                return None
            words.remove(redirect)
            to_stderr = True
    if not words or words[0] != "echo":
        return None
    if any(w.startswith("-") or any(c in w for c in _SHELL_SPECIALS) for w in words[1:]):
        return None
    return " ".join(words[1:]), to_stderr


def _certain_failure(script: str) -> tuple[int, str] | None:
    """(exit status, what it printed to stderr) for a script that is only
    `echo <literal words>` commands and then `exit <1-255>`; `None` for
    anything else."""
    commands = [c.strip() for line in script.splitlines() for c in line.split(";") if c.strip()]
    if not commands:
        return None
    leave = commands.pop().split()
    if len(leave) != 2 or leave[0] != "exit" or not leave[1].isdigit() or not 1 <= int(leave[1]) <= 255:
        return None
    stderr = []
    for command in commands:
        echoed = _echoes(command)
        if echoed is None:
            return None
        if echoed[1]:
            stderr.append(echoed[0])
    return int(leave[1]), "\n".join(stderr)


def _prints_a_json_bool(script: str) -> bool | None:
    """`True` for a script that is exactly `echo true` or `echo false`
    (optionally `; exit 0`), else `None`."""
    commands = [c.strip() for line in script.splitlines() for c in line.split(";") if c.strip()]
    if commands and commands[-1] == "exit 0":
        commands.pop()
    if len(commands) != 1:
        return None
    echoed = _echoes(commands[0])
    return True if echoed is not None and not echoed[1] and echoed[0] in ("true", "false") else None


def _relation_get_runs(access: _Access, tree: ast.Module, names: _Names, ops) -> bool:
    """Whether ops certainly gets as far as running `relation-get` for this
    access, rather than raising `RuntimeError` (application data on a Juju
    version without it, which is any when `JUJU_VERSION` is unset) or
    `TypeError` (a direct call that does not bind, or a non-bool `is_app`)."""
    node = access.node
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "relation_get":
        func = inspect.getattr_static(ops.model._ModelBackend, "relation_get", None)
        if not inspect.isfunction(func) or any(isinstance(a, ast.Starred) for a in node.args):
            return False
        if any(k.arg is None for k in node.keywords):
            return False
        try:
            bound = inspect.signature(func).bind(None, *node.args, **{k.arg: k.value for k in node.keywords})
        except TypeError:
            return False
        is_app = bound.arguments.get("is_app")
        if not (isinstance(is_app, ast.Constant) and isinstance(is_app.value, bool)):
            return False
        if not is_app.value:
            return True
    return _juju_version_set_first(access, tree, names, ops)


def _juju_version_set_first(access: _Access, tree: ast.Module, names: _Names, ops) -> bool:
    """Whether the test sets `JUJU_VERSION` to a literal version with
    application data, at its top, before every `_ModelBackend(...)` in the
    file, and nothing else in the file names `JUJU_VERSION`."""
    mentions = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "JUJU_VERSION"]
    if len(mentions) != 1:
        return False
    body = access.function.body
    set_at = None
    for index, statement in enumerate(body[: access.index]):
        value = None
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "setenv"
                and len(call.args) == 2
                and not call.keywords
                and call.args[0] is mentions[0]
            ):
                value = call.args[1]
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Subscript)
            and statement.targets[0].slice is mentions[0]
            and names.dotted(statement.targets[0].value) == "os.environ"
        ):
            value = statement.value
        if value is not None:
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                return False
            try:
                if not ops.JujuVersion(value.value).has_app_data():
                    return False
            except Exception:
                return False
            set_at = index
            break
    if set_at is None:
        return False
    after = {id(n) for statement in body[set_at + 1 : access.index + 1] for n in ast.walk(statement)}
    return all(id(n) in after for n in ast.walk(tree) if _is_backend_call(n, names))


def _backend_keywords(tree: ast.Module) -> list[str]:
    """A keyword argument a `_ModelBackend` method does not take.

    On `#2709`, one fake-hook-tool test called
    `backend.relation_get(relation_id=2, unit_name=..., app_name=False)` and
    died on the `TypeError` (`spike-step-5/static-retry/RESULT.md` §16), the
    same guess at an API as the `ops.testing` keywords `_testing_names()`
    rejects. The signature comes from the installed ops, and the reason lists
    what the method does take.

    Only on a name assigned once to `_ModelBackend(...)`, or
    `<model>._backend` for a name assigned once to `ops.Model(...)` over one,
    in a file that builds no `testing.Context` (whose backend is a different
    class). A method that is not a plain function on `_ModelBackend`, or that
    takes `**kwargs`, is not checked.
    """
    if any(_is_context_call(node) for node in ast.walk(tree)):
        return []
    counts = _file_binding_counts(tree)
    ops = _ops()
    if counts is None or ops is None:
        return []
    names = _Names(tree, counts)
    backends = _backends(names)
    models = _models(names)
    reasons = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        on_backend = (isinstance(owner, ast.Name) and owner.id in backends) or (
            isinstance(owner, ast.Attribute)
            and owner.attr == "_backend"
            and isinstance(owner.value, ast.Name)
            and owner.value.id in models
        )
        if not on_backend:
            continue
        method = node.func.attr
        accepted = _backend_parameters(method)
        if accepted is None:
            continue
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in accepted[0]:
                reasons.append(
                    f"line {node.lineno}: `{ast.unparse(node.func)}(...)` has no `{keyword.arg}` "
                    f"argument; `_ModelBackend.{method}()` takes {accepted[1]}"
                )
    return list(dict.fromkeys(reasons))


@functools.cache
def _backend_parameters(method: str) -> tuple[frozenset[str], str] | None:
    """(the keywords `_ModelBackend.<method>` accepts, them spelled out in
    order), or `None` when that is not knowable."""
    ops = _ops()
    func = inspect.getattr_static(ops.model._ModelBackend, method, None) if ops is not None else None
    if not inspect.isfunction(func):
        return None
    try:
        params = list(inspect.signature(func).parameters.values())[1:]
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    keywords = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
    if not keywords:
        return frozenset(), "no keyword arguments"
    spelled = [
        f"keyword-only `{p.name}`" if p.kind is inspect.Parameter.KEYWORD_ONLY else f"`{p.name}`" for p in keywords
    ]
    return frozenset(p.name for p in keywords), _and_list(spelled)


def _line_of(reason: str) -> int:
    head = reason.split(":", 1)[0]
    return int(head.removeprefix("line ")) if head.startswith("line ") else 0
