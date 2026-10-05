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
import inspect

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
    if not any(reason.endswith(_NO_SUCH_CONTEXT_ATTRIBUTE) for reason in result.reasons):
        return []
    allowed = context_attributes()
    if allowed is None:
        return []
    public = ", ".join(
        f"`{name}`" for name in sorted(allowed - _UNLISTED_CONTEXT_ATTRIBUTES) if not name.startswith("_")
    )
    return [
        f"The `testing.Context` attributes a test can use are: {public}. None of "
        "them is the charm or anything on it. To compare something only the charm can "
        "see (`self.charm_dir`, `self.framework`, `self.model`, `os.getcwd()` during "
        "the hook), read it in an event handler, store it in a module-level dict, and "
        "assert on the dict after `ctx.run(...)` returns."
    ]


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
    found: set[type | None] = set()
    for base in cls.__mro__:
        if base is object:
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


def _line_of(reason: str) -> int:
    head = reason.split(":", 1)[0]
    return int(head.removeprefix("line ")) if head.startswith("line ") else 0
