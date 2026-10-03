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
    public = ", ".join(f"`{name}`" for name in sorted(allowed) if not name.startswith("_"))
    return [
        f"The public attributes of `testing.Context` are, in full: {public}. None of "
        "them is the charm or anything on it. To compare something only the charm can "
        "see (`self.charm_dir`, `self.framework`, `self.model`, `os.getcwd()` during "
        "the hook), read it in an event handler, store it in a module-level dict, and "
        "assert on the dict after `ctx.run(...)` returns."
    ]


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


def check_commands(commands: list[str]) -> StaticCheckResult | None:
    """Check the test file a heredoc in `commands[]` writes, or `None` when
    there is none (nothing to check: the synthesiser may still write one)."""
    test_file = embedded_test_file(commands)
    if test_file is None:
        return None
    return check(test_file.body, path=test_file.path)


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
                    return []
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
    return [
        f"line {line}: `{name}` is used but never imported or defined"
        for name, line in sorted(loads.items(), key=lambda item: item[1])
        if name not in bound
    ]


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
