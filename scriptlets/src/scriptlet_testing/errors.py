# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Exceptions raised by the raw layer.

Each runner error kind maps to one class, through the single table
``KIND_TO_EXCEPTION``. The class names follow PEP 8 (an ``Error`` suffix);
the kind strings are the runner's own and don't change with them.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ._result import RawResult


@dataclasses.dataclass(frozen=True)
class Position:
    """A location in a scriptlet source, as a load path (``lib/checks.star``)."""

    path: str
    line: int
    col: int

    def __str__(self) -> str:
        return f'{self.path}:{self.line}:{self.col}'


class ScriptletTestingError(Exception):
    """Base class for everything this package raises."""


class RunnerError(ScriptletTestingError):
    """The runner process failed: it couldn't start, exited, or broke the protocol."""


class NotMeasuredError(ScriptletTestingError):
    """Steps or allocations were read from a result that wasn't measured."""


class ScriptletError(ScriptletTestingError):
    """A structured error reported by the runner.

    ``kind`` is the runner's error kind, and ``phase`` one of ``input``,
    ``charm``, ``load`` or ``event``. ``result`` is the full result when the
    error came from a run, so a test can still read its logs and budgets.
    """

    kind: str
    phase: str
    message: str
    juju_log: str
    backtrace: str
    position: Position | None
    builtin: str
    result: RawResult | None

    def __init__(
        self,
        message: str,
        *,
        kind: str = '',
        phase: str = '',
        juju_log: str = '',
        backtrace: str = '',
        position: Position | None = None,
        builtin: str = '',
        result: RawResult | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.phase = phase
        self.message = message
        self.juju_log = juju_log
        self.backtrace = backtrace
        self.position = position
        self.builtin = builtin
        self.result = result

    def __str__(self) -> str:
        where = f' at {self.position}' if self.position else ''
        text = f'{self.kind or "error"} ({self.phase or "?"} phase){where}: {self.message}'
        if self.backtrace:
            text += '\n' + self.backtrace
        return text


class ScriptletSyntaxError(ScriptletError):
    """A parse or resolve error, including dialect violations such as ``while``."""


class ScriptletRuntimeError(ScriptletError):
    """A Starlark runtime error in top-level code, ``init()`` or a handler."""


class BuiltinArgumentError(ScriptletError):
    """A builtin such as ``juju.set_status`` rejected its arguments."""


class StepLimitError(ScriptletError):
    """The load or the event ran out of steps."""


class AllocLimitError(ScriptletError):
    """The load or the event ran out of allocation budget."""


class LoadPathError(ScriptletError):
    """An invalid scriptlet file name or ``load()`` path, or a missing file."""


class InvalidEventError(ScriptletError):
    """An event name starform rejects (too short, too long, invalid characters)."""


class InvalidCharmError(ScriptletError):
    """A charm Juju would refuse, or fail to start a runner for."""


class InvalidInputError(ScriptletError):
    """Input the runner couldn't use: a bad path, attributes out of range, a bad request."""


class InterpreterPanicError(ScriptletError):
    """A panic inside starform or Starlark, recovered by the runner (Juju doesn't)."""


class RunnerInternalError(ScriptletError, RunnerError):
    """An error the runner couldn't classify.

    Also a RunnerError, so ``except RunnerError`` catches every failure that is
    the runner's fault rather than the charm's.
    """


# The single mapping from the runner's error kinds to exception classes. A
# kind not in the table raises the base ScriptletError, so a newer runner
# doesn't break an older package.
KIND_TO_EXCEPTION: dict[str, type[ScriptletError]] = {
    'syntax_error': ScriptletSyntaxError,
    'runtime_error': ScriptletRuntimeError,
    'builtin_argument_error': BuiltinArgumentError,
    'step_limit': StepLimitError,
    'alloc_limit': AllocLimitError,
    'load_path_error': LoadPathError,
    'invalid_event': InvalidEventError,
    'charm_invalid': InvalidCharmError,
    'invalid_input': InvalidInputError,
    'interpreter_panic': InterpreterPanicError,
    'internal': RunnerInternalError,
}


def error_from_json(data: dict[str, Any], result: RawResult | None = None) -> ScriptletError:
    """Build the exception for one runner ``error`` object."""
    kind = str(data.get('kind', ''))
    cls = KIND_TO_EXCEPTION.get(kind, ScriptletError)
    pos = cast('dict[str, Any] | None', data.get('position'))
    position = None
    if pos:
        position = Position(
            path=str(pos.get('path', '')), line=int(pos.get('line', 0)), col=int(pos.get('col', 0))
        )
    return cls(
        str(data.get('message', '')),
        kind=kind,
        phase=str(data.get('phase', '')),
        juju_log=str(data.get('juju_log', '')),
        backtrace=str(data.get('backtrace', '')),
        position=position,
        builtin=str(data.get('builtin', '')),
        result=result,
    )
