# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit-test Juju scriptlet (unitless) charms with Juju's own executor code.

The raw layer runs an event by name with a dict of attributes,
because Juju hasn't defined event payloads yet.
"""

from ._context import Context
from ._result import Budget, Intent, LoadInfo, LogEntry, RawResult, Versions
from .errors import (
    KIND_TO_EXCEPTION,
    AllocLimitError,
    BuiltinArgumentError,
    InterpreterPanicError,
    InvalidCharmError,
    InvalidEventError,
    InvalidInputError,
    LoadPathError,
    NotMeasuredError,
    Position,
    RunnerError,
    RunnerInternalError,
    ScriptletError,
    ScriptletRuntimeError,
    ScriptletSyntaxError,
    ScriptletTestingError,
    StepLimitError,
)

__all__ = [
    'KIND_TO_EXCEPTION',
    'AllocLimitError',
    'Budget',
    'BuiltinArgumentError',
    'Context',
    'Intent',
    'InterpreterPanicError',
    'InvalidCharmError',
    'InvalidEventError',
    'InvalidInputError',
    'LoadInfo',
    'LoadPathError',
    'LogEntry',
    'NotMeasuredError',
    'Position',
    'RawResult',
    'RunnerError',
    'RunnerInternalError',
    'ScriptletError',
    'ScriptletRuntimeError',
    'ScriptletSyntaxError',
    'ScriptletTestingError',
    'StepLimitError',
    'Versions',
]
