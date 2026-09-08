# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

"""Exceptions raised by the framework.

Note that these exceptions are not meant to be caught by charm authors. They are
used by the framework to signal errors or inconsistencies in the charm tests
themselves.
"""

from __future__ import annotations


class ContextSetupError(RuntimeError):
    """Raised by Context when setup fails."""


class AlreadyEmittedError(RuntimeError):
    """Raised when ``run()`` is called more than once."""


class ScenarioRuntimeError(RuntimeError):
    """Base class for exceptions raised by the runtime module."""


class UncaughtCharmError(ScenarioRuntimeError):
    """Error raised if the charm raises while handling the event being dispatched.

    Set the ``SCENARIO_BARE_CHARM_ERRORS`` environment variable to ``1`` or ``true``
    (case insensitive) to disable wrapping charm errors with ``UncaughtCharmError``.
    """


class InconsistentScenarioError(ScenarioRuntimeError):
    """Error raised when the combination of state and event is inconsistent."""


class StateValidationError(RuntimeError):
    """Raised when individual parts of the State are inconsistent."""

    # as opposed to InconsistentScenario error where the **combination** of
    # several parts of the State are.


class MetadataNotFoundError(RuntimeError):
    """Raised when a metadata file can't be found in the provided charm root."""


class ActionMissingFromContextError(Exception):
    """Raised when the user attempts to invoke action hook commands outside an action context."""

    # This is not an ops error: in ops, you'd have to go exceptionally out of
    # your way to trigger this flow.


class NoObserverError(RuntimeError):
    """Error raised when the event being dispatched has no registered observers."""


class BadOwnerPath(RuntimeError):  # ruff: ignore[error-suffix-on-exception-name]  # Need to keep the name for backwards compatibility.
    """Error raised when the owner path does not lead to a valid ObjectEvents instance."""


class IsolationError(RuntimeError):
    """Raised when a charm event fails inside the isolated worker subprocess.

    This wraps any exception raised by the worker - either an uncaught charm
    exception or a worker-infrastructure error (for example, the charm module
    could not be imported, or the worker process crashed without producing a
    response).

    The original traceback from the worker is included in the message.
    """


class StateVersionMismatchError(RuntimeError):
    """Raised when a payload's producing ``ops.testing`` version differs from this process's.

    Every per-charm venv is required to carry the same ``ops.testing`` version
    as the parent test process; there is no cross-version negotiation. Install
    a matching ``ops.testing`` (part of the ``ops[testing]`` extra) in the
    charm's venv.
    """


class JujuError(RuntimeError):
    """Raised when an operation cannot be performed on a :class:`~ops.testing.Juju`.

    For example: removing the last unit of an application, referring to an
    application that was never deployed, removing units in a form that does
    not match the application's substrate, or a convergence loop that does not
    terminate.
    """
