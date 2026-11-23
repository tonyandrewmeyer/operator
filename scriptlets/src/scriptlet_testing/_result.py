"""Result types for the raw layer, built from the runner's JSON."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

from .errors import NotMeasuredError


@dataclasses.dataclass(frozen=True)
class Intent:
    """One intent, in Juju's ``{Type, Args}`` shape."""

    type: str
    args: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class LogEntry:
    """One ``print`` or ``debug`` call."""

    level: str  # 'print' or 'debug'
    event: str  # '<load>' during loading
    path: str
    line: int
    message: str
    juju: str  # the line Juju's log adapter writes
    juju_level: str  # 'INFO' or 'DEBUG'

    @classmethod
    def _from_json(cls, data: Mapping[str, Any]) -> LogEntry:
        return cls(
            level=str(data.get('level', '')),
            event=str(data.get('event', '')),
            path=str(data.get('path', '')),
            line=int(data.get('line', 0)),
            message=str(data.get('message', '')),
            juju=str(data.get('juju', '')),
            juju_level=str(data.get('juju_level', '')),
        )


@dataclasses.dataclass(frozen=True)
class Budget:
    """One limit, and how much of it a phase used.

    ``used`` is None when the run wasn't measured, or when the phase still
    failed at ``cap`` (the runner searches up to ``cap`` when the real limit
    was hit).
    """

    limit: int
    used: int | None
    limit_hit: bool
    measured: bool
    cap: int | None = None

    @classmethod
    def _from_json(cls, data: Mapping[str, Any] | None) -> Budget | None:
        if data is None:
            return None
        used = data.get('used')
        cap = data.get('cap')
        return cls(
            limit=int(data.get('limit', 0)),
            used=None if used is None else int(used),
            limit_hit=bool(data.get('limit_hit', False)),
            measured=bool(data.get('measured', False)),
            cap=None if cap is None else int(cap),
        )


@dataclasses.dataclass(frozen=True)
class LoadInfo:
    """Loading the sources and running every ``init()``."""

    sources: list[str]
    log_entries: list[LogEntry]
    step_budget: Budget | None
    alloc_budget: Budget | None
    cached: bool

    @property
    def logs(self) -> list[str]:
        """The load phase's ``print`` and ``debug`` messages."""
        return [e.message for e in self.log_entries]

    @classmethod
    def _from_json(cls, data: Mapping[str, Any] | None) -> LoadInfo | None:
        if data is None:
            return None
        return cls(
            sources=[str(s) for s in _items(data, 'sources')],
            log_entries=[LogEntry._from_json(e) for e in _items(data, 'logs')],
            step_budget=Budget._from_json(data.get('steps')),
            alloc_budget=Budget._from_json(data.get('allocs')),
            cached=bool(data.get('cached', False)),
        )


def _items(data: Mapping[str, Any], key: str) -> list[Any]:
    return cast('list[Any]', data.get(key) or [])


def _used(budget: Budget | None, what: str) -> int:
    if budget is None or not budget.measured:
        raise NotMeasuredError(
            f'{what} were not measured: pass measure=True to run_raw() or Context.from_charm()'
        )
    if budget.used is None:
        raise NotMeasuredError(
            f'{what} used is unknown: the run still failed at {budget.cap} (the search cap)'
        )
    return budget.used


@dataclasses.dataclass(frozen=True)
class RawResult:
    """What one event did: intents, logs, and (when measured) steps and allocations."""

    event: str
    ok: bool
    observed: bool
    intents: list[Intent]
    log_entries: list[LogEntry]
    step_budget: Budget | None
    alloc_budget: Budget | None
    load: LoadInfo | None
    raw: Mapping[str, Any] = dataclasses.field(repr=False, compare=False)

    @property
    def logs(self) -> list[str]:
        """The event's ``print`` and ``debug`` messages (load output is in ``load.logs``)."""
        return [e.message for e in self.log_entries]

    @property
    def load_logs(self) -> list[str]:
        """The load phase's messages, or an empty list if loading didn't happen."""
        return self.load.logs if self.load else []

    @property
    def measured(self) -> bool:
        """Whether this run bisected for steps and allocations."""
        return self.step_budget is not None and self.step_budget.measured

    @property
    def steps(self) -> int:
        """Steps the event used. Raises NotMeasuredError if the run wasn't measured."""
        return _used(self.step_budget, 'steps')

    @property
    def allocs(self) -> int:
        """Bytes the event allocated. Raises NotMeasuredError if the run wasn't measured."""
        return _used(self.alloc_budget, 'allocations')

    @property
    def load_steps(self) -> int:
        """Steps the load used. Raises NotMeasuredError if the run wasn't measured."""
        return _used(self.load.step_budget if self.load else None, 'load steps')

    def intents_of_type(self, type: str) -> list[Intent]:
        """The intents of one type, in order."""
        return [i for i in self.intents if i.type == type]

    @classmethod
    def _from_json(cls, data: Mapping[str, Any]) -> RawResult:
        intents: list[Intent] = []
        for i in _items(data, 'intents'):
            item = cast('dict[str, Any]', i)
            intents.append(Intent(str(item.get('type', '')), item.get('args') or {}))
        return cls(
            event=str(data.get('event', '')),
            ok=bool(data.get('ok', False)),
            observed=bool(data.get('observed', False)),
            intents=intents,
            log_entries=[LogEntry._from_json(e) for e in _items(data, 'logs')],
            step_budget=Budget._from_json(data.get('steps')),
            alloc_budget=Budget._from_json(data.get('allocs')),
            load=LoadInfo._from_json(data.get('load')),
            raw=data,
        )


@dataclasses.dataclass(frozen=True)
class Versions:
    """What the runner reports about itself."""

    protocol: int
    juju_commit: str
    juju_tag: str | None
    """The Juju release tag that ``juju_commit`` is, or None if it isn't in a release yet."""
    juju_version: str
    """The version in Juju's ``core/version/version.go`` at ``juju_commit``."""
    starform: str
    starlark: str
    go: str

    @property
    def juju_short(self) -> str:
        """The Juju commit, abbreviated."""
        return self.juju_commit[:7]

    @property
    def juju(self) -> str:
        """The Juju the runner's copies came from: the tag, or the version and commit."""
        return _juju_label(self.juju_tag, self.juju_version, self.juju_commit)

    @classmethod
    def _from_json(cls, data: Mapping[str, Any]) -> Versions:
        return cls(
            protocol=int(data.get('protocol', 0)),
            juju_commit=str(data.get('juju_commit', '')),
            juju_tag=None if data.get('juju_tag') is None else str(data['juju_tag']),
            juju_version=str(data.get('juju_version', '')),
            starform=str(data.get('starform', '')),
            starlark=str(data.get('starlark', '')),
            go=str(data.get('go', '')),
        )


def _juju_label(tag: str | None, version: str, commit: str) -> str:
    # 'v4.2.0 (234537adf971)', or '4.2-beta1 (234537adf971, untagged)' for a
    # commit that isn't in a release yet.
    if tag:
        return f'{tag} ({commit[:12]})'
    return f'{version or "?"} ({commit[:12] or "?"}, untagged)'
