# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.

"""Hypothesis property-based tests for scenario.state.

Pilot for the operator-hypothesis-tests investigation.  Five properties are
tested here against the State / Port / _EntityStatus / Secret data model;
see non-roadmap/operator-hypothesis-tests/STEP1-2.md for the design notes.
"""

from __future__ import annotations

import dataclasses

from hypothesis import given, settings
from hypothesis import strategies as st
from scenario.state import (
    ActiveStatus,
    BlockedStatus,
    ErrorStatus,
    ICMPPort,
    MaintenanceStatus,
    Secret,
    State,
    TCPPort,
    UDPPort,
    WaitingStatus,
    _EntityStatus,
)

# ── strategies ──────────────────────────────────────────────────────────────
#
# Strategy design is where Hypothesis pays off or doesn't (per the plan).
# The choices below are deliberately broad so the tool can discover input
# combinations the authors wouldn't reach for by hand.

# All settable status types (UnknownStatus is excluded: its __init__ takes no
# args, so _EntityStatus.from_status_name('unknown', msg) raises TypeError —
# a known limitation documented in from_status_name's own docstring).
_status_strategy = st.one_of(
    st.builds(ActiveStatus, message=st.text()),
    st.builds(BlockedStatus, message=st.text()),
    st.builds(MaintenanceStatus, message=st.text()),
    st.builds(WaitingStatus, message=st.text()),
    st.builds(ErrorStatus, message=st.text()),
)

# All three concrete Port types; TCPPort/UDPPort accept validated port numbers.
_port_strategy = st.one_of(
    st.integers(min_value=1, max_value=65535).map(TCPPort),
    st.integers(min_value=1, max_value=65535).map(UDPPort),
    st.just(ICMPPort()),
)

# Config values are typed str | int | float | bool; exclude NaN/inf to stay
# within JSON-safe territory (the same range ops itself enforces).
_config_value_strategy: st.SearchStrategy[str | int | float | bool] = st.one_of(
    st.text(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)

# Secret content must be non-empty dict[str, str]; keys must be non-empty
# so they satisfy Juju's requirement that secret keys are valid identifiers.
_secret_content_strategy = st.dictionaries(
    keys=st.text(min_size=1),
    values=st.text(),
    min_size=1,
)


# ── tests ────────────────────────────────────────────────────────────────────


@settings(max_examples=200)
@given(status=_status_strategy)
def test_entity_status_ops_roundtrip(status: _EntityStatus) -> None:
    """_EntityStatus → ops.StatusBase → _EntityStatus preserves name and message.

    This bridges the scenario and ops status representations; any mismatch in
    the name/message mapping would silently corrupt the visible status after a
    charm event.
    """
    round_tripped = _EntityStatus.from_ops(status._to_ops())
    assert round_tripped == status
    assert round_tripped.name == status.name
    assert round_tripped.message == status.message


@settings(max_examples=200)
@given(
    config=st.dictionaries(st.text(), _config_value_strategy),
    leader=st.booleans(),
    planned_units=st.integers(min_value=0, max_value=100),
    workload_version=st.text(),
)
def test_state_replace_is_idempotent(
    config: dict[str, str | int | float | bool],
    leader: bool,
    planned_units: int,
    workload_version: str,
) -> None:
    """dataclasses.replace(state) with no field overrides equals the original.

    State.__post_init__ normalises several fields (frozenset conversion, status
    coercion).  A round-trip through replace() re-runs __post_init__; if
    normalisation is not idempotent the two States diverge.
    """
    state = State(
        config=config,
        leader=leader,
        planned_units=planned_units,
        workload_version=workload_version,
    )
    assert dataclasses.replace(state) == state


@settings(max_examples=200)
@given(
    ports=st.lists(
        _port_strategy,
        max_size=5,
        # Deduplicate by (port, protocol): the same logical port appears at most
        # once, so list → frozenset and frozenset → frozenset give the same set.
        unique_by=lambda p: (p.port, p.protocol),
    ),
)
def test_state_opened_ports_list_equals_frozenset(
    ports: list[ICMPPort | TCPPort | UDPPort],
) -> None:
    """State built with opened_ports as list equals the same State rebuilt with a frozenset.

    __post_init__ normalises both inputs to frozenset.  Using dataclasses.replace
    keeps all other fields (including the randomly-generated Model) identical, so
    only the opened_ports construction path differs.
    """
    state = State(opened_ports=ports)
    state_via_frozenset = dataclasses.replace(state, opened_ports=frozenset(ports))
    assert state == state_via_frozenset


@settings(max_examples=200)
@given(port=_port_strategy)
def test_port_to_ops_preserves_attributes(port: TCPPort | UDPPort | ICMPPort) -> None:
    """Port._to_ops() faithfully copies port number and protocol to ops.Port.

    If the mapping were to swap protocol strings or clamp the port number, ops
    would open the wrong port at runtime.
    """
    ops_port = port._to_ops()
    assert ops_port.port == port.port
    assert ops_port.protocol == port.protocol


@settings(max_examples=200)
@given(tracked_content=_secret_content_strategy)
def test_secret_latest_content_defaults_to_tracked(
    tracked_content: dict[str, str],
) -> None:
    """When latest_content is not supplied, it defaults to equal tracked_content.

    This default is implemented in Secret.__post_init__ via object.__setattr__
    (bypassing the frozen dataclass guard) followed by _deepcopy_mutable_fields.
    Hypothesis explores the full space of valid dict[str, str] payloads,
    including Unicode content and multi-key secrets.
    """
    secret = Secret(tracked_content=tracked_content)
    assert secret.latest_content is not None
    assert secret.latest_content == tracked_content
