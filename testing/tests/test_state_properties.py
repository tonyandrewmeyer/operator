# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.

"""Hypothesis property-based tests for scenario.state.

Two properties are tested here against the _EntityStatus / Port conversion
glue between scenario and ops; see
non-roadmap/operator-hypothesis-tests/STEP1-2.md and STEP3-4.md in the
canonical-work-queue staging repo for the design notes and the measurement
that motivated keeping exactly these two.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from scenario.state import (
    ActiveStatus,
    BlockedStatus,
    ErrorStatus,
    ICMPPort,
    MaintenanceStatus,
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
@given(port=_port_strategy)
def test_port_to_ops_preserves_attributes(port: TCPPort | UDPPort | ICMPPort) -> None:
    """Port._to_ops() faithfully copies port number and protocol to ops.Port.

    If the mapping were to swap protocol strings or clamp the port number, ops
    would open the wrong port at runtime.
    """
    ops_port = port._to_ops()
    assert ops_port.port == port.port
    assert ops_port.protocol == port.protocol
