# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit tests for the event half of the isolated-worker wire codec."""

from __future__ import annotations

import json
import re

import pytest
from scenario import Context, Relation
from scenario._isolated_serde import decode_event, encode_event
from scenario.errors import StateVersionMismatchError

import ops
import ops.version


class _Charm(ops.CharmBase):
    pass


def _event(name: str = 'install', **kwargs: object):
    ctx = Context(_Charm, meta={'name': 'ec', 'requires': {'db': {'interface': 'db'}}})
    return getattr(ctx.on, name)(**kwargs)


def test_plain_event_round_trips():
    event = _event()
    assert decode_event(encode_event(event)) == event


def test_event_carrying_a_relation_round_trips():
    relation = Relation(endpoint='db', remote_app_name='pg')
    event = _event('relation_changed', relation=relation)
    decoded = decode_event(encode_event(event))
    assert decoded == event
    assert decoded.relation == relation


def test_payload_stamps_the_running_version():
    data = json.loads(encode_event(_event()))
    assert data['ops_testing_version'] == ops.version.version


def test_mismatched_version_raises():
    payload = json.dumps({'ops_testing_version': '0.0.0-does-not-exist', 'event': {}})
    with pytest.raises(StateVersionMismatchError, match=re.escape('0.0.0-does-not-exist')):
        decode_event(payload)


def test_mismatched_version_names_both_versions():
    payload = json.dumps({'ops_testing_version': '0.0.0-does-not-exist', 'event': {}})
    with pytest.raises(StateVersionMismatchError, match=re.escape(ops.version.version)):
        decode_event(payload)


def test_payload_that_is_not_an_event_raises():
    payload = json.dumps({'ops_testing_version': ops.version.version, 'event': 'not-an-event'})
    with pytest.raises(TypeError, match='not an _Event'):
        decode_event(payload)
