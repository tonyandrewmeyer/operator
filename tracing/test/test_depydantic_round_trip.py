# Copyright 2026 Canonical Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Round-trip tests for the depydantic'd vendored charm libs.

These cover the dump → load identity for the three shapes that
ops_tracing exercises, and pin the sorted-set wire format that
the dump helpers promise.
"""

from __future__ import annotations

import json

from ops_tracing.vendor.charms.certificate_transfer_interface.v1.certificate_transfer import (
    ProviderApplicationData,
)
from ops_tracing.vendor.charms.tempo_coordinator_k8s.v0.tracing import (
    ProtocolType,
    Receiver,
    TracingProviderAppData,
    TracingRequirerAppData,
    TransportProtocolType,
)


def test_tracing_requirer_app_data_round_trip():
    data = TracingRequirerAppData(receivers=['otlp_http'])
    assert TracingRequirerAppData.load(data.dump()) == data


def test_tracing_provider_app_data_round_trip_with_nested_dataclass_and_enum():
    data = TracingProviderAppData(
        receivers=[
            Receiver(
                url='http://a:4318',
                protocol=ProtocolType(name='otlp_http', type=TransportProtocolType.http),
            ),
            Receiver(
                url='grpc://b:4317',
                protocol=ProtocolType(name='otlp_grpc', type=TransportProtocolType.grpc),
            ),
        ],
    )
    assert TracingProviderAppData.load(data.dump()) == data


def test_provider_application_data_round_trip_populated_and_empty():
    populated = ProviderApplicationData(certificates={'cert-a', 'cert-b', 'cert-c'})
    empty = ProviderApplicationData()

    assert ProviderApplicationData.load(populated.dump()) == populated
    assert ProviderApplicationData.load(empty.dump()) == empty

    dumped = populated.dump()
    assert json.loads(dumped['certificates']) == ['cert-a', 'cert-b', 'cert-c']
