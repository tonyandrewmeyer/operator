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
#
# The integration tests use the Jubilant library and the pytest-jubilant plugin.
# See https://canonical.com/juju/docs/ops/latest/howto/write-integration-tests-for-a-charm/
#
# pytest-jubilant provides a module-scoped `juju` fixture that creates a temporary Juju model.
# The `charm` fixture is defined in conftest.py.

import logging
import pathlib

import jubilant
import pytest
import yaml

logger = logging.getLogger(__name__)

METADATA = yaml.safe_load(pathlib.Path("charmcraft.yaml").read_text())
APP_NAME = METADATA["name"]


@pytest.mark.juju_setup
def test_deploy(charm: pathlib.Path, juju: jubilant.Juju):
    """Deploy the charm under test."""
    resources = {
        "demo-server-image": METADATA["resources"]["demo-server-image"]["upstream-source"]
    }
    juju.deploy(charm, app=APP_NAME, resources=resources)
    juju.wait(jubilant.all_active)


def test_workload_version_is_set(charm: pathlib.Path, juju: jubilant.Juju):
    """Verify that the workload version has been set."""
    version = juju.status().apps[APP_NAME].version
    assert version == "1.0.4"  # Hardcoded for simplicity.


# The tests below extend beyond the tutorial scope, but demonstrate how to
# exercise the `server-port` config option added in the "Make your charm
# configurable" chapter from integration tests. See the "How to write
# integration tests for a charm" guide for background.


def test_configure_server_port(charm: pathlib.Path, juju: jubilant.Juju):
    """Setting a non-default `server-port` should leave the charm active."""
    juju.config(APP_NAME, {"server-port": 8080})
    juju.wait(jubilant.all_active)
    juju.config(APP_NAME, reset="server-port")
    juju.wait(jubilant.all_active)


def test_reserved_server_port_blocks(charm: pathlib.Path, juju: jubilant.Juju):
    """The charm rejects port 22 (reserved for SSH) and reports blocked."""
    juju.config(APP_NAME, {"server-port": 22})
    juju.wait(jubilant.all_blocked)
    juju.config(APP_NAME, reset="server-port")
    juju.wait(jubilant.all_active)
