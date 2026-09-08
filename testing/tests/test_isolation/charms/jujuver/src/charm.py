#!/usr/bin/env python3
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Charm that reports the Juju version it is running against."""

import ops


class JujuVersionCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_install)

    def _on_install(self, _event: ops.EventBase):
        self.unit.status = ops.ActiveStatus(str(ops.JujuVersion.from_environ()))


if __name__ == '__main__':
    ops.main(JujuVersionCharm)
