#!/usr/bin/env python3
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Charm that reports whether it is trusted, and where its charm directory is."""

import ops


class EnvCheckCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_install)

    def _on_install(self, _event: ops.EventBase):
        try:
            self.model.get_cloud_spec()
        except ops.ModelError as e:
            trusted = 'untrusted' if 'not trusted' in str(e) else 'trusted'
        else:
            trusted = 'trusted'
        self.unit.status = ops.ActiveStatus(f'{trusted} {self.charm_dir}')


if __name__ == '__main__':
    ops.main(EnvCheckCharm)
