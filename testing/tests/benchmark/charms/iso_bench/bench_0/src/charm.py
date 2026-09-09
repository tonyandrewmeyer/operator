#!/usr/bin/env python3
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Benchmark charm 0: a modest, representative per-event reconcile."""

import ops


class BenchCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        for event in (
            self.on.install,
            self.on.start,
            self.on.config_changed,
            self.on.update_status,
        ):
            framework.observe(event, self._reconcile)

    def _reconcile(self, event: ops.EventBase):
        level = self.config.get('log-level', 'info')
        self.unit.set_workload_version('1.0')
        self.unit.status = ops.ActiveStatus(f'{event.handle.kind} level={level}')


if __name__ == '__main__':
    ops.main(BenchCharm)
