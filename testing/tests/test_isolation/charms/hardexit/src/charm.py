#!/usr/bin/env python3
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Hard-exit charm: kills the worker on ``start``, behaves on ``install``.

``os._exit`` terminates the interpreter immediately, bypassing the worker's
try/except.  The parent therefore sees the framed-protocol stream close with no
response — a genuine worker crash, as opposed to a charm exception (which the
worker catches and reports as an error response).
"""

import os

import ops


class HardExitCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_install)
        framework.observe(self.on.start, self._on_start)

    def _on_install(self, _event: ops.EventBase):
        self.unit.status = ops.ActiveStatus('installed ok')

    def _on_start(self, _event: ops.EventBase):
        # Hard-exit the worker process (not a catchable exception).
        os._exit(70)


if __name__ == '__main__':
    ops.main(HardExitCharm)
