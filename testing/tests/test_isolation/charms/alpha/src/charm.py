#!/usr/bin/env python3
# Copyright 2025 Canonical Ltd.
# See LICENSE file for licensing details.

"""Alpha charm: depends on confdep v1's API.

Importing confdep at module level binds the v1 module in *this* process.  When
run isolated via ops.testing.IsolatedContext, this process is a separate
subprocess, so a beta charm using confdep v2 in its own subprocess cannot clash.
"""

import ops
import confdep

# v1-only attribute: this does not exist in confdep v2, so if v2 were the one
# importable here the import-time access would raise AttributeError immediately.
assert confdep.VERSION == '1.0', f'alpha requires confdep v1, got {confdep.VERSION}'
_LEGACY = confdep.LEGACY_NAME


class AlphaCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_any)
        framework.observe(self.on.start, self._on_any)
        framework.observe(self.on.config_changed, self._on_any)

    def _on_any(self, _event: ops.EventBase):
        self.unit.status = ops.ActiveStatus(
            f'confdep={confdep.VERSION} legacy={_LEGACY} compute={confdep.compute()}'
        )


if __name__ == '__main__':
    ops.main(AlphaCharm)
