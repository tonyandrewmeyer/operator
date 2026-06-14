#!/usr/bin/env python3
# Copyright 2025 Canonical Ltd.
# See LICENSE file for licensing details.

"""Beta charm: depends on confdep v2's API.

Uses NEW_NAME and compute()==2, which only exist in confdep v2.  v2 removed
LEGACY_NAME, so alpha and beta genuinely cannot share one confdep import in the
same Python process.
"""

import confdep

import ops

# v2-only attribute: removed in v1, so v1-era code breaks against this.
assert confdep.VERSION == '2.0', f'beta requires confdep v2, got {confdep.VERSION}'
_NEW = confdep.NEW_NAME


class BetaCharm(ops.CharmBase):
    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._on_any)
        framework.observe(self.on.start, self._on_any)
        framework.observe(self.on.config_changed, self._on_any)

    def _on_any(self, _event: ops.EventBase):
        self.unit.status = ops.ActiveStatus(
            f'confdep={confdep.VERSION} new={_NEW} compute={confdep.compute()}'
        )


if __name__ == '__main__':
    ops.main(BetaCharm)
