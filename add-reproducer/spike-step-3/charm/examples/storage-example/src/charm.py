#!/usr/bin/env python3
"""Generated scratch charm -- do not hand-edit, re-run render.py."""

import logging

import ops

logger = logging.getLogger(__name__)


class ReproducerCharm(ops.CharmBase):
    """Scaffolding charm wired for one reproduction hypothesis."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.config_changed, self._observe)

    def _observe(self, event: ops.EventBase) -> None:
        """Surface whatever the reported bug is about, for inspection.

        Sets unit status to a short summary of what fired so `juju status`
        (or `ctx.run()`'s returned state, under ops.testing) is enough to
        compare against the hypothesis's `observed` field -- no need to
        shell into the unit for the common case.
        """
        logger.info('reproducer surface fired: %s', "config_changed")
        self.unit.status = ops.ActiveStatus("observed")


if __name__ == "__main__":
    ops.main(ReproducerCharm)
