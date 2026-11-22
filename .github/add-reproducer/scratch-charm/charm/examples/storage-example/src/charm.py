#!/usr/bin/env python3
"""Generated scratch charm -- do not hand-edit, re-run render.py."""

import logging

import ops

logger = logging.getLogger(__name__)


class ReproducerCharm(ops.CharmBase):
    """Scaffolding charm wired for one reproduction hypothesis."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._ready)
        framework.observe(self.on.config_changed, self._observe)

    def _ready(self, event: ops.EventBase) -> None:
        """Mark the unit up and observing.

        Nothing else sets a status before the surface fires, so without
        this a runner cannot tell a charm that is ready from one still
        installing -- and stimulating too early records the event before
        the observer is listening.
        """
        self.unit.status = ops.ActiveStatus("reproducer ready")

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
