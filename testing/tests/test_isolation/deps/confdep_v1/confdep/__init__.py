# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""confdep v1 — an intentionally API-incompatible test dependency.

v1 exposes ``LEGACY_NAME`` and ``compute()`` returning 1.
Used by the isolation test suite to demonstrate that two charms with
conflicting dependency versions can coexist in one test run when each runs in
its own isolated subprocess.
"""

VERSION = '1.0'  # noqa: RUF067
LEGACY_NAME = 'alpha-only-name'  # noqa: RUF067


def compute() -> int:  # noqa: RUF067
    return 1
