"""confdep v2 — an intentionally API-incompatible test dependency.

v2 *removes* ``LEGACY_NAME`` (so v1-era code breaks against it) and exposes
``NEW_NAME`` plus ``compute()`` returning 2.
Used by the isolation test suite to demonstrate that two charms with
conflicting dependency versions can coexist in one test run when each runs in
its own isolated subprocess.
"""

VERSION = '2.0'
NEW_NAME = 'beta-only-name'


def compute() -> int:
    return 2
