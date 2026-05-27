"""confdep v1 — an intentionally API-incompatible test dependency.

v1 exposes ``LEGACY_NAME`` and ``compute()`` returning 1.
Used by the isolation test suite to demonstrate that two charms with
conflicting dependency versions can coexist in one test run when each runs in
its own isolated subprocess.
"""

VERSION = '1.0'
LEGACY_NAME = 'alpha-only-name'


def compute() -> int:
    return 1
