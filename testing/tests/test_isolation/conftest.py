# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

# The charm and dependency trees under this directory are fixtures loaded by
# subprocess workers, not test modules. Keep --doctest-modules from importing
# them (they pull in deps that only exist inside the isolated environments).
collect_ignore_glob = ['charms/*', 'deps/*']
