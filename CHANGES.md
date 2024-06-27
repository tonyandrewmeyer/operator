# 2.15.0 - 27 Jun 2024

## Fixes

* Add connect timeout for exec websockets to avoid hanging (#1247)
* Adjust Harness secret behaviour to align with Juju (#1248)

## Tests

* Fix TypeError when running test.pebble_cli (#1245)
* Properly clean up after running setup_root_logging in test_log (#1259)
* Verify that defer() is not usable on stop,remove,secret-expired,secret-rotate (#1233)

## Documentation

* Fix HACKING.md link on PyPI, and internal links (#1261, #1236)
* Add a section to HACKING.md on PR titles (commit messages to main) (#1252)
* Add release step to update pinned charm tests (#1213)
* Add a security policy (#1266)

## CI

* Only run tests once on push to PR (#1242)
* Validate PR title against conventional commit rules in (#1262)
* Only update ops, not all dependencies, in charm tests in (#1275)
* Add artefact attestation (#1267)

# 2.14.0 - 29 May 2024

## Features

* Add a `__str__` to ActionFailed, for better unexpected failure output (#1209)

## Fixes

* The `other` argument to `RelatationDataContent.update(...)` should be optional (#1226)

## Documentation

* Use the actual emoji character rather than GitHub markup, to show properly on PyPI (#1221)
* Clarify that SecretNotFound may be raised for permission errors (#1231)

## Refactoring

* Refactor tests to pytest style (#1199, #1200, #1203, #1206)
* Use `ruff` formatter and reformat all code (#1224)
* Don't use f-strings in logging calls (#1227, 1234)

# 2.13.0 - 30 Apr 2024

## Features

* Added support for user secrets in Harness (#1176)

## Fixes

* Corrected the model config types (#1183)
* In Harness, only inspect the source file if it will be used - this fixed using Harness in a Python REPL (#1181)

## Documentation

* Updated publishing a release in HACKING.md (#1173)
* Added `tox -e docs-deps` to compile requirements.txt (#1172)
* Updated doc to note deprecated functionality in (#1178)

## Tests

* First stage of converting tests from unittest to pytest (#1191, #1192, #1196, #1193, #1195)
* Added `pebble.CheckInfo.change_id` field (#1197)

# 2.12.0 - 28 Mar 2024

## Features

* Added `Model.get_cloud_spec` which uses the `credential-get` hook tool to get details of the cloud where the model is deployed (#1152)

## Fixes

* Update Pebble Notices `get_notices` parameter name to `users=all` (previously `select=all`) (#1146)
* Warn when an observer weakref is lost (#1142)
* More robust validation of observer signatures (#1147)
* Change `Model.relation.app` type from `Application|None` to `Application` (#1151)
* Fix attaching storage in Harness before `begin` (#1150)
* Fixed an issue where `pebble.Client.exec` might leak a `socket.timeout` (`builtins.TimeoutError`) exception (#1155)
* Add a consistency check and default network to `add_relation` (#1138)
* Don't special-case `get_relation` behaviour in `leader-elected` (#1156)
* Accept `type: secret` for config options (#1167)

## Refactoring

* Refactor main.py, creating a new `_Manager` class (#1085)

## Documentation

* Use "integrate with" rather than "relate to" (#1145)
* Updated code examples in the docstring of `ops.testing` from unittest to pytest style (#1157)
* Add peer relation details in `Harness.add_relation` docstring (#1168)
* Update Read the Docs Sphinx Furo theme to use Canonical's latest styling (#1163, #1164, #1165)

# 2.11.0 - 29 Feb 2024

## Features

* `StopEvent`, `RemoveEvent`, and all `LifeCycleEvent`s are no longer deferrable, and will raise a `RuntimeError` if `defer()` is called on the event object (#1122)
* Add `ActionEvent.id`, exposing the JUJU_ACTION_UUID environment variable (#1124)
* Add support for creating `pebble.Plan` objects by passing in a `pebble.PlanDict`, the
  ability to compare two `Plan` objects with `==`, and the ability to create an empty Plan with `Plan()` (#1134)

## Fixes

* The remote app name (and its databag) is now consistently available in relation-broken events (#1130)

## Documentation

* Improve the `can_connect()` API documentation (#1123)

## Tooling

* Use ruff for linting (#1120, #1139, #1114)

# 2.10.0 - 31 Jan 2024

## Features

* Add support for Pebble Notices (`PebbleCustomNoticeEvent`, `get_notices`, and so on) (#1086, #1100)
* Add `Relation.active`, and excluded inactive relations from `Model.relations` (#1091)
* Add full support for charm metadata v2 (in particular, extended `ContainerMeta`,
  and various info links in `CharmMeta`) (#1106)
* When handling actions, print uncaught exceptions to stderr (#1087)
* Raise `ModelError` in Harness if an invalid status is set (#1107)

## Fixes

* Add Pebble log targets and checks to testing plans (#1111)
* CollectStatusEvent is now a LifecycleEvent (#1080)

## Documentation

* Update README to reflect charmcraft init changes (#1089)
* Add information on pushing locked/bind-mount files (#1094)
* Add instructions for using a custom version of ops to HACKING (#1092)

## Tooling

* Use pyproject.toml for building (#1068)
* Update to the latest version of Pyright (#1105)

# 2.9.0 - 30 Nov 2023

## Features

* Add log target support to `ops.pebble` layers and plans (#1074)
* Add `Harness.run_action()`, `testing.ActionOutput`, and `testing.ActionFailed` (#1053)

## Fixes

* Secret owners no longer auto-peek, and can use refresh, in Harness, and corrected secret access for non-leaders (#1067, #1076)
* Test suite adjustments to pass with Python 3.12 (#1081)

## Documentation

* Refresh README (#1052)
* Clarify how custom events are emitted (#1072)
* Fix the `Harness.get_filesystem_root` example (#1065)

# 2.8.0 - 25 Oct 2023

## Features

* Add `Unit.reboot()` and `Harness.reboot_count` (#1041)
* Add `RelationMeta.optional` (#1038)
* Raise a clearer exception when the Pebble socket is missing (#1049)

## Fixes

* The type of a `Handle`'s `key` was expanded from `str` to `str|None`
* Narrow types of `app` and `unit` in relation events to exclude `None` where applicable
* `push_path` and `pull_path` now include empty directories (#1024)
* Harness's `evaluate_status` resets collected statuses (#1048)

## Documentation

* Notes that status changes are immediate (#1029)
* Clarifies `set_results` maximum size (#1047)
* Expands documentation on when exceptions may be raised (#1044)
* Makes `pebble.Client.remove_path` and `Container.remove_path` docs consistent (#1031)

## Tooling

* Adds type hinting across the test suite (#1017, #1015, #1022, #1023, #1025, #1028, #1030, #1018, #1034, #1032)

# 2.7.0 - 29 Sept 2023

## Features

* Adds Unit.set_ports() (#1005)
* Type checks now allow comparing a `JujuVersion` to a `str`
* Rename `OpenPort` to `Port` (`OpenPort` remains as an alias)

## Documentation

* Reduces the amount of detail in open/close port methods (#1006)
* Removes you/your from docstrings (#1003)
* Minor improvements to HACKING (#1016)

## Tooling

* Extends the use of type hints in the test suite (#1008, #1009, #1011, #1012, #1013, #1014, #1004)
