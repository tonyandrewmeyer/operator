# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Rewrite the version strings across the repository for a release.

Called by the propose-release workflow, which has already worked out which
version is being released, and again by the create-draft-release workflow's
post-release job, which has worked out which development version the branch
goes back to. Both run it from their own checkout of the default branch, so a
maintenance branch doesn't need a copy. This script only writes files: it does
not decide the version, run git, or talk to GitHub, and it deliberately does
not run `uv lock` either, so that the lockfile update is a visible workflow
step rather than something buried in here.

The relationships between the packages' versions, and the fan-out over the
packages, are the parts of the release that are specific to this repository, so they stay
here rather than moving to the shared changelog package.

Reads nothing but the files it rewrites:

    python3 .github/release/update-release-versions.py --version 3.9.0
    python3 .github/release/update-release-versions.py --version 3.9.0.dev0 --post-release

The two differ in one thing: `--post-release` leaves the tool-versions table
alone. That table records when a major.minor was released and when it goes
out of support, and a post-release bump is neither - it is the tree saying it
is no longer the version that just shipped. Writing a row for a development
version would put a release date on a release that has not happened.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import sys

# Matches the version strings already written into the files, which may carry
# a pre-release or a dev suffix even when the version being released does not.
EXISTING_VERSION = r'(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.dev\d+)?)'

# Matches a version this script will accept, and takes it apart.
VERSION = re.compile(r'^(\d+)\.(\d+)\.(\d+)((?:a|b|rc)\d+)?(\.dev\d+)?$')

VERSION_FILES = {
    'ops/src': pathlib.Path('ops/version.py'),
    'ops/pyproject': pathlib.Path('pyproject.toml'),
    'testing/src': pathlib.Path('testing/src/scenario/_version.py'),
    'testing': pathlib.Path('testing/pyproject.toml'),
    'tracing': pathlib.Path('tracing/pyproject.toml'),
    'scriptlets': pathlib.Path('scriptlets/pyproject.toml'),
    'versions_doc': pathlib.Path('docs/explanation/versions.md'),
}

# How far ahead of ops the ops-scenario version runs. Exactly, and forever:
# ops 3.1.2 is scenario 8.1.2, and any pre-release or dev suffix is carried
# across unchanged.
SCENARIO_MAJOR_OFFSET = 5

# Which ops-testing-scriptlets minor version goes with which ops minor version.
# It's released with ops, every time, but has its own version: each ops minor
# release is a scriptlets minor release, and the patch number and any
# pre-release or dev suffix are carried across. For the pair here, ops 3.12.1
# is scriptlets 0.2.1. Change the pair when scriptlets goes to 1.0, and when
# ops changes major version.
SCRIPTLETS_BASE = ((3, 11), (0, 1))

# Where the runner's Juju release is recorded. It's empty while the runner is
# built from a Juju commit that isn't in a release.
JUJU_TAG_FILE = pathlib.Path('scriptlets/runner/main.go')
JUJU_TAG = re.compile(r'^const JujuTag = "([^"]*)"$', re.MULTILINE)


def scenario_version(ops_version: str) -> str:
    """Return the ops-scenario version that goes with an ops version."""
    match = VERSION.match(ops_version)
    if not match:
        raise ValueError(f'Not a version this script can take apart: {ops_version!r}')
    major, minor, patch, pre, dev = match.groups()
    return f'{int(major) + SCENARIO_MAJOR_OFFSET}.{minor}.{patch}{pre or ""}{dev or ""}'


def scriptlets_version(ops_version: str) -> str:
    """Return the ops-testing-scriptlets version that goes with an ops version."""
    match = VERSION.match(ops_version)
    if not match:
        raise ValueError(f'Not a version this script can take apart: {ops_version!r}')
    major, minor, patch, pre, dev = match.groups()
    (ops_major, ops_minor), (base_major, base_minor) = SCRIPTLETS_BASE
    if int(major) != ops_major or int(minor) < ops_minor:
        raise ValueError(
            f'No ops-testing-scriptlets version goes with ops {ops_version}: '
            f'update SCRIPTLETS_BASE, which starts at ops {ops_major}.{ops_minor}'
        )
    return f'{base_major}.{base_minor + int(minor) - ops_minor}.{patch}{pre or ""}{dev or ""}'


def check_juju_tag() -> None:
    """Refuse a release whose runner isn't built from a Juju release."""
    match = JUJU_TAG.search(JUJU_TAG_FILE.read_text())
    if not match:
        raise ValueError(f'No JujuTag in {JUJU_TAG_FILE}')
    if not match.group(1):
        raise ValueError(
            f'The scriptlet runner is not built from a Juju release: JujuTag is empty in '
            f'{JUJU_TAG_FILE}'
        )


def update_pyproject(path: pathlib.Path, version: str | None, deps: dict[str, str]) -> None:
    """Rewrite a pyproject.toml's pins on our own packages, and its own version if given.

    The version is None for a package whose pyproject.toml reads it from the
    package's version module.
    """
    content = path.read_text()
    updated = content
    if version is not None:
        updated = re.sub(
            rf'^version = "{EXISTING_VERSION}"$',
            f'version = "{version}"',
            updated,
            flags=re.MULTILINE,
        )
    for package, pinned in deps.items():
        updated = re.sub(rf'{package}=={EXISTING_VERSION}', f'{package}=={pinned}', updated)
    if content == updated:
        target = f'pins {deps}' if version is None else version
        raise ValueError(f'Nothing changed in {path}: is it already at {target}?')
    path.write_text(updated)
    if version is None:
        print(f'{path}: pins {deps}')
    else:
        print(f'{path}: version {version}, pins {deps}')


def update_version_module(path: pathlib.Path, version: str) -> None:
    """Rewrite the `version: str = '...'` line of a package's version module."""
    content = path.read_text()
    updated = re.sub(
        rf"^version: str = '{EXISTING_VERSION}'$",
        f"version: str = '{version}'",
        content,
        flags=re.MULTILINE,
    )
    if content == updated:
        raise ValueError(f'Nothing changed in {path}: is it already at {version}?')
    path.write_text(updated)
    print(f'{path}: version {version}')


def update_ops(ops_version: str, testing_version: str) -> None:
    """Rewrite ops/version.py and the top-level pyproject.toml."""
    update_version_module(VERSION_FILES['ops/src'], ops_version)
    update_pyproject(
        VERSION_FILES['ops/pyproject'],
        ops_version,
        deps={'ops-scenario': testing_version, 'ops-tracing': ops_version},
    )


def update_testing(ops_version: str, testing_version: str) -> None:
    """Rewrite scenario/_version.py and the ops pin in testing/pyproject.toml."""
    update_version_module(VERSION_FILES['testing/src'], testing_version)
    update_pyproject(VERSION_FILES['testing'], None, deps={'ops': ops_version})


def update_tracing(ops_version: str) -> None:
    """Rewrite tracing/pyproject.toml."""
    update_pyproject(VERSION_FILES['tracing'], ops_version, deps={'ops': ops_version})


def update_scriptlets(version: str | None) -> None:
    """Rewrite scriptlets/pyproject.toml.

    The version is None on a maintenance branch from before the package
    existed. The package doesn't pin, and isn't pinned by, the others.
    """
    path = VERSION_FILES['scriptlets']
    if version is None:
        print(f'{path}: not present, skipping')
        return
    update_pyproject(path, version, deps={})


def update_versions_doc(ops_version: str, released: datetime.date) -> None:
    """Rewrite this major version's row in the tool-versions table.

    The row carries the release date and an end-of-life date a year later.
    A missing doc is not an error: it is documentation rather than something
    the release depends on.
    """
    match = VERSION.match(ops_version)
    if not match:
        raise ValueError(f'Not a version this script can take apart: {ops_version!r}')
    major, minor = match.group(1), match.group(2)

    try:
        eol = released.replace(year=released.year + 1)
    except ValueError:
        # 29 February has no anniversary in a non-leap year.
        eol = datetime.date(released.year + 1, 3, 1)

    path = VERSION_FILES['versions_doc']
    if not path.exists():
        print(f'{path}: not present, skipping')
        return
    content = path.read_text()
    # A row can carry a label after the version, such as `Ops 2.23 (LTS)`.
    pattern = rf'(\| Ops {major})\.\d+( \([^)|]*\))? (\| [^|]+ \|) [^|]+ \| [^|]+ \|'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f'Nothing to update in {path}: is there an "Ops {major}.x" row?')
    # An LTS row's dates are the support commitment made when it became LTS,
    # not a year from the latest release, so a release doesn't move them.
    if match.group(2) == ' (LTS)':
        print(f'{path}: Ops {major} is an LTS row, leaving it alone')
        return
    prefix, label, status = match.group(1), match.group(2) or '', match.group(3)
    row = f'{prefix}.{minor}{label} {status} {released:%Y-%m-%d} | {eol:%Y-%m-%d} |'
    path.write_text(content[: match.start()] + row + content[match.end() :])
    print(f'{path}: Ops {major}.{minor}, released {released:%Y-%m-%d}, EOL {eol:%Y-%m-%d}')


def main(argv: list[str] | None = None) -> int:
    """Rewrite every version file for the release, and say what changed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--version',
        required=True,
        metavar='X.Y.Z',
        help='The ops version being released. The other versions follow from it.',
    )
    parser.add_argument(
        '--date',
        type=datetime.date.fromisoformat,
        default=None,
        metavar='YYYY-MM-DD',
        help="The release date for the versions doc. Defaults to today's date, in UTC.",
    )
    parser.add_argument(
        '--post-release',
        action='store_true',
        help='Bump back to a development version: the same fan-out, without the versions doc.',
    )
    args = parser.parse_args(argv)

    released = args.date or datetime.datetime.now(datetime.timezone.utc).date()
    try:
        testing_version = scenario_version(args.version)
        # Both of these are checked before anything is written.
        scriptlets = None
        if VERSION_FILES['scriptlets'].exists():
            scriptlets = scriptlets_version(args.version)
            if not args.post_release:
                check_juju_tag()
        update_ops(args.version, testing_version)
        update_testing(args.version, testing_version)
        update_tracing(args.version)
        update_scriptlets(scriptlets)
        if args.post_release:
            print(f'{VERSION_FILES["versions_doc"]}: post-release, leaving it alone')
        else:
            update_versions_doc(args.version, released)
    except ValueError as exc:
        print(f'update-release-versions: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
