#!/usr/bin/env -S uv run --script --no-project
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///

# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Generate testing/src/scenario/_charmcraft_extensions.py.

For each charmcraft extension profile (django-framework, fastapi-framework, etc.),
and each base that the extension supports, this script runs `charmcraft init` and
`charmcraft expand-extensions` in a temp directory, then extracts the metadata,
config, and actions that the extension adds.

Charmcraft can expand the same extension differently depending on the charm's base
(for example, the 12-factor extensions have a V1 for Ubuntu 22.04 and 24.04 and a V2
for Ubuntu 26.04). Bases that expand identically are grouped into one variant, keyed
by the oldest base in the group.

The output module contains:
  BASES:    dict[str, dict[str, str]]             - each supported base, mapped to its variant
  METADATA: dict[str, dict[str, dict]]            - containers, peers, provides, requires,
                                                    resources, assumes, per variant
  CONFIG:   dict[str, dict[str, dict]]            - config options added by each variant
  ACTIONS:  dict[str, dict[str, dict]]            - actions added by each variant
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import yaml

# Keys from the expanded YAML that belong to "metadata" (the charm's metadata.yaml equivalent).
METADATA_KEYS = {
    'assumes',
    'containers',
    'peers',
    'provides',
    'requires',
    'resources',
}

# Keys that are user-provided or build-related (not extension-contributed metadata).
IGNORED_KEYS = {
    'name',
    'summary',
    'description',
    'type',
    'bases',
    'base',
    'platforms',
    'extensions',
    'parts',
    'charm-libs',
}

OUTPUT_FILE = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'testing'
    / 'src'
    / 'scenario'
    / '_charmcraft_extensions.py'
)


def get_extensions() -> dict[str, list[str]]:
    """Get the available charmcraft extensions and the bases each one supports.

    Uses ``charmcraft list-extensions``. Experimental bases are included, because
    charmcraft will still expand the extension for them.
    """
    result = subprocess.check_output(
        ['charmcraft', 'list-extensions', '--format', 'json'], text=True
    )
    extensions: dict[str, list[str]] = {}
    for ext in json.loads(result):
        bases = [*ext['bases'], *ext['experimental_bases']]
        extensions[ext['name']] = sorted(set(bases), key=base_sort_key)
    return dict(sorted(extensions.items()))


def base_sort_key(base: str) -> tuple[str, tuple[int, ...]]:
    """Sort bases like ``ubuntu@22.04`` by name and then numerically by channel."""
    name, _, channel = base.partition('@')
    return name, tuple(int(part) for part in channel.split('.') if part.isdigit())


def run_charmcraft(profile: str, base: str, workdir: pathlib.Path) -> dict[str, Any]:
    """Run charmcraft init + expand-extensions and return the expanded YAML as a dict.

    Not every supported base has its own init template (for example, there is no
    22.04 template for the extensions that support 22.04), so if init fails for the
    base, use the default template. Either way, set the base in charmcraft.yaml, since
    that is what charmcraft uses to decide how to expand the extension.
    """
    init = ['charmcraft', 'init', '--profile', profile, '--name', 'test-charm']
    try:
        subprocess.check_call([*init, '--base', base], cwd=workdir)
    except subprocess.CalledProcessError:
        subprocess.check_call(init, cwd=workdir)
    charmcraft_yaml = workdir / 'charmcraft.yaml'
    project = yaml.safe_load(charmcraft_yaml.read_text())
    project.pop('bases', None)
    project['base'] = base
    charmcraft_yaml.write_text(yaml.safe_dump(project))
    output = subprocess.check_output(
        ['charmcraft', 'expand-extensions'],
        cwd=workdir,
        text=True,
    )
    return yaml.safe_load(output)


def extract_extension_data(
    expanded: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract metadata, config options, and actions from expanded charmcraft YAML."""
    metadata = {key: expanded[key] for key in METADATA_KEYS if key in expanded}
    config = expanded.get('config', {}).get('options', {})
    actions: dict[str, Any] = expanded.get('actions', {})

    return metadata, config, actions


_ExtensionData = tuple[dict[str, Any], dict[str, Any], dict[str, Any]]


def group_variants(
    per_base: dict[str, _ExtensionData],
) -> tuple[dict[str, str], dict[str, _ExtensionData]]:
    """Group bases that expand identically, keying each group by its oldest base.

    Returns a mapping of base to variant key, and a mapping of variant key to data.
    """
    bases: dict[str, str] = {}
    variants: dict[str, _ExtensionData] = {}
    for base in sorted(per_base, key=base_sort_key):
        data = per_base[base]
        for key, existing in variants.items():
            if existing == data:
                bases[base] = key
                break
        else:
            bases[base] = base
            variants[base] = data
    return bases, variants


def generate_module(
    all_data: dict[str, dict[str, _ExtensionData]],
) -> str:
    """Generate the Python module source code."""
    lines = [
        '# Copyright 2026 Canonical Ltd.',
        '# See LICENSE file for licensing details.',
        '"""Charmcraft extension metadata, config, and actions.',
        '',
        'Auto-generated by .github/generate_charmcraft_extensions.py',
        'Do not edit manually.',
        '"""',
        '',
        'from __future__ import annotations',
        '',
        'from typing import Any, TypedDict',
        '',
        '',
        'class _ExtensionMetadata(TypedDict, total=False):',
        '    assumes: list[str]',
        '    containers: dict[str, Any]',
        '    peers: dict[str, Any]',
        '    provides: dict[str, Any]',
        '    requires: dict[str, Any]',
        '    resources: dict[str, Any]',
        '',
        '',
    ]

    # Build the top-level dicts, pre-sorting keys.
    base_entries: dict[str, dict[str, str]] = {}
    metadata_entries: dict[str, dict[str, dict[str, Any]]] = {}
    config_entries: dict[str, dict[str, dict[str, Any]]] = {}
    action_entries: dict[str, dict[str, dict[str, Any]]] = {}
    for profile, per_base in sorted(all_data.items()):
        bases, variants = group_variants(per_base)
        base_entries[profile] = bases
        metadata_entries[profile] = {}
        config_entries[profile] = {}
        action_entries[profile] = {}
        for variant, (metadata, config, actions) in variants.items():
            metadata_entries[profile][variant] = {k: metadata[k] for k in sorted(metadata)}
            config_entries[profile][variant] = {k: config[k] for k in sorted(config)}
            action_entries[profile][variant] = {k: actions[k] for k in sorted(actions)}

    for var_name, data, type_str, comment in [
        (
            'BASES',
            base_entries,
            'dict[str, dict[str, str]]',
            'The bases each charmcraft extension supports, oldest first, mapped to the\n'
            '# variant (keyed by its oldest base) that charmcraft expands for that base.',
        ),
        (
            'METADATA',
            metadata_entries,
            'dict[str, dict[str, _ExtensionMetadata]]',
            'Metadata added by each variant of each charmcraft extension.',
        ),
        (
            'CONFIG',
            config_entries,
            'dict[str, dict[str, dict[str, Any]]]',
            'Config options added by each variant of each charmcraft extension.',
        ),
        (
            'ACTIONS',
            action_entries,
            'dict[str, dict[str, dict[str, Any]]]',
            'Actions added by each variant of each charmcraft extension.',
        ),
    ]:
        lines.append(f'# {comment}')
        lines.append(f'{var_name}: {type_str} = {data!r}')
        lines.append('')

    return '\n'.join(lines)


def main() -> int:  # ruff: ignore[undocumented-public-function]
    extensions = get_extensions()
    print(f'Found extensions: {", ".join(extensions)}')

    all_data: dict[str, dict[str, _ExtensionData]] = {}

    for profile, bases in extensions.items():
        all_data[profile] = {}
        for base in bases:
            print(f'Processing {profile} on {base}...')
            with tempfile.TemporaryDirectory() as tmpdir:
                expanded = run_charmcraft(profile, base, pathlib.Path(tmpdir))
                all_data[profile][base] = extract_extension_data(expanded)

    module_source = generate_module(all_data)
    OUTPUT_FILE.write_text(module_source)
    print(f'Written to {OUTPUT_FILE}')

    print('Running tox -e format...')
    subprocess.run(
        ['tox', '-e', 'format', '--', str(OUTPUT_FILE)],
        cwd=OUTPUT_FILE.parent,
        check=True,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
