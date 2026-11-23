# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Measure the runner's steps and allocations on the test fixtures, and compare platforms.

Usage::

    python measure_fixtures.py measure OUT.json
    python measure_fixtures.py compare BASE.json OTHER.json [OTHER.json ...]

``measure`` runs every fixture charm in ``runner/testdata`` against every
event in ``runner/testdata/events`` with the runner that ``scriptlet_testing``
finds (the bundled one, in an installed wheel), and writes the steps and
allocations of the load and the event.

``compare`` prints a Markdown report comparing each file with the first.
Steps must be the same on every platform, so a difference in steps (or in
the outcome) fails. Allocations are starlark's estimate, and can differ
between platforms, so a difference is reported but doesn't fail.
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys
from typing import Any

TESTDATA = pathlib.Path(__file__).resolve().parent.parent / 'runner' / 'testdata'


def _used(budget: dict[str, Any] | None) -> int | None:
    return None if budget is None else budget.get('used')


def measure(out: pathlib.Path) -> None:
    """Run every fixture and write the budgets to ``out``."""
    from scriptlet_testing._process import find_runner  # Not needed by compare.

    runner = find_runner()
    version = json.loads(
        subprocess.run([runner, 'version'], capture_output=True, check=True).stdout
    )
    charms = sorted(p.parent for p in TESTDATA.glob('*/charmcraft.yaml'))
    events = sorted(TESTDATA.glob('events/*.json'))
    cases: dict[str, dict[str, Any]] = {}
    for charm in charms:
        for event in events:
            proc = subprocess.run(
                [runner, 'run', '--charm', charm, '--event', event],
                capture_output=True,
                check=True,
            )
            result: dict[str, Any] = json.loads(proc.stdout)
            load: dict[str, Any] = result.get('load') or {}
            error: dict[str, Any] = result.get('error') or {}
            cases[f'{charm.name}/{event.stem}'] = {
                'outcome': error.get('kind', 'ok'),
                'steps': _used(result.get('steps')),
                'allocs': _used(result.get('allocs')),
                'load_steps': _used(load.get('steps')),
                'load_allocs': _used(load.get('allocs')),
            }
    data = {
        'platform': f'{sys.platform}/{platform.machine()}',
        'runner': version,
        'cases': cases,
    }
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(f'{len(cases)} cases on {data["platform"]} written to {out}')


def compare(base_path: pathlib.Path, others: list[pathlib.Path]) -> int:
    """Print a Markdown comparison with the base, and return the exit status."""
    base = json.loads(base_path.read_text())
    status = 0
    print(f'## Runner budgets compared with {base["platform"]}\n')
    print(f'{len(base["cases"])} fixture cases.\n')
    for path in others:
        other = json.loads(path.read_text())
        name = other['platform']
        mismatches: list[str] = []
        alloc_diffs: list[str] = []
        for case in sorted(set(base['cases']) | set(other['cases'])):
            b = base['cases'].get(case)
            o = other['cases'].get(case)
            if b is None or o is None:
                mismatches.append(f'| `{case}` | only on {"base" if o is None else name} |')
                continue
            for key in ('outcome', 'steps', 'load_steps'):
                if b[key] != o[key]:
                    mismatches.append(f'| `{case}` | {key}: {b[key]} vs {o[key]} |')
            for key in ('allocs', 'load_allocs'):
                if b[key] != o[key]:
                    delta = (
                        f'{o[key] - b[key]:+}'
                        if isinstance(b[key], int) and isinstance(o[key], int)
                        else 'n/a'
                    )
                    alloc_diffs.append(f'| `{case}` | {key} | {b[key]} | {o[key]} | {delta} |')
        print(f'### {name}\n')
        if mismatches:
            status = 1
            print('Steps or outcomes differ (this fails the job):\n')
            print('| Case | Difference |\n|---|---|')
            print('\n'.join(mismatches) + '\n')
        else:
            print('Steps and outcomes: all the same.\n')
        if alloc_diffs:
            print(f'Allocation differences: {len(alloc_diffs)} (recorded, not a failure).\n')
            print(f'| Case | Budget | {base["platform"]} | {name} | Difference |')
            print('|---|---|---|---|---|')
            print('\n'.join(alloc_diffs) + '\n')
        else:
            print('Allocations: all the same.\n')
    return status


def main(argv: list[str]) -> int:
    """Run ``measure`` or ``compare``."""
    if len(argv) == 2 and argv[0] == 'measure':
        measure(pathlib.Path(argv[1]))
        return 0
    if len(argv) >= 3 and argv[0] == 'compare':
        return compare(pathlib.Path(argv[1]), [pathlib.Path(p) for p in argv[2:]])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
