"""Print the per-step `elapsed_s` table for a `--dump-run` output directory.

The same shape `spike-step-5/substrate-2026-09-04/RESULT.md` §1 records for
the two cold multipass runs, so a GHA figure can be read against them
without transcription.
"""

from __future__ import annotations

import json
import pathlib
import sys


def rows(run: dict):
    for command in run.get("commands", []):
        yield command.get("step"), command.get("exit_code"), command.get("elapsed_s")


def main() -> int:
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    dumps = sorted(out_dir.glob("*-run.json"))
    if not dumps:
        print(f"No `*-run.json` under {out_dir} -- the run did not reach the runner.")
        return 0
    for dump in dumps:
        run = json.loads(dump.read_text())
        print(f"### `{dump.name}`\n")
        print("| step | exit | elapsed |")
        print("|---|---|---|")
        total = 0.0
        for step, exit_code, elapsed in rows(run):
            elapsed = elapsed or 0.0
            total += elapsed
            print(f"| {step} | {exit_code} | {elapsed:.1f}s |")
        print(f"| **total** | | **{total:.1f}s ({total / 60:.1f}m)** |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
