#!/usr/bin/env python3
"""Write a sweep's execution ledger (Markdown) from its ``sweep_status.json``.

The ledger lists every planned (candidate, seed) run with its status, attempt
count, exact command and evidence paths, as required by LOG.md's sweep records.

Example (from the paper folder)::

    python utils/sweep_ledger.py --sweep-dir outdir/sweeps/e2_fig2b > results/e2_fig2b_ledger.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]


def ledger(sweep_dir: Path) -> str:
    """Return the Markdown ledger of one sweep."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    relative_dir = (
        sweep_dir.relative_to(PAPER_DIR)
        if sweep_dir.is_relative_to(PAPER_DIR)
        else sweep_dir
    )
    completed = sum(r["status"] == "COMPLETED" for r in status["runs"])
    lines = [
        f"### Sweep Execution Ledger — `{status['sweep_id']}`",
        "",
        f"Sweep dir `{relative_dir}`; status {status['status']}; {completed}/{status['expected_runs']} runs "
        f"completed; code {status['code']['commit'][:10]} (dirty={status['code']['dirty']}); "
        f"started {status['started_at']}, completed {status['completed_at']}.",
        "",
        "| Candidate | Seed | Attempts | Status | Run log | Run status | Config snapshot | Metrics |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for run in status["runs"]:
        lines.append(
            f"| {run['candidate']} | {run['seed']} | {run.get('attempts', 1)} | {run['status']} | "
            f"`{run['run_log']}` | `{run['run_status']}` | `{run['config_path']}` | `{run['metrics_path']}` |"
        )
    commands = {
        r["candidate"]: r.get("command") for r in status["runs"] if r.get("command")
    }
    lines += [
        "",
        "Exact command per candidate (seed and output folder vary per run):",
        "",
    ]
    for candidate, command in commands.items():
        lines.append(f"- `{candidate}`: `{command}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    sweep_dir = (
        args.sweep_dir if args.sweep_dir.is_absolute() else PAPER_DIR / args.sweep_dir
    )
    print(ledger(sweep_dir.resolve()), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
