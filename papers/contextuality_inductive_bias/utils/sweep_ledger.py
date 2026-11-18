"""Print a Markdown execution ledger and coverage check for finished sweeps.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/sweep_ledger.py <sweep_dir> [<sweep_dir> ...]

For each sweep, the coverage line compares the planned candidate x repetition
set with the COMPLETED runs and reports duplicates, failures and retries. The
per-run table lists status, attempts, elapsed time, the run directory and the
metric file, all read from ``sweep_status.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ledger(sweep_dir: Path) -> str:
    """Return the Markdown ledger of one sweep."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    runs = status["runs"]
    planned = {(json.dumps(r["grid_point"], sort_keys=True), r["seed"]) for r in runs}
    completed = [
        (json.dumps(r["grid_point"], sort_keys=True), r["seed"])
        for r in runs
        if r["status"] == "COMPLETED"
    ]
    duplicates = len(completed) - len(set(completed))
    coverage = (
        "COMPLETE SWEEP"
        if set(completed) == planned
        and not duplicates
        and len(planned) == status["expected_runs"]
        else "PARTIAL SWEEP"
    )
    code = status["code"]
    lines = [
        f"#### `{status['sweep_id']}`",
        "",
        f"- Status {status['status']}; {status['started_at']} to {status['completed_at']}; "
        f"commit {code['commit'][:10]} (dirty={code['dirty']}); paper_code_sha256 {code['paper_code_sha256'][:12]}"
        + (
            f" -> {code['paper_code_sha256_end'][:12]}"
            if "paper_code_sha256_end" in code
            else " (unchanged)"
        ),
        f"- Coverage: {coverage}. Planned {len(planned)}, expected {status['expected_runs']}, completed {len(completed)}, "
        f"duplicates {duplicates}, failed {sum(r['status'] == 'FAILED' for r in runs)}, "
        f"runs with a retry {sum(len(r['attempts']) > 1 for r in runs)}",
        f"- Summary: `{status['summary_path']}`. Command: `{runs[0]['attempts'][-1]['command'].split(' --run-id')[0]} ...`",
        "",
        "| Run ID | Candidate | Seed | Attempts | Status | Elapsed (s) | Run dir |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for run in runs:
        last = run["attempts"][-1] if run["attempts"] else {}
        lines.append(
            f"| {run['run_id']} | `{json.dumps(run['grid_point'], sort_keys=True)}` | {run['seed']} | {len(run['attempts'])} | "
            f"{run['status']} | {last.get('elapsed_s', '')} | `{run['run_dir']}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    for sweep_dir in args.sweep_dirs:
        print(ledger(sweep_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
