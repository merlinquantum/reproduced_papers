"""Validate a finished sweep and print its per-candidate table.

``python utils/report_sweep.py <sweep_dir> --validator <validate_logging.py>`` runs the
logging validator on the sweep and every run directory, then builds the table of final
test accuracy, AUC and F1 (mean and standard deviation over seeds) from the runs'
``metrics.json`` files. The table is also saved as ``report.md`` in the sweep directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPORTED_METRICS = ("test_accuracy", "test_auc", "test_f1")


def validate(sweep_dir: Path, validator: Path, status: dict) -> tuple[int, str]:
    """Run ``validate_logging.py`` on the sweep and all of its run directories."""
    command = [
        sys.executable,
        str(validator),
        "--sweep-dir",
        str(sweep_dir),
        "--require-training",
        "--require-evaluation",
    ]
    for run in status["runs"]:
        command += ["--run-dir", str(sweep_dir / run["run_dir"])]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode, (result.stdout + result.stderr).strip()


def build_table(sweep_dir: Path, status: dict) -> tuple[str, dict]:
    """Aggregate final-round metrics per candidate; returns markdown and raw numbers."""
    grouped: dict[str, dict] = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        final = metrics["history"][-1]
        entry = grouped.setdefault(
            run["candidate_name"],
            {
                **{name: [] for name in REPORTED_METRICS},
                "parameters": metrics["trainable_parameters"],
                "seconds": [],
                "n_clients": metrics["n_clients"],
            },
        )
        for name in REPORTED_METRICS:
            entry[name].append(final[name])
        entry["seconds"].append(metrics["wall_clock_seconds"])

    def fmt(values: list[float], scale: float = 100.0) -> str:
        spread = statistics.stdev(values) if len(values) > 1 else float("nan")
        return f"{scale * statistics.fmean(values):.2f} +- {scale * spread:.2f}"

    lines = [
        "| Candidate | Clients | Parameters | Runs | Test accuracy % | Test AUC % | Test F1 % | Mean wall-clock s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in grouped.items():
        lines.append(
            f"| `{name}` | {entry['n_clients']} | {entry['parameters']} | "
            f"{len(entry['test_accuracy'])} | {fmt(entry['test_accuracy'])} | "
            f"{fmt(entry['test_auc'])} | {fmt(entry['test_f1'])} | "
            f"{statistics.fmean(entry['seconds']):.0f} |"
        )
    return "\n".join(lines), grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--validator", type=Path, required=True)
    args = parser.parse_args(argv)
    status = json.loads((args.sweep_dir / "sweep_status.json").read_text())
    completed = sum(run["status"] == "COMPLETED" for run in status["runs"])
    ledger = json.loads((args.sweep_dir / "ledger.json").read_text())["attempts"]
    failed = sum(attempt["outcome"] != "COMPLETED" for attempt in ledger)
    code, message = validate(args.sweep_dir, args.validator, status)
    table, _ = build_table(args.sweep_dir, status)
    report = (
        f"sweep {status['sweep_id']}: {status['status']}, {completed}/{len(status['runs'])} "
        f"runs completed, {len(ledger)} attempts ({failed} failed), "
        f"{status['started_at']} to {status['completed_at']}\n"
        f"logging validation exit {code}: {message}\n\n{table}\n"
    )
    (args.sweep_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
