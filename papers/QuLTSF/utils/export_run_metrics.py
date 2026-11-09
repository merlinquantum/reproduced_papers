"""Export one row per included run to ``results/run_metrics.csv``.

The notebook and README show individual seeds from this durable file, since
raw run folders under ``outdir/`` are not committed. Runs are collected with
``utils/compare_results.py::collect_runs`` (ledger and run status both
COMPLETED, each run ID once).

Usage (from the paper folder)::

    python utils/export_run_metrics.py --sweep-dir outdir/sweep_table1_r2 ... \
        --out results/run_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "utils"))

import compare_results as cr  # noqa: E402

COLUMNS = (
    "run_id",
    "model",
    "seq_len",
    "pred_len",
    "seed",
    "sweep",
    "test_mse",
    "test_mae",
    "test_mse_drop_last",
    "test_mae_drop_last",
    "val_mse",
    "epochs_run",
    "best_epoch",
    "stop_reason",
    "wall_clock_seconds",
    "trainable_parameters",
)


def rows(runs: dict[str, dict]) -> list[dict]:
    """Flatten the collected runs into CSV rows, sorted by run ID."""
    table = []
    for run_id in sorted(runs):
        run = runs[run_id]
        metrics = run["metrics"]
        training = metrics["training"]
        table.append(
            {
                "run_id": run_id,
                "model": run["name"],
                "seq_len": run["L"],
                "pred_len": run["T"],
                "seed": run["seed"],
                "sweep": run["sweep"],
                "test_mse": repr(metrics["test"]["mse"]),
                "test_mae": repr(metrics["test"]["mae"]),
                "test_mse_drop_last": repr(metrics["test"]["mse_drop_last"]),
                "test_mae_drop_last": repr(metrics["test"]["mae_drop_last"]),
                "val_mse": repr(metrics["val"]["mse"]),
                "epochs_run": training["epochs_run"],
                "best_epoch": training.get("best_epoch", ""),
                "stop_reason": training["stop_reason"],
                "wall_clock_seconds": repr(metrics["wall_clock_seconds"]),
                "trainable_parameters": metrics["trainable_parameters"],
            }
        )
    return table


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_DIR / "results" / "run_metrics.csv"
    )
    args = parser.parse_args(argv)
    table = rows(cr.collect_runs(args.sweep_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {len(table)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
