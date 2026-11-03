"""Export a compact per-run record of every sweep into ``results/runs.json``.

``python utils/export_runs.py <sweeps_dir> <results_dir>`` reads each sweep's
``sweep_status.json`` and the runs' ``metrics.json`` and keeps, per COMPLETED run: sweep,
candidate, seed, task classes, mode, client count, parameter count, wall-clock, the
final-round test (and, when present, validation and train) metrics and the per-round
test accuracy. The notebook loads this file, because raw runs in ``outdir/`` are not
kept in Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_record(sweep: str, run: dict, metrics: dict, config: dict) -> dict:
    final = metrics["history"][-1]
    record = {
        "sweep": sweep,
        "candidate": run["candidate_name"],
        "seed": run["seed"],
        "classes": config["dataset"]["classes"],
        "mode": metrics["mode"],
        "n_clients": metrics["n_clients"],
        "model": config["model"]["name"],
        "parameters": metrics["trainable_parameters"],
        "wall_clock_seconds": metrics["wall_clock_seconds"],
        "test_accuracy": final["test_accuracy"],
        "test_auc": final["test_auc"],
        "test_f1": final["test_f1"],
        "test_accuracy_per_round": [
            entry["test_accuracy"] for entry in metrics["history"]
        ],
    }
    if "validation_accuracy" in final:
        record["validation_accuracy"] = final["validation_accuracy"]
    if "train_metrics" in metrics["final_report"]:
        record["train_accuracy"] = metrics["final_report"]["train_metrics"]["accuracy"]
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    records = []
    for sweep_dir in sorted(
        p for p in args.sweeps_dir.iterdir() if (p / "sweep_status.json").is_file()
    ):
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        for run in status["runs"]:
            if run["status"] != "COMPLETED":
                continue
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            config = json.loads((sweep_dir / run["config_path"]).read_text())
            records.append(run_record(sweep_dir.name, run, metrics, config))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "runs.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8"
    )
    print(
        f"exported {len(records)} runs from {len({r['sweep'] for r in records})} sweeps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
