"""Aggregate a sweep's per-run ``metrics.json`` files into a summary table.

Called by ``run_sweep.py`` at the end of a sweep, or standalone::

    python3 papers/fermi_dirac_neurons/utils/summarize_sweep.py <SWEEP_DIR>

Only structured metrics are read (never logs). For ``train`` runs, every
(config, model) pair gets the number of completed seeds and the mean, standard
deviation, median, minimum and maximum of its final metrics on each split
(full precision). Runs of other tasks are listed without aggregation.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


def _collect(sweep_dir: Path, runs: list[dict]) -> dict:
    table: dict = {}
    for run in runs:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        if metrics.get("task") != "train":
            continue
        config = run["config"]
        for model, record in metrics["models"].items():
            entry = table.setdefault(
                (config, model),
                {
                    "seeds": [],
                    "diverged_seeds": [],
                    "values": {},
                    "n_parameters": record["n_parameters"],
                },
            )
            entry["seeds"].append(run["seed"])
            # Phase 5 runs mark a model whose training went non-finite as
            # diverged; its metrics are not results and are left out.
            if record.get("diverged_at_epoch") is not None:
                entry["diverged_seeds"].append(run["seed"])
                continue
            for split, split_metrics in record["final"].items():
                for name, value in split_metrics.items():
                    entry["values"].setdefault(f"{split}_{name}", []).append(value)
    return table


def _stats(values: list[float]) -> dict:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) != len(values):
        raise ValueError("Non-finite metric in sweep results")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(sweep_dir: Path, status: dict) -> Path:
    """Write ``summary.json`` and ``summary.csv`` and return the JSON path."""
    table = _collect(sweep_dir, status["runs"])
    expected_seeds = sorted({run["seed"] for run in status["runs"]})
    rows = []
    for (config, model), entry in sorted(table.items()):
        seeds = sorted(entry["seeds"])
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds for {config}/{model}: {seeds}")
        row = {
            "config": config,
            "model": model,
            "n_parameters": entry["n_parameters"],
            "completed_seeds": len(seeds),
            "expected_seeds": len(expected_seeds),
            "coverage": "COMPLETE" if seeds == expected_seeds else "PARTIAL",
            "diverged_seeds": sorted(entry["diverged_seeds"]),
        }
        for name, values in sorted(entry["values"].items()):
            for statistic, value in _stats(values).items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)
    summary = {
        "sweep_id": status["sweep_id"],
        "expected_runs": status["expected_runs"],
        "completed_runs": sum(run["status"] == "COMPLETED" for run in status["runs"]),
        "failed_runs": [
            run["run_id"] for run in status["runs"] if run["status"] != "COMPLETED"
        ],
        "rows": rows,
    }
    json_path = sweep_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if rows:
        columns = sorted(
            {key for row in rows for key in row},
            key=lambda key: (key not in ("config", "model"), key),
        )
        with (sweep_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return json_path


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    print(
        summarize(directory, json.loads((directory / "sweep_status.json").read_text()))
    )
