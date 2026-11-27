"""Build a sweep's result table from its structured run artifacts.

Reads ``sweep_status.json`` and each COMPLETED run's ``metrics.json`` (never
logs), checks coverage against the plan, and writes
``sweep_summary.json`` and ``sweep_summary.csv`` next to it: one row per
candidate with successful/expected run counts, mean and sample std of test
MSE/MAE (all windows and the upstream drop-last subset), validation MSE,
epochs run, wall-clock and trainable parameters.

Usage::

    python papers/QuLTSF/utils/summarise_sweep.py <sweep_dir>
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

METRICS = (
    ("test_mse", ("test", "mse")),
    ("test_mae", ("test", "mae")),
    ("test_mse_drop_last", ("test", "mse_drop_last")),
    ("test_mae_drop_last", ("test", "mae_drop_last")),
    ("val_mse", ("val", "mse")),
    ("epochs_run", ("training", "epochs_run")),
    ("wall_clock_seconds", ("wall_clock_seconds",)),
)


def _get(metrics: dict, path: tuple[str, ...]) -> float:
    value = metrics
    for key in path:
        value = value[key]
    return float(value)


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def summarise_sweep(sweep_dir: Path) -> Path:
    """Write the sweep summary files and return the JSON path.

    Parameters
    ----------
    sweep_dir : pathlib.Path
        Folder holding ``sweep_status.json``.

    Returns
    -------
    pathlib.Path
        Path of ``sweep_summary.json``.

    Raises
    ------
    ValueError
        If an included metric is non-finite or a run is listed twice.
    """
    sweep_dir = Path(sweep_dir)
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    expected = defaultdict(int)
    rows = defaultdict(list)
    seen = set()
    for run in status["runs"]:
        expected[run["candidate"]] += 1
        if run["status"] != "COMPLETED":
            continue
        if run["run_id"] in seen:
            raise ValueError(f"Duplicate run {run['run_id']}")
        seen.add(run["run_id"])
        metrics = json.loads(
            (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
        )
        values = {name: _get(metrics, path) for name, path in METRICS}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Non-finite metric in {run['metrics_path']}")
        values["trainable_parameters"] = int(metrics["trainable_parameters"])
        values["seed"] = run["seed"]
        rows[run["candidate"]].append(values)

    table = []
    for candidate in expected:
        observed = rows.get(candidate, [])
        record = {
            "candidate": candidate,
            "successful_runs": len(observed),
            "expected_runs": expected[candidate],
            "status": "COMPLETE" if len(observed) == expected[candidate] else "PARTIAL",
            "seeds": [r["seed"] for r in observed],
        }
        if observed:
            record["trainable_parameters"] = observed[0]["trainable_parameters"]
            for name, _ in METRICS:
                mean, std = _mean_std([r[name] for r in observed])
                record[f"{name}_mean"] = mean
                record[f"{name}_std"] = std
        table.append(record)

    coverage = (
        "COMPLETE SWEEP"
        if all(r["status"] == "COMPLETE" for r in table)
        else "PARTIAL SWEEP"
    )
    summary = {
        "sweep_id": status["sweep_id"],
        "coverage": coverage,
        "metric_space": "standardised (train-fitted scaler)",
        "std": "sample std over seeds (ddof=1); null with one run",
        "candidates": table,
    }
    json_path = sweep_dir / "sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fields = sorted(
        {key for record in table for key in record if key != "seeds"},
        key=lambda k: (k != "candidate", k),
    )
    with (sweep_dir / "sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    return json_path


if __name__ == "__main__":
    print(summarise_sweep(Path(sys.argv[1])))
