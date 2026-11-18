"""Build a sweep's result table programmatically from structured metrics.

Usage::

    python3 papers/contextuality_inductive_bias/utils/summarize_sweep.py <sweep_dir>/sweep_status.json

Reads ``metrics.json`` of every COMPLETED run listed in ``sweep_status.json``,
groups runs by planned grid point, and writes ``summary.json`` and
``summary.csv`` next to the status file. For each candidate and each metric in
``spec.plan.metrics`` (dotted paths into ``metrics.json``, e.g.
``final.test_kl``) it reports the successful/expected run counts, mean,
sample standard deviation (``ddof = 1``; 0 for a single run), minimum and
maximum, all at full precision. Sweeps here are descriptive (models x
protocols x seeds, or L2 strengths); no winner is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _key(grid_point: dict) -> str:
    return json.dumps(grid_point, sort_keys=True)


def metric_value(metrics: dict, dotted_name: str):
    """Look up ``"a.b.c"`` in nested ``metrics``; ``None`` if absent."""
    value = metrics
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def summarize(status_path: Path, status: dict | None = None) -> Path:
    """Write ``summary.json``/``summary.csv`` and return the JSON path.

    Raises
    ------
    ValueError
        If a completed run's metric is missing or non-finite, or a planned
        run ID is duplicated.
    """
    status_path = Path(status_path)
    sweep_dir = status_path.parent
    status = status if status is not None else json.loads(status_path.read_text())
    plan = status["spec"]["plan"]
    metric_names = plan["metrics"]
    run_ids = [run["run_id"] for run in status["runs"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run IDs in sweep status")
    expected_repetitions = len(status["spec"]["repetitions"])

    groups: dict[str, dict] = {}
    for run in status["runs"]:
        key = _key(run["grid_point"])
        group = groups.setdefault(
            key, {"grid_point": run["grid_point"], "runs": [], "missing": []}
        )
        if run["status"] != "COMPLETED":
            group["missing"].append({"run_id": run["run_id"], "status": run["status"]})
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        values = {}
        for name in metric_names:
            value = metric_value(metrics, name)
            if value is None or isinstance(value, (list, dict)):
                raise ValueError(
                    f"{run['run_id']}: metric {name!r} missing or not scalar"
                )
            values[name] = float(value)
        non_finite = [
            name for name, value in values.items() if not math.isfinite(value)
        ]
        if non_finite:
            # A completed run whose metric is inf/nan (for example a HardTanh
            # surrogate giving probability 0 to a validation label) is kept
            # visible as INVALID and excluded from the aggregates.
            group["missing"].append(
                {
                    "run_id": run["run_id"],
                    "status": f"INVALID (non-finite {', '.join(non_finite)})",
                }
            )
            continue
        group["runs"].append(
            {
                "run_id": run["run_id"],
                "seed": run["seed"],
                "repetition": run["repetition"],
                **values,
            }
        )

    candidates = []
    for group in groups.values():
        entry = {
            "grid_point": group["grid_point"],
            "successful_runs": len(group["runs"]),
            "expected_runs": expected_repetitions,
            "status": "COMPLETE"
            if len(group["runs"]) == expected_repetitions
            else "PARTIAL",
            "missing": group["missing"],
            "runs": group["runs"],
            "aggregates": {},
        }
        for name in metric_names:
            values = np.array([run[name] for run in group["runs"]], dtype=float)
            entry["aggregates"][name] = (
                {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n": int(values.size),
                }
                if values.size
                else None
            )
        candidates.append(entry)

    complete = (
        all(c["status"] == "COMPLETE" for c in candidates)
        and len(candidates) * expected_repetitions == status["expected_runs"]
    )
    summary = {
        "sweep_id": status["sweep_id"],
        "coverage": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "expected_runs": status["expected_runs"],
        "successful_runs": sum(c["successful_runs"] for c in candidates),
        "plan": plan,
        "selected_candidates": [],
        "selection_note": "Descriptive sweep: every candidate is reported; no winner is selected.",
        "candidates": candidates,
    }
    json_path = sweep_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    axis_names = sorted({k for c in candidates for k in c["grid_point"]})
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = axis_names + ["successful_runs", "expected_runs", "status"]
        for name in metric_names:
            header += [f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max"]
        writer.writerow(header)
        for c in candidates:
            row = [c["grid_point"].get(a) for a in axis_names] + [
                c["successful_runs"],
                c["expected_runs"],
                c["status"],
            ]
            for name in metric_names:
                agg = c["aggregates"][name]
                row += (
                    [agg["mean"], agg["std"], agg["min"], agg["max"]]
                    if agg
                    else ["", "", "", ""]
                )
            writer.writerow(row)
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize one sweep from its structured metrics"
    )
    parser.add_argument("status", type=Path, help="Path to sweep_status.json")
    args = parser.parse_args(argv)
    path = summarize(args.status)
    summary = json.loads(path.read_text())
    print(
        f"{summary['coverage']}: {summary['successful_runs']}/{summary['expected_runs']} runs -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
