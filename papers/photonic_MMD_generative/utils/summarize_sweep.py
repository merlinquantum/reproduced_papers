"""Aggregate a sweep's structured run metrics into a candidate table.

Reads ``sweep_status.json`` and every included run's ``metrics.json``; never
parses logs. Writes ``summary.json`` and ``summary.csv`` next to the sweep
status. Comparison sweeps in this reproduction declare no winner: every
candidate is reported with mean, standard deviation (ddof=1) and run counts.

Usage::

    python papers/photonic_MMD_generative/utils/summarize_sweep.py <sweep_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

MODEL_METRICS = (
    "walsh_mmd",
    "gaussian_mmd_raw",
    "gaussian_mmd_binarized",
    "collision_fraction",
)


def flatten_metrics(metrics: dict) -> dict:
    """Pick the scalar metrics compared across candidates."""
    if metrics.get("backend") == "rbm_tuned":
        row = {"total_train_seconds": metrics["tuning"]["total_train_seconds"]}
        for sigma, entries in metrics["sample_metrics_by_sigma"].items():
            selected = metrics["tuning"]["selected"][sigma]
            for key in ("hidden", "learning_rate", "gibbs_steps", "validation_mmd"):
                row[f"rbm_tuned_sigma{sigma}.selected_{key}"] = selected[key]
            for key in MODEL_METRICS:
                row[f"rbm_tuned_sigma{sigma}.{key}"] = entries["rbm_tuned"][key]
        return row
    row = {
        "test_mmd_estimator": metrics["test_mmd_estimator"],
        "init_test_mmd_estimator": metrics["init"]["test_mmd_estimator"],
        "final_train_mmd_estimate": metrics["training"]["final_train_mmd_estimate"],
        "steps_run": metrics["training"]["steps_run"],
        "seconds_per_step": metrics["training"]["seconds_per_step"],
        "unitarity_error": metrics["unitarity_error"],
    }
    for model, entry in metrics.get("sample_metrics", {}).items():
        for key in MODEL_METRICS:
            if key in entry:
                row[f"{model}.{key}"] = entry[key]
    return row


def summarize(sweep_dir: Path) -> dict:
    """Build the candidate table from structured artifacts.

    Returns
    -------
    dict
        ``planned``, ``observed`` and ``missing`` (candidate, seed) pairs,
        ``coverage`` ("COMPLETE SWEEP" or "PARTIAL SWEEP") and ``candidates``
        with per-metric mean, std, n.

    Raises
    ------
    ValueError
        On duplicate or non-finite included results.
    """
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    planned = {
        (json.dumps(r["candidate"], sort_keys=True), r["seed"]) for r in status["runs"]
    }
    rows = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        key = (json.dumps(run["candidate"], sort_keys=True), run["seed"])
        if key in rows:
            raise ValueError(f"duplicate completed run for {key}")
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        flat = flatten_metrics(metrics)
        bad = [
            k for k, v in flat.items() if isinstance(v, float) and not math.isfinite(v)
        ]
        if bad:
            raise ValueError(f"non-finite metrics {bad} in {run['run_id']}")
        rows[key] = flat
    expected = status["expected_runs"]
    observed = set(rows)
    coverage = (
        "COMPLETE SWEEP"
        if len(observed) == expected and observed == planned
        else "PARTIAL SWEEP"
    )
    table = {}
    for candidate_json in sorted({k[0] for k in planned}):
        seeds = sorted(s for c, s in observed if c == candidate_json)
        values = [rows[(candidate_json, s)] for s in seeds]
        stats = {}
        for metric in sorted({m for v in values for m in v}):
            series = np.array([v[metric] for v in values if metric in v], dtype=float)
            stats[metric] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else None,
                "n": int(len(series)),
            }
        table[candidate_json] = {"seeds": seeds, "metrics": stats}
    return {
        "sweep_id": status["sweep_id"],
        "expected_runs": expected,
        "completed_runs": len(observed),
        "coverage": coverage,
        "missing": sorted([list(k) for k in planned - observed]),
        "candidates": table,
        "selection": "none: comparison sweep, every candidate reported",
    }


def write_summary(sweep_dir: Path) -> Path:
    """Write ``summary.json`` and ``summary.csv``; return the JSON path."""
    summary = summarize(sweep_dir)
    json_path = sweep_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "metric", "mean", "std", "n"])
        for candidate, entry in summary["candidates"].items():
            for metric, stats in entry["metrics"].items():
                writer.writerow(
                    [
                        candidate,
                        metric,
                        repr(stats["mean"]),
                        repr(stats["std"]),
                        stats["n"],
                    ]
                )
    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize one sweep from its structured artifacts."
    )
    parser.add_argument("sweep_dir", type=Path)
    path = write_summary(parser.parse_args().sweep_dir.resolve())
    print(
        json.dumps(
            {
                k: v
                for k, v in json.loads(path.read_text()).items()
                if k != "candidates"
            },
            indent=2,
        )
    )
