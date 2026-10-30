"""Build the result table of a sweep from its runs' ``metrics.json`` files.

Usage (from the repository root)::

    python papers/QuaRK/utils/summarize_sweep.py papers/QuaRK/outdir/sweep_<name>

Writes ``sweep_summary.json`` and ``sweep_summary.md`` in the sweep directory.

Rules (frozen in LOG.md "Fair Baseline Plan"):

* metric: test MSE at the ``reg`` selected on validation data (``validation``), and,
  reported separately, at the test-selected ``reg`` (the authors' Fig. 4 rule);
* aggregation: mean and sample standard deviation over seeds;
* comparison against the reference candidate, per task: paired difference
  ``delta = MSE(reference) - MSE(other)`` over the shared seeds, with
  ``SE = std(delta) / sqrt(n)``; "reference better" if ``mean < -2 SE``,
  "other better" if ``mean > +2 SE``, else "no clear separation";
* coverage: a candidate-task cell counts only completed runs with finite metrics;
  the sweep is complete only if every planned run completed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if values.size > 1 else None
    return float(values.mean()), std


def _paired(reference, other):
    seeds = sorted(set(reference) & set(other))
    delta = np.array([reference[s] - other[s] for s in seeds])
    if delta.size < 2:
        return {"seeds": seeds, "label": "not enough paired seeds"}
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / math.sqrt(delta.size))
    if mean < -2 * se:
        label = "reference better"
    elif mean > 2 * se:
        label = "other better"
    else:
        label = "no clear separation"
    return {"seeds": seeds, "mean_delta": mean, "se_delta": se, "label": label}


def summarize(sweep_dir: Path) -> dict:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    spec = json.loads((sweep_dir / "sweep_spec.json").read_text())
    reference = spec.get("reference")
    cells: dict = {}
    invalid = []
    for run in status["runs"]:
        name = run["candidate"]["name"]
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        for task, result in metrics["tasks"].items():
            values = {sel: result["selections"][sel] for sel in ("validation", "test")}
            numbers = [
                values[s][k]
                for s in values
                for k in ("mse_test", "normalized_mse_test")
            ]
            if not all(math.isfinite(x) for x in numbers):
                invalid.append(f"{run['run_id']}:{task}")
                continue
            cell = cells.setdefault(name, {}).setdefault(task, {"seeds": {}})
            cell["seeds"][run["seed"]] = {
                "mse_test": values["validation"]["mse_test"],
                "normalized_mse_test": values["validation"]["normalized_mse_test"],
                "mse_test_test_selected": values["test"]["mse_test"],
                "decay_slope": values["validation"]["decay_slope"],
                "decay_slope_test_selected": values["test"]["decay_slope"],
                "mse_train_at_smallest_reg": result["mse_train_at_smallest_reg"],
            }

    expected_per_candidate = len(spec["seeds"])
    table = []
    for candidate in spec["candidates"]:
        name = candidate["name"]
        for task, cell in sorted(cells.get(name, {}).items()):
            seeds = cell["seeds"]
            row = {
                "candidate": name,
                "task": task,
                "successful_runs": len(seeds),
                "expected_runs": expected_per_candidate,
            }
            for key in (
                "mse_test",
                "normalized_mse_test",
                "mse_test_test_selected",
                "decay_slope",
                "decay_slope_test_selected",
                "mse_train_at_smallest_reg",
            ):
                vals = [v[key] for v in seeds.values() if v[key] is not None]
                if vals:
                    row[f"{key}_mean"], row[f"{key}_std"] = _mean_std(vals)
            row["status"] = (
                "complete" if len(seeds) == expected_per_candidate else "partial"
            )
            table.append(row)

    comparisons = []
    if reference in cells:
        for name, tasks in cells.items():
            if name == reference:
                continue
            for task in tasks:
                if task not in cells[reference]:
                    continue
                ref = {
                    s: v["mse_test"] for s, v in cells[reference][task]["seeds"].items()
                }
                oth = {s: v["mse_test"] for s, v in tasks[task]["seeds"].items()}
                comparisons.append(
                    {
                        "reference": reference,
                        "other": name,
                        "task": task,
                        **_paired(ref, oth),
                    }
                )

    lowest = {}
    for task in sorted({r["task"] for r in table}):
        rows = [r for r in table if r["task"] == task and "mse_test_mean" in r]
        best = min(rows, key=lambda r: r["mse_test_mean"])
        lowest[task] = best["candidate"]

    completed = sum(r["status"] == "COMPLETED" for r in status["runs"])
    summary = {
        "sweep_id": status["sweep_id"],
        "coverage": "COMPLETE SWEEP"
        if completed == status["expected_runs"] and not invalid
        else "PARTIAL SWEEP",
        "completed_runs": completed,
        "expected_runs": status["expected_runs"],
        "missing_runs": [
            r["run_id"] for r in status["runs"] if r["status"] != "COMPLETED"
        ],
        "invalid_cells": invalid,
        "table": table,
        "comparisons": comparisons,
        "lowest_observed_mean_test_mse": lowest,
    }
    return summary


def _markdown(summary: dict) -> str:
    def fmt(row, key, digits=4):
        mean = row.get(f"{key}_mean")
        std = row.get(f"{key}_std")
        if mean is None:
            return "—"
        return (
            f"{mean:.{digits}g} ± {std:.2g}"
            if std is not None
            else f"{mean:.{digits}g}"
        )

    lines = [
        f"# Sweep summary: {summary['sweep_id']}",
        "",
        f"{summary['coverage']}: {summary['completed_runs']}/{summary['expected_runs']} runs completed.",
        "",
        "Test MSE at the validation-selected ridge, mean ± std over seeds; normalized = MSE / variance of the test labels.",
        "",
        "| Candidate | Task | Runs | Test MSE | Normalized | Test MSE (test-selected) | Decay slope |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["table"]:
        lines.append(
            f"| {row['candidate']} | {row['task']} | {row['successful_runs']}/{row['expected_runs']} | "
            f"{fmt(row, 'mse_test')} | {fmt(row, 'normalized_mse_test', 3)} | "
            f"{fmt(row, 'mse_test_test_selected')} | {fmt(row, 'decay_slope', 3)} |"
        )
    if summary["comparisons"]:
        lines += [
            "",
            "Paired comparison with the reference (delta = MSE(reference) − MSE(other), 2·SE rule):",
            "",
            "| Reference | Other | Task | Mean delta | SE | Label |",
            "|---|---|---|---:|---:|---|",
        ]
        for c in summary["comparisons"]:
            if "mean_delta" in c:
                lines.append(
                    f"| {c['reference']} | {c['other']} | {c['task']} | {c['mean_delta']:.4g} | {c['se_delta']:.2g} | {c['label']} |"
                )
            else:
                lines.append(
                    f"| {c['reference']} | {c['other']} | {c['task']} | — | — | {c['label']} |"
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    sweep_dir = Path(sys.argv[1]).resolve()
    summary = summarize(sweep_dir)
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    (sweep_dir / "sweep_summary.md").write_text(_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
