"""Build a sweep's result table from its runs' ``metrics.json`` (never from logs).

Rule (frozen before the sweeps): for each candidate config,
method and model, the aggregate is the mean over seeds of the 5-fold mean
accuracy; variability is the standard deviation over seeds (``ddof=1``); the
mean fold standard deviation (the paper's ``±``) is reported alongside. A
candidate is complete only when every planned seed has a COMPLETED run with a
finite metric.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/summarize_sweep.py <sweep_dir>
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _models(outcome: dict) -> dict[str, dict]:
    """``{model label: summary}`` for one method of one run."""

    models = {"kernel_svm": outcome["kernel_svm"]}
    pca = outcome.get("kernel_pca")
    if pca is None:
        return models
    if "best_on_test_folds" in pca:
        # Paper protocol: the paper's statistic (best component count chosen on
        # the evaluation folds) and every fixed component count.
        models["kernel_pca_best_on_test_folds"] = pca["best_on_test_folds"]
        for count, summary in pca["per_component_count"].items():
            models[f"kernel_pca_{count}_components"] = summary
    else:
        models["kernel_pca_nested_cv"] = pca
    return models


def collect(sweep_dir: Path) -> dict:
    """Coverage check and aggregates for one sweep directory."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    planned = defaultdict(set)
    values = defaultdict(
        lambda: defaultdict(dict)
    )  # (candidate, method, model) -> metric -> {seed: (mean, std)}
    extra = defaultdict(dict)
    problems = []
    for entry in status["runs"]:
        candidate = Path(entry["candidate"]["config"]).stem
        planned[candidate].add(entry["seed"])
        if entry["status"] != "COMPLETED":
            problems.append(f"{entry['run_id']}: status {entry['status']}")
            continue
        metrics = json.loads((sweep_dir / entry["metrics_path"]).read_text())
        if metrics["seed"] != entry["seed"]:
            problems.append(
                f"{entry['run_id']}: metrics seed {metrics['seed']} != {entry['seed']}"
            )
            continue
        for method, outcome in metrics["methods"].items():
            for model, summary in _models(outcome).items():
                for metric in ("accuracy", "balanced_accuracy"):
                    mean, std = summary[metric]["mean"], summary[metric]["std"]
                    if not (math.isfinite(mean) and math.isfinite(std)):
                        problems.append(
                            f"{entry['run_id']}: non-finite {method}/{model}/{metric}"
                        )
                        continue
                    values[(candidate, method, model)][metric][entry["seed"]] = (
                        mean,
                        std,
                    )
                if "components" in summary:
                    extra[(candidate, method, model)].setdefault("components", {})[
                        entry["seed"]
                    ] = summary["components"]
    rows = []
    for (candidate, method, model), per_metric in sorted(values.items()):
        row = {
            "candidate": candidate,
            "method": method,
            "model": model,
            "expected_runs": len(planned[candidate]),
        }
        for metric, per_seed in per_metric.items():
            seeds = sorted(per_seed)
            means = np.array([per_seed[s][0] for s in seeds])
            fold_stds = np.array([per_seed[s][1] for s in seeds])
            row[metric] = {
                "seed_mean": float(means.mean()),
                "seed_std": float(means.std(ddof=1))
                if len(means) > 1
                else float("nan"),
                "mean_fold_std": float(fold_stds.mean()),
                "per_seed": {str(s): float(per_seed[s][0]) for s in seeds},
            }
            row["successful_runs"] = len(seeds)
        row["status"] = (
            "COMPLETE"
            if set(per_metric["accuracy"]) == planned[candidate]
            else "PARTIAL"
        )
        if (candidate, method, model) in extra:
            row["components_per_seed"] = {
                str(k): v
                for k, v in extra[(candidate, method, model)]["components"].items()
            }
        rows.append(row)
    complete = not problems and all(row["status"] == "COMPLETE" for row in rows)
    return {
        "sweep_id": status["sweep_id"],
        "coverage": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "planned": {k: sorted(v) for k, v in planned.items()},
        "problems": problems,
        "rule": "mean over seeds of the 5-fold mean accuracy; seed_std = std over seeds (ddof=1); mean_fold_std = mean of the per-seed fold std",
        "rows": rows,
    }


def summarize(sweep_dir: Path) -> Path:
    """Write ``summary.json`` and ``summary.md`` in ``sweep_dir``; return the JSON path."""

    summary = collect(Path(sweep_dir))
    path = Path(sweep_dir) / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Sweep {summary['sweep_id']}: {summary['coverage']}",
        "",
        f"Rule: {summary['rule']}.",
        "",
        "| Candidate | Method | Model | Runs | Accuracy (seed mean) | Std over seeds | Mean fold std | Balanced accuracy | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        acc, bal = row["accuracy"], row["balanced_accuracy"]
        lines.append(
            f"| {row['candidate']} | {row['method']} | {row['model']} | {row['successful_runs']}/{row['expected_runs']} "
            f"| {acc['seed_mean']:.4f} | {acc['seed_std']:.4f} | {acc['mean_fold_std']:.4f} | {bal['seed_mean']:.4f} | {row['status']} |"
        )
    if summary["problems"]:
        lines += ["", "Problems:", *[f"- {p}" for p in summary["problems"]]]
    (Path(sweep_dir) / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    print(summarize(Path(sys.argv[1])).read_text())
