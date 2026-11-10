"""Independent recalculation of ``results/paper_seeds_v1_comparisons.csv``.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/recheck_comparisons.py \
        papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1 \
        papers/fermi_dirac_neurons/results/paper_seeds_v1_comparisons.csv

Reads every run's ``metrics.json`` directly (no import of
``utils/sweep_statistics.py``) and recomputes the paired median, mean, min,
max and seed count of each CSV row with the metric definitions declared in
``configs/sweeps/paper_seeds_v1.json``: ``train_loss`` is the training loss
logged at the last epoch, before the final update (field
``last_logged_train_loss``, the convention of the authors' CSVs), not the loss
after the final update (``final.train.loss``). Exits 1 on any difference
above 1e-12.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def raw_value(metrics: dict, model: str, metric: str) -> float:
    """Return one model's declared metric from a run's metrics.json."""
    record = metrics["models"][model]
    if metric == "train_loss":
        return record["last_logged_train_loss"]
    return record["final"]["validation"][metric.removeprefix("validation_")]


def main() -> int:
    """Recompute every CSV row and report the largest difference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("comparisons_csv", type=Path)
    args = parser.parse_args()
    status = json.loads((args.sweep_dir / "sweep_status.json").read_text())
    metrics_by_run = {
        (run["config"], run["seed"]): json.loads(
            (args.sweep_dir / run["metrics_path"]).read_text()
        )
        for run in status["runs"]
        if run["status"] == "COMPLETED"
    }
    seeds = sorted(set(status["spec"]["seeds"]))
    worst, rows = 0.0, 0
    for row in csv.DictReader(args.comparisons_csv.open()):
        metric = row["metric"]
        accuracy = metric.startswith("validation_accuracy")
        differences = []
        for seed in seeds:
            better = raw_value(
                metrics_by_run[(row["better_config"], seed)],
                row["better_model"],
                metric,
            )
            other = raw_value(
                metrics_by_run[(row["other_config"], seed)], row["other_model"], metric
            )
            differences.append(
                100 * (better - other)
                if accuracy
                else math.log10(other) - math.log10(better)
            )
        differences = np.array(differences)
        recomputed = {
            "median_advantage": np.median(differences),
            "mean_advantage": np.mean(differences),
            "min_advantage": np.min(differences),
            "max_advantage": np.max(differences),
            "seeds_favouring": int(np.sum(differences > 0)),
            "n_pairs": len(differences),
        }
        for key, value in recomputed.items():
            worst = max(worst, abs(float(row[key]) - float(value)))
        rows += 1
    print(f"{rows} rows recomputed; largest absolute difference {worst:.3g}")
    return 0 if worst <= 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
