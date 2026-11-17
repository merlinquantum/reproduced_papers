"""Plot mean test accuracy per model against the paper's reported accuracies.

Usage::

    python utils/plot_accuracy.py outdir/run_A outdir/run_B ... --name accuracy_1d

Reads ``metrics.json`` of ``classification`` runs on one dataset. Bars are the
mean over instances (U_0 seeds, dataset variants) of the mean test accuracy over
the 100 partitions; error bars are the spread over instances when there are
several, otherwise the std over partitions. Paper values are drawn as stars.
The figure goes to every run directory and to ``results/<name>.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from summarize_results import classification_rows  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    rows = [row for run in args.run_dirs for row in classification_rows(run)]
    datasets = {row["dataset"] for row in rows}
    if len(datasets) != 1:
        raise ValueError(f"runs mix datasets {sorted(datasets)}")
    validities = sorted({row["validity"] for row in rows})

    arms = ["paper", "tuned"]
    models = list(dict.fromkeys(row["model"] for row in rows))
    figure, axis = plt.subplots(figsize=(max(6, 0.75 * len(models)), 4.2))
    positions = np.arange(len(models))
    width = 0.38
    for offset, arm in zip((-width / 2, width / 2), arms):
        means, errors = [], []
        for model in models:
            match = [r for r in rows if r["model"] == model and r["arm"] == arm]
            if not match:
                means.append(np.nan)
                errors.append(0.0)
                continue
            row = match[0]
            means.append(row["mean_test_accuracy"])
            errors.append(
                row["std_over_instances"]
                if row["n_instances"] > 1
                else row["pooled_std_over_partitions"]
            )
        label = "paper arm (C = 50)" if arm == "paper" else "tuned arm (inner CV)"
        axis.bar(positions + offset, means, width, yerr=errors, capsize=2, label=label)
    for index, model in enumerate(models):
        paper = next(r["paper_accuracy"] for r in rows if r["model"] == model)
        if paper is not None:
            axis.plot(index, paper, "k*", markersize=11)
    axis.plot([], [], "k*", label="paper reported")
    axis.axhline(0.5, color="grey", lw=0.8, ls=":")
    axis.set_xticks(positions, models, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("mean test accuracy (100 partitions, 80/20)")
    axis.set_ylim(0, 1.05)
    axis.set_title(f"{rows[0]['dataset']} - " + "; ".join(validities), fontsize=8)
    axis.legend(fontsize=7, loc="lower right")

    RESULTS_DIR.mkdir(exist_ok=True)
    targets = [run / f"{args.name}.png" for run in args.run_dirs]
    for target in targets + [RESULTS_DIR / f"{args.name}.png"]:
        figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(RESULTS_DIR / f"{args.name}.png")


if __name__ == "__main__":
    main()
