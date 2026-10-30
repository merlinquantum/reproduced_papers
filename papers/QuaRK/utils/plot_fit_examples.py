"""Fig. 3: predicted and true labels on the first 50 training windows, one panel per task.

Predictions come from the kernel ridge fit at the tuning ridge (10⁻⁶) on the full
training set, as in the authors' figure.

Usage (from the repository root)::

    python papers/QuaRK/utils/plot_fit_examples.py <run_dir> --label "<scope>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_style import (
    TASK_COLORS,
    TASK_LABELS,
    TEXT,
    TEXT_SECONDARY,
    apply_style,
    load_run,
    ordered_tasks,
    save,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    apply_style()
    metrics, arrays = load_run(args.run_dir)
    examples = arrays["fit_examples"]
    tasks = ordered_tasks(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    for ax, task in zip(axes.flat, tasks):
        truth = examples[f"{task}__truth"]
        prediction = examples[f"{task}__prediction"]
        index = np.arange(truth.size)
        ax.plot(index, truth, color=TEXT, linewidth=1.5)
        ax.plot(index, prediction, color=TASK_COLORS[task], linestyle="--")
        mse = metrics["tasks"][task]["fit_examples_mse"]
        ax.set_title(
            f"{TASK_LABELS[task]} · training MSE {mse:.1e}", fontsize=9, loc="left"
        )
    for ax in axes.flat[len(tasks) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Training window (first 50, authors' order)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Label y")
    handles = [
        Line2D([], [], color=TEXT, linewidth=1.5, label="True label"),
        Line2D(
            [],
            [],
            color=TEXT_SECONDARY,
            linestyle="--",
            label="Prediction (task color)",
        ),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(
        f"Fit on training windows at λ_reg = 10⁻⁶ (paper Fig. 3) · {args.label}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    saved = save(fig, args.run_dir, "fig3_fit_examples")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
