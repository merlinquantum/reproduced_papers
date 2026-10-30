"""QuaRK's test MSE against the shot budget, relative to exact expectations, per seed.

Each seed's test MSE is divided by the test MSE of the same seed with exact
expectations (no shot noise), so 1 means no measurement penalty. Thin lines are seeds,
thick lines their mean. The 1000-shot and exact runs come from the main comparison
sweep, the 100- and 10000-shot runs from the shot-noise ablation.

Usage (from the repository root)::

    python papers/QuaRK/utils/plot_shot_budget.py \\
        papers/QuaRK/outdir/sweep_main_comparison \\
        papers/QuaRK/outdir/sweep_shot_noise_ablation --label "<scope>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_style import (
    PAPER_TASKS,
    TASK_LABELS,
    TEXT_SECONDARY,
    apply_style,
    load_sweep,
    quark_runs,
    save,
    selected,
)

BUDGETS = (100, 1000, 10000)
ESTIMATORS = (
    ("per_observable_shots", "Per-observable noise (authors' code)", "#2a78d6"),
    ("random_pauli_shadows", "Random-Pauli shadows", "#eb6834"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("main_sweep", type=Path)
    parser.add_argument("shot_sweep", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    apply_style()
    index = quark_runs(load_sweep(args.main_sweep), load_sweep(args.shot_sweep))
    exact = index[("exact", None, 5, 3)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, task in zip(axes, PAPER_TASKS):
        for estimator, _name, color in ESTIMATORS:
            groups = [index[(estimator, b, 5, 3)] for b in BUDGETS]
            seeds = sorted(exact)
            curves = np.array(
                [
                    [
                        selected(g[s], task, "mse_test")
                        / selected(exact[s], task, "mse_test")
                        for g in groups
                    ]
                    for s in seeds
                ]
            )
            for curve in curves:
                ax.plot(BUDGETS, curve, color=color, lw=0.8, alpha=0.35)
            ax.plot(BUDGETS, curves.mean(axis=0), color=color, marker="o", ms=5)
        ax.axhline(1.0, color=TEXT_SECONDARY, ls="--", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(BUDGETS, [f"{b:,}" for b in BUDGETS])
        ax.set_xlabel("Shots per observable or snapshots per circuit")
        ax.set_title(TASK_LABELS[task], fontsize=9, loc="left")
    axes[0].set_ylabel(
        "Test MSE / test MSE with exact\nexpectations, same seed (log scale)"
    )
    handles = [
        Line2D([], [], color=c, marker="o", ms=5, label=n) for _e, n, c in ESTIMATORS
    ]
    handles += [
        Line2D([], [], color=TEXT_SECONDARY, ls="--", label="No measurement penalty"),
        Line2D([], [], color=TEXT_SECONDARY, lw=0.8, alpha=0.5, label="One seed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        f"QuaRK test error relative to exact expectations, against the measurement budget (paper: 1000) · {args.label}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    saved = save(fig, args.shot_sweep, "shot_budget")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
