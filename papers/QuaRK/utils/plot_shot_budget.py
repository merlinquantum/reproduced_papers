"""QuaRK's test MSE against the measurement budget, relative to exact expectations, per seed.

Each seed's test MSE is divided by the test MSE of the same seed with exact
expectations (no shot noise), so 1 means no measurement penalty. Thin lines are seeds,
thick lines their mean; hollow rings mark the paper's setting (1000). The 1000-shot and
exact runs come from the main comparison sweep, the 100- and 10000-shot runs from the
shot-noise ablation.

The x-axis counts all measurements per window and sub-reservoir, so the two noise
models are compared at equal cost: the authors' model draws its shots for each of the
105 observables separately (1000 per observable = 105,000 measurements), while
random-Pauli shadows share their snapshots among all observables (1000 = 1000).

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
from matplotlib.ticker import NullFormatter
from plot_style import (
    TEXT_SECONDARY,
    apply_style,
    load_sweep,
    quark_runs,
    save,
    selected,
)
from report_statistics import PAPER_TASKS, TASK_LABELS

BUDGETS = (100, 1000, 10000)
PAPER_BUDGET = 1000
# 1- and 2-qubit Pauli observables on n = 5 qubits: 3 * 5 + 9 * 10.
OBSERVABLES = 105
# (estimator, legend, colour, measurements per window and sub-reservoir per unit budget)
ESTIMATORS = (
    (
        "per_observable_shots",
        "Per-observable noise (authors' code): shots × 105 observables",
        "#2a78d6",
        OBSERVABLES,
    ),
    (
        "random_pauli_shadows",
        "Random-Pauli shadows (paper's text): snapshots shared by all observables",
        "#eb6834",
        1,
    ),
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
        for estimator, _name, color, per_unit in ESTIMATORS:
            cost = np.array(BUDGETS) * per_unit
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
                ax.plot(cost, curve, color=color, lw=0.8, alpha=0.35)
            mean = curves.mean(axis=0)
            ax.plot(cost, mean, color=color, marker="o", ms=5)
            paper = BUDGETS.index(PAPER_BUDGET)
            ax.plot(
                cost[paper], mean[paper], "o", ms=11, mfc="none", mec=color, mew=1.5
            )
        ax.axhline(1.0, color=TEXT_SECONDARY, ls="--", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        low, high = ax.get_ylim()
        if high / low < 10:  # less than a decade: plain factors instead of 2 × 10⁰
            factors = [f for f in (1, 1.5, 2, 2.5, 3) if low <= f <= high]
            ax.set_yticks(factors, [f"{f:g}" for f in factors])
            ax.yaxis.set_minor_formatter(NullFormatter())
        ticks = [10**k for k in range(2, 7)]
        ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
        ax.set_xlabel("Measurements per window and sub-reservoir")
        ax.set_title(TASK_LABELS[task], fontsize=9, loc="left")
    axes[0].set_ylabel(
        "Measurement penalty (×, log scale):\ntest MSE with shot noise ÷ without"
    )
    handles = [
        Line2D([], [], color=c, marker="o", ms=5, label=n)
        for _e, n, c, _u in ESTIMATORS
    ]
    handles.append(
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=11,
            mfc="none",
            mec=TEXT_SECONDARY,
            label="Paper's setting (1000)",
        )
    )
    handles += [
        Line2D([], [], color=TEXT_SECONDARY, ls="--", label="No measurement penalty"),
        Line2D([], [], color=TEXT_SECONDARY, lw=0.8, alpha=0.5, label="One seed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        f"How much shot noise multiplies QuaRK's test MSE (same seed, ÷ the MSE without shot noise), against the number of measurements · {args.label}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    saved = save(fig, args.shot_sweep, "shot_budget")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
