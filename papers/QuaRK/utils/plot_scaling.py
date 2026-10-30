"""QuaRK's normalized test MSE against the number of qubits n and of sub-reservoirs R.

One factor at a time around the paper's setting (n = 5, R = 3), per-observable noise
with 1000 shots. Thin lines are seeds, thick lines their mean. The paper's point comes
from the main comparison sweep, the others from the scaling sweep.

Usage (from the repository root)::

    python papers/QuaRK/utils/plot_scaling.py \\
        papers/QuaRK/outdir/sweep_main_comparison \\
        papers/QuaRK/outdir/sweep_scaling --label "<scope>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import (
    PAPER_TASKS,
    TASK_COLORS,
    TASK_LABELS,
    TEXT_SECONDARY,
    apply_style,
    load_sweep,
    quark_runs,
    save,
    selected,
    spread_log_labels,
)

AXES = (
    ("Qubits per sub-reservoir n (R = 3)", (3, 5, 7), 5, lambda v: (v, 3)),
    ("Sub-reservoirs R (n = 5)", (1, 3, 6), 3, lambda v: (5, v)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("main_sweep", type=Path)
    parser.add_argument("scaling_sweep", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    apply_style()
    index = quark_runs(load_sweep(args.main_sweep), load_sweep(args.scaling_sweep))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (xlabel, values, paper_value, settings) in zip(axes, AXES):
        groups = [index[("per_observable_shots", 1000, *settings(v))] for v in values]
        ends = {}
        for task in PAPER_TASKS:
            seeds = sorted(groups[0])
            curves = np.array(
                [
                    [selected(g[s], task, "normalized_mse_test") for g in groups]
                    for s in seeds
                ]
            )
            color = TASK_COLORS[task]
            for curve in curves:
                ax.plot(values, curve, color=color, lw=0.8, alpha=0.35)
            mean = curves.mean(axis=0)
            ax.plot(values, mean, color=color, marker="o", ms=5)
            ends[task] = mean[-1]
        for task, y in spread_log_labels(ends, min_ratio=1.22).items():
            ax.annotate(
                TASK_LABELS[task].split(" (")[0],
                (values[-1], ends[task]),
                xytext=(values[-1] + 0.25, y),
                textcoords="data",
                va="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
        ax.axvline(paper_value, color=TEXT_SECONDARY, ls=":", lw=1.2)
        ax.annotate(
            "paper setting",
            (paper_value, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -12),
            textcoords="offset points",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
        ax.set_yscale("log")
        ax.set_xticks(values)
        ax.set_xlim(values[0] - 0.4, values[-1] + 2.2)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Normalized test MSE (log scale)")
    fig.suptitle(
        f"QuaRK test error against reservoir size; thin lines are seeds · {args.label}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    saved = save(fig, args.scaling_sweep, "scaling")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
