"""Fig. 2: training MSE against the ridge regularization, one line per task.

Usage (from the repository root)::

    python papers/QuaRK/utils/plot_regularization_sweep.py <run_dir> --label "<scope>" \\
        [--release-run <authors' run dir>]
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
    load_release,
    load_run,
    ordered_tasks,
    save,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--label", required=True, help="Result scope, e.g. 'paper-accurate, seed 12345'"
    )
    parser.add_argument("--release-run", type=Path, default=None)
    args = parser.parse_args()

    apply_style()
    metrics, arrays = load_run(args.run_dir)
    sweep = arrays["regularization_sweep"]
    grid = sweep["reg_grid"]
    release = load_release(args.release_run) if args.release_run else None

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ends = {}
    for task in ordered_tasks(metrics):
        values = sweep[f"{task}__mse_train"]
        color = TASK_COLORS[task]
        ax.plot(
            grid, values, color=color, marker="o", markersize=4, label=TASK_LABELS[task]
        )
        ends[task] = float(values[-1])
        if release is not None and task in PAPER_TASKS:
            ax.plot(
                release["reg_grid"],
                release["mse_train"][PAPER_TASKS.index(task)],
                linestyle="none",
                marker="x",
                markersize=8,
                color=TEXT_SECONDARY,
            )
    if release is not None:
        ax.plot(
            [],
            [],
            linestyle="none",
            marker="x",
            markersize=8,
            color=TEXT_SECONDARY,
            label="Authors' release",
        )
    # Direct end labels, pushed apart so close plateaus do not collide.
    label_y, previous = {}, None
    for task, y in sorted(ends.items(), key=lambda item: item[1]):
        y = y if previous is None else max(y, previous + 0.06)
        label_y[task], previous = y, y
    for task, y in label_y.items():
        ax.annotate(
            TASK_LABELS[task].split(" (")[0],
            (grid[-1], ends[task]),
            xytext=(grid[-1] * 2.5, y),
            textcoords="data",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Ridge regularization λ_reg (Gram diagonal + λ_reg)")
    ax.set_ylabel("Training MSE")
    ax.set_title(
        f"Training MSE vs regularization (paper Fig. 2) · {args.label}",
        fontsize=10,
        loc="left",
    )
    ax.set_xlim(grid[0] / 3, grid[-1] * 2000)
    ax.set_ylim(bottom=-0.02)
    ax.legend(loc="upper left", fontsize=8)
    saved = save(fig, args.run_dir, "fig2_regularization_sweep")
    print("\n".join(str(p) for p in saved))
    print(
        "max training MSE at the smallest λ_reg:",
        max(float(np.asarray(sweep[f"{t}__mse_train"])[0]) for t in metrics["tasks"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
