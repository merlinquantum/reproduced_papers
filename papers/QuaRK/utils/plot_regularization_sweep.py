"""Fig. 3: training MSE against the ridge strength, one line per paper task.

The release overlay marks the authors' values.

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
    PAPER_FIGURE_TITLES,
    TASK_COLORS,
    TEXT_SECONDARY,
    apply_style,
    load_release,
    load_run,
    save,
)
from report_statistics import PAPER_TASKS


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
    for task in PAPER_TASKS:
        values = sweep[f"{task}__mse_train"]
        color = TASK_COLORS[task]
        ax.plot(grid, values, color=color, marker="o", markersize=4)
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
    ax.plot([], [], color=TEXT_SECONDARY, marker="o", markersize=4, label="Ours")
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
            PAPER_FIGURE_TITLES[task],
            (grid[-1], ends[task]),
            xytext=(grid[-1] * 2.5, y),
            textcoords="data",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Ridge strength (log scale)")
    ax.set_ylabel("Training MSE")
    ax.set_title(
        f"Training MSE vs ridge strength (paper Fig. 3) · {args.label}",
        fontsize=10,
        loc="left",
    )
    ax.set_xlim(grid[0] / 3, grid[-1] * 2000)
    ax.set_ylim(bottom=-0.02)
    ax.legend(loc="upper left", fontsize=8)
    saved = save(fig, args.run_dir, "fig3_regularization_sweep")
    print("\n".join(str(p) for p in saved))
    print(
        "max training MSE at the smallest ridge strength:",
        max(float(np.asarray(sweep[f"{t}__mse_train"])[0]) for t in PAPER_TASKS),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
