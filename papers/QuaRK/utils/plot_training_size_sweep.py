"""Fig. 4: test MSE against the number of training windows, one panel per task.

Shows both ridge selections of the run: on validation data (our protocol) and on
the test set (the authors' rule). The release overlay marks the authors' values.

Usage (from the repository root)::

    python papers/QuaRK/utils/plot_training_size_sweep.py <run_dir> --label "<scope>" \\
        [--release-run <authors' run dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
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
    parser.add_argument("--label", required=True)
    parser.add_argument("--release-run", type=Path, default=None)
    args = parser.parse_args()

    apply_style()
    metrics, arrays = load_run(args.run_dir)
    sweep = arrays["training_size_sweep"]
    sizes = sweep["sizes"]
    release = load_release(args.release_run) if args.release_run else None
    tasks = ordered_tasks(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    for ax, task in zip(axes.flat, tasks):
        color = TASK_COLORS[task]
        ax.plot(
            sizes,
            sweep[f"{task}__test__mse_test"],
            color=color,
            marker="o",
            markersize=5,
            label="λ_reg chosen on test (authors' rule)",
        )
        ax.plot(
            sizes,
            sweep[f"{task}__validation__mse_test"],
            color=color,
            linestyle="--",
            marker="s",
            markersize=4,
            alpha=0.8,
            label="λ_reg chosen on validation",
        )
        if release is not None and task in PAPER_TASKS:
            ax.plot(
                release["sizes"],
                release["size_mse_test"][PAPER_TASKS.index(task)],
                linestyle="none",
                marker="x",
                markersize=8,
                color=TEXT_SECONDARY,
                label="Authors' release",
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(TASK_LABELS[task], fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=7)
    for ax in axes.flat[len(tasks) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Number of training windows N")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test MSE (2000 windows)")
    fig.suptitle(
        f"Test MSE vs training-set size (paper Fig. 4) · {args.label}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    saved = save(fig, args.run_dir, "fig4_training_size_sweep")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
