"""Fig. 5: test MSE against the number of training windows, one panel per paper task.

The ridge strength is chosen on the test set, as in the authors' code at the paper's
commit, so the curves can be compared with their release (crosses). The
effect of choosing it on validation data instead is in ``results/result_tables.md``.

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
    parser.add_argument("--label", required=True)
    parser.add_argument("--release-run", type=Path, default=None)
    args = parser.parse_args()

    apply_style()
    _metrics, arrays = load_run(args.run_dir)
    sweep = arrays["training_size_sweep"]
    sizes = sweep["sizes"]
    release = load_release(args.release_run) if args.release_run else None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, task in zip(axes, PAPER_TASKS):
        color = TASK_COLORS[task]
        ax.plot(
            sizes,
            sweep[f"{task}__test__mse_test"],
            color=color,
            marker="o",
            markersize=5,
            label="Ours",
        )
        if release is not None:
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
        ax.set_title(PAPER_FIGURE_TITLES[task], fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=7)
    for ax in axes:
        ax.set_xlabel("Number of training windows")
    axes[0].set_ylabel("Test MSE (2000 windows)")
    fig.suptitle(
        f"Test MSE vs training-set size (paper Fig. 5) · {args.label}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    saved = save(fig, args.run_dir, "fig5_training_size_sweep")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
