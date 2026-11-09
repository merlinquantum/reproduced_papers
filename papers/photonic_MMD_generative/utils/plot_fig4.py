"""Fig. 4a/4b reproduction: training MMD^2 curves by initialization (mean over seeds).

Usage::

    python papers/photonic_MMD_generative/utils/plot_fig4.py <fig4a sweep_dir> <fig4b sweep_dir>

Each curve is the seed mean of the per-step training estimate; runs stopped
early by the convergence rule are padded with their last value (as a stopped
run's model stays fixed). Log y axis, as in the paper.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from plot_common import TEXT, load_sweep_runs, plt, save, style_axes

COLORS = {  # fixed categorical order of the reference palette
    "weak_unbiased": "#2a78d6",
    "data_dependent": "#eb6834",
    "identity": "#1baf7a",
    "pairwise_uniform": "#eda100",
    "block_uniform": "#eda100",
    "random": "#e87ba4",
}


def curves(sweep_dir: Path):
    by_init = defaultdict(list)
    for run in load_sweep_runs(sweep_dir):
        rows = list(csv.reader((run["run_dir"] / "losses.csv").open()))[1:]
        by_init[run["candidate"]["init.strategy"]].append(
            np.array([float(r[1]) for r in rows])
        )
    return by_init


def main() -> None:
    """Plot Fig. 4a and 4b."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fig4a_dir", type=Path)
    parser.add_argument("fig4b_dir", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sweep_dir, title in zip(
        axes,
        (args.fig4a_dir.resolve(), args.fig4b_dir.resolve()),
        ("(a) Pairwise sushi", "(b) Mushroom blocks"),
        strict=True,
    ):
        style_axes(ax)
        for init, runs in curves(sweep_dir).items():
            length = max(len(r) for r in runs)
            padded = np.array(
                [np.pad(r, (0, length - len(r)), mode="edge") for r in runs]
            )
            ax.plot(
                np.arange(1, length + 1),
                np.clip(padded.mean(axis=0), 1e-4, None),
                color=COLORS[init],
                linewidth=2,
                label=init.replace("_", " "),
            )
        ax.set_yscale("log")
        ax.set_xlabel("Training step", color=TEXT, fontsize=9)
        ax.set_title(title, fontsize=9, color=TEXT)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Training MMD² (mean over 5 seeds)", color=TEXT, fontsize=9)
    fig.suptitle(
        "Fig. 4 reproduction — block-structured toy datasets, Haar, Gaussian σ=2\n"
        "Paper settings · seeds stopped earlier by the convergence rule are held at their last value; each curve ends at its longest run",
        fontsize=9,
        color=TEXT,
        y=1.04,
    )
    for path in save(fig, args.fig4a_dir.resolve(), "fig4_init_curves"):
        print(path)


if __name__ == "__main__":
    main()
