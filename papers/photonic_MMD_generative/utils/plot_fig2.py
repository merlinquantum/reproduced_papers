"""Fig. 2 reproduction: training-loss curves for |K|=|Z|=2000 vs 5000 on boson-sampling data.

Usage::

    python papers/photonic_MMD_generative/utils/plot_fig2.py <sweep_dir of fig2_bs>

Two panels with the paper's y range (0 to 2e-3) and an inset on steps
500-1250, as in the paper. Reads each run's ``losses.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from plot_common import SERIES, TEXT, load_sweep_runs, plt, save, style_axes


def main() -> None:
    """Plot the Fig. 2 sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    sweep_dir = parser.parse_args().sweep_dir.resolve()
    runs = sorted(load_sweep_runs(sweep_dir), key=lambda r: r["candidate"]["label"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    color = SERIES["model"][1]
    for ax, run in zip(axes, runs, strict=True):
        rows = list(csv.reader((run["run_dir"] / "losses.csv").open()))[1:]
        steps = np.array([int(r[0]) for r in rows])
        loss = np.array([float(r[1]) for r in rows])
        style_axes(ax)
        ax.plot(steps, loss, color=color, linewidth=1.2)
        ax.set_ylim(-2e-4, 2e-3)
        ax.set_xlabel("Training step", color=TEXT, fontsize=9)
        k = run["metrics"]["experiment"]["name"]
        size = 2000 if "2a" in k or "K2000" in run["candidate"]["label"] else 5000
        tail = np.diff(loss[500:])
        ax.set_title(
            f"|K| = |Z| = {size} · step-to-step std after step 500: {tail.std():.1e}",
            fontsize=9,
            color=TEXT,
        )
        inset = ax.inset_axes([0.45, 0.45, 0.5, 0.45])
        window = (steps >= 500) & (steps <= 1250)
        inset.plot(steps[window], loss[window], color=color, linewidth=0.9)
        style_axes(inset)
        inset.tick_params(labelsize=6)
    axes[0].set_ylabel("Training MMD² estimate", color=TEXT, fontsize=9)
    fig.suptitle(
        "Fig. 2 reproduction — boson sampling m=256, n=16, σ=3, Haar, near-identity init, seed 0\n"
        "Paper settings · BS data regenerated (new unitary) · single seed (preliminary per run)",
        fontsize=9,
        color=TEXT,
        y=1.06,
    )
    for path in save(fig, sweep_dir, "fig2_bs_loss_curves"):
        print(path)


if __name__ == "__main__":
    main()
