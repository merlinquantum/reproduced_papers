"""Plot a sigma-by-E heatmap of QKS test accuracy.

Usage:
    python utils/plot_sigma_E_sweep.py outdir/run_YYYYMMDD-HHMMSS

Reads the ``metrics.json`` produced by ``lib.runner.train_and_evaluate`` and
writes a contour plot of mean test accuracy (mean across seeds) vs. (sigma, E).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", type=str, default="QKS test accuracy")
    args = parser.parse_args()

    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    results = metrics["results"]

    # mean accuracy across seeds for each (sigma, E)
    buckets: dict[tuple[float, int], list[float]] = defaultdict(list)
    for r in results:
        buckets[(float(r["sigma"]), int(r["n_episodes"]))].append(
            float(r["test_accuracy"])
        )
    sigmas = sorted({s for s, _ in buckets})
    Es = sorted({e for _, e in buckets})
    Z = np.full((len(Es), len(sigmas)), np.nan)
    Zerr = np.full_like(Z, np.nan)
    for (sigma, E), accs in buckets.items():
        i = Es.index(E)
        j = sigmas.index(sigma)
        Z[i, j] = float(np.mean(accs))
        Zerr[i, j] = float(np.std(accs))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    pcm = ax.pcolormesh(
        sigmas, Es, Z, shading="nearest", cmap="viridis", vmin=0.5, vmax=1.0
    )
    fig.colorbar(pcm, ax=ax, label="mean test accuracy")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Number of episodes $E$")
    ax.set_yscale("log")
    ax.set_title(args.title)
    for i, E in enumerate(Es):
        for j, sigma in enumerate(sigmas):
            if not np.isnan(Z[i, j]):
                ax.text(
                    sigma,
                    E,
                    f"{Z[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="w" if Z[i, j] < 0.85 else "k",
                )
    out = args.out or args.run_dir / "sigma_E_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
