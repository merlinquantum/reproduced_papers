"""Plot the heat equation prediction vs RK45 reference.

Layout follows Figure 11 of the paper: side-by-side (x, t) → T heatmaps
plus an absolute-error heatmap.

Usage:

    python utils/plot_heat.py <run_dir> --out results/heat_qpinn_grid.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    predictions = json.loads((args.run_dir / "predictions.json").read_text())
    x = np.array(predictions["x"])
    t = np.array(predictions["t"])
    u_pred = np.array(predictions["u_pred"])
    u_ref = np.array(predictions["u_ref"])
    err = np.abs(u_pred - u_ref)

    extent = (float(x.min()), float(x.max()), float(t.min()), float(t.max()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    pcm = axes[0].imshow(u_pred, extent=extent, origin="lower", aspect="auto",
                          cmap="inferno", vmin=0, vmax=0.5)
    axes[0].set_title(f"Prediction ({summary['experiment']})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.colorbar(pcm, ax=axes[0])
    pcm = axes[1].imshow(u_ref, extent=extent, origin="lower", aspect="auto",
                          cmap="inferno", vmin=0, vmax=0.5)
    axes[1].set_title("Reference (RK45)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    plt.colorbar(pcm, ax=axes[1])
    pcm = axes[2].imshow(err, extent=extent, origin="lower", aspect="auto",
                          cmap="viridis")
    axes[2].set_title(f"Abs error (RMSE={summary['metrics']['rmse']:.2e})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    plt.colorbar(pcm, ax=axes[2])
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
