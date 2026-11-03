"""Render the paper's Fig. 2 style panel from a mixture-of-Gaussians run.

Usage
-----
    python utils/plot_mixture.py outdir/run_YYYYMMDD-HHMMSS --seed 1

Reads the ``samples_<latent>_seed<k>.npy`` artifacts written by
``lib.experiments.run_mixture_of_gaussians`` and writes ``mixture_panel.png``
next to them (or wherever ``--out`` points).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Left-to-right ordering follows the paper's figure: most classical first.
PANEL_ORDER = ("gaussian", "bernoulli", "distinguishable", "boson")
TITLES = {
    "gaussian": "Gaussian",
    "bernoulli": "Bernoulli",
    "distinguishable": "Distinguishable sampler",
    "boson": "Boson sampler",
}


def build_panel(run_dir: Path, seed: int, out: Path) -> Path:
    """Draw one row of scatter plots, one per latent distribution.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run directory holding ``summary.json`` and the sample artifacts.
    seed : int
        Which repeat to plot.
    out : pathlib.Path
        Destination image path.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    FileNotFoundError
        If the run directory has no ``summary.json``.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no summary.json in {run_dir}")
    centers = np.asarray(json.loads(summary_path.read_text())["centers"])
    # Axis limits follow the data, not a hard-coded ring size.
    limit = 1.35 * float(np.linalg.norm(centers, axis=1).max())

    fig, axes = plt.subplots(
        1, len(PANEL_ORDER), figsize=(14, 3.6), sharex=True, sharey=True
    )
    for ax, kind in zip(axes, PANEL_ORDER):
        path = run_dir / f"samples_{kind}_seed{seed}.npy"
        if not path.exists():
            ax.set_axis_off()
            ax.set_title(f"{TITLES[kind]}\n(missing)")
            continue
        points = np.load(path)
        ax.scatter(
            points[:, 0], points[:, 1], s=2, alpha=0.25, c="#3b6ea5", linewidths=0
        )
        ax.scatter(
            centers[:, 0], centers[:, 1], s=40, marker="x", c="#d1495b", linewidths=1.5
        )
        ax.set_title(TITLES[kind])
        ax.set_aspect("equal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.tick_params(length=0, labelsize=8)

    fig.suptitle(
        f"GAN samples by latent distribution, seed {seed} (red x = true mixture modes)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory to read")
    parser.add_argument("--seed", type=int, default=0, help="Which repeat to plot")
    parser.add_argument("--out", type=Path, default=None, help="Output image path")
    args = parser.parse_args()
    out = args.out or args.run_dir / f"mixture_panel_seed{args.seed}.png"
    print("wrote", build_panel(args.run_dir, args.seed, out))


if __name__ == "__main__":
    main()
