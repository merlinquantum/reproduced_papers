"""
tau_alpha_grid_run.py – minimal benchmark of τ/α hyper‑parameters
=================================================================
Train **MerlinReuploadingClassifier** on a Cartesian grid of
(τ, λ) values for a fixed re‑uploading *design* "AA" and plot the
resulting decision maps & feature axes.  A fresh figure is saved
for each circuit depth ℓ.

Motivation
----------
* Explore how the Fisher‑loss temperature **τ** (fit argument) and
  the input phase‑scaling **α** (constructor argument `alpha`) shape
  the learned decision boundary.
* Provide *Publication‑ready*  grids similar to *architecture_grid_run.py*.
* Keep the script **tiny & tweakable** – all knobs live at the top.

Usage
-----
```bash
python tau_alpha_grid_run.py          # saves tau_alpha_grid_1.png … _L.png
```
Tweak `DEPTHS`, `LIST_TAU`, `LIST_alpha`, dataset, or training
hyper‑parameters as you wish.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .paper_datasets import CirclesDataset, MoonsDataset  # swap for CirclesDataset, …
from .reuploading_experiment import MerlinReuploadingClassifier

# ── global knobs ──────────────────────────────────────────────────────────────

DEPTHS = range(2, 5)  # ℓ = 1 … 4  (set to [1] for a quick test)
# DATASET = CirclesDataset(n_train=400, n_test=100)
DATASET = MoonsDataset(n_train=400, n_test=100)

LIST_TAU = np.logspace(-3, 2, 6)
LIST_alpha = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]) * np.pi

HYPERPARAMS = {
    "max_epochs": 10_000,
    "learning_rate": 1e-3,
    "batch_size": 400,
    "patience": 1_000,
    "track_history": False,  # disable loss tracking to speed things up
}

# ── little helpers (copied & slimmed from architecture_grid_run.py) ──────────


def _classification_map(ax, model, X_tr, y_tr, X_te, y_te):
    res = 220
    all_pts = np.vstack([X_tr, X_te]) if X_te is not None else X_tr
    margin = 0.15 * (all_pts.max(0) - all_pts.min(0))
    xlim = (all_pts[:, 0].min() - margin[0], all_pts[:, 0].max() + margin[0])
    ylim = (all_pts[:, 1].min() - margin[1], all_pts[:, 1].max() + margin[1])

    xx, yy = np.meshgrid(np.linspace(*xlim, res), np.linspace(*ylim, res))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    prob = model.predict_proba(grid)
    diff = (prob[:, 1] - prob[:, 0]).reshape(xx.shape)

    cmap = LinearSegmentedColormap.from_list("rwb", ["#d62728", "#ffffff", "#1f77b4"])
    ax.contourf(
        xx,
        yy,
        diff,
        levels=np.linspace(-1, 1, 21),
        cmap=cmap,
        alpha=0.75,
        extend="both",
    )
    ax.contour(xx, yy, diff, levels=[0], colors="k", linewidths=2)

    ax.scatter(*X_tr[y_tr == 0].T, c="darkred", s=30, lw=0.6, ec="w")
    ax.scatter(*X_tr[y_tr == 1].T, c="darkblue", s=30, lw=0.6, ec="w")
    if X_te is not None:
        ax.scatter(*X_te[y_te == 0].T, c="darkred", s=40, marker="s", lw=1, ec="w")
        ax.scatter(*X_te[y_te == 1].T, c="darkblue", s=40, marker="s", lw=1, ec="w")

    ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])


def _probability_axis(ax, model, X_tr, y_tr, X_te, y_te):
    feat_tr = model.get_quantum_features(X_tr)[:, 0]
    off = 0.12
    ax.scatter(
        feat_tr[y_tr == 0],
        -off * np.ones_like(feat_tr[y_tr == 0]),
        c="red",
        s=22,
        alpha=0.85,
    )
    ax.scatter(
        feat_tr[y_tr == 1],
        off * np.ones_like(feat_tr[y_tr == 1]),
        c="blue",
        s=22,
        alpha=0.85,
    )
    if X_te is not None:
        feat_te = model.get_quantum_features(X_te)[:, 0]
        ax.scatter(
            feat_te[y_te == 0],
            -2 * off * np.ones_like(feat_te[y_te == 0]),
            c="red",
            s=28,
            marker="s",
            alpha=0.85,
        )
        ax.scatter(
            feat_te[y_te == 1],
            2 * off * np.ones_like(feat_te[y_te == 1]),
            c="blue",
            s=28,
            marker="s",
            alpha=0.85,
        )
    ax.set_xlim(0, 1)
    ax.axis("off")


def run_tau_alpha_grid(cfg, run_dir):
    """Run the tau/alpha grid experiment using configuration."""
    import logging

    logger = logging.getLogger(__name__)

    # Extract configuration
    dataset_name = cfg["experiment"]["dataset"]
    if dataset_name == "circles":
        dataset = CirclesDataset(n_train=400, n_test=100)
    elif dataset_name == "moons":
        dataset = MoonsDataset(n_train=400, n_test=100)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_tr, y_tr = dataset.train
    X_te, y_te = dataset.test

    # Use configuration parameters
    depths = cfg["experiment"].get("depths", range(2, 5))
    list_tau = cfg["experiment"].get("tau_values", np.logspace(-3, 2, 6))
    list_alpha = cfg["experiment"].get(
        "alpha_values", np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]) * np.pi
    )

    hyperparams = {
        "max_epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["lr"],
        "batch_size": cfg["dataset"]["batch_size"],
        "patience": cfg["training"].get("patience", 1000),
        "track_history": False,
    }

    for L in depths:
        logger.info(f"Processing depth {L}...")
        n_rows, n_cols = len(list_tau), len(list_alpha)
        fig_h = 4.5 * n_rows  # height heuristic
        fig_w = 4.0 * n_cols  # width  heuristic
        fig, axes = plt.subplots(
            2 * n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            gridspec_kw={
                "height_ratios": [4, 1] * n_rows,
                "hspace": 0.25,
                "wspace": 0.15,
            },
        )

        total_models = len(list_tau) * len(list_alpha)
        model_count = 0

        for r, tau in enumerate(list_tau):
            for c, lam in enumerate(list_alpha):
                model_count += 1
                logger.info(
                    f"  Training model {model_count}/{total_models}: tau={tau:.3f}, alpha={lam / np.pi:.3f}π"
                )

                mdl = MerlinReuploadingClassifier(
                    dimension=X_tr.shape[1], num_layers=L, design="AA", alpha=lam
                )
                mdl.fit(X_tr, y_tr, tau=tau, **hyperparams)

                ax_map = axes[2 * r, c]
                ax_prob = axes[2 * r + 1, c]
                _classification_map(ax_map, mdl, X_tr, y_tr, X_te, y_te)
                _probability_axis(ax_prob, mdl, X_tr, y_tr, X_te, y_te)

                # label left‑most column with τ
                if c == 0:
                    ax_map.set_ylabel(
                        f"$\\tau = {tau:.2f}$",
                        rotation=0,
                        labelpad=40,
                        fontsize=12,
                        va="center",
                    )

            # add λ titles once per column (top row)
            for c, lam in enumerate(list_alpha):
                if r == 0:
                    axes[0, c].set_title(
                        f"$\\alpha = {lam / np.pi:.2f} \\pi$", fontsize=13, pad=16
                    )

        fig.suptitle(f"AA design – {L} layer(s)", fontsize=16, y=0.93)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out = run_dir / f"tau_alpha_grid_{L}_{dataset_name}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        logger.info(f"✓ saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Default standalone execution
    X_tr, y_tr = DATASET.train
    X_te, y_te = DATASET.test

    for L in DEPTHS:
        n_rows, n_cols = len(LIST_TAU), len(LIST_alpha)
        fig_h = 4.5 * n_rows  # height heuristic
        fig_w = 4.0 * n_cols  # width  heuristic
        fig, axes = plt.subplots(
            2 * n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            gridspec_kw={
                "height_ratios": [4, 1] * n_rows,
                "hspace": 0.25,
                "wspace": 0.15,
            },
        )

        for r, tau in enumerate(LIST_TAU):
            for c, lam in enumerate(LIST_alpha):
                mdl = MerlinReuploadingClassifier(
                    dimension=X_tr.shape[1], num_layers=L, design="AA", alpha=lam
                )
                mdl.fit(X_tr, y_tr, tau=tau, **HYPERPARAMS)
                print(
                    f"{r * len(LIST_TAU) + c + 1} / {len(LIST_TAU) * len(LIST_alpha)}"
                )

                ax_map = axes[2 * r, c]
                ax_prob = axes[2 * r + 1, c]
                _classification_map(ax_map, mdl, X_tr, y_tr, X_te, y_te)
                _probability_axis(ax_prob, mdl, X_tr, y_tr, X_te, y_te)

                # label left‑most column with τ
                if c == 0:
                    ax_map.set_ylabel(
                        f"$\\tau = {tau:.2f}$",
                        rotation=0,
                        labelpad=40,
                        fontsize=12,
                        va="center",
                    )

            # add λ titles once per column (top row)
            for c, lam in enumerate(LIST_alpha):
                if r == 0:
                    axes[0, c].set_title(
                        f"$\\alpha = {lam / np.pi:.2f} \\pi$", fontsize=13, pad=16
                    )

        fig.suptitle(f"AA design – {L} layer(s)", fontsize=16, y=0.93)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out = f"results/tau_alpha_grid_{L}_moons.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"✓ saved {out}")
