"""Per-run figures (paper-style score curves and Frozen Lake Q* error)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import padded_score_matrix  # noqa: E402


def plot_score_curve(
    scores: list[list[float]], max_score: float, title: str, path: Path
) -> None:
    """Mean +- std of padded per-episode scores over agents (Figs. 3a-6)."""
    length = max((len(agent_scores) for agent_scores in scores), default=0)
    if length == 0:
        raise ValueError("No episodes to plot")
    matrix = padded_score_matrix(scores, length)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    episodes = np.arange(length)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(episodes, mean, color="tab:green", lw=1)
    ax.fill_between(
        episodes,
        np.clip(mean - std, 0, None),
        np.clip(mean + std, None, max_score),
        color="tab:green",
        alpha=0.3,
        lw=0,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_q_star_mae(mae: list[list[float]], title: str, path: Path) -> None:
    """Mean +- standard error of MAE(Q, Q*) over agents per time step (Fig. 3b)."""
    length = max(len(values) for values in mae)
    matrix = padded_score_matrix(mae, length)
    mean = matrix.mean(axis=0)
    sem = matrix.std(axis=0) / np.sqrt(max(matrix.shape[0], 1))
    steps = np.arange(length)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, mean, color="tab:blue", lw=1)
    ax.fill_between(steps, mean - sem, mean + sem, color="tab:blue", alpha=0.3, lw=0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("MAE(Q, Q*)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
