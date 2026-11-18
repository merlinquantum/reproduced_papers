"""Score-curve and solve-time summaries, following the reference plotting code.

Paper curves (Figs. 3-6) show the raw per-episode score averaged over agents;
an agent that stopped (solved) keeps contributing its last score for the
remaining episodes, and the band is +- one standard deviation clipped to the
score range (reference ``src/utils/plots.py``).
"""

from __future__ import annotations

import numpy as np


def padded_score_matrix(scores: list[list[float]], length: int) -> np.ndarray:
    """Agents x episodes matrix; each row is padded with its last score.

    Parameters
    ----------
    scores : list of list of float
        Per-agent episode scores.
    length : int
        Number of episodes in the matrix.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_agents, length)``.
    """
    matrix = np.zeros((len(scores), length))
    for row, agent_scores in enumerate(scores):
        values = np.asarray(agent_scores[:length], dtype=float)
        fill = values[-1] if len(values) else 0.0
        matrix[row, :] = fill
        matrix[row, : len(values)] = values
    return matrix


def solve_summary(
    solved_at: list[int | None], max_episodes: int
) -> dict[str, float | int | None]:
    """Summaries of the solve episode index across agents.

    Unsolved agents are censored at ``max_episodes``: ``*_censored`` statistics
    include them at that value, the plain statistics use solved agents only.

    Parameters
    ----------
    solved_at : list of int or None
        0-based episode index at which each agent met the solve rule
        (reference-code ``env_solved_at``), ``None`` if never.
    max_episodes : int
        Training cap.

    Returns
    -------
    dict
        Counts, mean / std / median / min of solved agents and censored mean /
        median.
    """
    solved = [value for value in solved_at if value is not None]
    censored = [value if value is not None else max_episodes for value in solved_at]
    summary: dict[str, float | int | None] = {
        "n_agents": len(solved_at),
        "n_solved": len(solved),
        "solved_fraction": len(solved) / len(solved_at) if solved_at else 0.0,
        "mean_solved_episode_index": float(np.mean(solved)) if solved else None,
        "std_solved_episode_index": float(np.std(solved)) if solved else None,
        "median_solved_episode_index": float(np.median(solved)) if solved else None,
        "min_solved_episode_index": int(min(solved)) if solved else None,
        "mean_episode_index_censored": float(np.mean(censored)) if censored else None,
        "median_episode_index_censored": float(np.median(censored))
        if censored
        else None,
    }
    return summary


def normalised_area_under_curve(
    scores: list[list[float]], episodes: int, max_score: float
) -> list[float]:
    """Per-agent mean padded score over the first ``episodes`` episodes / ``max_score``."""
    matrix = padded_score_matrix(scores, episodes)
    return (matrix.mean(axis=1) / max_score).tolist()
