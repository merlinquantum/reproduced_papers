"""Analytic optimal Q-values for deterministic Frozen Lake (paper Sec. 5.1).

The paper states ``Q*(s, a) = gamma^beta`` with ``beta`` the number of steps
on the shortest path to the goal from the successor state ``s'``. That equals
the Bellman fixed point of the deterministic 4x4 lake with reward 1 on
entering the goal and absorbing (zero-valued) holes and goal, which is what
``optimal_q_values`` computes. In particular ``Q*(14, RIGHT) = 1``; the
paper's sentence "Q*(14, R) = gamma" contradicts its own formula, and the
reference code also uses 1 (see LOG.md).
"""

from __future__ import annotations

import numpy as np

FROZENLAKE_MAP = ("SFFF", "FHFH", "FFFH", "HFFG")
# Gymnasium / gym action encoding.
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def _tile(state: int) -> str:
    return FROZENLAKE_MAP[state // 4][state % 4]


def terminal_states() -> list[int]:
    """Holes and the goal: episodes end on entering them."""
    return [state for state in range(16) if _tile(state) in "HG"]


def deterministic_successor(state: int, action: int) -> int:
    """Successor on the non-slippery lake; moves into a wall leave the agent in place."""
    row, col = divmod(state, 4)
    if action == LEFT:
        col = max(col - 1, 0)
    elif action == DOWN:
        row = min(row + 1, 3)
    elif action == RIGHT:
        col = min(col + 1, 3)
    elif action == UP:
        row = max(row - 1, 0)
    else:
        raise ValueError(f"Unknown action {action}")
    return row * 4 + col


def optimal_q_values(gamma: float, iterations: int = 200) -> np.ndarray:
    """Optimal Q-table ``(16, 4)``; rows of terminal states are zero.

    Parameters
    ----------
    gamma : float
        Discount factor (paper: 0.8 for Frozen Lake).
    iterations : int
        Value-iteration sweeps; the lake converges in fewer than 10. Default
        value is 200.

    Returns
    -------
    numpy.ndarray
        ``Q*[s, a]``.
    """
    terminals = set(terminal_states())
    q_values = np.zeros((16, 4))
    for _ in range(iterations):
        updated = np.zeros_like(q_values)
        for state in range(16):
            if state in terminals:
                continue
            for action in range(4):
                successor = deterministic_successor(state, action)
                reward = 1.0 if _tile(successor) == "G" else 0.0
                bootstrap = (
                    0.0 if successor in terminals else gamma * q_values[successor].max()
                )
                updated[state, action] = reward + bootstrap
        if np.allclose(updated, q_values):
            break
        q_values = updated
    return q_values
