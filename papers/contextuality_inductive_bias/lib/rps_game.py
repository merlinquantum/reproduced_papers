"""Three-player rock-paper-scissors (RPS) zero-sum game of Sec. 5.

Actions are indexed R = 0, P = 1, S = 2. Player k (0-based) has the special
action k: player 1 (index 0) R, player 2 (index 1) P, player 3 (index 2) S.

Payoff rule (Eq. (60)): E[Y_k | a] = l_k(a) / 2, where l_k(a) is the number of
players that player k beats minus the number of players that beat player k.
Different actions follow the usual rule (R > S, P > R, S > P); two players with
the same action are separated only when the common action is one of their
special actions, in which case its owner wins.

A strategy is a 3x3 row-stochastic matrix x whose row k is the distribution of
player k over (R, P, S). Because l_k is a sum of pairwise contests, the
expected payoff under independent strategies is bilinear in the rows:
E[l_k | x] = sum_{j != k} x_k^T A[k, j] x_j.
"""

from __future__ import annotations

import itertools

import numpy as np

N_PLAYERS = 3
N_ACTIONS = 3
ACTION_NAMES = "RPS"
_BEATS = {(0, 2), (1, 0), (2, 1)}  # (winner action, loser action)

# Deterministic action profiles whose payoff behaviours are the six vertices of
# the zero-sum hexagon (Sec. 5.2). Ordered to match the paper's v_1..v_6 of
# Eq. (31): v_1 = (1, 0, 1/2), v_2 = (0, 1, 1/2), v_3 = (1/2, 1, 0),
# v_4 = (1/2, 0, 1), v_5 = (0, 1/2, 1), v_6 = (1, 1/2, 0).
EXTREMAL_PROFILES = ("RSS", "SRS", "RPR", "RRP", "PPS", "SPP")
HEXAGON_VERTICES = np.array(
    [
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 1.0, 0.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.5, 0.0],
    ]
)


def pairwise_points(
    player: int, opponent: int, action: int, opponent_action: int
) -> int:
    """Return +1 if ``player`` beats ``opponent``, -1 if beaten, 0 for a draw.

    Parameters
    ----------
    player, opponent : int
        Player indices in {0, 1, 2}, with ``player != opponent``.
    action, opponent_action : int
        Action indices in {0 (R), 1 (P), 2 (S)}.

    Returns
    -------
    int
        Points awarded to ``player`` from this pairwise contest.
    """
    if action != opponent_action:
        if (action, opponent_action) in _BEATS:
            return 1
        if (opponent_action, action) in _BEATS:
            return -1
        return 0
    if action == player:
        return 1
    if action == opponent:
        return -1
    return 0


def points_tables() -> np.ndarray:
    """Pairwise points tables A[k, j][a_k, a_j] for all ordered player pairs.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3, 3, 3, 3)``; ``A[k, k]`` is zero.
    """
    tables = np.zeros((N_PLAYERS, N_PLAYERS, N_ACTIONS, N_ACTIONS))
    for k, j in itertools.permutations(range(N_PLAYERS), 2):
        for a, b in itertools.product(range(N_ACTIONS), repeat=2):
            tables[k, j, a, b] = pairwise_points(k, j, a, b)
    return tables


POINTS = points_tables()


def score_differences(actions: tuple[int, int, int]) -> np.ndarray:
    """Return l_k(a) for a deterministic action profile.

    Parameters
    ----------
    actions : tuple of int
        Action of each player.

    Returns
    -------
    numpy.ndarray
        Integer score differences of shape ``(3,)``; they sum to zero.
    """
    return np.array(
        [
            sum(
                POINTS[k, j, actions[k], actions[j]] for j in range(N_PLAYERS) if j != k
            )
            for k in range(N_PLAYERS)
        ]
    )


def win_probabilities(strategies: np.ndarray) -> np.ndarray:
    """Probability that each player receives a positive payoff.

    Parameters
    ----------
    strategies : numpy.ndarray
        Row-stochastic strategy matrices of shape ``(N, 3, 3)`` (or ``(3, 3)``).

    Returns
    -------
    numpy.ndarray
        P(Y_k = +1 | x) = (1 + E[l_k | x] / 2) / 2, shape ``(N, 3)`` (or ``(3,)``).
    """
    strategies = np.asarray(strategies, dtype=np.float64)
    single = strategies.ndim == 2
    batch = strategies[None] if single else strategies
    expected_scores = np.einsum("nka,kjab,njb->nk", batch, POINTS, batch)
    probabilities = (1.0 + expected_scores / 2.0) / 2.0
    return probabilities[0] if single else probabilities


def deterministic_strategy(profile: str) -> np.ndarray:
    """Strategy matrix of a deterministic action profile such as ``"RSS"``.

    Parameters
    ----------
    profile : str
        Three characters from ``"RPS"``, one per player.

    Returns
    -------
    numpy.ndarray
        One-hot strategy matrix of shape ``(3, 3)``.
    """
    if len(profile) != N_PLAYERS or any(c not in ACTION_NAMES for c in profile):
        raise ValueError(f"Profile must be three letters from 'RPS', got {profile!r}")
    return np.eye(N_ACTIONS)[[ACTION_NAMES.index(c) for c in profile]]
