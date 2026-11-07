"""Linear programs for the noncontextuality bound of Sec. 4.3 (experiment E5).

A noncontextual ontological model of the m = 3 scenario is characterised in
the proof of Lemma 4.1 by weights nu_j(i) >= 0, sum_i nu_j(i) = 1, over the six
extremal response vectors xi^i (the permutations of (1, 0, 1/2), i.e. the
hexagon vertices) for the six preparation densities s_1..s_6, with

    P(E^k_+ | s_j) = sum_i nu_j(i) xi^i_k                       (Eq. (48))
    nu_1 + nu_2 = nu_3 + nu_4 = nu_5 + nu_6   (elementwise)     (Eq. (50))

the last line being the preparation equivalence (43). Two programs follow:

``max_lemma_lhs``
    maximises P(E1+|s1) + P(E2+|s3) + P(E3+|s5). Lemma 4.1 bounds it by 5/2;
    the LP checks that bound and whether it is attained. Dropping the
    preparation equivalence is a control of ours (not in the paper): the
    trivial bound is then 3.
``max_noncontextual_eta``
    the largest eta for which the behaviours u_j(eta) = eta v_j +
    (1 - eta)(1/2, 1/2, 1/2) admit such a model. Theorem 4.1 implies it is at
    most 2/3; the LP gives the exact value for this scenario.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .rps_game import HEXAGON_VERTICES

N_PREPARATIONS = 6
N_RESPONSES = 6
RESPONSE_VERTICES = HEXAGON_VERTICES  # xi^i: the six permutations of (1, 0, 1/2)
CENTRE = np.full(3, 0.5)


def _index(preparation: int, response: int) -> int:
    return preparation * N_RESPONSES + response


def _base_constraints(with_preparation_equivalence: bool, extra_columns: int = 0):
    n_vars = N_PREPARATIONS * N_RESPONSES + extra_columns
    rows, rhs = [], []
    for j in range(N_PREPARATIONS):
        row = np.zeros(n_vars)
        row[[_index(j, i) for i in range(N_RESPONSES)]] = 1.0
        rows.append(row)
        rhs.append(1.0)
    if with_preparation_equivalence:
        # nu_1 + nu_2 - nu_3 - nu_4 = 0 and nu_3 + nu_4 - nu_5 - nu_6 = 0 (0-based pairs).
        for (a, b), (c, d) in (((0, 1), (2, 3)), ((2, 3), (4, 5))):
            for i in range(N_RESPONSES):
                row = np.zeros(n_vars)
                row[_index(a, i)] += 1.0
                row[_index(b, i)] += 1.0
                row[_index(c, i)] -= 1.0
                row[_index(d, i)] -= 1.0
                rows.append(row)
                rhs.append(0.0)
    return np.array(rows), np.array(rhs), n_vars


def max_lemma_lhs(with_preparation_equivalence: bool = True) -> dict:
    """Maximise the left-hand side of the Lemma 4.1 inequality (Eq. (44)).

    Parameters
    ----------
    with_preparation_equivalence : bool
        Impose Eq. (50). Default value is True.

    Returns
    -------
    dict
        ``optimum`` (float), ``status`` (str) and ``weights`` (6 x 6 nested list).
    """
    a_eq, b_eq, n_vars = _base_constraints(with_preparation_equivalence)
    objective = np.zeros(n_vars)
    # LHS = P(E^1_+ | s_1) + P(E^2_+ | s_3) + P(E^3_+ | s_5)
    for preparation, task in ((0, 0), (2, 1), (4, 2)):
        for i in range(N_RESPONSES):
            objective[_index(preparation, i)] = RESPONSE_VERTICES[i, task]
    result = linprog(-objective, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Lemma LP failed: {result.message}")
    return {
        "optimum": float(-result.fun),
        "status": result.message,
        "weights": result.x.reshape(N_PREPARATIONS, N_RESPONSES).tolist(),
    }


def lemma_lhs_for_eta(eta: float) -> float:
    """LHS of Eq. (44) on the behaviours u_j(eta): 3/2 (eta + 1) (Sec. 4.3)."""
    behaviours = eta * HEXAGON_VERTICES + (1.0 - eta) * CENTRE
    return float(behaviours[0, 0] + behaviours[2, 1] + behaviours[4, 2])


def max_noncontextual_eta() -> dict:
    """Largest eta whose behaviours u_j(eta) admit a noncontextual model.

    Returns
    -------
    dict
        ``eta_max`` (float), ``status`` (str).
    """
    a_eq, b_eq, n_vars = _base_constraints(True, extra_columns=1)
    eta_column = n_vars - 1
    rows, rhs = list(a_eq), list(b_eq)
    # sum_i nu_j(i) xi^i_k - eta (v_j,k - 1/2) = 1/2 for every preparation j and task k.
    for j in range(N_PREPARATIONS):
        for k in range(3):
            row = np.zeros(n_vars)
            for i in range(N_RESPONSES):
                row[_index(j, i)] = RESPONSE_VERTICES[i, k]
            row[eta_column] = -(HEXAGON_VERTICES[j, k] - CENTRE[k])
            rows.append(row)
            rhs.append(CENTRE[k])
    objective = np.zeros(n_vars)
    objective[eta_column] = -1.0
    bounds = [(0, None)] * (n_vars - 1) + [(0, 1)]
    result = linprog(
        objective,
        A_eq=np.array(rows),
        b_eq=np.array(rhs),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Eta LP failed: {result.message}")
    return {"eta_max": float(result.x[eta_column]), "status": result.message}
