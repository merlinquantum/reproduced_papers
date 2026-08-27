"""QUBO representations: generic helpers, and the problem-type dispatch.

The canonical internal form of a QUBO is a **symmetric matrix**: it scores in one
expression (``x @ Q @ x``), it is what ObliQ encodes, and it carries no bookkeeping
about which triangle a coefficient landed in. The ``{(i, j): coeff}`` dict survives
in two places only: the problem formulations build one (they accumulate terms,
which reads better as a dict), and D-Wave's samplers require one --
:func:`qubo_matrix_to_dict` converts back at that single boundary.

The two problems are defined in :mod:`utils.max_cut` and :mod:`utils.max_clique`;
:func:`build_qubo` and :func:`to_quadratic_program` are the single place that maps
a ``problem_type`` string onto them. (CVaR-VQE builds its own matrix from the
graph; see :mod:`models.cvar_vqe`.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from networkx import Graph

if TYPE_CHECKING:  # torch is imported lazily; this is for annotations only
    import torch

from utils.max_clique import calculate_beta_max_clique, create_qubo_max_clique
from utils.max_cut import calculate_beta_max_cut, create_qubo_max_cut, exact_max_cut


def build_qubo(problem_type: str, graph: Graph) -> np.ndarray:
    """Build the QUBO matrix for a problem type.

    Raises:
        NotImplementedError: for an unknown problem type.
    """
    if problem_type == "max-cut":
        Q_dict = create_qubo_max_cut(graph)
    elif problem_type == "max-clique":
        Q_dict = create_qubo_max_clique(graph)
    else:
        raise NotImplementedError(
            f"Provided problem type {problem_type} is not implemented"
        )
    return qubo_dict_to_matrix(Q_dict, len(graph))


def exact_optimum(problem_type: str, graph: Graph) -> int:
    """Exact optimal objective of an instance -- the denominator of exact beta.

    One definition for the whole harness. The sweep (``--exact``) and the plotter
    (``-e``) both come here, so a stored optimum and a recomputed one cannot
    disagree. Note that both are exhaustive at benchmark sizes: Max-Cut enumerates
    partitions up to :data:`utils.max_cut.EXACT_MAX_CUT_LIMIT` nodes, above which
    it degrades to a greedy bound.

    Raises:
        NotImplementedError: for an unknown problem type.
    """
    import networkx as nx

    if problem_type == "max-cut":
        return exact_max_cut(graph)
    if problem_type == "max-clique":
        return int(nx.max_weight_clique(graph, weight=None)[1])
    raise NotImplementedError(
        f"Provided problem type {problem_type} is not implemented"
    )


def calculate_beta(
    problem_type: str,
    graph_or_size: Graph | int,
    objective: float,
    seed: int | None = None,
) -> float:
    """Q-score beta for one result.

    The normalization follows ``graph_or_size``, as it does in
    :mod:`utils.max_cut` and :mod:`utils.max_clique`: pass the **graph** to score
    against that instance's own optimum (exact), or the **size** to use the Q-score
    standard's asymptotic constants. A failed run (``nan``) scores 0.

    ``seed`` seeds the sampled random baseline on the exact Max-Clique path.

    Raises:
        NotImplementedError: for an unknown problem type.
    """
    if objective is None or (isinstance(objective, float) and math.isnan(objective)):
        return 0.0
    if problem_type == "max-cut":
        return calculate_beta_max_cut(graph_or_size, objective)
    if problem_type == "max-clique":
        return calculate_beta_max_clique(graph_or_size, objective, seed=seed)
    raise NotImplementedError(
        f"Provided problem type {problem_type} is not implemented"
    )


def to_quadratic_program(problem_type: str, graph: Graph):
    """Build the Qiskit ``QuadraticProgram`` used by the QAOA solver.

    Qiskit is imported lazily: it is a heavy dependency and only the QAOA path
    needs it, so a photonic run should not pay for it on import.

    Raises:
        NotImplementedError: for an unknown problem type.
    """
    from qiskit_optimization.applications import Clique, Maxcut

    if problem_type == "max-cut":
        return Maxcut(graph).to_quadratic_program()
    if problem_type == "max-clique":
        return Clique(graph).to_quadratic_program()
    raise NotImplementedError(
        f"Provided problem type {problem_type} is not implemented"
    )


def qubo_dict_to_matrix(
    Q_dict: Mapping[tuple[int, int], float], size: int
) -> np.ndarray:
    """Convert a ``{(i, j): coeff}`` QUBO into a symmetric matrix.

    Off-diagonal coefficients are split evenly across both triangles so that
    ``x @ Q @ x`` reproduces the dict's objective for any binary ``x``.
    """
    matrix = np.zeros((size, size), dtype=float)
    for (i, j), value in Q_dict.items():
        if i == j:
            matrix[i, i] += value
        else:
            coeff = value / 2.0
            matrix[i, j] += coeff
            matrix[j, i] += coeff
    return matrix


def normalize_qubo(Q: torch.Tensor) -> torch.Tensor:
    """Min-max scale the off-diagonal terms of a (batched) QUBO into [0, 1].

    The diagonal is zeroed first: a constant diagonal shifts every assignment
    equally, and a non-constant one is expected to have been moved onto an ancilla
    already (see :func:`models.obliq.augment_qubo`). An all-zero off-diagonal is
    returned unchanged rather than dividing by a zero range.
    """
    import torch

    Q = Q.clone()
    Q.diagonal(dim1=-2, dim2=-1).zero_()
    Q_min = Q.min()
    Q_max = Q.max()
    denom = Q_max - Q_min
    if torch.isclose(denom, torch.tensor(0.0, device=Q.device, dtype=denom.dtype)):
        return torch.zeros_like(Q)
    return (Q - Q_min) / (Q_max - Q_min)


def qubo_matrix_to_dict(Q: np.ndarray) -> dict:
    """Convert a symmetric QUBO matrix back into ``{(i, j): coeff}`` form.

    Inverse of :func:`qubo_dict_to_matrix`: the two triangles are recombined into
    one upper-triangular coefficient. Needed only at the D-Wave boundary, whose
    samplers take a QUBO dict.
    """
    size = Q.shape[0]
    Q_dict = {}
    for i in range(size):
        if Q[i, i]:
            Q_dict[(i, i)] = float(Q[i, i])
        for j in range(i + 1, size):
            coeff = float(Q[i, j] + Q[j, i])
            if coeff:
                Q_dict[(i, j)] = coeff
    return Q_dict


def qubo_objective(Q: np.ndarray, x: Sequence[int]) -> float:
    """Evaluate the QUBO energy ``x^T Q x`` for a binary vector ``x``.

    Negate it for the benchmark's *maximization* objective (cut / clique size).
    """
    vector = np.asarray(x, dtype=float)
    return float(vector @ Q @ vector)


def solve_qubo_bruteforce(Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Exhaustive QUBO minimization. Only for very small instances (tests)."""
    from itertools import product

    size = Q.shape[0]
    best_solution = np.zeros(size, dtype=int)
    best_value = float("inf")
    for combination in product([0, 1], repeat=size):
        x = np.array(combination)
        value = qubo_objective(Q, x)
        if value < best_value:
            best_value = value
            best_solution = x
    return best_solution, best_value
