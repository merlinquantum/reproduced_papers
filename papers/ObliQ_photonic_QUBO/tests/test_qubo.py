"""QUBO representations: matrix is canonical, the dict is a boundary format."""

import numpy as np
import pytest
from utils.max_clique import create_qubo_max_clique
from utils.max_cut import create_qubo_max_cut, exact_max_cut
from utils.qubo import (
    build_qubo,
    exact_optimum,
    qubo_dict_to_matrix,
    qubo_matrix_to_dict,
    qubo_objective,
    solve_qubo_bruteforce,
)


def test_dict_matrix_round_trip():
    Q_dict = {(0, 0): -1.0, (1, 1): -1.0, (2, 2): -1.0, (0, 1): 2.0, (1, 2): 3.0}
    assert qubo_matrix_to_dict(qubo_dict_to_matrix(Q_dict, 3)) == Q_dict


def test_matrix_scores_the_same_as_the_dict_form():
    """The matrix must reproduce the dict's objective for every binary vector.

    This is the guarantee that made the dict removable: off-diagonal coefficients
    are halved into both triangles, so ``x @ Q @ x`` equals the dict's sum.
    """
    Q_dict = {(0, 0): -1.0, (1, 1): -1.0, (2, 2): -1.0, (0, 1): 2.0, (1, 2): 3.0}
    Q = qubo_dict_to_matrix(Q_dict, 3)
    for bits in np.ndindex(2, 2, 2):
        from_dict = sum(
            coeff * bits[i] if i == j else coeff * bits[i] * bits[j]
            for (i, j), coeff in Q_dict.items()
        )
        assert qubo_objective(Q, bits) == pytest.approx(from_dict)


@pytest.mark.parametrize("problem_type", ["max-cut", "max-clique"])
def test_build_qubo_matches_the_problem_formulation(problem_type, clique_graph):
    builder = (
        create_qubo_max_cut if problem_type == "max-cut" else create_qubo_max_clique
    )
    expected = qubo_dict_to_matrix(builder(clique_graph), len(clique_graph))
    assert np.array_equal(build_qubo(problem_type, clique_graph), expected)


def test_build_qubo_rejects_unknown_problems(clique_graph):
    with pytest.raises(NotImplementedError):
        build_qubo("travelling-salesman", clique_graph)


def test_max_clique_qubo_optimum_is_the_maximum_clique(clique_graph):
    """The QUBO's minimizer must be an actual maximum clique."""
    Q = build_qubo("max-clique", clique_graph)
    bits, energy = solve_qubo_bruteforce(Q)
    assert -energy == exact_optimum("max-clique", clique_graph)
    selected = [node for node, bit in enumerate(bits) if bit]
    subgraph = clique_graph.subgraph(selected)
    assert subgraph.number_of_edges() == len(selected) * (len(selected) - 1) / 2


def test_max_cut_qubo_optimum_is_the_maximum_cut(clique_graph):
    Q = build_qubo("max-cut", clique_graph)
    _bits, energy = solve_qubo_bruteforce(Q)
    assert -energy == exact_max_cut(clique_graph)


def test_max_cut_diagonal_varies_but_max_clique_is_constant(clique_graph):
    """Why augmentation exists: only Max-Cut has a non-constant diagonal."""
    assert len(set(build_qubo("max-cut", clique_graph).diagonal())) > 1
    assert len(set(build_qubo("max-clique", clique_graph).diagonal())) == 1
