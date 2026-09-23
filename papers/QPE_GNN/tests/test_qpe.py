"""Tests for the QPE feature computations."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.pe_factory import pe_batch  # noqa: E402
from lib.qpe import (  # noqa: E402
    adjacency_from_edges,
    correlation_matrix_on_state,
    cqrw_features,
    ground_state_correlation_eigvecs,
    ising_ground_state,
    ladder_ising_ground_states,
    qirw_features,
    rrwp,
    xy_in_k_subspace,
)


def _square_graph():
    return adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])


def _explicit_qirw2_reference(adjacency: np.ndarray, num_features: int) -> np.ndarray:
    num_nodes = adjacency.shape[0]
    basis = list(combinations(range(num_nodes), 2))
    basis_indices = {state: index for index, state in enumerate(basis)}
    occupation_adjacency = np.zeros((len(basis), len(basis)), dtype=np.float64)
    for source_index, source_state in enumerate(basis):
        occupied_nodes = set(source_state)
        for occupied_node in source_state:
            for neighbor in np.flatnonzero(adjacency[occupied_node]):
                if neighbor in occupied_nodes:
                    continue
                destination_state = tuple(
                    sorted((occupied_nodes - {occupied_node}) | {int(neighbor)})
                )
                occupation_adjacency[
                    source_index, basis_indices[destination_state]
                ] += 2.0

    row_sums = occupation_adjacency.sum(axis=1)
    transition = np.divide(
        occupation_adjacency,
        row_sums[:, None],
        out=np.zeros_like(occupation_adjacency),
        where=row_sums[:, None] != 0,
    )
    graph_edge_states = [
        basis_indices[(i, j)]
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
        if adjacency[i, j] != 0 or adjacency[j, i] != 0
    ]
    initial_state = np.zeros(len(basis), dtype=np.float64)
    initial_state[graph_edge_states] = 1.0 / len(graph_edge_states)

    output = np.zeros((num_features, num_nodes, num_nodes), dtype=np.float64)
    for step in range(num_features):
        evolved_state = np.linalg.matrix_power(transition, step) @ initial_state
        for state_index, (i, j) in enumerate(basis):
            output[step, i, j] = evolved_state[state_index]
            output[step, j, i] = evolved_state[state_index]
    return output


def test_rrwp_row_stochastic():
    A = _square_graph()
    P = rrwp(A, 5)
    # P[k] for k >= 1 should have rows that sum to 1 (random walk transition).
    for k in range(1, 5):
        np.testing.assert_allclose(P[k].sum(axis=1), 1.0, atol=1e-10)


def test_cqrw1_unitary_preserves_probability():
    A = _square_graph()
    P = cqrw_features(A, 1, [0.0, 0.5, 1.3])
    # Each row of P[t] is the probability distribution over output nodes when
    # starting from row index. Should sum to 1.
    for k in range(P.shape[0]):
        np.testing.assert_allclose(P[k].sum(axis=1), 1.0, atol=1e-10)


def test_cqrw1_at_t0_is_identity():
    A = _square_graph()
    P = cqrw_features(A, 1, [0.0])
    np.testing.assert_allclose(P[0], np.eye(A.shape[0]), atol=1e-10)


def test_qirw2_path_has_hand_computed_edge_initial_state_and_powers():
    adjacency = adjacency_from_edges(3, [(0, 1), (1, 2)])
    expected = np.array(
        [
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]],
        ]
    )

    features = qirw_features(adjacency, 2, num_features=3)

    np.testing.assert_allclose(features, expected, atol=1e-12)


def test_qirw2_cycle_matches_explicit_matrix_power_reference():
    adjacency = adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])

    features = qirw_features(adjacency, 2, num_features=5)
    expected = _explicit_qirw2_reference(adjacency, num_features=5)

    np.testing.assert_allclose(features, expected, atol=1e-12)


def test_qirw2_irregular_graph_matches_explicit_matrix_power_reference():
    adjacency = adjacency_from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])

    features = qirw_features(adjacency, 2, num_features=4)
    expected = _explicit_qirw2_reference(adjacency, num_features=4)

    np.testing.assert_allclose(features, expected, atol=1e-12)


def test_qirw2_feature_count_includes_identity_power_at_index_zero():
    adjacency = adjacency_from_edges(3, [(0, 1), (1, 2)])
    adjacency_batch = torch.from_numpy(adjacency).float().unsqueeze(0)
    mask = torch.ones((1, 3), dtype=torch.bool)

    features = pe_batch(adjacency_batch, mask, "qirw2", K=4)

    assert features.shape == (1, 3, 3, 4)
    expected_identity_power = adjacency / 2.0
    np.testing.assert_allclose(
        features[0, :, :, 0].numpy(), expected_identity_power, atol=1e-12
    )


def test_combined_cqrw_uses_separate_rrwp_and_qpe_dimensions():
    adjacency = _square_graph()
    adjacency_batch = torch.from_numpy(adjacency).float().unsqueeze(0)
    mask = torch.ones((1, 4), dtype=torch.bool)
    times = [0.1, 0.7, 1.4]

    features = pe_batch(
        adjacency_batch,
        mask,
        "rrwp+cqrw1",
        K=5,
        times=times,
        rrwp_dim=2,
        qpe_dim=3,
    )
    expected = np.concatenate(
        [rrwp(adjacency, 2), cqrw_features(adjacency, 1, times)], axis=0
    )

    np.testing.assert_allclose(
        features[0].numpy(), expected.transpose(1, 2, 0), atol=1e-7
    )


def test_combined_qirw_uses_separate_rrwp_and_qpe_dimensions():
    adjacency = _square_graph()
    adjacency_batch = torch.from_numpy(adjacency).float().unsqueeze(0)
    mask = torch.ones((1, 4), dtype=torch.bool)

    features = pe_batch(
        adjacency_batch,
        mask,
        "rrwp+qirw2",
        K=5,
        rrwp_dim=2,
        qpe_dim=3,
    )
    expected = np.concatenate(
        [rrwp(adjacency, 2), qirw_features(adjacency, 2, num_features=3)], axis=0
    )

    np.testing.assert_allclose(
        features[0].numpy(), expected.transpose(1, 2, 0), atol=1e-7
    )


def test_qirw2_rejects_graph_without_initial_edges():
    with pytest.raises(ValueError, match="at least one edge"):
        qirw_features(np.zeros((3, 3)), 2, num_features=3)


def test_xy_subspace_dimensions():
    A = _square_graph()
    H1 = xy_in_k_subspace(A, 1)
    H2 = xy_in_k_subspace(A, 2)
    assert H1.shape == (4, 4)
    assert H2.shape == (6, 6)
    # H1 == 2 * A (single-particle XY hamiltonian = 2 * adjacency).
    np.testing.assert_allclose(H1, 2.0 * A, atol=1e-12)


def test_ising_ground_state_has_unit_norm():
    A = _square_graph()
    psi = ising_ground_state(A)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-10)


def test_correlation_matrix_is_symmetric_and_diag_one():
    A = _square_graph()
    psi = ising_ground_state(A)
    C = correlation_matrix_on_state(psi, A.shape[0])
    np.testing.assert_allclose(C, C.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-10)  # Z_i^2 = I


def test_gs_eigvecs_orthonormal():
    A = _square_graph()
    feats = ground_state_correlation_eigvecs(A, 3)
    # Eigenvectors of a symmetric matrix should be orthonormal.
    # For top-3 eigvecs of a 4x4 matrix the gram should be (close to) identity
    # *in the subspace spanned by non-zero eigenvalues*.
    nonzero_cols = np.where(np.linalg.norm(feats, axis=0) > 1e-8)[0]
    sub = feats[:, nonzero_cols]
    sub_gram = sub.T @ sub
    np.testing.assert_allclose(sub_gram, np.eye(sub.shape[1]), atol=1e-9)


def test_ladder_transfer_solver_matches_explicit_weighted_reference():
    from lib.data import make_type2

    num_nodes, edges = make_type2(5)
    transfer_states = ladder_ising_ground_states(num_nodes, edges)
    crossing_edges = {(0, 3), (6, 9)}
    explicit_energies = []
    explicit_states = []
    for state_index in range(1 << num_nodes):
        spins = np.asarray(
            [1 - 2 * ((state_index >> node) & 1) for node in range(num_nodes)]
        )
        energy = sum(
            (2 if (first_node, second_node) in crossing_edges else 1)
            * spins[first_node]
            * spins[second_node]
            for first_node, second_node in edges
        )
        explicit_energies.append(energy)
        explicit_states.append(spins)
    minimum_energy = min(explicit_energies)
    explicit_ground_states = {
        tuple(state)
        for energy, state in zip(explicit_energies, explicit_states)
        if energy == minimum_energy
    }
    assert {tuple(state) for state in transfer_states} == explicit_ground_states


def test_type2_figure_ground_state_count():
    from lib.data import make_type2

    num_nodes, edges = make_type2(7)
    # Figure 7 lists nine states after identifying the global spin reversal.
    assert len(ladder_ising_ground_states(num_nodes, edges)) == 18
