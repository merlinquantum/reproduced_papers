"""ObliQ's encoding and decoding: augmentation, anchor angles, and readout."""

import numpy as np
import pytest
import torch
from models.circuits import expected_coeff_count
from models.obliq import augment_qubo, distribution_to_result, qubo_norm_to_theta
from utils.qubo import build_qubo, normalize_qubo, qubo_objective
from utils.readout import EnergyTable, number_mapping


def test_augment_leaves_a_constant_diagonal_alone(clique_graph):
    Q = build_qubo("max-clique", clique_graph)
    Q_out, augmented = augment_qubo(Q)
    assert not augmented
    assert np.array_equal(Q_out, Q)


def test_augment_moves_a_varying_diagonal_onto_one_ancilla(clique_graph):
    Q = build_qubo("max-cut", clique_graph)
    n = Q.shape[0]
    Q_aug, augmented = augment_qubo(Q)

    assert augmented
    assert Q_aug.shape == (n + 1, n + 1)
    assert np.allclose(np.diag(Q_aug), 0.0), (
        "augmented QUBO must be purely off-diagonal"
    )
    # The linear terms are now couplings to the always-occupied ancilla mode.
    assert np.allclose(Q_aug[:n, n], np.diag(Q) / 2)
    assert np.allclose(Q_aug[n, :n], np.diag(Q) / 2)


def test_augmented_qubo_preserves_the_objective(clique_graph):
    """With the ancilla pinned to 1, the augmented energy must match the original."""
    Q = build_qubo("max-cut", clique_graph)
    Q_aug, _ = augment_qubo(Q)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, size=Q.shape[0])
        assert qubo_objective(Q_aug, [*x, 1]) == pytest.approx(qubo_objective(Q, x))


def test_normalize_zeroes_the_diagonal_and_bounds_the_rest(clique_graph):
    Q = torch.from_numpy(build_qubo("max-cut", clique_graph)).float().unsqueeze(0)
    Q_norm = normalize_qubo(Q)
    assert torch.allclose(Q_norm.diagonal(dim1=-2, dim2=-1), torch.zeros(1))
    assert float(Q_norm.min()) >= 0.0 and float(Q_norm.max()) <= 1.0


def test_normalize_handles_an_all_zero_offdiagonal():
    """A QUBO with no couplings must not divide by a zero range."""
    Q = torch.zeros(1, 3, 3)
    assert torch.allclose(normalize_qubo(Q), torch.zeros(1, 3, 3))


def test_theta_ordering_matches_the_upper_triangle():
    """The anchor layer consumes theta in strict-upper-triangular row-major order.

    ``_add_anchor_layers`` walks pairs as ``(0,1), (0,2), ..., (1,2), ...``; if the
    encoder emitted a different order every beam splitter would get the wrong
    coefficient, so this pins the two together.
    """
    size, num_rep = 4, 10
    Q_norm = torch.zeros(1, size, size)
    pairs = [(a, b) for a in range(size) for b in range(a + 1, size)]
    for idx, (row, col) in enumerate(pairs):
        Q_norm[0, row, col] = Q_norm[0, col, row] = (idx + 1) / 10

    theta = qubo_norm_to_theta(Q_norm, num_rep)[0]
    assert theta.numel() == size * (size - 1) // 2
    # Monotone in the pair index, because the weights were assigned that way.
    assert torch.all(theta[1:] > theta[:-1])


def test_theta_implements_the_anchor_relation():
    """theta = 0.5 * arccos(sqrt(1 - w)) with w = Q_ij / num_rep**2."""
    num_rep = 10
    # float64 so this checks the formula rather than float32 rounding -- the models
    # themselves run in float32, where the same value agrees only to ~1e-6.
    Q_norm = torch.tensor([[[0.0, 0.64], [0.64, 0.0]]], dtype=torch.float64)
    theta = qubo_norm_to_theta(Q_norm, num_rep)[0, 0]
    weight = 0.64 / num_rep**2
    assert float(theta) == pytest.approx(
        0.5 * np.arccos(np.sqrt(1 - weight)), rel=1e-12
    )


def test_theta_is_clamped_at_full_coupling():
    """w = 1 would make arccos(sqrt(0)) exact; clamping must keep it finite."""
    Q_norm = torch.ones(1, 2, 2)
    theta = qubo_norm_to_theta(Q_norm, 1)
    assert torch.isfinite(theta).all()


@pytest.mark.parametrize("size,expected", [(1, 2), (2, 10), (5, 34), (8, 58)])
def test_expected_coeff_count(size, expected):
    assert expected_coeff_count(size) == expected


def test_decode_picks_the_best_bitstring(clique_graph):
    """A distribution peaked on the optimum must decode to the optimum."""
    from utils.qubo import solve_qubo_bruteforce

    Q = build_qubo("max-clique", clique_graph)
    best_bits, best_energy = solve_qubo_bruteforce(Q)

    # Fock keys are the bitstrings themselves here (one photon per selected mode).
    keys = [
        tuple(int(b) for b in np.binary_repr(v, Q.shape[0]))
        for v in range(2 ** Q.shape[0])
    ]
    distribution = [0.0] * len(keys)
    distribution[keys.index(tuple(int(b) for b in best_bits))] = 1.0

    result = distribution_to_result(distribution, Q, 0, keys)
    assert result.objective == pytest.approx(-best_energy)
    assert result.bitstring == [int(b) for b in best_bits]


def test_decode_rejects_a_mismatched_basis(clique_graph):
    Q = build_qubo("max-clique", clique_graph)
    with pytest.raises(ValueError, match="output"):
        distribution_to_result([1.0, 0.0], Q, 0, [(1, 1, 1, 1, 1)])


def test_energy_table_matches_direct_evaluation(clique_graph):
    """Lazy and dense paths must agree with a plain objective evaluation."""
    Q = build_qubo("max-clique", clique_graph)
    keys = [tuple(int(b) for b in np.binary_repr(v, Q.shape[0])) for v in range(8)]
    table = EnergyTable(Q, keys)

    dense = table.full()
    assert dense.shape == (len(keys),)
    for index, key in enumerate(keys):
        expected = qubo_objective(Q, [1 if n >= 1 else 0 for n in key])
        assert float(dense[index]) == pytest.approx(expected, abs=1e-5)
    # for_indices is the same table, restricted.
    assert torch.allclose(table.for_indices([2, 5]), dense[[2, 5]])


def test_energy_table_ignores_the_ancilla_mode(clique_graph):
    """Augmented circuits emit n+1 modes; the extra one must not reach the energy."""
    Q = build_qubo("max-clique", clique_graph)
    n = Q.shape[0]
    keys = [tuple([1] * n + [photons]) for photons in (0, 1, 2)]
    energies = EnergyTable(Q, keys).full()
    assert len(set(energies.tolist())) == 1


def test_number_mapping_thresholds_on_photon_presence():
    assert number_mapping((3, 0, 2, 1), 4).tolist() == [1, 0, 1, 1]


def test_number_mapping_drops_the_ancilla_mode():
    """Slicing to ``size`` is what makes an augmented circuit decode correctly."""
    assert number_mapping((1, 0, 1, 1), 3).tolist() == [1, 0, 1]


def test_number_mapping_invert_matches_the_upstream_formula():
    """Behaviour lock against the reference readout.

    Upstream (``run_photonic_cvarvqe._parify_samples_threshold``) computes
    ``(int(n == 0) + j) % 2``. That is this mapping for ``j = 1`` and its inverse
    for ``j = 0`` -- not photon-number parity. Any drift here silently changes the
    CVaR-VQE baseline.
    """
    for occupation in [(0, 1, 2, 0), (1, 1, 1, 1), (3, 0, 2, 1), (0, 0, 0, 0)]:
        for j in (0, 1):
            upstream = [(int(n == 0) + j) % 2 for n in occupation]
            assert (
                number_mapping(occupation, len(occupation), invert=(j == 0)).tolist()
                == upstream
            )
