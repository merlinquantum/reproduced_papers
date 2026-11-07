from __future__ import annotations

import common  # noqa: F401  (puts the paper package on sys.path)
import numpy as np
from qelm_concentration.lib.eigentasks import (
    fit_local_eigentasks,
    nsr_cutoff_mask,
    resolvable_expressive_capacity,
    subset_eigentasks,
)
from qelm_concentration.lib.observables import (
    PAULI_MATRICES,
    TWO_QUBIT_PAULIS,
    pair_povm_effects,
    pair_povm_probabilities,
    pauli_expectations_from_rdms,
    sample_pair_frequencies,
    shadow_gaussian_noise,
    shadow_pair_paulis,
    shadow_single_paulis,
    single_qubit_frequencies,
)


def _random_rdms(n_samples: int, n_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_samples, n_pairs, 4, 4)) + 1j * rng.normal(
        size=(n_samples, n_pairs, 4, 4)
    )
    rho = raw @ np.conj(np.swapaxes(raw, -1, -2))
    return rho / np.trace(rho, axis1=-2, axis2=-1)[..., None, None]


def test_povm_effects_are_complete():
    np.testing.assert_allclose(pair_povm_effects().sum(axis=0), np.eye(4), atol=1e-12)


def test_shadow_estimators_are_exact_at_infinite_shots():
    rdms = _random_rdms(5, 3, seed=0)
    probabilities = pair_povm_probabilities(rdms)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    exact_pairs = pauli_expectations_from_rdms(rdms, TWO_QUBIT_PAULIS)
    np.testing.assert_allclose(
        shadow_pair_paulis(probabilities, TWO_QUBIT_PAULIS), exact_pairs, atol=1e-12
    )

    singles = shadow_single_paulis(single_qubit_frequencies(probabilities), (1, 2, 3))
    first_qubit = pauli_expectations_from_rdms(rdms, [(1, 0), (2, 0), (3, 0)])
    second_qubit = pauli_expectations_from_rdms(rdms, [(0, 1), (0, 2), (0, 3)])
    np.testing.assert_allclose(singles[:, 0::2], first_qubit, atol=1e-12)
    np.testing.assert_allclose(singles[:, 1::2], second_qubit, atol=1e-12)


def test_shadow_estimator_variance_matches_formula():
    rdms = _random_rdms(1, 1, seed=1)
    probabilities = pair_povm_probabilities(rdms)
    rng = np.random.default_rng(2)
    shots = 400
    estimates = np.stack(
        [
            shadow_pair_paulis(
                sample_pair_frequencies(probabilities, shots, rng), [(3, 3), (1, 0)]
            )[0, 0]
            for _ in range(4000)
        ]
    )
    exact = pauli_expectations_from_rdms(rdms, [(3, 3), (1, 0)])[0, 0]
    expected_var = (np.array([9.0, 3.0]) - exact**2) / shots
    np.testing.assert_allclose(estimates.var(axis=0), expected_var, rtol=0.1)
    noisy = shadow_gaussian_noise(
        np.tile(exact, (4000, 1)), np.array([2.0, 1.0]), shots, rng
    )
    np.testing.assert_allclose(noisy.var(axis=0), expected_var, rtol=0.1)


def test_eigentasks_solve_generalised_problem():
    rdms = _random_rdms(300, 1, seed=3)
    probabilities = pair_povm_probabilities(rdms)[:, 0]
    shots = 1000
    frequencies = sample_pair_frequencies(
        probabilities[:, None], shots, np.random.default_rng(4)
    )[:, 0]
    coefficients, nsr = subset_eigentasks(frequencies, shots, operator_dim=16)
    assert coefficients.shape == (36, 15)
    assert np.all(np.diff(nsr) >= -1e-9)

    n = len(frequencies)
    second = frequencies.T @ frequencies / n
    noise = (np.diag(frequencies.mean(0)) - second) * shots / (shots - 1)
    gram = second - noise / shots
    np.testing.assert_allclose(
        coefficients.T @ gram @ coefficients, np.eye(15), atol=1e-8
    )
    np.testing.assert_allclose(
        noise @ coefficients, gram @ coefficients * nsr, atol=1e-8
    )


def test_local_eigentask_counts_per_subset():
    rdms = _random_rdms(200, 2, seed=5)
    frequencies = sample_pair_frequencies(
        pair_povm_probabilities(rdms), 500, np.random.default_rng(6)
    )
    pairs = fit_local_eigentasks(frequencies, 500, subset_size=2)
    singles = fit_local_eigentasks(
        single_qubit_frequencies(frequencies), 500, subset_size=1
    )
    assert pairs.nsr.size == 2 * 15
    assert singles.nsr.size == 4 * 3
    assert pairs.transform(frequencies).shape == (200, 30)


def test_cutoff_and_rec():
    nsr = np.array([10.0, 100.0, 1000.0, 1e5])
    np.testing.assert_array_equal(
        nsr_cutoff_mask(nsr, shots=100, cutoff=1.0), [True, True, False, False]
    )
    assert np.isclose(
        resolvable_expressive_capacity(nsr, 100), 1 / 1.1 + 1 / 2 + 1 / 11 + 1 / 1001
    )
    assert np.isclose(resolvable_expressive_capacity(nsr, 100, cutoff=1.0), 1 / 1.1)


def test_pauli_matrices_are_standard():
    np.testing.assert_allclose(
        PAULI_MATRICES[1] @ PAULI_MATRICES[2], 1j * PAULI_MATRICES[3]
    )


def test_empty_cutoff_readout_falls_back_to_intercept_only():
    from qelm_concentration.lib.landsat_task import _evaluate_readout

    rdms = _random_rdms(60, 2, seed=7)
    frequencies = sample_pair_frequencies(
        pair_povm_probabilities(rdms), 200, np.random.default_rng(8)
    )
    labels = np.array([1, 2, 3] * 20)
    labels[:5] = 1
    result = _evaluate_readout(
        "w2_etcut_scaled",
        frequencies[:40],
        frequencies[40:],
        labels[:40],
        labels[40:],
        200,
        1e-9,
        np.logspace(-3, 1, 5),
    )
    assert result["empty_readout"] and result["n_features"] == 0
    assert result["n_eigentasks_retained"] == 0
