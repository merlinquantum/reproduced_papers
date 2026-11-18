"""Pauli features, randomized-measurement POVMs and finite-shot estimators.

Conventions:

- Single-qubit Pauli labels ``0, 1, 2, 3 = I, X, Y, Z``; measurement bases ``0, 1, 2 =
  X, Y, Z``; outcome bit ``o`` means eigenvalue ``(-1)**o``.
- Pair RDMs come from :func:`lib.reservoir.pair_reduced_density_matrices` with the
  first qubit of the pair as the most significant bit.
- Randomized local Pauli measurement on a pair (paper App. "Measurement model"): each
  shot draws a uniformly random basis per qubit. Its 36 POVM effects are
  ``M[(a, b), (o1, o2)] = (1/9) P_a^{o1} (x) P_b^{o2}`` with flat index
  ``k = ((3 a + b) * 4) + 2 o1 + o2``.
"""

from __future__ import annotations

import numpy as np

PAULI_MATRICES = np.array(
    [
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)
PAULI_LABELS = "IXYZ"
# The 15 non-identity two-qubit Pauli strings, as (left, right) label indices.
TWO_QUBIT_PAULIS = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]
# Strings built from {I, X} only: the paper's "X" readout on a pair.
TWO_QUBIT_X_PAULIS = [(1, 0), (0, 1), (1, 1)]
N_PAIR_EFFECTS = 36
N_SINGLE_EFFECTS = 6


def _basis_projectors() -> np.ndarray:
    """Projectors ``(I + (-1)**o P_a) / 2`` for bases X, Y, Z; shape ``(3, 2, 2, 2)``."""
    projectors = np.empty((3, 2, 2, 2), dtype=complex)
    for basis in range(3):
        for outcome in range(2):
            sign = 1 - 2 * outcome
            projectors[basis, outcome] = (
                PAULI_MATRICES[0] + sign * PAULI_MATRICES[basis + 1]
            ) / 2
    return projectors


def pair_povm_effects() -> np.ndarray:
    """The 36 randomized-measurement effects on a pair; shape ``(36, 4, 4)``."""
    projectors = _basis_projectors()
    effects = np.empty((N_PAIR_EFFECTS, 4, 4), dtype=complex)
    for basis_a in range(3):
        for basis_b in range(3):
            for out_a in range(2):
                for out_b in range(2):
                    k = (3 * basis_a + basis_b) * 4 + 2 * out_a + out_b
                    effects[k] = (
                        np.kron(projectors[basis_a, out_a], projectors[basis_b, out_b])
                        / 9
                    )
    return effects


def pauli_expectations_from_rdms(
    rdms: np.ndarray, strings: list[tuple[int, int]]
) -> np.ndarray:
    """Exact ``<P_a (x) P_b>`` for every RDM.

    Parameters
    ----------
    rdms : numpy.ndarray
        Shape ``(n_samples, n_pairs, 4, 4)``.
    strings : list of tuple of int
        Pauli label pairs.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, n_pairs, len(strings))``.
    """
    operators = np.stack(
        [np.kron(PAULI_MATRICES[a], PAULI_MATRICES[b]) for a, b in strings]
    )
    return np.einsum("spij,oji->spo", rdms, operators).real


def tuning_features(ring_rdms: np.ndarray) -> np.ndarray:
    """Features of the hyperparameter study: ``X_i, Y_i, Z_i`` and ring ``XX, YY, ZZ``.

    Parameters
    ----------
    ring_rdms : numpy.ndarray
        RDMs of the ring pairs ``(i, i+1 mod N)``; shape ``(n_samples, N, 4, 4)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, 6 N)``: ``3 N`` weight-1 then ``3 N`` weight-2 features.
    """
    weight1 = pauli_expectations_from_rdms(ring_rdms, [(1, 0), (2, 0), (3, 0)])
    weight2 = pauli_expectations_from_rdms(ring_rdms, [(1, 1), (2, 2), (3, 3)])
    n_samples = ring_rdms.shape[0]
    return np.concatenate(
        [weight1.reshape(n_samples, -1), weight2.reshape(n_samples, -1)], axis=1
    )


def nearest_neighbour_features(ring_rdms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full nearest-neighbour Pauli set used in the larger experiments.

    Returns ``(features, weights)``: ``3 N`` single-qubit Paulis ``X_i, Y_i, Z_i`` followed
    by the ``9 N`` two-body strings ``P_i P'_{i+1}`` on the ring, and the Pauli weight
    (1 or 2) of each feature.
    """
    n_samples = ring_rdms.shape[0]
    singles = [(a, 0) for a in (1, 2, 3)]
    doubles = [(a, b) for a in (1, 2, 3) for b in (1, 2, 3)]
    weight1 = pauli_expectations_from_rdms(ring_rdms, singles).reshape(n_samples, -1)
    weight2 = pauli_expectations_from_rdms(ring_rdms, doubles).reshape(n_samples, -1)
    weights = np.concatenate([np.ones(weight1.shape[1]), 2 * np.ones(weight2.shape[1])])
    return np.concatenate([weight1, weight2], axis=1), weights


def shadow_gaussian_noise(
    features: np.ndarray, weights: np.ndarray, shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise with the classical-shadow estimator variance.

    A weight-``w`` Pauli estimated from ``S`` randomized local measurements has variance
    ``(3**w - <P>**2) / S``. Covariances between features are neglected (stated
    approximation for overlapping ring pairs, which cannot be sampled independently).
    """
    variance = (3.0 ** weights[None, :] - features**2) / shots
    return features + rng.standard_normal(features.shape) * np.sqrt(variance)


def pair_povm_probabilities(disjoint_rdms: np.ndarray) -> np.ndarray:
    """Born probabilities of the 36 effects for each disjoint pair.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, n_pairs, 36)``, real, clipped at 0 and renormalised to absorb
        float round-off.
    """
    effects = pair_povm_effects()
    probabilities = np.einsum("spij,kji->spk", disjoint_rdms, effects).real
    probabilities = np.clip(probabilities, 0.0, None)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def sample_pair_frequencies(
    probabilities: np.ndarray, shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Multinomial finite-shot frequencies of the pair POVM.

    Sampling each disjoint pair from its exact marginal reproduces the marginal
    statistics of full-register randomized measurements; correlations between pairs
    are not needed because every readout here is local to one pair.
    """
    counts = rng.multinomial(shots, probabilities)
    return counts / shots


def single_qubit_frequencies(pair_frequencies: np.ndarray) -> np.ndarray:
    """Marginalise pair frequencies to the 6 single-qubit effects of each qubit.

    Returns shape ``(n_samples, 2 * n_pairs, 6)`` ordered by qubit, with flat index
    ``2 * basis + outcome``.
    """
    n_samples, n_pairs, _ = pair_frequencies.shape
    grid = pair_frequencies.reshape(n_samples, n_pairs, 3, 3, 2, 2)
    first = grid.sum(axis=(3, 5)).reshape(n_samples, n_pairs, 6)
    second = grid.sum(axis=(2, 4)).reshape(n_samples, n_pairs, 6)
    return np.stack([first, second], axis=2).reshape(n_samples, 2 * n_pairs, 6)


def shadow_pair_paulis(
    pair_frequencies: np.ndarray, strings: list[tuple[int, int]]
) -> np.ndarray:
    """Classical-shadow estimates of two-qubit Pauli strings from pair frequencies.

    ``<P_a (x) P_b>`` is ``9 * sum (-1)**(o1+o2) f[(a, b), o]``; strings with an identity
    use ``3 * sum (-1)**o f`` over all settings of the other qubit.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, n_pairs, len(strings))``.
    """
    n_samples, n_pairs, _ = pair_frequencies.shape
    grid = pair_frequencies.reshape(n_samples, n_pairs, 3, 3, 2, 2)
    sign = np.array([1.0, -1.0])
    estimates = []
    for left, right in strings:
        if left and right:
            value = 9 * np.einsum(
                "spxy,x,y->sp", grid[:, :, left - 1, right - 1], sign, sign
            )
        elif left:
            value = 3 * np.einsum("spbxy,x->sp", grid[:, :, left - 1], sign)
        else:
            value = 3 * np.einsum("spaxy,y->sp", grid[:, :, :, right - 1], sign)
        estimates.append(value)
    return np.stack(estimates, axis=-1)


def shadow_single_paulis(
    single_frequencies: np.ndarray, bases: tuple[int, ...]
) -> np.ndarray:
    """Shadow estimates ``3 * sum_o (-1)**o f[a, o]`` for the given bases (1=X, 2=Y, 3=Z).

    Returns shape ``(n_samples, n_qubits, len(bases))``.
    """
    grid = single_frequencies.reshape(*single_frequencies.shape[:2], 3, 2)
    sign = np.array([1.0, -1.0])
    return np.stack([3 * grid[:, :, basis - 1] @ sign for basis in bases], axis=-1)
