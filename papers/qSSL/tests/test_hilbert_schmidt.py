import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.linalg import matrix_power

# Ensure qSSL/ is on path so `lib` is importable
THIS_DIR = Path(__file__).resolve().parent
QSSL_DIR = THIS_DIR.parent
if str(QSSL_DIR) not in sys.path:
    sys.path.insert(0, str(QSSL_DIR))

from lib.training_utils import (  # noqa: E402
    compute_batch_hilbert_schmidt_metrics,
    compute_batch_probability_hilbert_schmidt_metrics,
)


def _random_state(rng, dim):
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def _brute_force_dhs_metrics(aug_1, aug_2):
    """Naive O(B^2) reference matching the positive-pair analysis in the
    original bjader/QSSL reproduction (train_simclr.py)."""
    positive_pairs = list(zip(aug_1, aug_2))
    rhos, sigmas = [], []
    for i, pair in enumerate(positive_pairs):
        rho = np.mean([np.outer(v, np.conj(v)) for v in pair], axis=0)
        rhos.append(rho)
        negatives = positive_pairs[:i] + positive_pairs[i + 1 :]
        sigma = np.mean(
            [np.outer(v, np.conj(v)) for neg_pair in negatives for v in neg_pair],
            axis=0,
        )
        sigmas.append(sigma)

    return {
        "rho_squared": float(
            np.mean([np.trace(matrix_power(rho, 2)) for rho in rhos]).real
        ),
        "sigma_squared": float(
            np.mean([np.trace(matrix_power(sigma, 2)) for sigma in sigmas]).real
        ),
        "rho_sigma": float(
            np.mean([np.trace(rho @ sigma) for rho, sigma in zip(rhos, sigmas)]).real
        ),
        "d_hs": float(
            np.mean(
                [
                    np.trace(matrix_power(rho - sigma, 2))
                    for rho, sigma in zip(rhos, sigmas)
                ]
            ).real
        ),
    }


def test_hilbert_schmidt_metrics_matches_brute_force_reference():
    rng = np.random.default_rng(0)
    batch_size, dim = 6, 8
    aug_1 = [_random_state(rng, dim) for _ in range(batch_size)]
    aug_2 = [_random_state(rng, dim) for _ in range(batch_size)]
    statevectors = np.array(aug_1 + aug_2)

    expected = _brute_force_dhs_metrics(aug_1, aug_2)
    actual = compute_batch_hilbert_schmidt_metrics(statevectors)

    for key in ("rho_squared", "sigma_squared", "rho_sigma", "d_hs"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-9)


def test_hilbert_schmidt_metrics_zero_for_identical_ensembles():
    # If every statevector in the batch is identical, positive and negative
    # ensembles coincide, so the Hilbert-Schmidt distance should vanish.
    rng = np.random.default_rng(1)
    dim = 4
    state = _random_state(rng, dim)
    batch_size = 5
    statevectors = np.tile(state, (2 * batch_size, 1))

    metrics = compute_batch_hilbert_schmidt_metrics(statevectors)

    assert metrics["d_hs"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rho_squared"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["sigma_squared"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["rho_sigma"] == pytest.approx(1.0, abs=1e-9)


def test_probability_hilbert_schmidt_metrics_matches_diagonal_density_matrices():
    probabilities = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
        ]
    )
    actual = compute_batch_probability_hilbert_schmidt_metrics(probabilities)

    # For diagonal density matrices, the metric is the squared Euclidean
    # distance between probability vectors, not the pure-state metric above.
    positive_0 = (probabilities[0] + probabilities[2]) / 2
    positive_1 = (probabilities[1] + probabilities[3]) / 2
    negative_0 = positive_1
    negative_1 = positive_0
    expected_distances = [
        np.dot(positive_0 - negative_0, positive_0 - negative_0),
        np.dot(positive_1 - negative_1, positive_1 - negative_1),
    ]
    assert actual["d_hs"] == pytest.approx(np.mean(expected_distances))
    assert actual["rho_squared"] == pytest.approx(
        np.mean([np.dot(positive_0, positive_0), np.dot(positive_1, positive_1)])
    )
    assert actual["sigma_squared"] == pytest.approx(actual["rho_squared"])
    assert actual["rho_sigma"] == pytest.approx(
        np.mean([np.dot(positive_0, negative_0), np.dot(positive_1, negative_1)])
    )


def test_probability_hilbert_schmidt_metrics_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="sum to one"):
        compute_batch_probability_hilbert_schmidt_metrics(
            np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 0.2], [0.8, 0.8]])
        )
