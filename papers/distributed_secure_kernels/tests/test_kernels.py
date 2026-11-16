from __future__ import annotations

import common  # noqa: F401
import numpy as np
from distributed_secure_kernels.lib.encoding import amplitude_encode, shared_permutation
from distributed_secure_kernels.lib.kernels import (
    cosine_kernel,
    estimate_from_zero_probability,
    squared_overlap_kernel,
)


def test_permutation_matches_python_random_shuffle_seed_42():
    import random

    indices = list(range(16))
    random.seed(42)
    random.shuffle(indices)
    np.testing.assert_array_equal(shared_permutation(16), indices)


def test_encoding_is_padded_unit_norm_and_overlap_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 13))
    states = amplitude_encode(x)
    assert states.shape == (5, 16)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0)
    unpermuted = amplitude_encode(x, permute=False)
    np.testing.assert_allclose(states @ states.T, unpermuted @ unpermuted.T)


def test_swap_test_kernel_is_the_squared_cosine_similarity():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 13))
    cosine = (x @ x.T) / np.outer(np.linalg.norm(x, axis=1), np.linalg.norm(x, axis=1))
    np.testing.assert_allclose(cosine_kernel(x, x), cosine)
    np.testing.assert_allclose(squared_overlap_kernel(x, x), cosine**2)


def test_shot_estimator_is_unbiased_symmetric_and_clipped():
    rng = np.random.default_rng(2)
    overlap = np.array([[1.0, 0.3], [0.3, 1.0]])
    draws = np.array(
        [
            estimate_from_zero_probability(0.5 + 0.5 * overlap, 1024, rng, True)
            for _ in range(2000)
        ]
    )
    assert np.all(draws[:, 0, 1] == draws[:, 1, 0])
    np.testing.assert_allclose(draws[:, 0, 0], 1.0)
    assert abs(draws[:, 0, 1].mean() - 0.3) < 0.003
    # Standard deviation 2 sqrt(p (1 - p) / M) of App. B.1, p = 0.65.
    assert abs(draws[:, 0, 1].std() - 2 * np.sqrt(0.65 * 0.35 / 1024)) < 0.002
    zero_overlap = estimate_from_zero_probability(
        np.full((50, 50), 0.5), 64, rng, False
    )
    assert zero_overlap.min() == 0.0  # authors' clipping of negative estimates
