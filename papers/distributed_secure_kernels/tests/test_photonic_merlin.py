from __future__ import annotations

import math

import common  # noqa: F401
import numpy as np
import pytest
import torch
from distributed_secure_kernels.lib.photonic_merlin import (
    PhotonicSwapTest,
    encode_registers,
    register_modes,
)


@pytest.mark.parametrize("features, modes", [(13, 6), (14, 6), (22, 8), (64, 12)])
def test_register_sizes_used_by_the_configs(features, modes):
    assert register_modes(features) == modes
    assert math.comb(modes, 2) >= features > math.comb(modes - 1, 2)


def _states(seed, count, modes):
    rng = np.random.default_rng(seed)
    return encode_registers(rng.normal(size=(count, math.comb(modes, 2) - 1)), modes)


def test_parity_probability_is_the_swap_test_law():
    test = PhotonicSwapTest(4)
    left, right = _states(0, 20, 4), _states(1, 20, 4)
    expected = 0.5 + 0.5 * np.einsum("ij,ij->i", left, right) ** 2
    np.testing.assert_allclose(test.even_probability(left, right), expected, atol=1e-9)
    np.testing.assert_allclose(
        test.even_probability(left[:3], left[:3]), 1.0, atol=1e-9
    )


def test_fock_space_with_four_photons_is_used():
    test = PhotonicSwapTest(4)
    keys = test.layer.output_keys
    assert all(sum(key) == 4 for key in keys)
    # FOCK: all occupations of 4 photons in 8 modes (bunched ones included).
    assert len(keys) == math.comb(8 + 4 - 1, 4)
    assert any(max(key) > 1 for key in keys)


def test_symmetric_matrix_matches_the_full_computation():
    test = PhotonicSwapTest(4)
    states = _states(2, 6, 4)
    full = test.zero_probability_matrix(states, states, symmetric=False)
    np.testing.assert_allclose(
        test.zero_probability_matrix(states, states, symmetric=True), full, atol=1e-12
    )
    np.testing.assert_allclose(full, full.T, atol=1e-12)


def test_merlin_shot_sampling_gives_bernoulli_parity():
    test = PhotonicSwapTest(4)
    left, right = _states(3, 1, 4), _states(4, 1, 4)
    p_even = test.even_probability(left, right)[0]
    test.layer.eval()
    torch.manual_seed(0)
    shots = 20_000
    with torch.no_grad():
        frequencies = test.layer(test._joint_states(left, right), shots=shots)
    observed = float((frequencies * test.even_mask).sum())
    if observed > 1.5:  # counts rather than frequencies
        observed /= shots
    assert abs(observed - p_even) < 4 * math.sqrt(p_even * (1 - p_even) / shots)


def test_runner_kernel_checks_exactness_and_uses_the_config():
    from distributed_secure_kernels.lib.runner import photonic_swap_test_kernel_fn

    method = {"shots": 1024, "exactness_tolerance": 1e-6, "device": "cpu"}
    kernel = photonic_swap_test_kernel_fn(method, np.random.default_rng(0))
    features = np.random.default_rng(5).normal(size=(5, 13))
    estimate = kernel(features, features, True)
    assert estimate.shape == (5, 5)
    np.testing.assert_allclose(np.diag(estimate), 1.0)
