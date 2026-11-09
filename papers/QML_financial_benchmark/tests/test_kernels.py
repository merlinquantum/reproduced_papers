from __future__ import annotations

import numpy as np
import torch
from lib import statevector as sv
from lib.kernels import amplitude_kernel, angle_feature_states, fidelity_kernel


def test_amplitude_kernel_equals_explicit_random_unitary_fidelity():
    rng = np.random.default_rng(0)
    features = rng.random((6, 8))
    unitary = sv.haar_random_unitary(8, torch.Generator().manual_seed(1), torch.float64)
    normalized = torch.as_tensor(
        features / np.linalg.norm(features, axis=1, keepdims=True),
        dtype=torch.complex128,
    )
    states = normalized @ unitary.transpose(0, 1)
    for beta in (1, 2, 3):
        assert np.allclose(
            fidelity_kernel(states, states, beta),
            amplitude_kernel(features, features, beta),
        )


def test_angle_kernel_without_layers_is_product_of_cosines():
    rng = np.random.default_rng(1)
    features = rng.random((5, 3))
    states = angle_feature_states(
        features, layers=0, topology="ring", angle_scale=np.pi, rotation_seed=0
    )
    angles = np.pi * features
    expected = np.prod(
        np.cos((angles[:, None, :] - angles[None, :, :]) / 2) ** 2, axis=-1
    )
    assert np.allclose(fidelity_kernel(states, states), expected)


def test_angle_kernel_with_layers_is_a_valid_gram_matrix():
    features = np.random.default_rng(2).random((7, 4))
    gram = fidelity_kernel(*(angle_feature_states(features, 2, "ring", np.pi, 3),) * 2)
    assert np.allclose(np.diag(gram), 1.0)
    assert np.allclose(gram, gram.T)
    assert np.linalg.eigvalsh(gram).min() > -1e-10
