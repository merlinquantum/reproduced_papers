"""Verify the scientifically significant loss direction and logarithm base."""

import numpy as np
import pytest
from lib.spsa import distribution_loss


def test_kl_is_model_to_target_in_bits_and_ignores_zero_model_mass():
    assert distribution_loss(np.array([1.0, 0.0]), np.array([0.25, 0.75]), "kl") == 2.0
    assert np.isinf(
        distribution_loss(np.array([0.25, 0.75]), np.array([1.0, 0.0]), "kl")
    )
    assert distribution_loss(np.array([0.5, 0.5]), np.array([0.5, 0.5]), "kl") == 0.0
    assert (
        distribution_loss(np.array([1.0, 0.0]), np.array([0.25, 0.75]), "tvd") == 0.75
    )


def test_loss_rejects_unnormalized_probabilities_instead_of_smoothing():
    with pytest.raises(ValueError, match="normalized"):
        distribution_loss(np.array([0.8, 0.8]), np.array([0.5, 0.5]), "kl")
