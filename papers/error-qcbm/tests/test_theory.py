"""Analytic checks for the independent permutation-sum reference."""

import numpy as np
import pytest
from lib.noise import load_profiles
from lib.theory import theoretical_distribution


def test_reference_hom_and_loss():
    unitary = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    hom, survival = 0.729, 0.028
    distribution = theoretical_distribution(
        unitary, (1, 1), survival, hom, threshold=False
    )
    assert distribution[(1, 1)] == pytest.approx(survival**2 * (1 - hom) / 2)
    assert distribution[(0, 0)] == pytest.approx((1 - survival) ** 2)
    assert sum(
        value for state, value in distribution.items() if sum(state) == 1
    ) == pytest.approx(2 * survival * (1 - survival))


def test_reference_ideal_hom_threshold_collision():
    unitary = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    distribution = theoretical_distribution(unitary, (1, 1))
    assert distribution[(1, 1)] == pytest.approx(0)
    assert distribution[(1, 0)] == pytest.approx(0.5)
    assert distribution[(0, 1)] == pytest.approx(0.5)


def test_noise_snapshot_missing_is_not_substituted(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profiles(tmp_path / "absent.json")
