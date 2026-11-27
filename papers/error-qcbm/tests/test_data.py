"""Independent checks of bin ordering and the known Gaussian target."""

from itertools import product
from math import comb, exp, pi, sqrt

import numpy as np
import pytest
from lib.data import collision_free_states, gaussian_target


def test_basis_matches_independent_binary_enumeration_and_author_key():
    states = collision_free_states(5, 2)
    expected = tuple(state for state in product((0, 1), repeat=5) if sum(state) == 2)
    assert states == expected
    author_keys = [
        sum(value * 6 ** (5 - i) for i, value in enumerate(state)) for state in states
    ]
    assert author_keys == sorted(author_keys)
    assert len(collision_free_states(12, 4)) == comb(12, 4) == 495


def test_default_gaussian_matches_scalar_density_definition():
    expected = np.array(
        [
            sum(
                exp(-((x - mean) ** 2) / (2 * 61**2)) / (61 * sqrt(2 * pi))
                for mean in (141, 353)
            )
            for x in range(495)
        ]
    )
    expected /= expected.sum()
    actual = gaussian_target()
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=1e-18)
    assert actual.dtype == np.float64
    assert np.all(actual > 0)
    assert actual.sum() == pytest.approx(1)
    assert actual[141] > actual[247]
    np.testing.assert_allclose(actual, actual[::-1], atol=1e-17)


@pytest.mark.parametrize("modes, photons", [(0, 0), (3, -1), (3, 4), (True, 1)])
def test_invalid_basis_is_rejected(modes, photons):
    with pytest.raises(ValueError):
        collision_free_states(modes, photons)


def test_invalid_gaussian_width_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        gaussian_target(standard_deviations=(0, 61))


def test_smaller_gaussian_preserves_the_author_integer_scaling():
    target = gaussian_target(num_bins=20)
    explicit = gaussian_target(
        num_bins=20, means=(2 * 20 // 7, 5 * 20 // 7), standard_deviations=(2, 2)
    )
    np.testing.assert_array_equal(target, explicit)
    with pytest.raises(ValueError, match="positive"):
        gaussian_target(num_bins=7)
