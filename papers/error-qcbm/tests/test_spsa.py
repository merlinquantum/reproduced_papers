"""Check the author's gain calibration, update schedule and failure semantics."""

import numpy as np
import pytest
from lib.spsa import SPSA

SETTINGS = {
    "alpha": 0.602,
    "gamma": 0.101,
    "c": 0.1,
    "stability_fraction": 0.1,
    "initial_step": 0.001,
    "gain_calibration": "author_absolute_mean",
}


def test_spsa_matches_author_schedule_and_rng_state():
    initial = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    calls = []

    def objective(parameters):
        calls.append(parameters.copy())
        return float(parameters @ parameters)

    optimizer = SPSA(initial, objective, SETTINGS, 1000, 4)
    rng = np.random.RandomState(4)

    def reference_gradient(parameters, radius):
        delta = rng.binomial(1, 0.5, len(parameters)) * 2 - 1
        positive, negative = parameters + radius * delta, parameters - radius * delta
        return (positive @ positive - negative @ negative) / (2 * radius * delta)

    gradient = reference_gradient(initial, 0.1)
    gain = 0.001 * 101**0.602 / abs(gradient.mean())
    expected = initial.copy()
    for step in range(1, 6):
        expected -= (
            gain
            / (step + 100) ** 0.602
            * reference_gradient(expected, 0.1 / step**0.101)
        )
        optimizer.step()
        assert optimizer.parameters == pytest.approx(expected, abs=1e-14)
    assert len(calls) == 12
    assert np.array_equal(initial, [1, 2, 3, 4, 5])
    checkpoint = optimizer.checkpoint()
    restored = np.random.RandomState()
    restored.set_state(
        (
            "MT19937",
            checkpoint["rng_keys"],
            checkpoint["rng_position"],
            checkpoint["rng_has_gaussian"],
            checkpoint["rng_cached_gaussian"],
        )
    )
    assert np.array_equal(restored.binomial(1, 0.5, 100), rng.binomial(1, 0.5, 100))


def test_degenerate_author_calibration_fails_without_epsilon_or_retry():
    with pytest.raises(ValueError, match="Degenerate"):
        SPSA(np.ones(5), lambda parameters: 0.5, SETTINGS, 1000, 1)


def test_in_memory_checkpoint_is_a_snapshot_and_restore_does_not_recalibrate():
    calls = []

    def objective(parameters):
        calls.append(1)
        return float(parameters @ parameters)

    optimizer = SPSA(np.arange(1.0, 6.0), objective, SETTINGS, 1000, 4)
    saved = optimizer.checkpoint()
    parameters = saved["parameters"].copy()
    keys = saved["rng_keys"].copy()
    optimizer.step()
    np.testing.assert_array_equal(saved["parameters"], parameters)
    np.testing.assert_array_equal(saved["rng_keys"], keys)
    count = len(calls)
    restored = SPSA.restore(saved, objective, SETTINGS)
    assert len(calls) == count
    assert restored.gain == saved["gain"]
    restored.step()
    np.testing.assert_array_equal(restored.parameters, optimizer.parameters)
