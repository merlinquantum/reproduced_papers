from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from qelm_concentration.lib.data import (
    landsat_directory,
    make_landsat_split,
    make_narma_sequence,
    narma_series,
)
from qelm_concentration.lib.mps import minimal_bond_dimension, truncated_mps_fidelity


def test_bond_dimension_of_simple_states():
    product = np.zeros(2**6, dtype=complex)
    product[0] = 1
    assert minimal_bond_dimension(product, 6, 0.9995) == 1

    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    zero = np.array([1, 0], dtype=complex)
    middle_bell = np.kron(np.kron(zero, bell), zero)
    assert minimal_bond_dimension(middle_bell, 4, 0.9995) == 2


def test_binary_search_matches_linear_scan():
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = rng.normal(size=2**8) + 1j * rng.normal(size=2**8)
        state /= np.linalg.norm(state)
        linear = next(
            chi for chi in range(1, 17) if truncated_mps_fidelity(state, 8, chi) >= 0.99
        )
        assert minimal_bond_dimension(state, 8, 0.99) == linear
    assert np.isclose(truncated_mps_fidelity(state, 8, 16), 1.0)


def test_narma_recursion_and_window_alignment():
    inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    outputs = narma_series(inputs, order=2)
    expected_y2 = 0.2 * 0 + 0.04 * (0 + 0) + 1.5 * inputs[0] * inputs[1] + 0.001
    assert np.isclose(outputs[2], expected_y2)
    expected_y3 = (
        0.2 * outputs[2]
        + 0.04 * (outputs[1] + outputs[2])
        + 1.5 * inputs[1] * inputs[2]
        + 0.001
    )
    assert np.isclose(outputs[3], expected_y3)

    sequence = make_narma_sequence(
        4, [1, 3], 10, 20, 15, "unit_interval", np.random.default_rng(0)
    )
    assert sequence.train_windows.shape == (20, 4) and sequence.test_targets.shape == (
        15,
        2,
    )
    first = sequence.first_window_index
    np.testing.assert_allclose(
        sequence.train_windows[0], 4 * sequence.raw_inputs[first - np.arange(4)] - 1
    )
    np.testing.assert_allclose(
        sequence.train_targets[0, 1], narma_series(sequence.raw_inputs, 3)[first + 1]
    )


def test_landsat_split_is_stratified_and_scaled():
    cfg = load_runtime_ready_config()
    if not (landsat_directory(cfg) / "sat.trn").exists():
        pytest.skip("Landsat data not downloaded (see README Data section)")
    data = make_landsat_split(cfg, split_seed=0)
    assert data.train_x.shape == (688, 72) and data.test_x.shape == (172, 72)
    assert np.isclose(data.train_x.min(), -1) and np.isclose(data.train_x.max(), 1)
    assert set(np.unique(data.train_y)) == {1, 2, 3, 4, 5, 7}
    again = make_landsat_split(cfg, split_seed=0)
    np.testing.assert_array_equal(again.subset_indices, data.subset_indices)
    assert PROJECT_DIR.name == "qelm_concentration"
