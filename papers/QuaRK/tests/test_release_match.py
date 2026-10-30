"""Our generator and featurizer against the authors' release (seed 12345).

Skipped when the release is not under the data root (see LOG.md "Data Acquisition Log").
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from common import PROJECT_DIR
from QuaRK.lib.data import check_against_release, generate_dataset, release_dataset_path
from QuaRK.lib.measurement import pauli_features
from QuaRK.lib.reservoir import draw_reservoir_parameters, reservoir_states

from runtime_lib.data_paths import resolve_data_root

RUN_ARRAYS = (
    "QuaRK/authors_release_v1.0.0/storage/results/reg_sweep/c-shadows/full/"
    "N=10000__w=25__d=3__s=100/2026-02-02_14-02-13/model/arrays.npz"
)


@pytest.fixture(scope="module")
def paper_data_cfg():
    cfg = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())["data"]
    cfg["num_windows"] = 10000
    if not release_dataset_path(cfg, None).is_file():
        pytest.skip("authors' release not downloaded")
    return cfg


def test_dataset_and_split_match_release(paper_data_cfg):
    dataset = generate_dataset(paper_data_cfg, 12345)
    check = check_against_release(dataset, paper_data_cfg, None)
    assert check["max_abs_diff_windows"] < 1e-12
    assert check["max_abs_diff_labels"] < 1e-12
    arrays = np.load(resolve_data_root(None) / RUN_ARRAYS)
    np.testing.assert_array_equal(arrays["train_idx"], dataset.train_idx)
    np.testing.assert_array_equal(arrays["test_idx"], dataset.test_idx)


def test_code_tasks_are_linear_in_the_window(paper_data_cfg):
    """F1 (code) is linear in the last point and F2 in the whole window: zero residual."""
    dataset = generate_dataset(paper_data_cfg, 12345)
    n = dataset.windows.shape[0]
    last = np.c_[dataset.windows[:, -1, :], np.ones(n)]
    window = np.c_[dataset.windows.reshape(n, -1), np.ones(n)]
    for design, task in ((last, 0), (window, 1)):
        coef, *_ = np.linalg.lstsq(design, dataset.labels[task], rcond=None)
        assert np.mean((design @ coef - dataset.labels[task]) ** 2) < 1e-20
    coef, *_ = np.linalg.lstsq(window, dataset.labels[2], rcond=None)
    assert (
        np.mean((window @ coef - dataset.labels[2]) ** 2) > 0.1
    )  # Volterra is nonlinear


def test_features_reproduce_authors_phi_full(paper_data_cfg):
    dataset = generate_dataset(paper_data_cfg, 12345)
    params = draw_reservoir_parameters(
        input_dim=3, num_qubits=5, num_reservoirs=3, first_contraction=0.1, seed=12345
    )
    states = reservoir_states(dataset.windows, params)
    features = pauli_features(
        states, locality=2, estimator="per_observable_shots", shots=1000, seed=12345
    )
    reference = np.load(resolve_data_root(None) / RUN_ARRAYS)
    np.testing.assert_array_equal(features, reference["Phi_full"])
