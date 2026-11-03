"""Ridge readouts, lambda selection, distributed partition and architecture specs."""

from __future__ import annotations

import common  # noqa: F401
import numpy as np
import pytest
import torch
from distributed_qrc.lib.architectures import build_cell_spec
from distributed_qrc.lib.kernels import bloch_vectors, gram_matrix
from distributed_qrc.lib.noise import IDEAL, devices_for_components
from distributed_qrc.lib.readouts import (
    fit_kernel_ridge,
    fit_linear_ridge,
    merge_predictions,
    partition_features,
    select_lambda,
)


def toy_problem(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    features = {s: rng.uniform(-1, 1, (n, d)) for s in ("train", "eval", "test")}
    coef = rng.normal(size=d)
    targets = {s: np.sin(x @ coef) * 0.3 + 0.5 for s, x in features.items()}
    return features, targets


def test_linear_ridge_matches_least_squares_for_tiny_lambda():
    features, targets = toy_problem()
    fitted = fit_linear_ridge(features, targets, [1e-12])
    design = np.c_[features["train"], np.ones(len(features["train"]))]
    coef, *_ = np.linalg.lstsq(design, targets["train"], rcond=None)
    expected = np.c_[features["test"], np.ones(len(features["test"]))] @ coef
    assert np.allclose(fitted.predictions["test"], expected, atol=1e-8)


def test_kernel_ridge_matches_direct_dual_solution():
    features, targets = toy_problem(n=200)
    lam = 1e-3
    fitted = fit_kernel_ridge(
        features,
        targets,
        [lam],
        n_qubits=3,
        noise=IDEAL,
        device="cpu",
        selection_subsample=None,
        fit_subsample=None,
    )
    b_train = torch.from_numpy(bloch_vectors(features["train"], 3))
    b_test = torch.from_numpy(bloch_vectors(features["test"], 3))
    gram = gram_matrix(b_train, 1.0).numpy()
    y_mean = targets["train"].mean()
    alpha = np.linalg.solve(gram + 200 * lam * np.eye(200), targets["train"] - y_mean)
    cross = gram_matrix(torch.cat([b_test, b_train]), 1.0).numpy()[:200, 200:]
    assert np.allclose(fitted.predictions["test"], cross @ alpha + y_mean, atol=1e-8)
    assert fitted.details["layers"] == 2


def test_lambda_selection_prefers_larger_lambda_within_tolerance():
    assert select_lambda({1e-6: 0.1000, 1e-3: 0.1005, 1.0: 0.2}) == 1e-3
    assert select_lambda({1e-6: 0.1000, 1e-3: 0.1200}) == 1e-6


def test_partition_and_merge():
    chunks = partition_features(10, 2)
    assert [c.tolist() for c in chunks] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert np.allclose(merge_predictions([np.ones(3), 3 * np.ones(3)], "mean"), 2.0)
    assert np.allclose(merge_predictions([np.ones(3), 3 * np.ones(3)], "sum"), 4.0)


@pytest.mark.parametrize(
    "architecture,size,reservoirs,instances,qubits",
    [
        ("srsr", 15, (15,), 1, 10),
        ("mrsr", "3x20", (20, 20, 20), 1, 10),
        ("srmr", [10, 2], (10,), 2, 5),
        ("srmr", [40, 2], (40,), 2, 10),
        ("srmr", [25, 5], (25,), 5, 5),
        ("mrmr", "5x5", (5,) * 5, 5, 5),
        ("mrmr", "2x20", (20, 20), 2, 10),
    ],
)
def test_cell_specs_follow_tables(architecture, size, reservoirs, instances, qubits):
    spec = build_cell_spec(architecture, size)
    assert (spec.reservoir_sizes, spec.n_instances, spec.kernel_qubits) == (
        reservoirs,
        instances,
        qubits,
    )


def test_table_iii_device_mapping():
    assert devices_for_components(3, "paper") == [
        "ibm_marrakesh",
        "ibm_brisbane",
        "ionq_aria1",
    ]
    assert devices_for_components(2, "ionq_aria1") == ["ionq_aria1", "ionq_aria1"]


def test_kernel_ridge_with_finite_shots_runs_and_degrades_gracefully():
    from distributed_qrc.lib.noise import DeviceNoise

    features, targets = toy_problem(n=300)
    noise = DeviceNoise("test", 2e-4, 7e-3, 1.6e-2)
    exact = fit_kernel_ridge(
        features, targets, [1e-6, 1e-4, 1e-2], 3, noise, "cpu", None, None
    )
    shot = fit_kernel_ridge(
        features,
        targets,
        [1e-6, 1e-4, 1e-2],
        3,
        noise,
        "cpu",
        None,
        None,
        shots=256,
        shot_seed=1,
    )
    assert shot.details["shots"] == 256
    assert np.isfinite(shot.predictions["test"]).all()

    def test_rmse(fitted):
        return np.sqrt(np.mean((fitted.predictions["test"] - targets["test"]) ** 2))

    assert test_rmse(shot) >= test_rmse(exact) * 0.9


def test_linear_ridge_without_intercept_is_paper_eq5():
    features, targets = toy_problem()
    lam = 1e-3
    fitted = fit_linear_ridge(features, targets, [lam], intercept=False)
    r, y = features["train"].T, targets["train"]
    n = y.size
    w = y @ r.T @ np.linalg.inv(r @ r.T + n * lam * np.eye(r.shape[0]))
    assert np.allclose(fitted.predictions["test"], features["test"] @ w, atol=1e-10)


def test_noise_scale_multiplies_error_rates():
    from distributed_qrc.lib.noise import device_from_config

    devices = {"dev": {"error_1q": 1e-4, "error_2q": 1e-2, "readout_error": 2e-2}}
    scaled = device_from_config("dev", devices, scale=3.0)
    assert (scaled.error_1q, scaled.error_2q, scaled.readout_error) == pytest.approx(
        (3e-4, 3e-2, 6e-2)
    )
    with pytest.raises(ValueError):
        device_from_config("dev", devices, scale=30.0)
