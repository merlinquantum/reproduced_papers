"""Additional classical baselines: RBF kernel ridge, persistence, tuned ESN."""

from __future__ import annotations

import json

import common  # noqa: F401
import numpy as np
import pytest
import torch
from common import load_runtime_ready_config
from distributed_qrc.lib.baselines_classical import (
    esn_settings,
    fit_persistence,
    fit_rbf_kernel_ridge,
    rbf_block,
    select_first_within_tolerance,
)


def _toy(n: int = 160, d: int = 5, seed: int = 6):
    rng = np.random.default_rng(seed)
    features = {s: rng.uniform(-1, 1, (n, d)) for s in ("train", "eval", "test")}
    targets = {s: 0.5 + 0.2 * np.sin(x.sum(1)) for s, x in features.items()}
    return features, targets


def test_rbf_block_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    expected = np.exp(-0.4 * ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    ours = rbf_block(torch.from_numpy(a), torch.from_numpy(b), 0.4).numpy()
    assert np.allclose(ours, expected, atol=1e-12)


def test_rbf_kernel_ridge_matches_direct_solution():
    features, targets = _toy()
    fitted = fit_rbf_kernel_ridge(features, targets, [1e-3], [1.0], "cpu", None, None)
    x, y = features["train"], targets["train"]
    gamma = 1.0 / x.shape[1]
    gram = np.exp(-gamma * ((x[:, None] - x[None]) ** 2).sum(-1))
    alpha = np.linalg.solve(gram + x.shape[0] * 1e-3 * np.eye(x.shape[0]), y - y.mean())
    cross = np.exp(-gamma * ((features["test"][:, None] - x[None]) ** 2).sum(-1))
    assert np.allclose(fitted.predictions["test"], cross @ alpha + y.mean(), atol=1e-8)
    assert fitted.details["gamma_factor_selected"] == pytest.approx(1.0)


def test_rbf_selection_is_joint_and_tie_breaks_to_smallest_gamma_then_largest_lambda():
    features, targets = _toy()
    grid, factors = [1e-6, 1e-4, 1e-2], [0.1, 1.0, 10.0]
    fitted = fit_rbf_kernel_ridge(features, targets, grid, factors, "cpu", 100, None)
    scores = {
        (float(g), float(lam)): v
        for g, row in fitted.details["selection_eval_rmse_by_gamma"].items()
        for lam, v in row.items()
    }
    assert len(scores) == 9 and fitted.details["selection_points"] == 100
    best = min(scores.values())
    tied = [k for k, v in scores.items() if v <= best * 1.01]
    gamma, lam = min(tied, key=lambda k: (k[0], -k[1]))
    assert fitted.details["gamma_selected"] == pytest.approx(gamma, rel=1e-3)
    assert fitted.lambda_selected == pytest.approx(lam, rel=1e-3)


def test_persistence_predicts_newest_lag_and_matches_shifted_target():
    cfg = load_runtime_ready_config()
    from distributed_qrc.lib.data import build_forecast_data

    data = build_forecast_data(cfg)
    fitted = fit_persistence(data.inputs, data.targets)
    for split in ("train", "eval", "test"):
        x, y = data.inputs[split], data.targets[split]
        assert np.array_equal(fitted.predictions[split], x[:, -1])
        # Window t+1 ends with the target of window t: the newest lag is x(t).
        assert np.array_equal(x[1:, -1], y[:-1])


def test_esn_grid_order_and_tie_rule():
    grid = load_runtime_ready_config()["classical_baselines"]["esn_grid"]
    settings = esn_settings(grid)
    assert len(settings) == 12
    assert settings[0] == {"sr": 0.5, "lr": 0.3, "input_scaling": 0.5}
    assert settings[1] == {"sr": 0.5, "lr": 0.3, "input_scaling": 1.0}
    assert settings[-1] == {"sr": 1.2, "lr": 1.0, "input_scaling": 1.0}
    assert select_first_within_tolerance([1.0, 0.995, 0.9, 0.905]) == 2


def test_tiny_baseline_grid_runs_and_esn_reuses_b0_matrices(tmp_path):
    from distributed_qrc.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    # raw_lags = no reservoir; esn_tuned = ESN tuned on eval.
    cfg["experiment"].update(
        sizes=[4],
        reservoir_types=["raw_lags", "classical", "esn_tuned"],
        readout_types=["persistence", "linear", "rbf_kernel"],
    )
    cfg["readout"].update(kernel_selection_subsample=150, kernel_fit_subsample=200)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="persistence needs the raw_lags"):
        paper_runner.train_and_evaluate(cfg, run_dir)

    cfg["experiment"].update(
        reservoir_types=["raw_lags", "classical", "esn_tuned"],
        readout_types=["linear", "rbf_kernel"],
    )
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)
    cells = json.loads((run_dir / "metrics.json").read_text())["cells"]
    assert [(c["reservoir_type"], c["readout_type"]) for c in cells] == [
        (r, o)
        for r in ("raw_lags", "classical", "esn_tuned")
        for o in ("linear", "rbf_kernel")
    ]
    assert all(0 < c["metrics"]["test"]["rmse"] < 1 for c in cells)
    esn = [c for c in cells if c["reservoir_type"] == "esn_tuned"][0]
    selection = esn["reservoir_selection"]
    assert len(selection["candidates"]) == 12
    scores = [c["eval_rmse_linear"] for c in selection["candidates"]]
    chosen = select_first_within_tolerance(scores)
    assert {
        k: selection["candidates"][chosen][k] for k in ("sr", "lr", "input_scaling")
    } == selection["selected"]
    rbf = [c for c in cells if c["readout_type"] == "rbf_kernel"][0]["instances"][0]
    assert rbf["kernel"] == "rbf" and rbf["selection_points"] == 150

    # The tuned ESN at the paper baseline's own setting reproduces the paper baseline's features exactly (same node seed).
    data = paper_runner.build_forecast_data(cfg)
    spec = paper_runner.build_cell_spec("srsr", 4)
    b0 = paper_runner._reservoir_features(cfg, data, spec, 0, "classical", [])
    again = paper_runner._reservoir_features(
        cfg, data, spec, 0, "classical", [], dict(cfg["classical_reservoir"])
    )
    assert all(np.array_equal(b0[s], again[s]) for s in b0)
