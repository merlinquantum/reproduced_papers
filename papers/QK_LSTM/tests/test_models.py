from __future__ import annotations

import numpy as np
import pytest
import torch
from lib.metrics import regression_metrics
from lib.models import (
    RidgeWindowRegressor,
    build_torch_model,
    count_trainable_parameters,
    sample_reference_inputs,
)

INPUT_SIZE = 11


def _kernel_cfg(model_type, gate_mode, n_references=4):
    return {
        "type": model_type,
        "params": {
            "gate_mode": gate_mode,
            "hidden_size": 16,
            "n_qubits": 4,
            "n_references": n_references,
            "circuit_layout": "text",
            "angle_map": "identity",
            "rbf_gamma": 1.0,
        },
    }


def _references(n=4):
    return sample_reference_inputs(
        np.random.default_rng(0).random((50, INPUT_SIZE)), n, seed=0
    )


@pytest.mark.parametrize(
    ("model_cfg", "expected"),
    [
        # Table II: QK-LSTM 209 (count-matched variant Q-B) and LSTM 1,873.
        (_kernel_cfg("qk_lstm", "shared"), 209),
        ({"type": "lstm", "params": {"hidden_size": 16}}, 1873),
        # Fair baselines (LOG.md, Fair Baseline Plan).
        ({"type": "lstm", "params": {"hidden_size": 3}}, 196),
        (_kernel_cfg("rbf_lstm", "shared"), 209),
        # Q-A: 112 + 4 * (16 N + 16) + 17.
        (_kernel_cfg("qk_lstm", "per_gate", 4), 449),
        (_kernel_cfg("qk_lstm", "per_gate", 16), 1217),
    ],
)
def test_parameter_counts(model_cfg, expected):
    n_references = model_cfg["params"].get("n_references", 4)
    model = build_torch_model(model_cfg, INPUT_SIZE, _references(n_references))
    assert count_trainable_parameters(model) == expected


@pytest.mark.parametrize("model_type", ["qk_lstm", "rbf_lstm"])
@pytest.mark.parametrize("gate_mode", ["shared", "per_gate"])
def test_kernel_models_forward_and_backward(model_type, gate_mode):
    torch.manual_seed(0)
    model = build_torch_model(
        _kernel_cfg(model_type, gate_mode), INPUT_SIZE, _references()
    )
    x = torch.rand(2, 3, INPUT_SIZE)
    out = model(x)
    assert out.shape == (2, 1)
    out.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    # The compression layer feeds the kernel, so its gradient must be non-zero.
    assert model.cell.compression.weight.grad.abs().sum() > 0
    # References are fixed buffers, not trainable parameters.
    assert "cell.reference_vectors" in dict(model.named_buffers())


def test_one_optimizer_step_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    model = build_torch_model(
        _kernel_cfg("qk_lstm", "shared"), INPUT_SIZE, _references()
    )
    x, y = torch.rand(8, 3, INPUT_SIZE), torch.rand(8, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    windows = rng.random((200, 3, INPUT_SIZE))
    weights = rng.normal(size=3 * INPUT_SIZE)
    targets = windows.reshape(200, -1) @ weights + 0.5
    ridge = RidgeWindowRegressor(alpha=1e-8).fit(windows, targets)
    assert ridge.n_parameters == 34
    assert np.allclose(ridge.predict(windows), targets, atol=1e-6)


def test_regression_metrics_match_definitions():
    y_true = np.array([10.0, 20.0, 40.0])
    y_pred = np.array([12.0, 18.0, 40.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["rmse"] == pytest.approx(np.sqrt(8 / 3))
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["mape"] == pytest.approx(100 * (0.2 + 0.1 + 0.0) / 3)
    assert metrics["r2"] == pytest.approx(1 - 8 / np.sum((y_true - y_true.mean()) ** 2))


def test_target_that_is_also_an_input_feature(tmp_path):
    """Known-answer configuration (target PM2.5, horizon 0) must load and score."""
    from common import load_runtime_ready_config
    from lib.data import prepare_series

    cfg = load_runtime_ready_config()
    cfg["dataset"]["target"] = "PM2.5"
    cfg["dataset"]["horizon"] = 0
    series = prepare_series(cfg)
    assert series.target.shape == (series.features.shape[0],)
    assert not np.isnan(series.target).any()
    # Horizon 0: the window's last input row is the target day.
    end = int(series.window_end["test"][0])
    assert end + series.horizon == end


def test_causal_fill_never_uses_later_values_and_interpolate_does():
    import pandas as pd
    from lib.data import fill_missing

    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    base = pd.DataFrame(
        {
            "a": [1.0, None, None, 4.0, 5.0, None, 7.0, 8.0],
            "b": [None, 2.0, 3.0, None, None, None, 7.0, 9.0],
        },
        index=dates,
    )
    changed_future = base.copy()
    changed_future.iloc[4:] = (
        changed_future.iloc[4:] * 100
    )  # alter every value after day 3
    for issue_day in range(1, 4):
        causal = fill_missing(base, "causal").iloc[: issue_day + 1]
        causal_changed = fill_missing(changed_future, "causal").iloc[: issue_day + 1]
        # Days after the first observation of each column must not depend on the future.
        assert causal["a"].equals(causal_changed["a"])
        assert causal["b"].iloc[1:].equals(causal_changed["b"].iloc[1:])
    # Interpolation fills day 2 of "a" from day 3 (=4.0), i.e. from a later observation.
    assert fill_missing(base, "interpolate").loc["2020-01-02", "a"] == 2.0
    assert fill_missing(base, "causal").loc["2020-01-02", "a"] == 1.0
    try:
        fill_missing(base, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown fill mode must raise")
