"""Optimiser equivalence with the authors' optax stack, metrics, and the training loop."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch
from common import load_runtime_ready_config
from contextuality_inductive_bias.lib import data as D
from contextuality_inductive_bias.lib.metrics import (
    average_kl,
    negative_log_likelihood,
    per_sample_kl,
)
from contextuality_inductive_bias.lib.quantum_models import BiasedQuantumModel
from contextuality_inductive_bias.lib.training import build_optimizer, train


def _quadratic_problem():
    target = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)
    return target, lambda w: (
        (w - target) ** 2 * torch.tensor([1.0, 10.0, 0.1], dtype=torch.float64)
    ).sum()


@pytest.mark.parametrize("name", ["adam", "sgd"])
def test_torch_optimisers_match_optax(name):
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    _, loss_torch = _quadratic_problem()
    weights = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = build_optimizer(
        torch.nn.ParameterList([weights]), {"optimizer": name, "lr": 0.05}
    )
    scale = jnp.array([1.0, 10.0, 0.1])
    target = jnp.array([0.3, -1.2, 2.0])
    tx = optax.adam(0.05) if name == "adam" else optax.sgd(0.05)
    params = jnp.zeros(3)
    state = tx.init(params)
    grad_fn = jax.grad(lambda w: jnp.sum((w - target) ** 2 * scale))
    for _ in range(50):
        optimizer.zero_grad()
        loss_torch(weights).backward()
        optimizer.step()
        updates, state = tx.update(grad_fn(params), state, params)
        params = optax.apply_updates(params, updates)
    np.testing.assert_allclose(weights.detach().numpy(), np.asarray(params), atol=1e-12)


def test_kl_matches_closed_form_and_floor():
    true_win = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float64)
    model_e = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    expected = np.mean(
        [
            p * math.log(p / 0.5) + (1 - p) * math.log((1 - p) / 0.5)
            for p in (0.2, 0.5, 0.8)
        ]
    )
    assert average_kl(true_win, model_e) == pytest.approx(expected)
    assert float(per_sample_kl(true_win, true_win * 2 - 1)[0]) == pytest.approx(
        0.0, abs=1e-15
    )
    saturated = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float64)
    assert math.isfinite(average_kl(true_win, saturated))


def test_nll_is_mean_over_labels_with_zero_meaning_win():
    expectations = torch.tensor([[0.5, -0.5, 0.0]], dtype=torch.float64)
    labels = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    expected = -np.mean([math.log(0.75), math.log(0.25), math.log(0.5)])
    assert float(negative_log_likelihood(expectations, labels)) == pytest.approx(
        expected
    )


def test_training_loop_writes_history_and_decreases_loss(tmp_path):
    cfg = load_runtime_ready_config()
    cfg["data"]["test"]["n"] = 200
    train_split, test_split = D.load_split(cfg, "train"), D.load_split(cfg, "test")
    model = BiasedQuantumModel(1, 1, torch.Generator().manual_seed(0), torch.float64)
    result = train(
        model,
        train_split,
        test_split,
        None,
        {"steps": 15, "optimizer": "adam", "lr": 0.05, "l2": 0.0},
        {"every": 5, "kl_floor": 1e-8},
        tmp_path / "history.csv",
    )
    with (tmp_path / "history.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["step"]) for r in rows] == [0, 5, 10, 14]
    assert float(rows[-1]["train_objective"]) < float(rows[0]["train_objective"])
    assert result["final"]["test_zero_sum_violation"] < 1e-12
    assert result["min_test_kl"]["test_kl"] <= result["final"]["test_kl"] + 1e-15


def test_non_finite_objective_raises(tmp_path):
    cfg = load_runtime_ready_config()
    cfg["data"]["test"]["n"] = 20
    train_split, test_split = D.load_split(cfg, "train"), D.load_split(cfg, "test")

    class Saturated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))

        def forward(self, features):
            return self.bias.expand(features.shape[0], 3)

    with pytest.raises(FloatingPointError):
        train(
            Saturated(),
            train_split,
            test_split,
            None,
            {"steps": 2, "optimizer": "sgd", "lr": 0.1, "l2": 0.0},
            {"every": 1, "kl_floor": 1e-8},
            tmp_path / "history.csv",
        )
