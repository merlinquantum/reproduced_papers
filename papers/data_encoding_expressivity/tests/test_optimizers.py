"""Optimiser protocols of the Phase 5 extensions (P4 adam_cosine, P5 lbfgs) and Adam parity."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from common import PROJECT_DIR
from lib.baselines import FourierSeriesModel
from lib.photonic_merlin import photonic_positive_frequencies
from lib.training import _build_optimizer, train_fit_model

GRID = {"start": -6.0, "stop": 6.0, "num": 70}
TARGET = {
    "c0": 0.0,
    "coefficients": [{"re": 0.05, "im": -0.05}] * 3,
    "sign": 1,
    "scaling": 1.0,
}


def _case(optimizer, lr, steps, full_batch):
    return {
        "grid": GRID,
        "target": TARGET,
        "training": {
            "optimizer": optimizer,
            "lr": lr,
            "steps": steps,
            "init_high": 2 * np.pi,
            "full_batch": full_batch,
            "batch_size": 25,
        },
    }


def test_adam_closure_path_matches_a_manual_adam_loop():
    torch.manual_seed(0)
    inputs = torch.linspace(-6, 6, 70, dtype=torch.float64)
    targets = torch.sin(inputs)
    models = []
    for _ in range(2):
        model = FourierSeriesModel([1, 2], 1.0, "zero", torch.float64)
        model.reset_parameters(torch.Generator().manual_seed(1), 2 * np.pi)
        models.append(model)
    optimizer, _ = _build_optimizer("adam", models[0], 0.1, 10, True)
    manual = torch.optim.Adam(models[1].parameters(), lr=0.1)
    for _ in range(10):

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((models[0](inputs) - targets) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)
        manual.zero_grad()
        torch.mean((models[1](inputs) - targets) ** 2).backward()
        manual.step()
    assert torch.equal(models[0].weights, models[1].weights)


def test_cosine_schedule_anneals_to_zero():
    model = FourierSeriesModel([1], 1.0, "zero", torch.float64)
    optimizer, scheduler = _build_optimizer("adam_cosine", model, 0.05, 4, True)
    rates = []
    for _ in range(4):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(0.05)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)
    assert rates == sorted(rates, reverse=True)


def test_lbfgs_requires_full_batch_and_unknown_names_raise():
    model = FourierSeriesModel([1], 1.0, "zero", torch.float64)
    with pytest.raises(ValueError):
        _build_optimizer("lbfgs", model, 1.0, 10, False)
    with pytest.raises(ValueError):
        _build_optimizer("sgd", model, 0.1, 10, True)


def test_lbfgs_fits_the_classical_baseline_exactly():
    model = FourierSeriesModel([1, 2, 3], 1.0, "zero", torch.float64)
    outcome = train_fit_model(
        model, _case("lbfgs", 1.0, 50, True), 0, torch.device("cpu"), "test"
    )
    # The basis is not orthogonal on [-6, 6], so even this convex fit converges gradually;
    # 1e-8 is far inside the success tolerance of 1e-4.
    assert outcome["final_mse"] < 1e-8
    assert len(outcome["full_grid_mse"]) == 51


@pytest.mark.parametrize(
    "config",
    ["e11_multiscale_square_merlin.json", "e12_fixed_readout_ceiling_merlin.json"],
)
def test_extension_photonic_degrees_are_contiguous(config):
    cases = json.loads((PROJECT_DIR / "configs" / config).read_text())["fit"]["cases"]
    for case in cases:
        frequencies = photonic_positive_frequencies(case["model"])
        assert frequencies == list(range(1, max(frequencies) + 1)), case["name"]
        if "_D" in case["name"]:
            assert max(frequencies) == int(case["name"].rsplit("_D", 1)[1])
