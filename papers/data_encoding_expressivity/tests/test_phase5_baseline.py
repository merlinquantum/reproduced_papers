"""Phase 5 additional fair baseline (B2 on the published Fig. 4 target) as used by the comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from common import PROJECT_DIR, load_named_config
from lib import runner
from lib.baselines import FourierSeriesModel, least_squares_oracle
from lib.data import build_dataset
from lib.training import build_fit_model, count_parameters, model_frequencies

CONFIG = "e10_fig4_published_classical.json"


def _cases():
    return json.loads((PROJECT_DIR / "configs" / CONFIG).read_text())["fit"]["cases"]


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
def test_frequency_set_and_parameter_count_match_the_degree(degree):
    case = next(c for c in _cases() if c["name"] == f"fig4s1_classical_degree{degree}")
    assert model_frequencies(case["model"]) == (list(range(1, degree + 1)), 1.0)
    model = build_fit_model("fourier_classical", case["model"], torch.float64)
    assert isinstance(model, FourierSeriesModel)
    assert count_parameters(model) == 2 * degree + 1


def test_target_is_the_published_fig4_target_used_by_compared_runs():
    published = json.loads(
        (PROJECT_DIR / "configs" / "e2_fig4_scaling1.json").read_text()
    )
    photonic = json.loads(
        (PROJECT_DIR / "configs" / "e8_fig4_improved_merlin.json").read_text()
    )
    reference = published["fit"]["cases"][0]["target"]
    assert all(case["target"] == reference for case in _cases())
    assert all(case["target"] == reference for case in photonic["fit"]["cases"])


def test_b2_can_represent_the_b1_solution_exactly():
    case = next(c for c in _cases() if c["name"] == "fig4s1_classical_degree5")
    cfg = {"grid": {"start": -6.0, "stop": 6.0, "num": 70}, "target": case["target"]}
    inputs, targets = build_dataset(cfg)
    oracle = least_squares_oracle(inputs.numpy(), targets.numpy(), [1, 2, 3, 4, 5], 1.0)
    model = FourierSeriesModel([1, 2, 3, 4, 5], 1.0, "uniform", torch.float64)
    with torch.no_grad():
        model.weights.copy_(torch.tensor(oracle["parameters"]))
        predictions = model(inputs).numpy()
    assert np.allclose(predictions, oracle["predictions"], atol=1e-12)


def test_gradients_and_seeding():
    case = next(c for c in _cases() if c["name"] == "fig4s1_classical_degree3")
    outputs = []
    for _ in range(2):
        model = build_fit_model("fourier_classical", case["model"], torch.float64)
        model.reset_parameters(torch.Generator().manual_seed(7), 2 * np.pi)
        output = model(torch.linspace(-6, 6, 9, dtype=torch.float64))
        outputs.append(output.detach())
    output.pow(2).sum().backward()
    assert model.weights.grad is not None and float(model.weights.grad.abs().max()) > 0
    assert torch.equal(outputs[0], outputs[1])


def test_tiny_full_batch_run_records_matched_parameter_counts(tmp_path):
    cfg = load_named_config(
        CONFIG, {"fit": {"seeds": [0], "training": {"steps": 2, "full_batch": True}}}
    )
    run_dir = tmp_path / "run_b2"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["training"]["full_batch"] is True
    for row in metrics["summary"]:
        degree = int(row["case"].rsplit("degree", 1)[1])
        assert row["n_parameters"] == 2 * degree + 1
        assert row["model_kind"] == "fourier_classical"


def test_zero_init_starts_at_zero_and_keeps_seed_alignment():
    cfg = json.loads(
        (
            PROJECT_DIR / "configs" / "e10_fig4_published_classical_zero_init.json"
        ).read_text()
    )
    case = cfg["fit"]["cases"][4]
    model = build_fit_model("fourier_classical", case["model"], torch.float64)
    generator = torch.Generator().manual_seed(3)
    model.reset_parameters(generator, 2 * np.pi)
    assert float(model.weights.abs().max()) == 0.0
    reference = torch.Generator().manual_seed(3)
    torch.rand(model.weights.shape, generator=reference, dtype=torch.float64)
    assert torch.equal(
        torch.rand(4, generator=generator), torch.rand(4, generator=reference)
    )


def test_unknown_coefficient_init_raises():
    with pytest.raises(ValueError):
        FourierSeriesModel([1], 1.0, "gaussian", torch.float64)
