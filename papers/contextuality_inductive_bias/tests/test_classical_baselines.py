"""Phase 5 fair baselines B2-B5: bias exactness, capacity matching, wiring."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config
from contextuality_inductive_bias.lib import data as D
from contextuality_inductive_bias.lib.classical_baselines import (
    BiasedMLP,
    BiasEncodedFourier,
    QuadraticModel,
    UnbiasedMLP,
    quadratic_monomials,
)
from contextuality_inductive_bias.lib.runner import build_model
from contextuality_inductive_bias.lib.surrogate import dense_features

DTYPE = torch.float64


def _features(n: int, seed: int) -> torch.Tensor:
    strategies = D.generate_strategies(np.random.RandomState(seed), n)
    return torch.tensor(D.compress_strategies(strategies, math.pi / 2), dtype=DTYPE)


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_parameter_counts_match_the_comparison_plan():
    gen = torch.Generator().manual_seed(0)
    assert _count(BiasedMLP(7, DTYPE)) == 97
    assert _count(UnbiasedMLP(9, DTYPE)) == 93
    assert _count(QuadraticModel(DTYPE)) == 84
    assert _count(BiasEncodedFourier(1, gen, DTYPE)) == 6 * 1458
    for count in (97, 93, 84):
        assert abs(count - 91) / 91 <= 0.10


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BiasedMLP(7, DTYPE),
        lambda: BiasEncodedFourier(
            1, torch.Generator().manual_seed(1), DTYPE, chunk_size=64
        ),
    ],
)
def test_bias_encoded_baselines_are_exactly_zero_sum(factory):
    torch.manual_seed(3)
    model = factory()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 3.0)
        expectations = model(_features(300, 4))
    assert float(expectations.sum(dim=1).abs().max()) < 1e-12
    assert float(expectations.abs().max()) <= 1.0


@pytest.mark.parametrize(
    "factory", [lambda: UnbiasedMLP(9, DTYPE), lambda: QuadraticModel(DTYPE)]
)
def test_unbiased_baselines_are_not_constrained(factory):
    torch.manual_seed(5)
    model = factory()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 1.0)
        expectations = model(_features(300, 6))
    assert float(expectations.sum(dim=1).abs().max()) > 0.05


def test_bias_fourier_logits_match_dense_features():
    model = BiasEncodedFourier(
        1, torch.Generator().manual_seed(2), DTYPE, chunk_size=13
    )
    with torch.no_grad():
        model.weights.normal_(generator=torch.Generator().manual_seed(8))
        x = _features(40, 7)
        dense = dense_features(x, 1) @ model.weights.T
        np.testing.assert_allclose(
            model.raw_output(x).numpy(), dense.numpy(), atol=1e-9
        )


def test_quadratic_class_contains_the_true_function():
    strategies = D.generate_strategies(np.random.RandomState(11), 400)
    x = torch.tensor(D.compress_strategies(strategies, math.pi / 2), dtype=DTYPE)
    true_expectation = 2.0 * D.win_probabilities(strategies) - 1.0
    design = np.hstack([np.ones((400, 1)), quadratic_monomials(x).numpy()])
    coefficients, *_ = np.linalg.lstsq(design, true_expectation, rcond=None)
    assert np.abs(design @ coefficients - true_expectation).max() < 1e-10


def test_gradients_flow_and_seeding_is_deterministic():
    x = _features(16, 9)
    for factory in (
        lambda: BiasedMLP(7, DTYPE),
        lambda: UnbiasedMLP(9, DTYPE),
        lambda: QuadraticModel(DTYPE),
    ):
        torch.manual_seed(12)
        first = factory()
        torch.manual_seed(12)
        second = factory()
        torch.testing.assert_close(first(x), second(x))
        first(x).pow(2).sum().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in first.parameters()
        )


@pytest.mark.parametrize(
    "config_name,kind,expected",
    [
        ("fair_mlp_biased_classical.json", "mlp_biased", 97),
        ("fair_mlp_unbiased_classical.json", "mlp_unbiased", 93),
        ("fair_quadratic_classical.json", "quadratic", 84),
        ("fair_bias_fourier_classical.json", "bias_fourier", 8748),
    ],
)
def test_configs_reach_the_models(config_name, kind, expected):
    cfg = load_runtime_ready_config()
    overlay = json.loads((PROJECT_DIR / "configs" / config_name).read_text())
    cfg["model"].update(overlay["model"])
    assert overlay["data"]["validation"]["enabled"] is True
    assert overlay["data"]["protocol"] == "paper"
    model = build_model(
        cfg["model"], torch.Generator().manual_seed(0), DTYPE, torch.device("cpu")
    )
    assert cfg["model"]["kind"] == kind and _count(model) == expected


def test_cli_exposes_hidden_and_validation_flags():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        ["--model", "mlp_unbiased", "--hidden", "9", "--validation", "true"]
    )
    assert args.hidden == 9 and args.validation is True


def test_validation_split_uses_training_scale_and_its_own_seed():
    cfg = load_runtime_ready_config()
    cfg["data"]["validation"]["enabled"] = True
    validation = D.load_split(cfg, "validation")
    train = D.load_split(cfg, "train")
    assert validation.input_scale == train.input_scale
    assert validation.provenance["seed"] == 201 and validation.labels is not None
