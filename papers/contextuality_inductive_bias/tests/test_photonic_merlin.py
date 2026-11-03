"""Phase 4 MerLin adaptation: bias exactness, readouts, wiring and seeding."""

from __future__ import annotations

import json
import math

import merlin as ml
import numpy as np
import pytest
import torch
from common import PROJECT_DIR, load_runtime_ready_config
from contextuality_inductive_bias.lib.photonic_merlin import (
    PhotonicZeroSumModel,
    validate_input_state,
)
from contextuality_inductive_bias.lib.runner import build_model

DTYPE = torch.float64
STATES = {2: [1, 0, 0, 1, 0, 0], 3: [1, 0, 1, 0, 1, 0]}


def _features(n: int, seed: int, scale: float = math.pi / 2) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 3, 3))
    s = raw / raw.sum(axis=2, keepdims=True)
    cols = [
        s[:, 0, 0],
        s[:, 1, 1],
        s[:, 2, 2],
        s[:, 0, 1] - s[:, 0, 2],
        s[:, 1, 2] - s[:, 1, 0],
        s[:, 2, 0] - s[:, 2, 1],
    ]
    return torch.tensor(scale * np.stack(cols, 1), dtype=DTYPE)


def _model(
    readout: str, n_photons: int, seed: int = 0, space: str = "unbunched"
) -> PhotonicZeroSumModel:
    torch.manual_seed(seed)
    return PhotonicZeroSumModel(
        readout, n_photons, STATES[n_photons], 2, DTYPE, computation_space=space
    )


@pytest.mark.parametrize("readout", ["vertex", "group"])
@pytest.mark.parametrize("n_photons", [2, 3])
def test_biased_readouts_satisfy_zero_sum_exactly(readout, n_photons):
    for seed in range(3):
        model = _model(readout, n_photons, seed)
        with torch.no_grad():
            for parameter in model.quantum.parameters():
                parameter.uniform_(0, 2 * math.pi)
            expectations = model(_features(300, seed))
        assert float(expectations.sum(dim=1).abs().max()) < 1e-12
        assert float(expectations.abs().max()) <= 1.0 + 1e-12


def test_unbiased_readout_breaks_zero_sum():
    model = _model("unbiased", 2)
    with torch.no_grad():
        model.readout_logits.normal_(0, 2.0)
        expectations = model(_features(200, 1))
    assert float(expectations.sum(dim=1).abs().max()) > 0.05


def test_group_readout_stays_in_noncontextual_triangle():
    model = _model("group", 2)
    with torch.no_grad():
        win = (1 + model(_features(500, 2))) / 2
    assert float(win.min()) >= 0.25 - 1e-12


def test_vertex_readout_matches_manual_computation():
    model = _model("vertex", 2)
    x = _features(4, 3)
    with torch.no_grad():
        probabilities = model.outcome_probabilities(x)
        expected = np.zeros((4, 3))
        vertices = model.vertices.numpy()
        for j, key in enumerate(model.quantum.output_keys):
            response = sum(n * vertices[m] for m, n in enumerate(key)) / 2
            expected += probabilities[:, j : j + 1].numpy() * response
        np.testing.assert_allclose(((1 + model(x)) / 2).numpy(), expected, atol=1e-12)


def test_shapes_parameters_and_gradients():
    for readout, n_params in (("vertex", 90), ("group", 90), ("unbiased", 108)):
        model = _model(readout, 2)
        assert sum(p.numel() for p in model.parameters()) == n_params
        x = _features(8, 4)
        out = model(x)
        assert out.shape == (8, 3) and out.dtype == DTYPE
        out.pow(2).sum().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )


def test_unbunched_space_and_collision_free_outcomes_are_used():
    model = _model("vertex", 3)
    assert (
        model.quantum.measurement_strategy.computation_space
        == ml.ComputationSpace.UNBUNCHED
    )
    keys = model.quantum.output_keys
    assert len(keys) == math.comb(6, 3)
    assert all(sum(k) == 3 and max(k) == 1 for k in keys)


def test_postselection_success_is_a_probability():
    model = _model("vertex", 2)
    success = model.postselection_success(_features(20, 5))
    assert success.shape == (20,)
    assert float(success.min()) > 0.0 and float(success.max()) <= 1.0 + 1e-12


def test_seeding_is_deterministic():
    x = _features(6, 6)
    first = _model("unbiased", 2, seed=7)(x).detach()
    second = _model("unbiased", 2, seed=7)(x).detach()
    other = _model("unbiased", 2, seed=8)(x).detach()
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, other)


@pytest.mark.parametrize(
    "bad", [[1, 1, 0, 0, 0], [2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
)
def test_input_state_is_validated_not_repaired(bad):
    with pytest.raises(ValueError):
        validate_input_state(bad, 6, 2)


def test_config_values_reach_the_model():
    cfg = load_runtime_ready_config()
    overlay = json.loads(
        (PROJECT_DIR / "configs" / "fig8_photonic_group_merlin.json").read_text()
    )
    cfg["model"].update(overlay["model"])
    cfg["model"]["n_photons"] = 3
    torch.manual_seed(0)
    model = build_model(
        cfg["model"], torch.Generator().manual_seed(0), DTYPE, torch.device("cpu")
    )
    assert isinstance(model, PhotonicZeroSumModel)
    assert (model.readout, model.n_photons, model.layers) == ("group", 3, 2)
    assert model.input_state == cfg["model"]["input_states"]["3"]


def test_missing_input_state_fails_clearly():
    cfg = load_runtime_ready_config()
    cfg["model"].update(
        {"kind": "photonic_merlin", "readout": "vertex", "n_photons": 4}
    )
    with pytest.raises(ValueError, match="input_states"):
        build_model(cfg["model"], torch.Generator(), DTYPE, torch.device("cpu"))


@pytest.mark.parametrize("readout", ["vertex", "group"])
def test_fock_detection_keeps_bias_without_postselection(readout):
    model = _model(readout, 2, seed=1, space="fock")
    assert (
        model.quantum.measurement_strategy.computation_space == ml.ComputationSpace.FOCK
    )
    assert any(max(k) == 2 for k in model.quantum.output_keys)
    x = _features(200, 9)
    with torch.no_grad():
        for parameter in model.quantum.parameters():
            parameter.uniform_(0, 2 * math.pi)
        assert torch.allclose(
            model.outcome_probabilities(x).sum(dim=1), torch.ones(200, dtype=DTYPE)
        )
        assert float(model(x).sum(dim=1).abs().max()) < 1e-12
        assert torch.all(model.postselection_success(x) == 1)


def test_unknown_computation_space_fails():
    with pytest.raises(ValueError, match="computation_space"):
        PhotonicZeroSumModel(
            "vertex", 2, STATES[2], 2, DTYPE, computation_space="dual_rail"
        )
