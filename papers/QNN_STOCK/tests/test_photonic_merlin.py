from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from common import PROJECT_DIR
from lib.baselines import context_features
from lib.photonic_merlin import (
    PhotonicContextModel,
    fit_photonic_model,
    spread_input_state,
)

PHOTONIC_CFG = {
    "n_photons": 2,
    "reuploads": 2,
    "computation_space": "fock",
    "distinguishable": False,
    "steps": 60,
    "lr": 0.05,
    "monitor_every": 20,
}


def _model(seed=0, **overrides):
    options = {
        "context_length": 3,
        "n_photons": 2,
        "reuploads": 2,
        "computation_space": "fock",
        "distinguishable": False,
        "dtype": torch.float64,
    } | overrides
    torch.manual_seed(seed)
    return PhotonicContextModel(**options)


def test_input_states_match_the_adaptation_plan():
    assert spread_input_state(4, 1) == [1, 0, 0, 0]
    assert spread_input_state(4, 2) == [1, 0, 1, 0]
    assert spread_input_state(6, 2) == [1, 0, 0, 1, 0, 0]
    assert spread_input_state(7, 2) == [1, 0, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        spread_input_state(2, 3)


def test_forward_gives_one_probability_per_context_and_depends_on_it():
    model = _model()
    bits = context_features(8, 2)
    prob_up = model(bits)
    assert prob_up.shape == (8,)
    assert torch.all((prob_up >= 0) & (prob_up <= 1))
    assert prob_up.max() - prob_up.min() > 1e-3
    assert model.n_params() == 36  # 3 meshes of 12 phases on 4 modes


def test_gradients_reach_every_mesh_phase():
    model = _model()
    model(context_features(8, 2)).sum().backward()
    gradients = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.any(g != 0) for g in gradients)


def test_same_seed_gives_same_model():
    bits = context_features(8, 2)
    assert torch.equal(_model(seed=3)(bits), _model(seed=3)(bits))
    assert not torch.equal(_model(seed=3)(bits), _model(seed=4)(bits))


def test_two_photon_interference_changes_the_readout():
    """Same mesh phases, photons distinguishable or not: the detection
    readout differs, so the readout is not the trivial linear one."""

    bits = context_features(8, 2)
    indistinguishable = _model()
    distinguishable = _model(distinguishable=True)
    distinguishable.load_state_dict(indistinguishable.state_dict())
    difference = (indistinguishable(bits) - distinguishable(bits)).abs().max()
    assert difference > 1e-3


def test_threshold_detection_discards_bunched_outcomes():
    bits = context_features(8, 2)
    fock = _model()
    unbunched = _model(computation_space="unbunched")
    unbunched.load_state_dict(fock.state_dict())
    assert torch.equal(fock.keep_rate(bits), torch.ones(8, dtype=torch.float64))
    keep = unbunched.keep_rate(bits)
    assert torch.all((keep > 0) & (keep < 1))
    # Renormalized collision-free outcomes: the readout is the FOCK
    # probability of "one photon on mode 0, the other elsewhere" over keep.
    probabilities = fock.quantum(fock.encode(bits))
    wanted = torch.tensor(
        [max(key) <= 1 and key[0] == 1 for key in fock.quantum.output_keys]
    )
    expected = probabilities[:, wanted].sum(dim=1) / keep
    assert torch.allclose(unbunched(bits), expected, atol=1e-10)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="computation_space"):
        _model(computation_space="dual_rail")
    with pytest.raises(ValueError, match="distinguishable"):
        _model(computation_space="unbunched", distinguishable=True)
    with pytest.raises(ValueError, match="reuploads"):
        _model(reuploads=0)
    counts = np.ones((8, 2))
    incomplete = {k: v for k, v in PHOTONIC_CFG.items() if k != "lr"}
    with pytest.raises(ValueError, match="photonic.lr"):
        fit_photonic_model(counts, incomplete, 0, torch.float64, "t", [])
    with pytest.raises(ValueError, match="binary"):
        fit_photonic_model(np.ones((8, 4)), PHOTONIC_CFG, 0, torch.float64, "t", [])


def test_training_lowers_the_weighted_log_loss():
    rng = np.random.default_rng(0)
    prob_up = rng.uniform(0.2, 0.8, size=8)
    totals = rng.integers(50, 150, size=8)
    counts = np.column_stack([totals * (1 - prob_up), totals * prob_up])
    history: list = []
    result = fit_photonic_model(counts, PHOTONIC_CFG, 0, torch.float64, "t", history)
    assert result["conditional"].shape == (8, 2)
    assert np.allclose(result["conditional"].sum(axis=1), 1.0)
    assert [row["stage_epoch"] for row in history] == [20, 40, 60]
    assert history[-1]["loss_estimate"] < history[0]["loss_estimate"]
    assert result["keep_rate_mean"] == 1.0
    assert result["input_state"] == [1, 0, 1, 0]


def test_photonic_configs_overlay_the_defaults():
    defaults = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())
    assert set(PHOTONIC_CFG) == set(defaults["photonic"])
    names = sorted(p.name for p in (PROJECT_DIR / "configs").glob("p4_photonic_*.json"))
    assert len(names) == 8
    for name in names:
        config = json.loads((PROJECT_DIR / "configs" / name).read_text())
        assert config["experiment"]["kind"] == "photonic"
        assert set(config["photonic"]) <= set(defaults["photonic"])
