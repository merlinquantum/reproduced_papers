"""Photonic (MerLin) adaptation of the virtual classifier: model and integration."""

from __future__ import annotations

import json

import pytest
import torch
from common import PROJECT_DIR, load_runtime_ready_config

from runtime_lib.config import deep_update, load_config

ml = pytest.importorskip("merlin")

from virtual_qnn.lib import vqc_merlin  # noqa: E402

KW = {
    "n_modes": 4,
    "input_state": [1, 0, 1, 0],
    "encoding_modes": [0, 1],
    "encoding_scale": torch.pi,
    "computation_space": "fock",
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}


def _model(budget=2.0, indistinguishability=0.9, seed=0):
    torch.manual_seed(seed)
    return vqc_merlin.PhotonicVirtualClassifier(
        budget=budget, indistinguishability=indistinguishability, **KW
    )


def test_shapes_and_branch_count():
    x = torch.rand(7, 2, dtype=torch.float64) * 2 - 1
    assert _model(budget=1.0).forward(x).shape == (7,)
    assert len(_model(budget=1.0).branches) == 1
    assert len(_model(budget=2.5).branches) == 2


def test_fock_space_and_readout_range():
    model = _model()
    keys = model.branches[0].output_keys
    assert any(max(k) == 2 for k in keys), (
        "bunched outcomes missing: FOCK space not used"
    )
    assert all(sum(k) == 2 for k in keys)
    z = model.branch_readouts(torch.rand(5, 2, dtype=torch.float64) * 2 - 1)[0]
    assert torch.all(z.abs() <= 1 + 1e-12)


def test_gradients_reach_every_branch():
    model = _model(budget=3.0)
    x = torch.rand(9, 2, dtype=torch.float64) * 2 - 1
    model(x).pow(2).mean().backward()
    for branch in model.branches:
        grads = [p.grad for p in branch.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_seeding_is_deterministic():
    x = torch.rand(4, 2, dtype=torch.float64)
    assert torch.equal(_model(seed=3)(x), _model(seed=3)(x))
    assert not torch.equal(_model(seed=3)(x), _model(seed=4)(x))


def test_noise_setting_reaches_the_layer():
    x = torch.rand(6, 2, dtype=torch.float64) * 2 - 1
    noisy, clean = (
        _model(indistinguishability=0.7, seed=1),
        _model(indistinguishability=None, seed=1),
    )
    assert not torch.allclose(noisy(x), clean(x)), "indistinguishability had no effect"


def test_linear_occupation_readout_is_blind_to_distinguishability():
    """Why the parity readout is needed (Phase 4 amendment)."""
    x = torch.rand(6, 2, dtype=torch.float64) * 2 - 1
    noisy, clean = (
        _model(indistinguishability=0.3, seed=2),
        _model(indistinguishability=None, seed=2),
    )
    weights = vqc_merlin.occupation_contrast_weights(
        noisy.branches[0].output_keys, 4, torch.float64
    )
    z_noisy = noisy.branches[0](x) @ weights
    z_clean = clean.branches[0](x) @ weights
    assert torch.allclose(z_noisy, z_clean, atol=1e-10)


def test_predictions_are_not_restricted_to_x1_minus_x2():
    """Guards amendment 2: encoding on all modes made outputs depend on x1 - x2 only."""
    points = torch.tensor([[0.3, 0.1], [0.7, 0.5], [-0.4, -0.6]], dtype=torch.float64)
    with torch.no_grad():
        predictions = _model(budget=1.0, indistinguishability=None, seed=0)(points)
    assert predictions.max() - predictions.min() > 1e-3


def test_single_photon_is_rejected():
    with pytest.raises(ValueError, match="at least two photons"):
        vqc_merlin.build_branch(
            4,
            [1, 0, 0, 0],
            [0, 1],
            torch.pi,
            "fock",
            None,
            torch.float64,
            torch.device("cpu"),
        )


def test_tiny_run_writes_contract_artifacts(tmp_path):
    from virtual_qnn.lib import runner

    cfg = deep_update(
        load_runtime_ready_config(),
        load_config(PROJECT_DIR / "configs" / "e4_vqc_merlin.json"),
    )
    cfg["data_root"] = str(PROJECT_DIR.parents[1] / "data")
    cfg["vqc_merlin"]["budget"] = 2.0
    cfg["training"]["iterations"] = 3
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["task"] == "vqc_merlin"
    assert metrics["hardware"]["computation_space"] == "fock"
    assert metrics["hardware"]["n_photons"] == 2
    assert metrics["n_parameters"] > 0
    log = (run_dir / "run.log").read_text()
    for event in (
        "RUN_STARTED",
        "DATASET_READY",
        "TRAINING_STARTED",
        "METRICS_WRITTEN",
        "RUN_COMPLETED",
    ):
        assert log.count(event) == 1, event
