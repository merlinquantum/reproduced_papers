"""Real-data training steps for the photonic and source gate models."""

import pytest
import torch
from common import MODEL_SETTINGS, make_config
from lib.data import load_task
from lib.training import predict_probabilities, train_model


def test_photonic_real_training_step_changes_weights():
    cfg = make_config(**MODEL_SETTINGS["merlin"])
    data = load_task(cfg)
    model, evidence = train_model(cfg, data["train"]["bits"], data["train"]["targets"])
    assert evidence["parameters"] == {"quantum": 120, "classical": 136, "total": 256}
    assert evidence["weight_change_l2"] > 0
    assert len(evidence["losses"]) == 1
    probabilities = predict_probabilities(model, cfg, data["test"]["bits"][:4])
    assert probabilities.shape == (4,)
    assert torch.isfinite(probabilities).all()


def test_original_source_spsa_real_iteration():
    cfg = make_config(**MODEL_SETTINGS["gate_source"])
    data = load_task(cfg)
    model, evidence = train_model(cfg, data["train"]["bits"], data["train"]["targets"])
    assert evidence["parameters"] == {"quantum": 81, "classical": 0, "total": 81}
    assert evidence["optimizer"] == "SPSA"
    assert evidence["updates"] == 1
    assert torch.isfinite(
        predict_probabilities(model, cfg, data["test"]["bits"][:4])
    ).all()


def test_source_labels_must_be_explicit():
    cfg = make_config(name="gate_source")
    data = load_task(cfg)
    with pytest.raises(KeyError):
        train_model(cfg, data["train"]["bits"], data["train"]["targets"])


def test_finite_shots_rejected_for_non_photonic_models():
    cfg = make_config(**MODEL_SETTINGS["classical"])
    data = load_task(cfg)
    model, _ = train_model(cfg, data["train"]["bits"], data["train"]["targets"])
    with pytest.raises(ValueError, match="MerLin"):
        predict_probabilities(model, cfg, data["test"]["bits"][:4], shots=10)
