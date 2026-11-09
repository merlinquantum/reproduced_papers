"""Focused tests of the Phase 5 classical baselines."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from common import PROJECT_DIR, SEMEION_FILE
from lib import runner
from lib.model_classical import ClassicalCentralModel, ClassicalVerticalFederatedModel
from utils.make_classical_configs import SIZES, TARGETS, TOLERANCE

from runtime_lib import run_from_project


def _pixels(batch=4):
    return (torch.rand(batch, 256) > 0.5).double() * torch.pi / 4


@pytest.mark.parametrize("size", list(SIZES))
def test_parameter_counts_match_the_frozen_targets(size):
    hc, hw, hn, hz, _ = SIZES[size]
    wide = ClassicalVerticalFederatedModel(list(range(8)), "server_wide", hc, hw)
    narrow = ClassicalVerticalFederatedModel([0], "server_narrow", hc, hn)
    central = ClassicalCentralModel(hz)
    counts = wide.trainable_parameter_counts()
    for part, value in (
        ("client", counts["client_0"]),
        ("wide", counts["server"]),
        ("narrow", narrow.trainable_parameter_counts()["server"]),
        ("central", central.trainable_parameter_counts()["total"]),
    ):
        assert abs(value - TARGETS[size][part]) <= TOLERANCE * TARGETS[size][part], part


def test_gate_matched_e1_total_is_close_to_the_quantum_budget():
    hc, hw, *_ = SIZES["gate"]
    total = ClassicalVerticalFederatedModel(list(range(8)), "server_wide", hc, hw)
    assert total.trainable_parameter_counts()["total"] == 72059  # gate model: 72000


def test_shapes_bounded_messages_and_gradients():
    model = ClassicalVerticalFederatedModel(
        [0, 1, 2, 3, 0, 1, 2, 3], "server_wide", 7, 12
    )
    pixels = _pixels()
    logits = model(pixels)
    assert logits.shape == (4, 10)
    message = model.clients[0](torch.rand(4, 32))
    assert message.shape == (4, 8) and message.abs().max() <= 1
    torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_central_reference_uses_all_pixels():
    model = ClassicalCentralModel(12)
    pixels = _pixels(2)
    base = model(pixels).detach()
    flipped = pixels.clone()
    flipped[:, 255] = torch.where(flipped[:, 255] > 0, 0.0, torch.pi / 4)
    assert (model(flipped).detach() - base).abs().sum() > 0


def test_same_seed_gives_same_model():
    torch.manual_seed(3)
    first = ClassicalVerticalFederatedModel([0], "server_narrow", 7, 14)
    torch.manual_seed(3)
    second = ClassicalVerticalFederatedModel([0], "server_narrow", 7, 14)
    pixels = _pixels()
    assert torch.equal(first(pixels), second(pixels))


def test_config_values_reach_the_model_and_missing_keys_fail():
    cfg = json.loads((PROJECT_DIR / "configs/classical/tuning_base.json").read_text())
    cfg["model"]["client_hidden"] = 9
    runner._validate_config(cfg)
    model = runner._build_model(cfg, torch.float32)
    assert model.clients[0][0][0].out_features == 9
    cfg["model"]["architecture"] = "central"
    assert isinstance(runner._build_model(cfg, torch.float32), ClassicalCentralModel)
    del cfg["model"]["central_hidden"]
    with pytest.raises(KeyError, match="model.central_hidden"):
        runner._validate_config(cfg)


def test_tuning_candidates_cover_the_frozen_grid():
    candidates = json.loads(
        (PROJECT_DIR / "configs/classical/tuning_candidates.json").read_text()
    )
    assert len(candidates) == 2 * 3 * 4
    assert {c["training.lr"] for c in candidates} == {0.001, 0.002, 0.005, 0.01}
    epochs = {(c["model.size"], c["training.epochs"]) for c in candidates}
    assert epochs == {("gate", 30), ("photonic_h", 100), ("photonic_f", 100)}


@pytest.mark.skipif(not SEMEION_FILE.exists(), reason="Semeion data not downloaded")
def test_classical_smoke_run_uses_the_shared_split(tmp_path):
    original = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                "configs/classical/tuning_base.json",
                "--outdir",
                str(tmp_path),
                "--epochs",
                "1",
            ],
        )
    finally:
        os.chdir(original)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert (
        status["dataset"]["splits"]["train"] == 1019
        and status["dataset"]["splits"]["test"] == 319
    )
    assert status["dataset"]["splits"]["validation"] == 255
    assert (
        metrics["backend"] == "classical"
        and metrics["trainable_parameters"]["total"] == 72059
    )
