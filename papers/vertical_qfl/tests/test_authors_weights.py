"""Validate the simulator and circuit layout against the authors' trained model."""

from __future__ import annotations

import json

import pytest
import torch
from common import AUTHORS_SETTING_1
from lib.model import VerticalQuantumFederatedModel

pytestmark = pytest.mark.skipif(
    not (AUTHORS_SETTING_1 / "models_tensors.pth").exists(),
    reason="authors' assets not downloaded",
)


def _angles(state_dict):
    prefix = "q." if "q.u_layers.0.params" in state_dict else ""
    out = torch.zeros(500, 2, 8, dtype=torch.float64)
    for layer in range(500):
        for kind in range(2):
            for wire in range(8):
                key = f"{prefix}u_layers.{wire + (2 * layer + kind) * 8}.params"
                out[layer, kind, wire] = state_dict[key].reshape(-1)[0]
    return out


def test_authors_8_client_weights_reproduce_their_test_accuracy():
    def load(name):
        return torch.load(
            AUTHORS_SETTING_1 / name, weights_only=True, map_location="cpu"
        )

    checkpoint, data = load("models_tensors.pth"), load("data_tensors.pt")
    recorded = json.loads((AUTHORS_SETTING_1 / "exp.json").read_text())["all_test_acc"][
        -1
    ]
    model = VerticalQuantumFederatedModel(
        list(range(8)), "server_wide", True, dtype=torch.float64
    ).double()
    for index, client in enumerate(model.clients):
        client.angles.data = _angles(checkpoint[f"client{index + 1}_model"])
    model.server.angles.data = _angles(checkpoint["server_model"])
    with torch.no_grad():
        scores = model(data["data_te"].double())
    accuracy = (scores.argmax(1) == data["label_te"]).double().mean().item()
    assert accuracy == pytest.approx(recorded, abs=1 / 319)
