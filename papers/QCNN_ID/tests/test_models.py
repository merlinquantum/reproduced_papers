"""Smoke tests for QCNN-ID model definitions.

These do not touch the real dataset; they only verify that every model
builds, accepts the expected input shape, and emits one binary logit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib.models import CNNClassifier, PhotonicClassifier, QCNNClassifier  # noqa: E402


def _assert_binary_affine_head(model, encoded_dim: int):
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.in_features == encoded_dim
    assert model.head.out_features == 1
    assert model.head.bias is not None
    assert model.head.bias.shape == (1,)


def test_cnn_classifier_forward():
    model = CNNClassifier(input_dim=23)
    x = torch.randn(4, 23)
    encoded = model.forward_features(x)
    out = model(x)
    assert encoded.shape == (4, 64)
    assert out.shape == (4, 1)
    _assert_binary_affine_head(model, 64)


def test_qcnn_classifier_forward_and_grad():
    model = QCNNClassifier(n_qubits=4, reps=1)
    x = torch.rand(3, 4)
    encoded = model.forward_features(x)
    out = model(x)
    assert encoded.shape == (3, 4)
    assert out.shape == (3, 1)
    _assert_binary_affine_head(model, 4)
    loss = out.sum()
    loss.backward()
    # At least one trainable parameter must receive a non-zero gradient.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )
    assert has_grad


@pytest.mark.parametrize("n_modes,n_photons", [(4, 2), (6, 3)])
def test_merlin_classifier_forward(n_modes, n_photons):

    model = PhotonicClassifier(
        n_modes=n_modes,
        n_photons=n_photons,
        seed=42,
        device=torch.device("cpu"),
    )
    assert model.output_size > 0
    x = torch.rand(2, n_modes)
    encoded = model.forward_features(x)
    out = model(x)
    assert encoded.shape == (2, model.output_size)
    assert out.shape == (2, 1)
    _assert_binary_affine_head(model, model.output_size)
