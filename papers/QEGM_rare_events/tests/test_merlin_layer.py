from __future__ import annotations

import sys

import pytest
import torch
from common import PROJECT_DIR

# The paper package lives under papers/, which pytest does not reliably put
# on sys.path at collection time; insert it explicitly.
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

pytest.importorskip("merlin", reason="merlinquantum not installed")

from QEGM_rare_events.lib.merlin_layer import MerlinPhotonicLayer  # noqa: E402


def test_output_shape_and_range():
    torch.manual_seed(42)
    layer = MerlinPhotonicLayer(n_qubits=4, n_modes=6, n_photons=3)
    z = torch.randn(8, 4)
    out = layer(z)
    assert out.shape == (8, 4)
    assert (out >= 0).all() and (out <= 1).all()


def test_output_size_matches_unbunched_subspace():
    import math

    layer = MerlinPhotonicLayer(n_qubits=4, n_modes=6, n_photons=3)
    # UNBUNCHED with n photons in m modes has C(m, n) outcomes.
    assert layer.output_size == math.comb(6, 3)


def test_gradients_reach_quantum_parameters():
    torch.manual_seed(0)
    layer = MerlinPhotonicLayer(n_qubits=2, n_modes=4, n_photons=2)
    z = torch.randn(4, 2)
    loss = layer(z).sum()
    loss.backward()
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_hardware_settings_reports_current_api():
    layer = MerlinPhotonicLayer(n_qubits=2, n_modes=4, n_photons=2)
    hw = layer.hardware_settings()
    assert hw["computation_space"] == "UNBUNCHED"
    assert hw["measurement_strategy"] == "MeasurementStrategy.probs(UNBUNCHED)"
    assert sum(hw["input_state"]) == 2
