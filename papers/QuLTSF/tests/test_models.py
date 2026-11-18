"""Model shapes, parameter counts and agreement with the PennyLane reference."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import pytest
import torch

from papers.QuLTSF.lib.models import (
    MLP,
    DLinear,
    Linear,
    NLinear,
    Persistence,
    QuLTSF,
    RankBottleneck,
    count_trainable_parameters,
    mlp_hidden_for_budget,
    qultsf_parameter_count,
)


def test_qultsf_parameter_count_matches_paper_setting():
    model = QuLTSF(
        336, 96, num_qubits=10, num_layers=3, circuit="trainable", dtype=torch.float32
    )
    assert count_trainable_parameters(model) == 346_234
    assert qultsf_parameter_count(336, 96, 10, 3) == 346_234


def test_identity_and_frozen_circuits_drop_only_the_circuit_weights():
    for circuit in ("identity", "frozen"):
        model = QuLTSF(
            336, 96, num_qubits=10, num_layers=3, circuit=circuit, dtype=torch.float32
        )
        assert count_trainable_parameters(model) == 346_234 - 90


@pytest.mark.parametrize(
    "model",
    [
        QuLTSF(
            24, 8, num_qubits=3, num_layers=2, circuit="trainable", dtype=torch.float32
        ),
        Linear(24, 8, torch.float32),
        NLinear(24, 8, torch.float32),
        DLinear(24, 8, torch.float32),
        Persistence(8),
        RankBottleneck(24, 8, rank=4, dtype=torch.float32),
        MLP(24, 8, hidden=5, dtype=torch.float32),
    ],
)
def test_models_map_lookback_to_horizon_per_channel(model):
    x = torch.randn(3, 24, 5)
    assert model(x).shape == (3, 8, 5)


def test_models_are_channel_independent():
    torch.manual_seed(0)
    model = QuLTSF(
        16, 4, num_qubits=3, num_layers=2, circuit="trainable", dtype=torch.float64
    )
    x = torch.randn(2, 16, 3, dtype=torch.float64)
    full = model(x)
    single = model(x[:, :, 1:2])
    torch.testing.assert_close(full[:, :, 1:2], single)


def test_persistence_repeats_last_value():
    x = torch.arange(12.0).reshape(1, 6, 2)
    torch.testing.assert_close(Persistence(3)(x), x[:, -1:, :].expand(1, 3, 2))


def test_nlinear_adds_back_last_value():
    model = NLinear(4, 2, torch.float64)
    torch.nn.init.zeros_(model.Linear.weight)
    torch.nn.init.zeros_(model.Linear.bias)
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    torch.testing.assert_close(model(x), x[:, -1:, :].expand(2, 2, 3))


def test_mlp_budget_is_close_to_qultsf():
    budget = qultsf_parameter_count(336, 96, 10, 3)
    hidden = mlp_hidden_for_budget(336, 96, budget)
    count = count_trainable_parameters(MLP(336, 96, hidden, torch.float32))
    assert abs(count - budget) <= 336 + 96 + 1


def test_torch_qultsf_matches_pennylane_reference():
    pytest.importorskip("pennylane")
    from papers.QuLTSF.lib.models import PennyLaneQuLTSF

    torch.manual_seed(7)
    fast = QuLTSF(
        12, 5, num_qubits=3, num_layers=3, circuit="trainable", dtype=torch.float64
    )
    torch.manual_seed(7)
    reference = PennyLaneQuLTSF(
        12,
        5,
        num_qubits=3,
        num_layers=3,
        qml_device="default.qubit",
        dtype=torch.float64,
    )

    # Same seed and creation order give the same initial weights as upstream.
    torch.testing.assert_close(
        fast.input_classical_layer.weight, reference.model.input_classical_layer.weight
    )
    torch.testing.assert_close(
        fast.hidden_quantum_layer.weights,
        reference.model.hidden_quantum_layer.weights.to(torch.float64),
    )
    x = torch.randn(2, 12, 4, dtype=torch.float64)
    torch.testing.assert_close(fast(x), reference(x), atol=1e-8, rtol=1e-8)
