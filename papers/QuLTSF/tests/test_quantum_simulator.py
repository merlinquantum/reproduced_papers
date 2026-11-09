"""The batched PyTorch circuit must equal the upstream PennyLane QNode."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import pytest
import torch

from papers.QuLTSF.lib.quantum_simulator import (
    AmplitudeSELayer,
    cnot_ring_permutation,
    default_ranges,
)

qml = pytest.importorskip("pennylane")


def _pennylane_expectations(features, weights, num_qubits):
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs, circuit_weights):
        qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), normalize=True)
        qml.StronglyEntanglingLayers(weights=circuit_weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return torch.stack(circuit(features, weights), dim=-1)


@pytest.mark.parametrize(("num_qubits", "num_layers"), [(3, 3), (4, 5), (5, 2)])
def test_expectations_match_pennylane(num_qubits, num_layers):
    torch.manual_seed(0)
    layer = AmplitudeSELayer(num_qubits, num_layers, dtype=torch.float64)
    features = torch.randn(7, 2**num_qubits, dtype=torch.float64)
    expected = _pennylane_expectations(features, layer.weights.detach(), num_qubits)
    torch.testing.assert_close(layer(features), expected, atol=1e-10, rtol=1e-10)


def test_gradients_match_pennylane():
    torch.manual_seed(1)
    num_qubits, num_layers = 3, 3
    layer = AmplitudeSELayer(num_qubits, num_layers, dtype=torch.float64)
    features = torch.randn(4, 2**num_qubits, dtype=torch.float64, requires_grad=True)
    reference_weights = layer.weights.detach().clone().requires_grad_(True)
    reference_features = features.detach().clone().requires_grad_(True)

    layer(features).sum().backward()
    _pennylane_expectations(
        reference_features, reference_weights, num_qubits
    ).sum().backward()

    torch.testing.assert_close(
        layer.weights.grad, reference_weights.grad, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        features.grad, reference_features.grad, atol=1e-10, rtol=1e-10
    )


def test_identity_circuit_reads_pauli_z_from_embedded_state():
    layer = AmplitudeSELayer(2, 3, circuit="identity", dtype=torch.float64)
    # |psi> = (|00> + |11>) / sqrt(2): <Z_0> = <Z_1> = 0.
    torch.testing.assert_close(
        layer(torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)),
        torch.zeros(1, 2, dtype=torch.float64),
    )
    # |psi> = |01>: <Z_0> = +1, <Z_1> = -1 (wire 0 is the most significant bit).
    torch.testing.assert_close(
        layer(torch.tensor([[0.0, 3.0, 0.0, 0.0]], dtype=torch.float64)),
        torch.tensor([[1.0, -1.0]], dtype=torch.float64),
    )


def test_frozen_circuit_has_no_trainable_parameters():
    layer = AmplitudeSELayer(3, 2, circuit="frozen")
    assert not any(p.requires_grad for p in layer.parameters())
    assert layer.weights.shape == (2, 3, 3)


def test_cnot_ring_permutation_matches_single_cnot():
    # One CNOT(0, 1) on two wires maps |10> -> |11> and |11> -> |10>.
    index = cnot_ring_permutation(2, 1)
    # The ring on 2 wires is CNOT(0, 1) then CNOT(1, 0).
    state = torch.eye(4)[2]  # |10>
    after = state[index]
    # CNOT(0,1): |10> -> |11>; CNOT(1,0): |11> -> |01>.
    torch.testing.assert_close(after, torch.eye(4)[1])


def test_default_ranges_follow_pennylane():
    assert default_ranges(10, 3) == [1, 2, 3]
    assert default_ranges(3, 4) == [1, 2, 1, 2]
