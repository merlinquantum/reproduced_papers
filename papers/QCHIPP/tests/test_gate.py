"""Gate QCNN against the IBM Q-CHIPP v1.0.0 circuits, plus objective checks."""

from __future__ import annotations

import ibm_qchipp_reference as ibm
import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

torch.set_num_threads(1)


def source_ansatz(n):
    return ibm.create_pyramid_ansatz(QuantumCircuit(n), n)


@pytest.mark.parametrize("n,expected", [(4, 21), (8, 54), (20, 168), (46, 399)])
def test_structural_parameter_count_and_source_order(n, expected):
    from lib import gate

    assert hasattr(gate, "GateQCNN"), "GateQCNN source pyramid is not implemented"
    model = gate.GateQCNN(n_bits=2 * n, dtype=torch.float64)
    circuit = source_ansatz(n)
    assert model.n_qubits == n
    assert model.weights.numel() == circuit.num_parameters == expected
    assert model.parameter_names == tuple(str(p) for p in circuit.parameters)


def oracle_scores(n, bits, weights):
    feature = ibm.full_circuit_21p_qrac(2 * n)
    ansatz = source_ansatz(n)
    circuit = feature.compose(ansatz)
    observable = SparsePauliOp.from_list(
        [("I" * (n // 2 - 1) + "Z" + "I" * (n // 2), 0.5)]
    )
    outputs = []
    for row in bits:
        binding = dict(zip(feature.parameters, row))
        binding.update(zip(ansatz.parameters, weights))
        outputs.append(
            Statevector.from_instruction(circuit.assign_parameters(binding))
            .expectation_value(observable)
            .real
        )
    return np.array(outputs)


@pytest.mark.parametrize("n", [4, 8, 10])
@pytest.mark.parametrize("binary", [False, True])
def test_forward_matches_exact_tagged_author_qiskit(n, binary):
    from lib.gate import GateQCNN

    model = GateQCNN(n_bits=2 * n, dtype=torch.float64)
    rng = np.random.default_rng(937 + n)
    bits = rng.integers(0, 2, (4, 2 * n)) if binary else rng.uniform(0, 1, (4, 2 * n))
    for _ in range(3):
        weights = rng.normal(size=model.weights.numel())
        with torch.no_grad():
            model.weights.copy_(torch.tensor(weights))
        actual = model(torch.tensor(bits))
        expected = oracle_scores(n, bits, weights)
        assert actual.shape == (4,)
        assert actual.dtype == torch.float64
        np.testing.assert_allclose(
            actual.detach().numpy(), expected, atol=2e-12, rtol=2e-12
        )
        assert actual.abs().max() <= 0.5 + 1e-12


def test_source_objective_exact_values_plateau_zero_and_true_gradient():
    from lib import gate

    assert hasattr(gate, "source_objective"), "exact source objective is missing"
    values = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5, 2.0], dtype=torch.float64)
    for target in (0.0, 1.0):
        signed = 2 * target - 1
        for score in values:
            scores = score.reshape(1).clone().requires_grad_()
            actual = gate.source_objective(scores, torch.tensor([target]))
            expected = -signed * np.log2(max(score.item(), 1e-10))
            assert actual.item() == pytest.approx(expected, abs=1e-12)
            actual.backward()
            derivative = 0.0 if score <= 1e-10 else -signed / (score.item() * np.log(2))
            assert scores.grad.item() == pytest.approx(derivative, abs=1e-12)
    scores = torch.tensor([-0.25, 0.25], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0.0, 1.0])
    assert gate.source_objective(scores, targets).item() == pytest.approx(
        -15.609640474436812
    )
    assert torch.autograd.gradcheck(
        lambda x: gate.source_objective(x, targets), (scores,), eps=1e-6
    )


def test_corrected_physical_probability_bce_is_distinct_and_finite():
    from lib import gate

    assert hasattr(gate, "corrected_objective"), "explicit corrected BCE is missing"
    model = gate.GateQCNN()
    bits = torch.zeros((3, 20))
    assert torch.allclose(model.probabilities(bits), 0.5 + model(bits))
    for dtype in (torch.float32, torch.float64):
        scores = torch.tensor(
            [-0.5, -0.25, 0.0, 0.25, 0.5], dtype=dtype, requires_grad=True
        )
        targets = torch.tensor([0, 1, 0, 1, 1])
        physical = 0.5 + scores.detach().double().numpy()
        expected = -np.mean(
            targets.numpy() * np.log2(np.maximum(physical, 1e-10))
            + (1 - targets.numpy()) * np.log2(np.maximum(1 - physical, 1e-10))
        )
        result = gate.corrected_objective(scores, targets)
        assert result.item() == pytest.approx(expected, rel=2e-6)
        assert result.item() != pytest.approx(
            gate.source_objective(scores, targets).item()
        )
        result.backward()
        assert torch.isfinite(scores.grad).all()
    scores = torch.tensor([-0.2, 0.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: gate.corrected_objective(x, torch.tensor([0.0, 1.0])), (scores,)
    )


@pytest.mark.parametrize("n", [4, 10])
def test_batched_score_autograd_matches_all_weight_finite_differences(n):
    from lib.gate import GateQCNN, corrected_objective, source_objective

    model = GateQCNN(n_bits=2 * n, dtype=torch.float64)
    bits = torch.tensor(np.random.default_rng(20).integers(0, 2, (3, 2 * n)))
    targets = torch.tensor([0.0, 1.0, 1.0])
    functions = [
        lambda s: (s * torch.tensor([1.0, 2.0, -1.0])).sum(),
        lambda s: source_objective(s, targets),
        lambda s: corrected_objective(s, targets),
    ]
    for objective in functions:
        model.zero_grad()
        objective(model(bits)).backward()
        analytic = model.weights.grad.detach().numpy().copy()
        finite = []
        with torch.no_grad():
            for i in range(model.weights.numel()):
                original = model.weights[i].item()
                model.weights[i] = original + 1e-6
                plus = objective(model(bits)).item()
                model.weights[i] = original - 1e-6
                minus = objective(model(bits)).item()
                model.weights[i] = original
                finite.append((plus - minus) / 2e-6)
        np.testing.assert_allclose(analytic, finite, atol=3e-7, rtol=3e-6)


def test_default_precision_initialization_batching_and_no_46_simulation():
    from lib.gate import GateQCNN

    model = GateQCNN()
    assert model.weights.dtype == torch.float32
    np.testing.assert_array_equal(
        model.weights.detach().numpy(),
        np.random.default_rng(49).random(model.weights.numel()).astype(np.float32),
    )
    bits = torch.tensor(np.random.default_rng(1).integers(0, 2, (5, 20)))
    batched = model(bits)
    independent = torch.cat([model(row[None]) for row in bits])
    assert torch.allclose(batched, independent, atol=2e-7)
    expected = oracle_scores(10, bits.numpy(), model.weights.detach().numpy())
    np.testing.assert_allclose(batched.detach().numpy(), expected, atol=5e-7, rtol=5e-6)
    with pytest.raises(ValueError, match="structural"):
        GateQCNN(n_bits=92)(torch.zeros(1, 92))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(20))
    with pytest.raises(ValueError):
        GateQCNN(n_bits=18)
    with pytest.raises(ValueError):
        GateQCNN(dtype=torch.float16)


@pytest.mark.parametrize("name", ["source_objective", "corrected_objective"])
def test_losses_reject_broadcast_target_shape(name):
    from lib import gate

    with pytest.raises(ValueError, match="shape"):
        getattr(gate, name)(torch.tensor([0.1, 0.2]), torch.tensor([[0], [1]]))
