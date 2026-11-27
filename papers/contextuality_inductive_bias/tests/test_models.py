"""Quantum models, surrogate and simulator (E6, E8, and the D3/D4/D5 conventions)."""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml
import pytest
import torch
from common import PROJECT_DIR  # noqa: F401  (puts papers/ on sys.path)
from contextuality_inductive_bias.lib import simulator as sim
from contextuality_inductive_bias.lib.quantum_models import (
    CNOT_PATTERNS,
    BiasedQuantumModel,
    GenericQuantumModel,
)
from contextuality_inductive_bias.lib.surrogate import (
    FourierSurrogate,
    dense_features,
    fourier_variables,
    frequency_grid,
)

DTYPE = torch.float64


def _features(n: int, seed: int, scale: float = math.pi / 2) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 3, 3))
    s = raw / raw.sum(axis=2, keepdims=True)
    cols = [
        s[:, 0, 0],
        s[:, 1, 1],
        s[:, 2, 2],
        s[:, 0, 1] - s[:, 0, 2],
        s[:, 1, 2] - s[:, 1, 0],
        s[:, 2, 0] - s[:, 2, 1],
    ]
    return torch.tensor(scale * np.stack(cols, 1), dtype=DTYPE)


def _pennylane_reference(
    weights, alpha, x, layers, blocks, biased, zz_repetitions=1, cnot_pattern="paper"
):
    """Independent PennyLane circuit following Fig. 7 / Fig. 10 gate by gate."""
    dev = qml.device("default.qubit", wires=3)

    def unitary(w):
        for b in range(blocks):
            if biased:
                for q in range(3):
                    qml.RZ(w[b, q], wires=q)
                for _ in range(zz_repetitions):
                    qml.PauliRot(w[b, 3], "ZZ", wires=[0, 1])
                    qml.PauliRot(w[b, 4], "ZZ", wires=[0, 2])
                    qml.PauliRot(w[b, 5], "ZZ", wires=[1, 2])
                for i, pair in enumerate(((0, 1), (1, 2), (0, 2))):
                    for pauli in ("XX", "YY", "ZZ"):
                        qml.PauliRot(w[b, 6 + i], pauli, wires=list(pair))
            else:
                for q in range(3):
                    qml.Rot(w[b, 3 * q], w[b, 3 * q + 1], w[b, 3 * q + 2], wires=q)
                for c, t in CNOT_PATTERNS[cnot_pattern]:
                    qml.CNOT(wires=[c, t])

    @qml.qnode(dev)
    def circuit():
        if biased:
            for q in range(3):
                qml.Hadamard(wires=q)
            qml.RY(alpha, wires=0)
            qml.RY(alpha - math.pi, wires=1)
        for layer in range(layers):
            unitary(weights[2 * layer])
            for q in range(3):
                qml.RZ(x[q], wires=q)
            unitary(weights[2 * layer + 1])
            qml.PauliRot(x[3] * x[4], "ZZ", wires=[0, 1])
            qml.PauliRot(x[4] * x[5], "ZZ", wires=[1, 2])
            qml.PauliRot(x[3] * x[5], "ZZ", wires=[0, 2])
        unitary(weights[2 * layers])
        return [qml.expval(qml.PauliZ(q)) for q in range(3)]

    return np.array(circuit(), dtype=float)


@pytest.mark.parametrize("layers,zz_repetitions", [(1, 1), (2, 1), (2, 3)])
def test_biased_model_matches_pennylane(layers, zz_repetitions):
    model = BiasedQuantumModel(
        layers,
        2,
        torch.Generator().manual_seed(3),
        DTYPE,
        zz_repetitions=zz_repetitions,
    )
    x = _features(4, 7)
    ours = model(x).detach().numpy()
    w = model.weights.detach().numpy()
    alpha = float(model.alpha.detach())
    for i in range(4):
        ref = _pennylane_reference(
            w, alpha, x[i].numpy(), layers, 2, True, zz_repetitions=zz_repetitions
        )
        np.testing.assert_allclose(ours[i], ref, atol=1e-12)


@pytest.mark.parametrize("pattern", sorted(CNOT_PATTERNS))
def test_generic_model_matches_pennylane(pattern):
    model = GenericQuantumModel(
        2, 2, torch.Generator().manual_seed(5), DTYPE, cnot_pattern=pattern
    )
    x = _features(3, 11)
    ours = model(x).detach().numpy()
    w = model.weights.detach().numpy()
    for i in range(3):
        ref = _pennylane_reference(
            w, None, x[i].numpy(), 2, 2, False, cnot_pattern=pattern
        )
        np.testing.assert_allclose(ours[i], ref, atol=1e-12)


def test_parameter_counts_match_paper():
    gen = torch.Generator().manual_seed(0)
    biased = BiasedQuantumModel(2, 2, gen, DTYPE)
    generic = GenericQuantumModel(2, 2, gen, DTYPE, cnot_pattern="paper")
    assert sum(p.numel() for p in biased.parameters()) == 91
    assert sum(p.numel() for p in generic.parameters()) == 90
    assert FourierSurrogate(1, gen, DTYPE).weights.numel() == 3 * 1458
    assert FourierSurrogate(2, gen, DTYPE).weights.numel() == 3 * 31250


def _bias_operator() -> torch.Tensor:
    z = np.diag([1.0, -1.0])
    eye = np.eye(2)
    total = (
        np.kron(np.kron(z, eye), eye)
        + np.kron(np.kron(eye, z), eye)
        + np.kron(np.kron(eye, eye), z)
    )
    return torch.tensor(total, dtype=torch.complex128)


def test_biased_model_satisfies_zero_sum_for_all_inputs_and_parameters():
    """E6: <Z1 + Z2 + Z3> = 0 for random x (several scales) and random parameters."""
    worst = 0.0
    for seed in range(5):
        model = BiasedQuantumModel(2, 2, torch.Generator().manual_seed(seed), DTYPE)
        for scale in (1.0, math.pi / 2, 3.0):
            x = _features(200, 100 + seed, scale)
            worst = max(worst, float(model(x).detach().sum(dim=1).abs().max()))
            state = model.final_state(x).detach()
            worst = max(
                worst,
                float(sim.expectation_operator(state, _bias_operator()).abs().max()),
            )
    assert worst < 1e-12


def test_generic_model_breaks_zero_sum():
    model = GenericQuantumModel(
        2, 2, torch.Generator().manual_seed(0), DTYPE, cnot_pattern="paper"
    )
    assert float(model(_features(200, 1)).detach().sum(dim=1).abs().max()) > 0.1


def test_biased_generators_commute_with_bias_operator():
    """E6: every trainable/encoding gate of Fig. 7 commutes with H = Z1 + Z2 + Z3."""
    bias = _bias_operator().numpy()
    identity = np.eye(8, dtype=complex)

    def gate_matrix(apply):
        columns = []
        for k in range(8):
            basis = torch.zeros((1, 2, 2, 2), dtype=torch.complex128)
            basis.view(1, 8)[0, k] = 1.0
            columns.append(apply(basis).reshape(8).numpy())
        return np.stack(columns, axis=1)

    gates = [lambda s, q=q: sim.apply_rz(s, 0.37, q) for q in range(3)]
    gates += [lambda s, p=p: sim.apply_zz(s, 0.81, p) for p in ((0, 1), (1, 2), (0, 2))]
    gates += [
        lambda s, p=p: sim.apply_swaprot(s, 1.23, p) for p in ((0, 1), (1, 2), (0, 2))
    ]
    for apply in gates:
        u = gate_matrix(apply)
        np.testing.assert_allclose(u.conj().T @ u, identity, atol=1e-12)
        np.testing.assert_allclose(u @ bias - bias @ u, 0, atol=1e-12)
    non_commuting = gate_matrix(lambda s: sim.apply_ry(s, 0.5, 0))
    assert np.abs(non_commuting @ bias - bias @ non_commuting).max() > 0.1


def test_fourier_spectrum_is_omega_l():
    """E8: an L = 1 model's outputs lie in span{cos, sin(omega . z), omega in Omega_1}."""
    model = BiasedQuantumModel(1, 2, torch.Generator().manual_seed(2), DTYPE)
    x = _features(3000, 21)
    basis = dense_features(x, 1).numpy()
    outputs = model(x).detach().numpy()
    coefficients, *_ = np.linalg.lstsq(basis, outputs, rcond=None)
    assert np.abs(basis @ coefficients - outputs).max() < 1e-9
    # With a smaller spectrum the fit must fail, so the check is not vacuous.
    reduced = np.concatenate(
        [np.ones((len(x), 1)), np.cos(fourier_variables(x).numpy())], axis=1
    )
    coefficients, *_ = np.linalg.lstsq(reduced, outputs, rcond=None)
    assert np.abs(reduced @ coefficients - outputs).max() > 1e-3


@pytest.mark.parametrize("spectrum,chunk", [(1, 7), (2, 64)])
def test_factorised_surrogate_matches_dense_features(spectrum, chunk):
    model = FourierSurrogate(
        spectrum, torch.Generator().manual_seed(4), DTYPE, chunk_size=chunk
    )
    with torch.no_grad():
        model.weights.normal_(generator=torch.Generator().manual_seed(9))
    x = _features(50, 3)
    dense = dense_features(x, spectrum) @ model.weights.T
    np.testing.assert_allclose(
        model.raw_output(x).detach().numpy(), dense.detach().numpy(), atol=1e-9
    )
    clipped = model(x)
    assert float(clipped.detach().abs().max()) <= 1.0


def test_surrogate_gradient_matches_dense():
    model = FourierSurrogate(1, torch.Generator().manual_seed(4), DTYPE, chunk_size=16)
    x = _features(40, 8)
    model.raw_output(x).pow(2).sum().backward()
    factorised_grad = model.weights.grad.clone()
    weights = model.weights.detach().clone().requires_grad_(True)
    (dense_features(x, 1) @ weights.T).pow(2).sum().backward()
    np.testing.assert_allclose(factorised_grad.numpy(), weights.grad.numpy(), atol=1e-9)


def test_frequency_grid_size():
    assert frequency_grid(1).shape == (729, 6)
    assert frequency_grid(2).shape == (15625, 6)


def test_surrogate_initial_scale_follows_app_f():
    model = FourierSurrogate(1, torch.Generator().manual_seed(0), DTYPE)
    assert float(model.weights.detach().abs().max()) <= 1.0 / 1458
