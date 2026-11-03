"""Cross-check the closed-form fidelity kernel against Qiskit statevectors and Aer."""

from __future__ import annotations

import math

import common  # noqa: F401
import numpy as np
import pytest
import torch
from distributed_qrc.lib.kernels import (
    bloch_vectors,
    contraction_factor,
    gram_matrix,
    kernel_block,
)
from distributed_qrc.lib.noise import IDEAL, DeviceNoise
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def feature_map(x, n_qubits: int) -> QuantumCircuit:
    """Fig. 3: two blocks of L layers of [H on all qubits, P(2 x) per qubit]."""

    layers = math.ceil(len(x) / n_qubits)
    padded = np.zeros(layers * n_qubits)
    padded[: len(x)] = x
    qc = QuantumCircuit(n_qubits)
    for _ in range(2):
        for layer in range(layers):
            for q in range(n_qubits):
                qc.h(q)
                qc.p(2 * padded[layer * n_qubits + q], q)
    return qc


def closed_form(xa, xb, n_qubits, noise=IDEAL):
    ba = torch.from_numpy(bloch_vectors(xa[None], n_qubits))
    bb = torch.from_numpy(bloch_vectors(xb[None], n_qubits))
    return float(
        kernel_block(ba, bb, contraction_factor(noise, len(xa), n_qubits))[0, 0]
    )


@pytest.mark.parametrize("n_features,n_qubits", [(5, 5), (10, 10), (20, 10), (15, 10)])
def test_ideal_kernel_matches_statevector_fidelity(n_features, n_qubits):
    rng = np.random.default_rng(n_features)
    for _ in range(4):
        xa, xb = rng.uniform(-1, 1, (2, n_features))
        sa = Statevector(feature_map(xa, n_qubits))
        sb = Statevector(feature_map(xb, n_qubits))
        assert closed_form(xa, xb, n_qubits) == pytest.approx(
            abs(sa.inner(sb)) ** 2, abs=1e-12
        )


def test_noisy_kernel_matches_aer_compute_uncompute():
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise = DeviceNoise("test", error_1q=0.01, error_2q=0.0, readout_error=0.0)
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(1 - noise.bloch_shrink_1q, 1), ["h"]
    )
    simulator = AerSimulator(method="density_matrix", noise_model=model)

    n_qubits, rng = 3, np.random.default_rng(7)
    for _ in range(3):
        xa, xb = rng.uniform(-1, 1, (2, 5))
        qc = feature_map(xa, n_qubits).compose(feature_map(xb, n_qubits).inverse())
        qc.save_probabilities()
        p_zero = simulator.run(qc).result().data()["probabilities"][0]
        assert closed_form(xa, xb, n_qubits, noise) == pytest.approx(p_zero, abs=1e-10)


def test_readout_error_factor_matches_independent_bit_flips():
    noise = DeviceNoise("test", error_1q=0.0, error_2q=0.0, readout_error=0.03)
    rng = np.random.default_rng(3)
    xa, xb = rng.uniform(-1, 1, (2, 6))
    ideal_per_qubit = [closed_form(xa[[q, q + 3]], xb[[q, q + 3]], 1) for q in range(3)]
    flipped = np.prod([(1 - 0.03) * f + 0.03 * (1 - f) for f in ideal_per_qubit])
    assert closed_form(xa, xb, 3, noise) == pytest.approx(flipped, abs=1e-12)


def test_gram_matrix_is_symmetric_with_unit_diagonal():
    x = np.random.default_rng(4).uniform(-1, 1, (50, 12))
    gram = gram_matrix(
        torch.from_numpy(bloch_vectors(x, 10)), 1.0, chunk_size=16
    ).numpy()
    assert np.allclose(gram, gram.T)
    assert np.allclose(np.diag(gram), 1.0)
    assert np.linalg.eigvalsh(gram).min() > -1e-10


def test_shot_noise_gram_is_symmetric_and_unbiased():
    from distributed_qrc.lib.kernels import ShotNoise

    x = np.random.default_rng(5).uniform(-1, 1, (300, 6))
    bloch = torch.from_numpy(bloch_vectors(x, 6))
    exact = gram_matrix(bloch, 0.95).numpy()
    generator = torch.Generator().manual_seed(0)
    noisy = gram_matrix(
        bloch, 0.95, chunk_size=64, shot_noise=ShotNoise(1024, generator)
    ).numpy()
    assert np.allclose(noisy, noisy.T)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    residual = noisy - exact
    assert abs(residual.mean()) < 1e-3
    expected_std = np.sqrt(exact * (1 - exact) / 1024).mean()
    assert 0.8 * expected_std < residual.std() < 1.2 * expected_std
