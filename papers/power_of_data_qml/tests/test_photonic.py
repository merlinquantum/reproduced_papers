"""MerLin photonic adaptation: checks against independent references."""

from __future__ import annotations

import math
from itertools import permutations

import common  # noqa: F401  (sets sys.path)
import numpy as np
import perceval as pcvl
import pytest
import torch
from lib import photonic_merlin as pm
from scipy.linalg import expm


def permanent(matrix: np.ndarray) -> complex:
    size = matrix.shape[0]
    return sum(
        np.prod([matrix[i, p[i]] for i in range(size)])
        for p in permutations(range(size))
    )


def fock_amplitude(unitary: np.ndarray, input_state, output_state) -> complex:
    """<out|U|in> for collision-free in/out via the permanent (bosons)."""
    rows = [i for i, v in enumerate(output_state) for _ in range(v)]
    cols = [i for i, v in enumerate(input_state) for _ in range(v)]
    norm = math.sqrt(np.prod([math.factorial(v) for v in output_state]))
    return permanent(unitary[np.ix_(rows, cols)]) / norm


def test_photon_count_and_input_state():
    assert pm.photon_count(4) == 2 and pm.photon_count(15) == 7
    assert pm.input_state(5) == [1, 0, 1, 0, 0]
    with pytest.raises(ValueError, match="at least 2"):
        pm.photon_count(3)


def test_haar_unitary_is_unitary_and_seeded():
    first = pm.haar_unitary(6, np.random.default_rng(0))
    np.testing.assert_allclose(first @ first.conj().T, np.eye(6), atol=1e-12)
    np.testing.assert_array_equal(first, pm.haar_unitary(6, np.random.default_rng(0)))


def test_pa_amplitudes_match_permanents():
    features = np.random.default_rng(1).standard_normal((3, 4))
    circuit, _ = pm.embedding_circuit("PA", 4, np.random.default_rng(2))
    layer = pm._layer(circuit, 4, pm.ml.MeasurementStrategy.AMPLITUDES)
    amplitudes = (
        layer(pm.circuit_inputs("PA", torch.as_tensor(features), 0.3)).detach().numpy()
    )
    keys = [tuple(k) for k in layer.output_keys]
    rng = np.random.default_rng(2)
    v, w = pm.haar_unitary(4, rng), pm.haar_unitary(4, rng)
    for row, x in enumerate(features):
        unitary = w @ np.diag(np.exp(1j * 2 * 0.3 * x)) @ v
        for out in [(1, 1, 0, 0), (0, 1, 0, 1), (0, 0, 2, 0)]:
            expected = fock_amplitude(unitary, pm.input_state(4), out)
            assert abs(amplitudes[row, keys.index(out)]) == pytest.approx(
                abs(expected), abs=1e-10
            )


def test_beam_splitter_convention_is_hopping_exponential():
    theta = 0.7
    bs = np.array(pcvl.BS(theta=theta).compute_unitary())
    hopping = np.array([[0, 1], [1, 0]])
    np.testing.assert_allclose(bs, expm(1j * theta / 2 * hopping), atol=1e-12)


@pytest.mark.parametrize("embedding", pm.PHOTONIC_EMBEDDINGS)
def test_bank_kernel_features_labels(embedding):
    features = np.random.default_rng(3).standard_normal((24, 4))
    bank = pm.compute_photonic_bank(
        embedding,
        features,
        input_scale=0.3,
        unitary_rng=np.random.default_rng(4),
        label_rng=np.random.default_rng(5),
        need_quantum_kernel=True,
        batch_size=7,
    )
    modes = pm.mode_count(embedding, 4)
    photons = pm.photon_count(modes)
    assert bank.n_qubits == modes
    assert bank.pauli_features.shape == (24, modes, 1)
    no_click = bank.pauli_features[:, :, 0]
    assert np.all((no_click >= -1e-12) & (no_click <= 1 + 1e-12))
    # Expected number of empty modes equals m - E[#occupied] >= m - k.
    assert np.all(no_click.sum(axis=1) >= modes - photons - 1e-9)
    kernel = bank.quantum_kernel
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() > -1e-9
    assert bank.qnn_labels.shape == (24,) and np.all(
        (bank.qnn_labels >= 0) & (bank.qnn_labels <= 1)
    )
    again = pm.compute_photonic_bank(
        embedding,
        features,
        input_scale=0.3,
        unitary_rng=np.random.default_rng(4),
        label_rng=np.random.default_rng(5),
        need_quantum_kernel=False,
    )
    # Different batch sizes only change floating-point summation order (~1e-16).
    np.testing.assert_allclose(again.pauli_features, bank.pauli_features, atol=1e-12)
    np.testing.assert_allclose(again.qnn_labels, bank.qnn_labels, atol=1e-12)
    assert again.quantum_kernel is None


def test_fock_space_and_input_scale_reach_the_model():
    features = np.random.default_rng(6).standard_normal((5, 6))
    circuit, _ = pm.embedding_circuit("PA", 6, np.random.default_rng(7))
    layer = pm._layer(circuit, 6, pm.ml.MeasurementStrategy.AMPLITUDES)
    assert layer.output_size == math.comb(6 + 3 - 1, 3)  # full Fock space, not C(6, 3)
    small = pm.compute_photonic_bank(
        "PA",
        features,
        input_scale=0.1,
        unitary_rng=np.random.default_rng(7),
        label_rng=None,
        need_quantum_kernel=True,
    )
    large = pm.compute_photonic_bank(
        "PA",
        features,
        input_scale=1.0,
        unitary_rng=np.random.default_rng(7),
        label_rng=None,
        need_quantum_kernel=True,
    )
    assert (
        small.quantum_kernel[np.triu_indices(5, 1)].mean()
        > large.quantum_kernel[np.triu_indices(5, 1)].mean()
    )


def test_pb_uses_n_plus_one_modes_and_every_feature():
    # Random base features so every bond mixes; then perturb only the last feature,
    # which drives the bond between modes n-1 and n (only present with n+1 modes).
    base = np.random.default_rng(8).standard_normal(4)
    features = np.stack([base, base])
    features[1, 3] += 1.0
    bank = pm.compute_photonic_bank(
        "PB",
        features,
        input_scale=1.0,
        unitary_rng=np.random.default_rng(0),
        label_rng=None,
        need_quantum_kernel=True,
    )
    assert bank.n_qubits == 5
    assert bank.quantum_kernel[0, 1] < 1 - 1e-6


def test_photon_cap_and_fock_dimension():
    assert pm.photon_count(20, 6) == 6 and pm.photon_count(8, 6) == 4
    assert sum(pm.input_state(20, 6)) == 6
    assert pm.input_state(20, 6)[:12] == [1, 0] * 6 and sum(pm.input_state(20)) == 10
    assert pm.fock_dimension(16, 8) == 490_314
    with pytest.raises(ValueError, match="at least 2"):
        pm.photon_count(20, 1)


def test_points_per_block_budget():
    assert pm.points_per_block(77_520, torch.complex128, None, 200) == 200
    assert pm.points_per_block(77_520, torch.complex128, 512, 200) == 200
    # 3.1 M amplitudes x 8 B = 23.8 MiB per point in complex64.
    assert pm.points_per_block(3_124_550, torch.complex64, 2048, 200) == 85
    assert pm.points_per_block(10**9, torch.complex64, 1, 200) == 1


@pytest.mark.parametrize("block_points", [1, 7, 24, 100])
def test_blocked_gram_matches_full_gram(block_points):
    from lib.embeddings import _fidelity_gram

    rng = np.random.default_rng(9)
    states = torch.as_tensor(
        rng.standard_normal((24, 35)) + 1j * rng.standard_normal((24, 35))
    )
    states = states / states.abs().square().sum(dim=1, keepdim=True).sqrt()
    np.testing.assert_allclose(
        pm.fidelity_gram_from_host(states, "cpu", block_points),
        _fidelity_gram(states),
        atol=1e-13,
    )


@pytest.mark.parametrize("embedding", pm.PHOTONIC_EMBEDDINGS)
def test_complex64_bank_matches_complex128(embedding):
    features = np.random.default_rng(10).standard_normal((16, 6))
    banks = [
        pm.compute_photonic_bank(
            embedding,
            features,
            input_scale=0.3,
            unitary_rng=np.random.default_rng(11),
            label_rng=np.random.default_rng(12),
            need_quantum_kernel=True,
            complex_dtype=dtype,
            batch_memory_mb=0.01,
        )
        for dtype in (torch.complex128, torch.complex64)
    ]
    np.testing.assert_allclose(
        banks[1].pauli_features, banks[0].pauli_features, atol=1e-5
    )
    np.testing.assert_allclose(banks[1].qnn_labels, banks[0].qnn_labels, atol=1e-5)
    np.testing.assert_allclose(
        banks[1].quantum_kernel, banks[0].quantum_kernel, atol=1e-5
    )


def test_capped_photon_bank_uses_the_capped_fock_space():
    features = np.random.default_rng(13).standard_normal((6, 8))
    timings = {}
    bank = pm.compute_photonic_bank(
        "PA",
        features,
        input_scale=0.3,
        unitary_rng=np.random.default_rng(14),
        label_rng=None,
        need_quantum_kernel=True,
        max_photons=2,
        timings=timings,
    )
    assert timings["n_photons"] == 2
    assert timings["fock_dimension"] == pm.fock_dimension(8, 2) == 36
    no_click = bank.pauli_features[:, :, 0]
    assert np.all(no_click.sum(axis=1) >= 8 - 2 - 1e-9)
    np.testing.assert_allclose(np.diag(bank.quantum_kernel), 1.0, atol=1e-12)
