"""Independent optical and finite-shot controls for the MerLin backend."""

from itertools import permutations
from math import factorial

import numpy as np
import perceval as pcvl
import pytest
import torch
from lib.backend import (
    DEMO_INPUT_STATE,
    PhotonicBackend,
    build_clements_circuit,
    initialize_parameters,
)


def profile(transmittance=1.0, indistinguishability=1.0, detector="threshold"):
    return {
        "name": "analytic_control",
        "transmittance": transmittance,
        "indistinguishability": indistinguishability,
        "detector": detector,
    }


def permanent(matrix):
    return sum(
        np.prod([matrix[row, column] for row, column in enumerate(order)])
        for order in permutations(range(len(matrix)))
    )


def test_author_clements_parameter_order_and_twenty_seed_initialization():
    circuit = build_clements_circuit()
    names = [parameter.name for parameter in circuit.get_parameters()]
    assert len(names) == 132
    assert names == [
        name for index in range(66) for name in (f"phi_a{index}_1", f"phi_b{index}_1")
    ]
    assert DEMO_INPUT_STATE == (0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0)
    values = initialize_parameters(132)
    assert values.shape == (20, 132)
    assert np.isfinite(values).all()
    assert len({tuple(row) for row in values}) == 20
    assert np.array_equal(values[0], np.random.RandomState(0).normal(0, 2 * np.pi, 132))
    assert np.array_equal(values, initialize_parameters(132))


def test_source_model_rejects_g2_beyond_two_photon_range():
    invalid = {**profile(), "g2": 0.8}
    with pytest.raises(ValueError, match="two-photon source model"):
        PhotonicBackend((1, 1), invalid)


def test_author_mzi_phase_last_and_rectangular_unitary_order():
    circuit = build_clements_circuit()
    phases = initialize_parameters(132, (7,))[0]
    for parameter, value in zip(circuit.get_parameters(), phases, strict=True):
        parameter.set_value(value)
    # Independent 2x2 algebra follows the author's phase-last circuit:
    # BS.Rx -> phase_a on lower mode -> BS.Rx -> phase_b on lower mode.
    balanced_rx = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)
    reference = np.eye(12, dtype=complex)
    mzi_index = 0
    for column in range(12):
        for upper_mode in range(column % 2, 11, 2):
            phase_a, phase_b = phases[2 * mzi_index : 2 * mzi_index + 2]
            lower_phase_a = np.diag([1, np.exp(1j * phase_a)])
            lower_phase_b = np.diag([1, np.exp(1j * phase_b)])
            mzi = lower_phase_b @ balanced_rx @ lower_phase_a @ balanced_rx
            embedded = np.eye(12, dtype=complex)
            embedded[upper_mode : upper_mode + 2, upper_mode : upper_mode + 2] = mzi
            reference = embedded @ reference
            mzi_index += 1
    assert mzi_index == 66
    actual = np.asarray(circuit.compute_unitary(), dtype=complex)
    assert np.allclose(actual, reference, atol=1e-12)


def test_all_small_fock_probabilities_match_independent_permanents():
    backend = PhotonicBackend((1, 1, 0, 0), profile(detector="pnr"))
    values = initialize_parameters(len(backend.parameter_names), (4,))[0]
    backend.set_parameters(values)
    assert sum(parameter.numel() for parameter in backend.layer.parameters()) == 12
    unitary = np.asarray(backend.circuit.compute_unitary(), dtype=complex)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-12)
    exact = backend.exact()
    assert len(exact) == 10
    for output, probability in exact.items():
        rows = [mode for mode, photons in enumerate(output) for _ in range(photons)]
        submatrix = unitary[np.ix_(rows, [0, 1])]
        expected = abs(permanent(submatrix)) ** 2
        expected /= np.prod([factorial(photons) for photons in output])
        assert probability == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("indistinguishability", [0.0, 0.729, 1.0])
@pytest.mark.parametrize("transmittance", [1.0, 0.028])
def test_hom_loss_and_threshold_follow_analytic_predictions(
    indistinguishability, transmittance
):
    circuit = pcvl.Circuit(2).add(0, pcvl.BS())
    backend = PhotonicBackend(
        (1, 1), profile(transmittance, indistinguishability), circuit=circuit
    )
    distribution = backend.exact()
    coincidence = transmittance**2 * (1 - indistinguishability) / 2
    single_click = transmittance * (1 - transmittance)
    single_click += transmittance**2 * (1 + indistinguishability) / 4
    assert distribution.get((0, 0), 0) == pytest.approx(
        (1 - transmittance) ** 2, abs=1e-12
    )
    assert distribution[(1, 1)] == pytest.approx(coincidence, abs=1e-12)
    assert distribution[(1, 0)] == pytest.approx(single_click, abs=1e-12)
    assert distribution[(0, 1)] == pytest.approx(single_click, abs=1e-12)
    assert sum(distribution.values()) == pytest.approx(1, abs=1e-12)


def test_loss_preserves_binomial_photon_sectors_before_threshold():
    backend = PhotonicBackend(
        (1, 1, 0, 0), profile(0.2, detector="pnr"), circuit=pcvl.Circuit(4)
    )
    distribution = backend.exact()
    totals = {
        photons: sum(
            value for key, value in distribution.items() if sum(key) == photons
        )
        for photons in range(3)
    }
    assert totals == pytest.approx({0: 0.64, 1: 0.32, 2: 0.04}, abs=1e-12)


def test_collision_is_detected_as_single_click_not_missing_probability():
    backend = PhotonicBackend(
        (1, 1), profile(), circuit=pcvl.Circuit(2).add(0, pcvl.BS())
    )
    distribution = backend.exact()
    assert distribution[(1, 0)] == pytest.approx(0.5, abs=1e-12)
    assert distribution[(0, 1)] == pytest.approx(0.5, abs=1e-12)
    assert distribution.get((0, 0), 0) == 0
    assert distribution.get((1, 1), 0) == pytest.approx(0, abs=1e-12)


def test_finite_shots_include_vacuum_are_seeded_and_match_theory():
    backend = PhotonicBackend(
        (1, 1), profile(0.028, 0.729), circuit=pcvl.Circuit(2).add(0, pcvl.BS())
    )
    expected = backend.exact()
    shots = 20_000
    state_before = torch.random.get_rng_state()
    default_counts = backend.sample(shots, 42)
    native = backend.sample_native(shots, 42)
    assert torch.equal(state_before, torch.random.get_rng_state())
    assert default_counts == backend.sample(shots, 42)
    assert default_counts == native
    assert backend.timings["sample_engine"] == "merlin_native_QuantumLayer_shots"
    assert native == backend.sample_native(shots, 42)
    assert sum(default_counts.values()) == sum(native.values()) == shots
    for counts in (default_counts, native):
        assert counts[(0, 0)] > 18_000
        for key, probability in expected.items():
            deviation = abs(counts.get(key, 0) / shots - probability)
            sigma = np.sqrt(probability * (1 - probability) / shots)
            assert deviation < 7 * sigma + 3 / shots


def test_g2_extension_conserves_probability_with_extra_photons():
    backend = PhotonicBackend(
        (1, 1),
        {**profile(0.028, 0.729, detector="pnr"), "g2": 0.011},
        circuit=pcvl.Circuit(2).add(0, pcvl.BS()),
    )
    distribution = backend.exact()
    assert sum(distribution.values()) == pytest.approx(1, abs=1e-12)
    assert any(sum(key) > 2 and value > 0 for key, value in distribution.items())


def test_invalid_parameters_and_noise_fail_clearly():
    with pytest.raises(KeyError, match="transmittance"):
        PhotonicBackend((1, 0), {"name": "incomplete"})
    with pytest.raises(ValueError, match="transmittance"):
        PhotonicBackend((1, 0), profile(transmittance=1.1))
    backend = PhotonicBackend((1, 0), profile())
    with pytest.raises(ValueError, match="Expected 2"):
        backend.set_parameters([0])
    with pytest.raises(ValueError, match="finite"):
        backend.set_parameters([0, np.nan])
    with pytest.raises(ValueError, match="positive integer"):
        backend.sample(0, 0)
