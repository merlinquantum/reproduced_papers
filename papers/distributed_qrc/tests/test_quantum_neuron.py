"""Cross-check the numpy quantum neuron against Qiskit (ideal) and Qiskit Aer (noisy)."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest
from distributed_qrc.lib.noise import IDEAL, DeviceNoise
from distributed_qrc.lib.quantum_neuron import (
    encoded_qubit_states,
    neuron_expectations,
    neuron_functional,
    observable_diagonal,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

# Exaggerated error rates make the noisy comparison sensitive.
NOISY = DeviceNoise("test", error_1q=0.01, error_2q=0.05, readout_error=0.0)
QISKIT_OBSERVABLES = {
    "z_all": "ZZZZ",
    "z0": "IIIZ",
}  # little-endian: last char is qubit 0


def neuron_circuit(x, w) -> QuantumCircuit:
    qc = QuantumCircuit(4)
    for q in range(4):
        qc.rx(x[q], q)
    for _ in range(2):
        for q in range(4):
            qc.rx(w[q], q)
        for t in range(1, 4):
            qc.cx(t - 1, t)
            qc.rz(w[t - 1] - w[t], t)
            qc.cx(t - 1, t)
    return qc


def numpy_expectation(x, w, noise, observable):
    functional = neuron_functional(w, noise, observable)[None]
    return neuron_expectations(functional, np.asarray(x)[None, None, :], noise)[0, 0]


@pytest.mark.parametrize("observable", ["z_all", "z0"])
def test_ideal_neuron_matches_qiskit_statevector(observable):
    rng = np.random.default_rng(1)
    for _ in range(5):
        x, w = rng.uniform(-1, 1, 4), rng.uniform(0, 2 * np.pi, 4)
        expected = Statevector(neuron_circuit(x, w)).expectation_value(
            SparsePauliOp(QISKIT_OBSERVABLES[observable])
        )
        assert numpy_expectation(x, w, IDEAL, observable) == pytest.approx(
            expected.real, abs=1e-10
        )


def test_noisy_neuron_matches_qiskit_aer_density_matrix():
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    shrink = NOISY.bloch_shrink_1q
    model = NoiseModel()
    # Qiskit's depolarizing parameter is 1 - (Bloch shrink); RX = 2 native gates.
    model.add_all_qubit_quantum_error(depolarizing_error(1 - shrink**2, 1), ["rx"])
    one = depolarizing_error(1 - shrink, 1)
    cx_error = depolarizing_error(1 - NOISY.depolarizing_2q, 2).compose(one.tensor(one))
    model.add_all_qubit_quantum_error(cx_error, ["cx"])
    simulator = AerSimulator(method="density_matrix", noise_model=model)

    rng = np.random.default_rng(2)
    for _ in range(3):
        x, w = rng.uniform(-1, 1, 4), rng.uniform(0, 2 * np.pi, 4)
        qc = neuron_circuit(x, w)
        qc.save_density_matrix()
        rho = DensityMatrix(simulator.run(qc).result().data()["density_matrix"])
        for observable, label in QISKIT_OBSERVABLES.items():
            expected = rho.expectation_value(SparsePauliOp(label)).real
            assert numpy_expectation(x, w, NOISY, observable) == pytest.approx(
                expected, abs=1e-10
            )


def test_readout_error_damps_each_measured_z():
    p = 0.02
    assert np.allclose(
        observable_diagonal("z_all", p),
        observable_diagonal("z_all", 0.0) * (1 - 2 * p) ** 4,
    )
    assert np.allclose(
        observable_diagonal("z0", p), observable_diagonal("z0", 0.0) * (1 - 2 * p)
    )


def test_encoded_states_are_valid_density_matrices():
    rho = encoded_qubit_states(np.linspace(-1, 1, 7), NOISY)
    assert np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1.0)
    assert np.allclose(rho, np.conj(np.swapaxes(rho, -1, -2)))
