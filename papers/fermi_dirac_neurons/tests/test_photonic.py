"""Phase 4 MerLin adaptation: photonic Fermi-Dirac neuron, baselines, E13 and the Kerr ablation."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
import scipy.linalg
from common import PROJECT_DIR, load_runtime_ready_config
from fermi_dirac_neurons.lib import photonic as ph

merlin = pytest.importorskip("merlin")
torch = pytest.importorskip("torch")

VALIDATOR = os.environ.get("ALPS_LOGGING_VALIDATOR", "")


def _dense_expectation(observable: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("mi,ij,mj->m", states.conj(), observable, states))


def _dense_activation(hamiltonian: np.ndarray, function) -> np.ndarray:
    values, vectors = np.linalg.eigh(hamiltonian)
    return vectors @ np.diag(function(values)) @ vectors.conj().T


def _random_theta(model, rng):
    return np.concatenate(
        [
            rng.uniform(0.0, 2.0 * np.pi, model.n_phases),
            rng.uniform(-2.0, 2.0, model.n_modes),
        ]
    )


@pytest.mark.parametrize(("n_modes", "n_photons"), [(3, 2), (4, 2), (5, 3), (3, 3)])
def test_fock_basis_matches_merlin_output_keys(n_modes, n_photons):
    builder = merlin.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer()
    layer = merlin.QuantumLayer(
        builder=builder,
        input_state=[1] * min(n_photons, n_modes)
        + [0] * (n_modes - min(n_photons, n_modes)),
        n_photons=n_photons,
        measurement_strategy=merlin.MeasurementStrategy.probs(
            computation_space=merlin.ComputationSpace.FOCK
        ),
    )
    keys = tuple(tuple(int(n) for n in key) for key in layer.output_keys)
    assert keys == ph.fock_basis(n_modes, n_photons)


def test_permanent_state_matches_matrix_exponential():
    rng = np.random.default_rng(0)
    n_modes, n_photons = 4, 3
    generator = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(
        size=(n_modes, n_modes)
    )
    generator = (generator + generator.conj().T) / 2
    unitary = scipy.linalg.expm(1j * generator)
    for input_state in [(1, 1, 1, 0), (2, 0, 1, 0), (0, 0, 0, 3)]:
        state = ph.interferometer_state(unitary, input_state, n_photons)
        basis = ph.fock_basis(n_modes, n_photons)
        reference = scipy.linalg.expm(1j * ph.fock_operator(generator, n_photons))[
            :, basis.index(input_state)
        ]
        np.testing.assert_allclose(state, reference, atol=1e-12)


@pytest.mark.parametrize(("n_modes", "n_photons"), [(4, 2), (4, 3), (5, 2)])
def test_merlin_neuron_equals_dense_activation_observable(n_modes, n_photons):
    """Tr[tanh(H/T) rho] = sum_n p_U(n) tanh(eps.n/T) with h = U^dag diag(eps) U."""
    rng = np.random.default_rng(n_modes + 10 * n_photons)
    temperature = 2.0
    d = len(ph.fock_basis(n_modes, n_photons))
    states = ph.haar_full_states(d, 6, rng)
    data = ph.PhotonicDataset(states, np.zeros(6))
    for activation, function in (
        ("tanh", lambda x: np.tanh(x / temperature)),
        ("linear", lambda x: x),
    ):
        model = ph.MerlinPassiveNeuron(n_modes, n_photons, activation, temperature)
        theta = _random_theta(model, rng)
        outputs = model.outputs(theta, data)
        unitary = model.unitary()
        np.testing.assert_allclose(
            unitary.conj().T @ unitary, np.eye(n_modes), atol=1e-12
        )
        single_particle = unitary.conj().T @ np.diag(theta[model.n_phases :]) @ unitary
        hamiltonian = ph.fock_operator(single_particle, n_photons)
        reference = _dense_expectation(_dense_activation(hamiltonian, function), states)
        np.testing.assert_allclose(outputs, reference, atol=1e-12)


def test_merlin_neuron_uses_fock_space_float64_and_configured_temperature():
    model = ph.build_photonic_model(
        {"kind": "merlin_passive", "activation": "tanh"}, 4, 2, 0.7
    )
    assert model.layer.computation_space == merlin.ComputationSpace.FOCK
    assert model.layer.output_size == len(ph.fock_basis(4, 2))
    assert model.temperature == 0.7
    assert model.n_parameters == 4 * 3 + 4
    rng = np.random.default_rng(1)
    data = ph.PhotonicDataset(ph.haar_full_states(10, 3, rng), np.zeros(3))
    probabilities = model.probabilities(data)
    assert probabilities.dtype == torch.float64
    np.testing.assert_allclose(
        probabilities.detach().numpy().sum(axis=1), 1.0, atol=1e-12
    )


def _finite_difference(model, theta, data, step=1e-6):
    gradient = np.zeros_like(theta)
    for index in range(len(theta)):
        shift = np.zeros_like(theta)
        shift[index] = step
        gradient[index] = (
            model.loss_and_gradient(theta + shift, data, "squared")[0]
            - model.loss_and_gradient(theta - shift, data, "squared")[0]
        ) / (2 * step)
    return gradient


@pytest.mark.parametrize(
    "kind", ["merlin_tanh", "merlin_linear", "count", "correlation"]
)
def test_gradients_match_finite_differences(kind):
    rng = np.random.default_rng(3)
    states = ph.haar_interferometer_states(4, 2, 5, rng)
    data = ph.PhotonicDataset(states, rng.normal(scale=0.2, size=5))
    if kind.startswith("merlin"):
        model = ph.MerlinPassiveNeuron(4, 2, kind.split("_")[1], 2.0)
        theta = _random_theta(model, rng)
    else:
        model = ph.build_photonic_model(
            {"kind": {"count": "photon_count"}.get(kind, kind)}, 4, 2, 2.0
        )
        theta = rng.uniform(-0.5, 0.5, model.n_parameters)
    _, gradient = model.loss_and_gradient(theta, data, "squared")
    assert np.all(np.isfinite(gradient))
    assert np.any(gradient[: max(1, len(gradient) - 4)] != 0)
    np.testing.assert_allclose(
        gradient, _finite_difference(model, theta, data), atol=1e-8
    )


def test_photon_count_neuron_is_the_commuting_reduction():
    """With a diagonal (commuting) H the neuron is a classical neuron on photon counts (eq. 12 analogue)."""
    rng = np.random.default_rng(4)
    model = ph.PhotonCountNeuron(3, 2, 2.0)
    theta = rng.uniform(-1, 1, model.n_parameters)
    states = ph.haar_full_states(6, 4, rng)
    data = ph.PhotonicDataset(states, np.zeros(4))
    hamiltonian = np.diag(model.design @ theta)
    reference = _dense_expectation(
        _dense_activation(hamiltonian, lambda x: np.tanh(x / 2.0)), states
    )
    np.testing.assert_allclose(model.outputs(theta, data), reference, atol=1e-12)


def test_correlation_features_are_generator_expectations():
    rng = np.random.default_rng(5)
    n_modes, n_photons = 3, 2
    states = ph.haar_full_states(6, 3, rng)
    data = ph.PhotonicDataset(states, np.zeros(3))
    features = data.correlation_features(n_modes, n_photons)
    model = ph.CorrelationNeuron(n_modes, n_photons, 2.0)
    for column, label in enumerate(model.labels):
        single = np.zeros((n_modes, n_modes), dtype=complex)
        if label.startswith("n"):
            single[int(label[1:]), int(label[1:])] = 1
        else:
            i, j = map(int, label[1:].split(","))
            if label.startswith("X"):
                single[i, j] = single[j, i] = 1
            else:
                single[i, j], single[j, i] = 1j, -1j
        reference = _dense_expectation(ph.fock_operator(single, n_photons), states)
        np.testing.assert_allclose(features[:, column], reference, atol=1e-12)


def test_teacher_targets_match_dense_activation():
    rng = np.random.default_rng(6)
    teacher = ph.build_teacher(4, 2, 2.0, rng)
    _, generators = ph.passive_generators(4)
    hamiltonian = ph.fock_operator(np.einsum("j,jab->ab", teacher.zeta, generators), 2)
    states = ph.haar_full_states(10, 5, rng)
    reference = _dense_expectation(
        _dense_activation(hamiltonian, lambda x: np.tanh(x / 2.0)), states
    )
    np.testing.assert_allclose(teacher.targets(states, 2.0), reference, atol=1e-12)


def test_readout_ablations():
    rng = np.random.default_rng(7)
    model = ph.MerlinPassiveNeuron(4, 2, "tanh", 2.0)
    theta = _random_theta(model, rng)
    basis = ph.fock_basis(4, 2)
    input_state = (1, 1, 0, 0)
    state = np.zeros(len(basis), dtype=complex)
    state[basis.index(input_state)] = 1
    data = ph.PhotonicDataset(state[None, :], np.zeros(1))
    np.testing.assert_allclose(
        model.readout_outputs(theta, data, "pnr"),
        model.outputs(theta, data),
        atol=1e-14,
    )
    # UNBUNCHED postselection must equal MerLin's own UNBUNCHED layer, renormalized.
    builder = merlin.CircuitBuilder(n_modes=4)
    builder.add_entangling_layer()
    unbunched_layer = merlin.QuantumLayer(
        builder=builder,
        input_state=list(input_state),
        n_photons=2,
        dtype=torch.float64,
        measurement_strategy=merlin.MeasurementStrategy.probs(
            computation_space=merlin.ComputationSpace.UNBUNCHED
        ),
    )
    with torch.no_grad():
        for target, source in zip(unbunched_layer.parameters(), model.phase_parameters):
            target.copy_(source)
        probabilities = unbunched_layer().numpy()[0]
    energies = (
        np.array(unbunched_layer.output_keys, dtype=float) @ theta[model.n_phases :]
    )
    reference = probabilities @ np.tanh(energies / 2.0) / probabilities.sum()
    np.testing.assert_allclose(
        model.readout_outputs(theta, data, "unbunched"), [reference], atol=1e-12
    )
    # Threshold: bunched outcomes count as one photon per clicked mode.
    with torch.no_grad():
        full = model.probabilities(data).numpy()[0]
    clicks = np.minimum(ph.occupations(4, 2), 1.0)
    expected = full @ np.tanh(clicks @ theta[model.n_phases :] / 2.0)
    np.testing.assert_allclose(
        model.readout_outputs(theta, data, "threshold"), [expected], atol=1e-12
    )
    with pytest.raises(ValueError, match="readout"):
        model.readout_outputs(theta, data, "homodyne")


def test_sampled_outputs_are_seeded_and_converge():
    rng = np.random.default_rng(8)
    model = ph.MerlinPassiveNeuron(4, 2, "tanh", 2.0)
    theta = _random_theta(model, rng)
    data = ph.PhotonicDataset(ph.haar_interferometer_states(4, 2, 4, rng), np.zeros(4))
    first = model.sampled_outputs(theta, data, 200, np.random.default_rng(1))
    second = model.sampled_outputs(theta, data, 200, np.random.default_rng(1))
    np.testing.assert_array_equal(first, second)
    many = model.sampled_outputs(theta, data, 200_000, np.random.default_rng(2))
    # Each shot lies in [-1, 1]: Hoeffding gives |error| < 0.01 w.p. > 1 - 1e-4.
    np.testing.assert_allclose(many, model.outputs(theta, data), atol=0.01)


def test_states_are_normalized_and_fock_set_is_complete():
    rng = np.random.default_rng(9)
    states = ph.build_states("haar_interferometer", 5, 2, 7, rng)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-12)
    fock = ph.build_states("fock", 3, 2, None, rng)
    np.testing.assert_array_equal(fock, np.eye(6))
    with pytest.raises(ValueError, match="all d basis states"):
        ph.build_states("fock", 3, 2, 4, rng)
    with pytest.raises(ValueError, match="Unknown photonic state set"):
        ph.build_states("coherent", 3, 2, 4, rng)


def test_kerr_gradient_matches_finite_differences():
    """The E13-K Daleckii-Krein gradient against finite differences of the dense f."""
    from fermi_dirac_neurons.lib.activations import (
        divided_differences,
        make_activation,
    )

    rng = np.random.default_rng(10)
    n_modes, n_photons, kappa, temperature = 3, 2, 2.0, 2.0
    labels, generators = ph.passive_generators(n_modes)
    dense = np.array([ph.fock_operator(g, n_photons) for g in generators])
    kerr = np.diag(ph.nearest_neighbour_kerr(n_modes, n_photons))
    theta = rng.uniform(-0.5, 0.5, len(labels))
    state = ph.haar_full_states(len(ph.fock_basis(n_modes, n_photons)), 1, rng)[0]

    def output(parameters):
        hamiltonian = np.einsum("j,jab->ab", parameters, dense) + kappa * kerr
        return float(
            _dense_expectation(
                _dense_activation(hamiltonian, lambda x: np.tanh(x / temperature)),
                state[None, :],
            )[0]
        )

    hamiltonian = np.einsum("j,jab->ab", theta, dense) + kappa * kerr
    values, vectors = np.linalg.eigh(hamiltonian)
    divided = divided_differences(make_activation("tanh", temperature), values)
    rotated_state = state @ vectors.conj()
    for index in (0, labels.index("X0,1")):
        kernel = divided * (vectors.conj().T @ dense[index] @ vectors)
        analytic = float(np.real(rotated_state.conj() @ kernel @ rotated_state))
        step = np.zeros_like(theta)
        step[index] = 1e-6
        numeric = (output(theta + step) - output(theta - step)) / 2e-6
        assert abs(analytic - numeric) < 1e-8


def _run_config(tmp_path, name):
    from fermi_dirac_neurons.lib import runner

    from runtime_lib.config import deep_update

    overlay = json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())
    cfg = deep_update(load_runtime_ready_config(), copy.deepcopy(overlay))
    cfg["models"] = overlay.get("models", cfg["models"])
    run_dir = tmp_path / name
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    return run_dir, json.loads((run_dir / "metrics.json").read_text())


@pytest.mark.parametrize(
    ("name", "training"),
    [("p4_smoke", True), ("p4_grad_smoke", False), ("p4_kerr_smoke", False)],
)
def test_photonic_tasks_write_contract_artifacts(tmp_path, name, training):
    run_dir, metrics = _run_config(tmp_path, name)
    if name == "p4_smoke":
        assert set(metrics["models"]) == {
            "photonic_tanh",
            "photonic_linear",
            "photon_count_tanh",
            "correlation_tanh",
        }
        assert set(metrics["readout_ablations"]) == {"photonic_tanh", "photonic_linear"}
        assert (run_dir / "history.csv").exists()
    else:
        assert all(np.isfinite(record["variance"]) for record in metrics["records"])
    if VALIDATOR and os.path.exists(VALIDATOR):
        command = [
            sys.executable,
            VALIDATOR,
            "--run-dir",
            str(run_dir),
            "--require-evaluation",
        ]
        if training:
            command.append("--require-training")
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_photonic_training_is_seeded(tmp_path):
    runs = []
    for label in ("a", "b"):
        (tmp_path / label).mkdir()
        runs.append(_run_config(tmp_path / label, "p4_smoke")[1])
    first, second = runs
    assert first["target"] == second["target"]
    for name in first["models"]:
        assert (
            first["models"][name]["final_parameters"]
            == second["models"][name]["final_parameters"]
        )
