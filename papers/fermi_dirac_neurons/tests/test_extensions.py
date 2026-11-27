"""Extension modules: gradient-variance study (E10) and Kirby-Love contextuality (E12)."""

from __future__ import annotations

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
from fermi_dirac_neurons.lib import contextuality as cx
from fermi_dirac_neurons.lib import trainability as tr
from fermi_dirac_neurons.lib.models import Dataset, HamiltonianNeuron
from fermi_dirac_neurons.lib.pauli import term_labels


def test_kirby_love_known_cases():
    assert cx.is_noncontextual(term_labels("tfim", 2, True))
    assert not cx.is_noncontextual(term_labels("tfim", 3, True))
    assert not cx.is_noncontextual(term_labels("heisenberg", 2, False))
    # Peres-Mermin square: the textbook contextual Pauli set.
    assert not cx.is_noncontextual(
        ["XI", "IX", "XX", "IZ", "ZI", "ZZ", "XZ", "ZX", "YY"]
    )
    # Commuting sets are trivially noncontextual.
    assert cx.is_noncontextual(term_labels("fcim", 4, False))


def test_commutation_matrix_matches_dense_matrices():
    from fermi_dirac_neurons.lib.pauli import pauli_string

    labels = ["XY", "ZZ", "YI", "IX", "XX"]
    matrix = cx.commutation_matrix(labels)
    for a, left in enumerate(labels):
        for b, right in enumerate(labels):
            product = pauli_string(left) @ pauli_string(right)
            reverse = pauli_string(right) @ pauli_string(left)
            assert matrix[a, b] == np.allclose(product, reverse)


def test_pauli_expansion_reconstructs_observable():
    from fermi_dirac_neurons.lib.pauli import pauli_string

    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    observable = matrix + matrix.conj().T
    labels = cx.all_pauli_labels(3)
    coefficients = cx.pauli_expansion(observable, labels)
    rebuilt = sum(c * pauli_string(label) for c, label in zip(coefficients, labels))
    np.testing.assert_allclose(rebuilt, observable, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_noncontextual_term_sets_stay_noncontextual_at_every_temperature(seed):
    rng = np.random.default_rng(seed)
    labels = cx.random_noncontextual_labels(3, 6, 2, rng)
    records = cx.temperature_scan(
        labels, rng.uniform(-1, 1, len(labels)), [50, 5, 1, 0.2, 0.05], [1e-8]
    )
    assert all(record["noncontextual"] for record in records)


def test_contextual_term_sets_stay_contextual():
    labels = term_labels("tfim", 3, True)
    records = cx.temperature_scan(
        labels,
        np.random.default_rng(1).uniform(-1, 1, len(labels)),
        [50, 1, 0.05],
        [1e-6],
    )
    assert not any(record["noncontextual"] for record in records)
    assert all(0.0 < record["contextual_fraction"] < 1.0 for record in records)


def test_variance_sweep_gradients_match_model_gradients():
    records = tr.variance_cell_sweep(
        "tfim",
        True,
        3,
        0.5,
        {"fixed": [2.0], "sqrt_n_factors": []},
        ["zero"],
        [0],
        2,
        np.random.default_rng(3),
    )
    by_quantity = {record["quantity"]: record for record in records}
    # The stream draws the target coefficients first, then the two theta
    # vectors; the |0...0> input is fixed.
    rng = np.random.default_rng(3)
    zeta = rng.uniform(-2.0, 2.0, 6)
    thetas = [rng.uniform(-0.5, 0.5, 6) for _ in range(2)]
    model = HamiltonianNeuron("tfim", 3, True, "tanh", 2.0)
    state = np.zeros((1, 8, 8), dtype=complex)
    state[0, 0, 0] = 1.0
    target = model.outputs(zeta, Dataset(state, np.zeros(1)))[0]
    output_gradients, loss_gradients = [], []
    for theta in thetas:
        output = model.outputs(theta, Dataset(state, np.zeros(1)))[0]
        _, gradient_zero_target = model.loss_and_gradient(
            theta, Dataset(state, np.zeros(1)), "squared"
        )
        # loss = f**2 for target 0, so d loss = 2 f d f.
        output_gradients.append(gradient_zero_target[0] / (2 * output))
        _, gradient = model.loss_and_gradient(
            theta, Dataset(state, np.array([target])), "squared"
        )
        loss_gradients.append(gradient[0])
    assert by_quantity["output"]["mean"] == pytest.approx(
        np.mean(output_gradients), rel=1e-10
    )
    assert by_quantity["output"]["variance"] == pytest.approx(
        np.var(output_gradients, ddof=1), rel=1e-8
    )
    assert by_quantity["loss"]["mean"] == pytest.approx(
        np.mean(loss_gradients), rel=1e-8
    )
    low, high = by_quantity["output"]["variance_ci95"]
    assert 0.0 <= low <= high


def test_tfim_global_adds_two_global_strings():
    assert term_labels("tfim_global", 3, False)[-2:] == ["XXX", "ZZZ"]
    assert (
        len(term_labels("tfim_global", 3, False))
        == len(term_labels("tfim", 3, False)) + 2
    )


def test_variance_sweep_rejects_single_draw():
    with pytest.raises(ValueError, match="draws"):
        tr.variance_cell_sweep(
            "tfim",
            True,
            2,
            0.5,
            {"fixed": [1.0], "sqrt_n_factors": []},
            ["zero"],
            [0],
            1,
            np.random.default_rng(0),
        )


def test_haar_data_variance_formula():
    record = tr.data_variance_check(
        "heisenberg", False, 4, 0.5, 1.0, 0, 20000, np.random.default_rng(4)
    )
    assert record["empirical_variance"] == pytest.approx(
        record["predicted_variance"], rel=0.05
    )
    assert record["empirical_mean"] == pytest.approx(
        record["predicted_mean"], abs=4 * np.sqrt(record["predicted_variance"] / 20000)
    )


def test_temperature_rules():
    pairs = dict(tr.temperatures_for(4, {"fixed": [1.0], "sqrt_n_factors": [0.5]}))
    assert pairs == {"fixed_1": 1.0, "sqrt_n_0.5": 1.0}


def test_plot_roles_cover_every_configured_model():
    import json

    from common import PROJECT_DIR
    from fermi_dirac_neurons.lib.plotting import model_role

    names = {
        model["name"]
        for path in (PROJECT_DIR / "configs").glob("fig*.json")
        for model in json.loads(path.read_text())["models"]
    }
    roles = {name: model_role(name) for name in names}
    assert roles["quantum_tfim"] == "quantum"
    assert roles["quantum_heis_linear"] == "linear"
    assert roles["b3_quantum_linear"] == "linear"
    assert roles["classical_fcim"] == "commuting"
    assert roles["b2_pauli_features"] == "b2"


def test_mrrr_eigh_matches_numpy_eigh():
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    hermitian = (matrix + matrix.conj().T) / 2
    values, vectors = tr.hermitian_eigh(hermitian)
    reference_values, _ = np.linalg.eigh(hermitian)
    np.testing.assert_allclose(values, reference_values, atol=1e-12)
    np.testing.assert_allclose(hermitian @ vectors, vectors * values, atol=1e-12)


@pytest.mark.parametrize("n_qubits", [3, 4, 6])
def test_ising_x0_is_noncontextual_noncommuting_with_tfim_parameter_count(n_qubits):
    labels = term_labels("ising_x0", n_qubits, False)
    assert len(labels) == len(term_labels("tfim", n_qubits, False)) == 2 * n_qubits - 1
    assert cx.is_noncontextual(labels)
    assert not cx.commutation_matrix(labels).all()
    assert not cx.is_noncontextual(term_labels("tfim", n_qubits, False))


@pytest.mark.parametrize("n_qubits", [2, 3])
def test_fast_pauli_expansion_matches_dense_traces(n_qubits):
    rng = np.random.default_rng(n_qubits)
    dim = 2**n_qubits
    matrix = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    observable = matrix + matrix.conj().T
    labels = cx.all_pauli_labels(n_qubits)
    np.testing.assert_allclose(
        cx.fast_pauli_expansion(observable),
        cx.pauli_expansion(observable, labels),
        atol=1e-12,
    )


def test_paulis_needed_counts_the_tail():
    coefficients = np.array([0.7, 0.5, -0.004, 0.003, 0.0, 0.2])
    assert cx.paulis_needed(coefficients, 1e-2) == 2
    assert cx.paulis_needed(coefficients, 1e-3) == 4
    assert cx.paulis_needed(coefficients, 1.0) == 0


@pytest.mark.parametrize("n_qubits", [3, 5, 8])
def test_constructed_noncontextual_sets(n_qubits):
    labels = cx.random_noncontextual_two_local(
        n_qubits, 2 * n_qubits - 1, np.random.default_rng(n_qubits)
    )
    assert len(set(labels)) == 2 * n_qubits - 1
    assert all(sum(c != "I" for c in label) <= 2 for label in labels)
    assert cx.is_noncontextual(labels) and not cx.commutation_matrix(labels).all()
