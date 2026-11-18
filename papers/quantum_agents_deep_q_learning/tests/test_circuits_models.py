"""Simulator and model correctness against an independent gate-by-gate reference."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from common import PROJECT_DIR
from lib import circuits
from lib.models import MLPQFunction, PQCQFunction


def load_config(name: str) -> dict:
    return json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())


# ----------------------------------------------------------- reference simulator
def _gate_on_qubit(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    operator = np.array([[1.0 + 0j]])
    for index in range(n_qubits):
        operator = np.kron(operator, gate if index == qubit else np.eye(2))
    return operator


def _rx(t):
    return np.array(
        [[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]]
    )


def _ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


def _cz(first: int, second: int, n_qubits: int) -> np.ndarray:
    diagonal = np.ones(2**n_qubits)
    for index in range(2**n_qubits):
        bits = [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[first] and bits[second]:
            diagonal[index] = -1
    return np.diag(diagonal)


def _z_product(qubits, n_qubits):
    operator = np.eye(1)
    for index in range(n_qubits):
        operator = np.kron(operator, np.diag([1, -1]) if index in qubits else np.eye(2))
    return operator


def reference_cartpole_q(
    theta, input_weights, output_weights, x, reuploading, activation
):
    """Gate-by-gate Fig. 1 circuit for one agent and one input (numpy, complex128)."""
    n = 4
    state = np.zeros(16, dtype=complex)
    state[0] = 1
    n_layers = theta.shape[0]
    for layer in range(n_layers):
        if reuploading or layer == 0:
            weights = input_weights[layer if reuploading else 0]
            for q in range(n):
                angle = activation(x[q] * weights[q])
                state = _gate_on_qubit(_rx(angle), q, n) @ state
        for q in range(n):
            state = _gate_on_qubit(_ry(theta[layer, q, 0]), q, n) @ state
            state = _gate_on_qubit(_rz(theta[layer, q, 1]), q, n) @ state
        for q in range(n):
            state = _cz(q, (q + 1) % n, n) @ state
    values = []
    for action, pair in enumerate(((0, 1), (2, 3))):
        expectation = np.real(state.conj() @ _z_product(pair, n) @ state)
        values.append((expectation + 1) / 2 * output_weights[action])
    return np.array(values)


# ------------------------------------------------------------------------ tests
def test_kron_ordering_qubit_zero_is_most_significant():
    x_gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)
    operator = circuits.kron_qubits(torch.stack([x_gate, identity, identity, identity]))
    state = torch.zeros(16, dtype=torch.complex128)
    state[0] = 1
    assert int(torch.argmax(circuits.apply_operator(state, operator).abs())) == 8


def test_rotation_matrices_are_unitary():
    angles = torch.linspace(-3, 3, 7, dtype=torch.float64)
    for builder in (circuits.rx_matrices, circuits.ry_matrices, circuits.rz_matrices):
        matrices = builder(angles, torch.complex128)
        products = matrices @ matrices.conj().transpose(-1, -2)
        assert torch.allclose(
            products, torch.eye(2, dtype=torch.complex128).expand_as(products)
        )


@pytest.mark.parametrize("reuploading", [True, False])
@pytest.mark.parametrize("activation", ["arctan", "tanh"])
def test_pqc_matches_gate_by_gate_reference(reuploading, activation):
    cfg = load_config("e2e3_cartpole_optimal_full_original")
    cfg["model"]["data_reuploading"] = reuploading
    cfg["model"]["encoding_activation"] = activation
    seeds = [3, 11]
    model = PQCQFunction(cfg, seeds, n_actions=2, real_dtype=torch.float64)
    with torch.no_grad():
        model.input_weights.uniform_(0.5, 1.5)
        model.output_weights.uniform_(1.0, 50.0)
    observations = torch.randn(
        2, 5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    q_values = model(observations).detach().numpy()
    activation_fn = np.arctan if activation == "arctan" else np.tanh
    for agent in range(2):
        for sample in range(5):
            expected = reference_cartpole_q(
                model.theta[agent].detach().numpy(),
                model.input_weights[agent].detach().numpy(),
                model.output_weights[agent].detach().numpy(),
                observations[agent, sample].numpy(),
                reuploading,
                activation_fn,
            )
            np.testing.assert_allclose(q_values[agent, sample], expected, atol=1e-10)


def test_frozenlake_basis_encoding_and_range():
    cfg = load_config("e1_frozenlake_l5_original")
    model = PQCQFunction(cfg, [0], n_actions=4, real_dtype=torch.float64)
    states = torch.arange(16, dtype=torch.float64).view(1, 16, 1)
    q_values = model(states)
    assert q_values.shape == (1, 16, 4)
    assert float(q_values.min()) >= 0.0 and float(q_values.max()) <= 1.0
    # With all angles zero the circuit is CZ only, so <Z_i> is the sign of bit i.
    with torch.no_grad():
        model.theta.zero_()
    q_values = model(states)[0]
    assert torch.allclose(
        q_values[1], torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    )
    assert torch.allclose(
        q_values[8], torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "layers, expected", [(5, 62), (10, 122), (15, 182), (20, 242), (25, 302), (30, 362)]
)
def test_pqc_parameter_counts_match_fig6(layers, expected):
    cfg = load_config(f"e4_cartpole_pqc_l{layers}_original")
    model = PQCQFunction(cfg, [0], n_actions=2, real_dtype=torch.float32)
    assert model.count_trainable_parameters() == expected


def test_ablation_parameter_counts():
    no_input = load_config("e2e3_cartpole_optimal_no_input_weights_original")
    assert (
        PQCQFunction(no_input, [0], 2, torch.float32).count_trainable_parameters() == 42
    )
    no_reupload = load_config("e2e3_cartpole_optimal_no_reuploading_original")
    assert (
        PQCQFunction(no_reupload, [0], 2, torch.float32).count_trainable_parameters()
        == 46
    )
    fixed = load_config("e2e3_cartpole_optimal_fixed_range_90_original")
    assert PQCQFunction(fixed, [0], 2, torch.float32).count_trainable_parameters() == 60
    # Reference-code Frozen Lake ansatz: 2 n (L + 1) angles (final rotation layer).
    frozen = load_config("e1_frozenlake_l5_original")
    assert (
        PQCQFunction(frozen, [0], 4, torch.float32).count_trainable_parameters() == 48
    )
    ring_ansatz = load_config("e1_frozenlake_l5_ringansatz_variant")
    assert (
        PQCQFunction(ring_ansatz, [0], 4, torch.float32).count_trainable_parameters()
        == 40
    )


def test_ring_ansatz_last_layer_cannot_change_z_readout():
    """Why Fig. 1's ring fails on Frozen Lake: the last RZ layer commutes with the Z readout."""
    cfg = load_config("e1_frozenlake_l5_ringansatz_variant")
    model = PQCQFunction(cfg, [0], 4, torch.float64)
    states = torch.arange(16, dtype=torch.float64).view(1, 16, 1)
    before = model(states)
    with torch.no_grad():
        model.theta[:, -1, :, 1] += 1.234  # RZ angles of the last layer
    assert torch.allclose(before, model(states))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("e4_cartpole_mlp_10_10_classical", 182),
        ("e4_cartpole_mlp_15_15_classical", 347),
        ("e4_cartpole_mlp_20_20_classical", 562),
        ("e4_cartpole_mlp_24_24_classical", 770),
        ("e4_cartpole_mlp_30_30_classical", 1142),
        ("e4_cartpole_mlp_64_64_classical", 4610),
        ("e4_cartpole_mlp_5_5_classical", 67),
        ("e4_cartpole_mlp_14_14_classical", 310),
        ("e1_frozenlake_mlp_classical", 49),
    ],
)
def test_mlp_parameter_counts(name, expected):
    cfg = load_config(name)
    n_actions = 4 if "frozenlake" in name else 2
    assert (
        MLPQFunction(cfg, [0], n_actions, torch.float32).count_trainable_parameters()
        == expected
    )


def test_fixed_output_range_bounds():
    cfg = load_config("e2e3_cartpole_optimal_fixed_range_90_original")
    model = PQCQFunction(cfg, [0, 1], 2, torch.float64)
    q_values = model(torch.randn(2, 64, 4, dtype=torch.float64))
    assert float(q_values.min()) >= 0.0 and float(q_values.max()) <= 90.0
    assert not model.output_weights.requires_grad


def test_agents_initialise_from_their_own_seed():
    cfg = deepcopy(load_config("e2e3_cartpole_optimal_full_original"))
    alone = PQCQFunction(cfg, [7], 2, torch.float32)
    together = PQCQFunction(cfg, [1, 7], 2, torch.float32)
    assert torch.equal(alone.theta[0], together.theta[1])
    assert float(alone.theta.min()) >= 0.0 and float(alone.theta.max()) <= math.pi


def test_frozenlake_init_reproduces_paper_initial_mae():
    """Near-zero init + 44 non-terminal pairs gives Fig. 3b's starting MAE of ~0.54."""
    from lib.reference import optimal_q_values, terminal_states

    cfg = load_config("e1_frozenlake_l5_original")
    assert cfg["model"]["theta_init"] == {"low": -0.05, "high": 0.05}
    model = PQCQFunction(cfg, list(range(40)), 4, torch.float64)
    q_values = model(
        torch.arange(16, dtype=torch.float64).view(1, 16, 1).expand(40, 16, 1)
    )
    nonterminal = [state for state in range(16) if state not in terminal_states()]
    errors = (q_values - torch.tensor(optimal_q_values(0.8)))[:, nonterminal].abs()
    assert float(errors.mean()) == pytest.approx(0.54, abs=0.02)
