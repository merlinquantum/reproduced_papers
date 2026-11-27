from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repository root on sys.path)
from lib.circuits import apply_ansatz, apply_share_and_specify, n_params
from lib.model import ContextualTask, per_sample_fidelity
from lib.optim import parameter_shift_gradient
from lib.statevector import (
    apply_1q,
    apply_cnot,
    apply_cswap,
    basis_state,
    probabilities,
    ry,
    rz,
)


def random_joint(rng, n_contexts=8, levels=2):
    joint = rng.random((n_contexts, levels))
    return joint / joint.sum()


def test_cnot_flips_target_only_when_control_set():
    state = basis_state(0b100, 3)  # qubit 0 = 1
    assert np.argmax(np.abs(apply_cnot(state, 0, 2, 3))) == 0b101
    state = basis_state(0b001, 3)  # control qubit 0 = 0
    assert np.argmax(np.abs(apply_cnot(state, 0, 2, 3))) == 0b001
    state = basis_state(0b001, 3)  # control after target
    assert np.argmax(np.abs(apply_cnot(state, 2, 0, 3))) == 0b101


def test_ry_on_qubit_zero_is_most_significant():
    state = apply_1q(basis_state(0, 2), ry(np.pi), 0, 2)
    assert np.isclose(abs(state[0b10]), 1.0)


def test_analytic_swap_test_matches_cswap_circuit():
    """P(ancilla = 0) of the explicit SWAP-test circuit equals (1 + F) / 2."""

    rng = np.random.default_rng(0)
    task = ContextualTask(random_joint(rng))
    params = rng.uniform(0, 2 * np.pi, n_params(4, 2))
    output = apply_ansatz(task.input_state, params, 4)

    n = 9  # ancilla + 4 output qubits + 4 target qubits
    state = np.kron(basis_state(0, 1), np.kron(output, task.target_state))
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    state = apply_1q(state, h, 0, n)
    for qubit in range(4):
        state = apply_cswap(state, 0, 1 + qubit, 5 + qubit, n)
    state = apply_1q(state, h, 0, n)
    p_zero = probabilities(state)[: 2 ** (n - 1)].sum()

    fidelity = task.fidelity(lambda s: apply_ansatz(s, params, 4))
    assert p_zero == pytest.approx(0.5 * (1 + fidelity), abs=1e-12)
    assert task.swap_test_loss(
        lambda s: apply_ansatz(s, params, 4), None, rng
    ) == pytest.approx(1 - p_zero, abs=1e-12)


def test_identity_circuit_reaches_fidelity_one_when_next_is_always_zero():
    joint = np.zeros((8, 2))
    joint[:, 0] = 1 / 8
    task = ContextualTask(joint)
    assert task.fidelity(lambda s: s) == pytest.approx(1.0)
    assert np.allclose(task.inference_conditional(lambda s: s), [[1.0, 0.0]] * 8)


def test_parameter_shift_matches_finite_difference():
    rng = np.random.default_rng(1)
    task = ContextualTask(random_joint(rng))
    params = rng.uniform(0, 2 * np.pi, n_params(4, 2))

    def loss(p):
        return task.swap_test_loss(lambda s: apply_ansatz(s, p, 4), None, rng)

    exact = parameter_shift_gradient(loss, params)
    step = 1e-6
    numeric = np.array(
        [
            (loss(params + step * e) - loss(params - step * e)) / (2 * step)
            for e in np.eye(len(params))
        ]
    )
    assert np.allclose(exact, numeric, atol=1e-7)


def _controlled_1q(gate: np.ndarray, data_qubit: int, label_value: int) -> np.ndarray:
    """32x32 unitary: ``gate`` on a data qubit when the label qubit (last) equals ``label_value``."""

    full = np.eye(1)
    for qubit in range(4):
        full = np.kron(full, gate if qubit == data_qubit else np.eye(2))
    on = np.zeros((2, 2))
    on[label_value, label_value] = 1
    return np.kron(full, on) + np.kron(np.eye(16), np.eye(2) - on)


def test_fig7_asset_blocks_match_share_and_specify_with_fixed_label():
    """Fig. 7 gate by gate: label-controlled RY then RZ per data qubit, no CNOT.

    U_1 acts when the label is 1; an X then flips the label, so U_2 acts when it
    was 0. For a fixed label this equals ``apply_share_and_specify`` with
    ``asset_block="rotations"`` and the matching asset's parameters (Eq. 30).
    """

    rng = np.random.default_rng(2)
    shared = rng.uniform(0, 2 * np.pi, 8)
    specific = {1: rng.uniform(0, 2 * np.pi, 8), 0: rng.uniform(0, 2 * np.pi, 8)}
    data_state = random_joint(rng).reshape(-1) ** 0.5 + 0j

    def asset_block(params):
        unitary = np.eye(32, dtype=np.complex128)
        for qubit in range(4):
            unitary = _controlled_1q(ry(params[qubit]), qubit, 1) @ unitary
        for qubit in range(4):
            unitary = _controlled_1q(rz(params[4 + qubit]), qubit, 1) @ unitary
        return unitary

    flip_label = np.kron(np.eye(16), np.array([[0, 1], [1, 0]]))
    asset_layer = (
        flip_label @ asset_block(specific[0]) @ flip_label @ asset_block(specific[1])
    )

    shared_out = apply_ansatz(data_state, shared, 4)
    for label in (0, 1):
        via_gates = asset_layer @ np.kron(shared_out, basis_state(label, 1))
        direct = apply_share_and_specify(
            data_state, shared, specific[label], 4, "rotations"
        )
        assert np.allclose(via_gates, np.kron(direct, basis_state(label, 1)))


def test_legacy_layer_asset_block_adds_the_ring_cnot():
    rng = np.random.default_rng(3)
    shared, specific = rng.uniform(0, 2 * np.pi, 8), rng.uniform(0, 2 * np.pi, 8)
    state = random_joint(rng).reshape(-1) ** 0.5 + 0j
    legacy = apply_share_and_specify(state, shared, specific, 4, "layer")
    expected = apply_ansatz(apply_ansatz(state, shared, 4), specific, 4)
    assert np.allclose(legacy, expected)
    fig7 = apply_share_and_specify(state, shared, specific, 4, "rotations")
    assert not np.allclose(legacy, fig7)
    with pytest.raises(ValueError):
        apply_share_and_specify(state, shared, specific, 4, "unknown")


def test_per_sample_fidelity_of_identity():
    assert per_sample_fidelity(lambda s: s, 5, 0, 8, 2) == pytest.approx(1.0)
    assert per_sample_fidelity(lambda s: s, 5, 1, 8, 2) == pytest.approx(0.0)
