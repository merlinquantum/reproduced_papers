"""Multi-step prediction (E14): path windows and the two context-preserving circuits."""

from __future__ import annotations

import numpy as np
import pytest
from lib.circuits import (
    apply_context_controlled,
    apply_context_entangling,
    apply_context_sequential,
    n_context_entangling_params,
    n_context_sequential_params,
)
from lib.data import joint_distribution, outcome_windows
from lib.metrics import path_step_metrics, sequence_metrics
from lib.model import ContextualTask
from lib.statevector import basis_state

CONTEXT_LENGTH = 3


def _random_joint(horizon, seed=0):
    rng = np.random.default_rng(seed)
    joint = rng.random((2**CONTEXT_LENGTH, 2**horizon))
    return joint / joint.sum()


def test_outcome_windows_encode_the_path_earliest_step_first():
    symbols = np.array([1, 0, 1, 1, 0, 1])
    windows = outcome_windows(symbols, context_length=3, horizon=2, levels=2)
    np.testing.assert_array_equal(windows, [[1, 0, 1, 2], [0, 1, 1, 1]])
    np.testing.assert_array_equal(outcome_windows(symbols, 3, 1, 2)[:, -1], symbols[3:])
    joint = joint_distribution(windows, 2, 4)
    assert joint.shape == (8, 4)
    assert joint[0b101, 0b10] == pytest.approx(1 / 2)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_parameter_counts(horizon):
    assert n_context_sequential_params(3, horizon) == sum(4 + k for k in range(horizon))
    assert n_context_entangling_params(3, horizon, 2) == 2 * horizon * 4


def test_sequential_circuit_with_one_step_is_the_order_one_circuit():
    rng = np.random.default_rng(1)
    params = rng.uniform(0, 2 * np.pi, 4)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    np.testing.assert_allclose(
        apply_context_sequential(state, params, 3, 1),
        apply_context_controlled(state, params, 3, 1),
        atol=1e-12,
    )


@pytest.mark.parametrize("horizon", [2, 3])
def test_sequential_circuit_is_the_chain_rule(horizon):
    rng = np.random.default_rng(2)
    params = rng.uniform(0, np.pi, n_context_sequential_params(3, horizon))
    for context in (0, 5, 7):
        bits = [(context >> (2 - q)) & 1 for q in range(3)]
        state = basis_state(context * 2**horizon, 3 + horizon)
        probs = np.abs(apply_context_sequential(state, params, 3, horizon)) ** 2
        probs = probs.reshape(8, 2**horizon)
        assert probs[context].sum() == pytest.approx(1.0)
        for path in range(2**horizon):
            path_bits = [(path >> (horizon - 1 - k)) & 1 for k in range(horizon)]
            expected, offset = 1.0, 0
            for step in range(horizon):
                angle = (
                    params[offset]
                    + np.dot(params[offset + 1 : offset + 4], bits)
                    + np.dot(params[offset + 4 : offset + 4 + step], path_bits[:step])
                )
                offset += 4 + step
                p_up = np.sin(angle / 2) ** 2
                expected *= p_up if path_bits[step] else 1 - p_up
            assert probs[context, path] == pytest.approx(expected)


@pytest.mark.parametrize(
    "circuit",
    [
        lambda state, rng: apply_context_sequential(
            state, rng.uniform(0, 2 * np.pi, n_context_sequential_params(3, 2)), 3, 2
        ),
        lambda state, rng: apply_context_entangling(
            state,
            rng.uniform(0, 2 * np.pi, n_context_entangling_params(3, 2, 2)),
            3,
            2,
            2,
        ),
    ],
)
def test_both_circuits_are_unitary_and_keep_the_context(circuit):
    rng = np.random.default_rng(3)
    state = rng.normal(size=32) + 1j * rng.normal(size=32)
    state /= np.linalg.norm(state)
    output = circuit(state, np.random.default_rng(4))
    assert np.linalg.norm(output) == pytest.approx(1.0)
    # The weight on every context is unchanged: the context register is only a control.
    np.testing.assert_allclose(
        (np.abs(output.reshape(8, 4)) ** 2).sum(axis=1),
        (np.abs(state.reshape(8, 4)) ** 2).sum(axis=1),
        atol=1e-12,
    )


def test_entangling_circuit_produces_negative_amplitudes_and_sequential_does_not():
    rng = np.random.default_rng(5)
    state = basis_state(0, 5)
    sequential = apply_context_sequential(
        state, rng.uniform(0, np.pi, n_context_sequential_params(3, 2)), 3, 2
    )
    assert sequential.real.min() >= -1e-12
    entangling = apply_context_entangling(
        state, rng.uniform(0, np.pi, n_context_entangling_params(3, 2, 2)), 3, 2, 2
    )
    assert np.allclose(entangling.imag, 0)
    assert entangling.real.min() < -1e-6


def test_contextual_task_handles_a_two_qubit_prediction_register():
    joint = _random_joint(2)
    task = ContextualTask(joint)
    assert (task.n_qubits, task.levels, task.n_context_qubits) == (5, 4, 3)
    params = np.random.default_rng(6).uniform(0, np.pi, 9)
    conditional = task.inference_conditional(
        lambda state: apply_context_sequential(state, params, 3, 2)
    )
    assert conditional.shape == (8, 4)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0)


def test_path_step_metrics_match_single_step_metrics_on_the_marginal():
    rng = np.random.default_rng(7)
    symbols = rng.integers(0, 2, size=400)
    windows = outcome_windows(symbols, 3, 2, 2)
    conditional = rng.random((8, 4))
    conditional /= conditional.sum(axis=1, keepdims=True)
    steps = path_step_metrics(windows, 2, conditional)
    first_step_table = np.column_stack(
        [conditional[:, :2].sum(axis=1), conditional[:, 2:].sum(axis=1)]
    )
    first_step_windows = np.column_stack([windows[:, :3], windows[:, 3] >> 1])
    reference = sequence_metrics(first_step_windows, 2, first_step_table)
    assert steps["step1_log_loss"] == pytest.approx(reference["log_loss"])
    assert steps["step1_accuracy"] == pytest.approx(reference["accuracy"])
