"""Context-preserving ansatz (E11): structure and readout equivalence."""

import common  # noqa: F401
import numpy as np
import pytest
from lib.circuits import (
    apply_context_controlled,
    context_subsets,
    n_context_controlled_params,
)
from lib.metrics import normalize_rows
from lib.model import ContextualTask
from lib.statevector import basis_state, probabilities
from lib.training import qstl_circuit_factory


def _random_joint(rng):
    joint = rng.uniform(0.2, 1.0, size=(8, 2))
    return joint / joint.sum()


@pytest.mark.parametrize("order, expected", [(0, 1), (1, 4), (2, 7), (3, 8)])
def test_parameter_count(order, expected):
    assert n_context_controlled_params(3, order) == expected
    assert len(context_subsets(3, order)) == expected


@pytest.mark.parametrize("order", [1, 2, 3])
def test_context_register_is_kept_and_angles_add(order):
    rng = np.random.default_rng(0)
    subsets = context_subsets(3, order)
    params = rng.uniform(0, 2 * np.pi, size=len(subsets))
    for context in range(8):
        output = apply_context_controlled(basis_state(2 * context, 4), params, 3, order)
        joint = probabilities(output).reshape(8, 2)
        assert joint[context].sum() == pytest.approx(1.0)
        bits = [(context >> (2 - q)) & 1 for q in range(3)]
        angle = sum(t for t, s in zip(params, subsets) if all(bits[q] for q in s))
        assert joint[context, 1] == pytest.approx(np.sin(angle / 2) ** 2)


def test_inference_equals_joint_readout_and_hellinger_fidelity():
    rng = np.random.default_rng(1)
    joint = _random_joint(rng)
    task = ContextualTask(joint)
    factory, count = qstl_circuit_factory(
        {"ansatz": "context_controlled", "interaction_order": 2},
        task.n_qubits,
        task.levels,
    )
    circuit = factory(rng.uniform(0, np.pi, size=count))
    inference = task.inference_conditional(circuit)
    np.testing.assert_allclose(
        inference, normalize_rows(task.output_joint(circuit)), atol=1e-12
    )
    # With the context kept, the QBGU overlap is the context-weighted sum of
    # sqrt(p(o|c)) times the signed model amplitude (LOG.md, Comparison Protocol).
    context = joint.sum(axis=1)
    conditional = joint / context[:, None]
    output = circuit(task.input_state).reshape(8, 2).real
    amplitudes = output / np.sqrt(context)[:, None]
    overlap = np.sum(context[:, None] * np.sqrt(conditional) * amplitudes)
    assert task.fidelity(circuit) == pytest.approx(overlap**2)


def test_full_order_represents_any_conditional():
    rng = np.random.default_rng(2)
    joint = _random_joint(rng)
    conditional = joint / joint.sum(axis=1, keepdims=True)
    target_angles = 2 * np.arcsin(np.sqrt(conditional[:, 1]))
    subsets = context_subsets(3, 3)
    # Solve theta from a_c = sum of theta_S over subsets contained in c's set bits.
    design = np.array(
        [[float(all((c >> (2 - q)) & 1 for q in s)) for s in subsets] for c in range(8)]
    )
    params = np.linalg.solve(design, target_angles)
    task = ContextualTask(joint)
    fidelity = task.fidelity(lambda s: apply_context_controlled(s, params, 3, 3))
    assert fidelity == pytest.approx(1.0)


def test_non_binary_levels_rejected():
    with pytest.raises(ValueError):
        qstl_circuit_factory(
            {"ansatz": "context_controlled", "interaction_order": 1}, 6, 4
        )
