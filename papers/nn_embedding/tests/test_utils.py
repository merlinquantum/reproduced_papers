import pytest
import torch
import numpy as np
import merlin as ml

import sys
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.utils.utils import calculate_distance, LinearLoss
from papers.nn_embedding.utils.merlin_model_utils import (
    ordered_variable_params,
    rename_params_in_current_order,
    strip_simple_negation_expressions,
    count_parameters_with_prefixes,
    compute_x2_permutation,
    assign_params,
)


@pytest.fixture
def trainable_merlin_layer():
    builder = ml.CircuitBuilder(4)
    builder.add_entangling_layer()
    builder.add_rotations()
    builder.add_entangling_layer()

    return ml.QuantumLayer(
        input_size=0,
        builder=builder,
        input_state=[1, 1, 0, 0],
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )


@pytest.fixture
def fidelity_layer(trainable_merlin_layer):
    quantum_embedding_layer = trainable_merlin_layer
    encoder_1 = deepcopy(quantum_embedding_layer)
    encoder_2 = deepcopy(quantum_embedding_layer)

    rename_params_in_current_order(encoder_1.circuit, "x_1_")
    rename_params_in_current_order(encoder_2.circuit, "x_2_")

    encoder_2.circuit.inverse(h=True)
    strip_simple_negation_expressions(encoder_2.circuit)

    fidelity_circuit = encoder_1.circuit.add(0, encoder_2.circuit, merge=False)
    input_prefixes = ["x_1_", "x_2_"]
    input_size = count_parameters_with_prefixes(fidelity_circuit, input_prefixes)

    fidelity_layer = ml.QuantumLayer(
        input_size=input_size,
        circuit=fidelity_circuit,
        input_state=encoder_1.input_state,
        n_photons=encoder_1.n_photons,
        amplitude_encoding=False,
        computation_space=encoder_1.computation_space,
        measurement_strategy=encoder_1.measurement_strategy,
        device=encoder_1.device,
        dtype=encoder_1.dtype,
        input_parameters=input_prefixes,
    )
    return (
        quantum_embedding_layer,
        encoder_1,
        encoder_2,
        fidelity_circuit,
        fidelity_layer,
    )


def _component_params(component):
    params = []
    if hasattr(component, "_params"):
        params.extend(component._params.values())
    if hasattr(component, "_components"):
        for _, subcomponent in component._components:
            params.extend(_component_params(subcomponent))
    return params


def test_calculate_distance():

    plus_state = torch.tensor(0.5 * np.array([[1, 1], [1, 1]]))
    minus_state = torch.tensor(0.5 * np.array([[1, -1], [-1, 1]]))
    zero_state = torch.tensor(np.array([[1, 0], [0, 0]]))

    # Trace distance
    assert calculate_distance(plus_state, minus_state) == 1
    assert calculate_distance(plus_state, zero_state) == 1 / np.sqrt(2)
    assert calculate_distance(plus_state, plus_state) == 0

    # Hs distance
    assert calculate_distance(plus_state, minus_state, distance="Hilbert-Schmidt") == 1
    assert calculate_distance(plus_state, zero_state, distance="Hilbert-Schmidt") == 0.5
    assert calculate_distance(plus_state, plus_state, distance="Hilbert-Schmidt") == 0


def test_LinearLoss():
    criterion = LinearLoss()

    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    predictions = torch.tensor([0.2, 0.7, 0.1], dtype=torch.float32)

    expected = torch.tensor(
        [
            0.2,  # y=0 -> loss = p
            0.3,  # y=1 -> loss = 1-p
            0.9,  # y=1 -> loss = 1-p
        ],
        dtype=torch.float32,
    ).mean()

    loss = criterion(labels, predictions)

    assert torch.isclose(loss, expected)


def test_ordered_variable_params(trainable_merlin_layer):
    layer = trainable_merlin_layer
    params = ordered_variable_params(layer.circuit)

    assert len(params) > 0
    assert all(not param.fixed for param in params)
    assert len({id(param) for param in params}) == len(params)


def test_rename_params_in_current_order(trainable_merlin_layer):
    layer = trainable_merlin_layer

    rename_params_in_current_order(layer.circuit, "x_1_")
    params = ordered_variable_params(layer.circuit)

    assert all(param.name.startswith("x_1_") for param in params)
    assert all(name.startswith("x_1_") for name in layer.circuit._params.keys())


def test_strip_simple_negation_expressions(trainable_merlin_layer):
    layer = trainable_merlin_layer
    inverse_layer = deepcopy(layer)
    rename_params_in_current_order(inverse_layer.circuit, "x_2_")
    inverse_layer.circuit.inverse(h=True)

    params_before = _component_params(inverse_layer.circuit)
    assert any(getattr(param, "_is_expression", False) for param in params_before)

    strip_simple_negation_expressions(inverse_layer.circuit)

    params_after = _component_params(inverse_layer.circuit)
    assert not any(
        getattr(param, "_is_expression", False) and param.name.startswith("(-x_2_")
        for param in params_after
    )


def test_count_parameters_with_prefixes(trainable_merlin_layer):
    layer = trainable_merlin_layer
    rename_params_in_current_order(layer.circuit, "x_1_")

    count = count_parameters_with_prefixes(layer.circuit, ["x_1_"])
    params = ordered_variable_params(layer.circuit)

    assert count == len(params)


def test_compute_x2_permutation(fidelity_layer):
    _, _, _, _, fidelity_layer = fidelity_layer

    perm_x2 = compute_x2_permutation(fidelity_layer)
    spec = fidelity_layer.computation_process.converter.spec_mappings

    x1_names = spec["x_1_"]
    x2_names = spec["x_2_"]

    reordered_x2 = [x2_names[i] for i in perm_x2]
    x1_suffixes = [name.removeprefix("x_1_") for name in x1_names]
    x2_suffixes = [name.removeprefix("x_2_") for name in reordered_x2]

    assert x1_suffixes == x2_suffixes


def test_assign_params(trainable_merlin_layer):
    layer = trainable_merlin_layer

    flat_size = 0
    for param in trainable_merlin_layer.parameters():
        param.requires_grad = False
        flat_size += param.numel()

    values = torch.linspace(0.1, 1.0, steps=flat_size, dtype=torch.float32)
    output = assign_params(layer, values)

    assigned = torch.cat([param.detach().reshape(-1) for param in layer.parameters()])
    assert torch.allclose(assigned, values)
    assert output.shape[-1] == layer.output_size

    batched_values = torch.stack([values, values + 1.0], dim=0)
    batched_output = assign_params(layer, batched_values)

    assigned_after_batch = torch.cat(
        [param.detach().reshape(-1) for param in layer.parameters()]
    )
    assert torch.allclose(assigned_after_batch, batched_values[-1])
    assert batched_output.shape == (2, layer.output_size)
