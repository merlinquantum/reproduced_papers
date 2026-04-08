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

from papers.nn_embedding.utils.utils import (
    calculate_distance,
    LinearLoss,
    create_random_pairs,
    pick_random_data,
    loss_lower_bound,
    get_error_bound,
    random_unitary_gate_based,
    haar_integral_gate_based,
    random_state_photonics,
    haar_integral_photonics,
    kron,
    kernel_variance,
    state_vector_to_density_matrix,
    randomize_trainable_parameters,
    TransparentModel,
    create_param_ensemble,
)
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


# ──────────────────────────────────────────────────────────────────────
# Tests for previously untested utils.py functions
# ──────────────────────────────────────────────────────────────────────


class TestCreateRandomPairs:
    def test_output_shapes(self):
        X = [torch.randn(4) for _ in range(20)]
        Y = [0] * 10 + [1] * 10
        X1, X2, Y_new = create_random_pairs(8, X, Y)
        assert X1.shape == (8, 4)
        assert X2.shape == (8, 4)
        assert Y_new.shape == (8,)

    def test_label_logic(self):
        X = [torch.tensor([float(i)]) for i in range(4)]
        Y = [0, 0, 1, 1]
        np.random.seed(42)
        X1, X2, Y_new = create_random_pairs(200, X, Y)
        for i in range(200):
            idx1 = int(X1[i].item())
            idx2 = int(X2[i].item())
            same_class = Y[idx1] == Y[idx2]
            assert Y_new[i].item() == (1 if same_class else 0)


class TestPickRandomData:
    def test_output_shapes(self):
        X = [torch.randn(3) for _ in range(50)]
        Y = list(range(50))
        X_batch, Y_batch = pick_random_data(10, X, Y)
        assert X_batch.shape == (10, 3)
        assert Y_batch.shape == (10,)


class TestLossLowerBound:
    def test_identical_distributions(self):
        rho = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(5, -1, -1)
        bound = loss_lower_bound(rho, rho)
        assert bound == pytest.approx(0.5, abs=1e-6)


class TestGetErrorBound:
    def test_known_kernel(self):
        N = 10
        K = np.eye(N, dtype=np.float64)
        Y = np.ones(N, dtype=np.float64)
        weights = np.array([0.1, 0.5, 1.0])
        errors = get_error_bound(weights, K, Y)
        assert errors.shape == (3,)
        assert np.all(errors >= 0)

    def test_psd_projection(self):
        np.random.seed(0)
        N = 5
        # Build a symmetric matrix that has at least one negative eigenvalue
        A = np.random.randn(N, N)
        K = (A + A.T) / 2
        # Confirm it is NOT positive semi-definite before projection
        assert np.min(np.linalg.eigvalsh(K)) < 0
        Y = np.array([1, -1, 1, -1, 1], dtype=np.float64)
        weights = np.array([0.5])
        errors = get_error_bound(weights, K, Y)
        assert np.all(np.isfinite(errors))
        assert np.all(errors >= 0)


class TestRandomUnitaryGateBased:
    def test_unitarity(self):
        np.random.seed(0)
        U = random_unitary_gate_based(2)
        assert U.shape == (4, 4)
        product = U @ U.conj().T
        assert np.allclose(product, np.eye(4), atol=1e-10)

    def test_determinant(self):
        np.random.seed(1)
        U = random_unitary_gate_based(1)
        assert abs(abs(np.linalg.det(U)) - 1.0) < 1e-10


class TestHaarIntegralGateBased:
    def test_shape(self):
        np.random.seed(0)
        result = haar_integral_gate_based(1, 50)
        assert result.shape == (4, 4)

    def test_trace_one(self):
        np.random.seed(0)
        result = haar_integral_gate_based(1, 200)
        assert abs(np.trace(result) - 1.0) < 0.01


class TestRandomStatePhotonics:
    def test_density_matrix_properties(self):
        np.random.seed(0)
        rho = random_state_photonics(3)
        assert rho.shape == (3, 3)
        assert abs(np.trace(rho) - 1.0) < 1e-10


class TestHaarIntegralPhotonics:
    def test_shape(self):
        np.random.seed(0)
        result = haar_integral_photonics(2, 50)
        assert result.shape == (4, 4)


class TestKron:
    def test_matches_numpy(self):
        a = torch.randn(2, 3)
        b = torch.randn(4, 5)
        result = kron(a, b)
        expected = torch.from_numpy(np.kron(a.numpy(), b.numpy()))
        assert torch.allclose(result, expected.float(), atol=1e-5)

    def test_batched(self):
        a = torch.randn(7, 2, 2)
        b = torch.randn(7, 3, 3)
        result = kron(a, b)
        assert result.shape == (7, 6, 6)
        expected_0 = torch.from_numpy(np.kron(a[0].numpy(), b[0].numpy()))
        assert torch.allclose(result[0], expected_0.float(), atol=1e-5)


class TestKernelVariance:
    def test_zero_variance_uniform(self):
        K = torch.ones(4, 4) * 0.5
        var = kernel_variance(K)
        assert var == pytest.approx(0.0, abs=1e-10)

    def test_positive_variance(self):
        K = torch.tensor([[1.0, 0.3, 0.7], [0.3, 1.0, 0.5], [0.7, 0.5, 1.0]])
        var = kernel_variance(K)
        assert var > 0


class TestStateVectorToDensityMatrix:
    def test_torch_batched(self):
        psi = torch.tensor(
            [[1.0, 0.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]], dtype=float
        )
        rho = state_vector_to_density_matrix(psi)
        assert rho.shape == (2, 2, 2)
        assert torch.allclose(
            rho[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=float)
        )
        assert torch.allclose(
            rho[1], torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=float)
        )


class TestRandomizeTrainableParameters:
    def test_changes_single_layer(self):
        model = torch.nn.Linear(4, 2)
        params_before = [p.clone() for p in model.parameters()]
        randomize_trainable_parameters(model)
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, model.parameters())
        )
        assert changed

    def test_changes_sequential(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        params_before = [p.clone() for p in model.parameters()]
        randomize_trainable_parameters(model)
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, model.parameters())
        )
        assert changed


class TestTransparentModel:
    def test_identity(self):
        model = TransparentModel()
        x = torch.randn(5, 3)
        out = model(x)
        assert torch.equal(out, x)


class TestCreateParamEnsemble:
    def test_shapes(self):
        params = [torch.randn(3, 2), torch.randn(4)]
        d = 10
        ensemble = create_param_ensemble(params, d, epsilon=0.5, num_samples=10)
        assert len(ensemble) == 10
        assert ensemble[0][0].shape == (3, 2)
        assert ensemble[0][1].shape == (4,)

    def test_within_ball(self):
        center = torch.zeros(5)
        params = [center]
        eps = 1.0
        ensemble = create_param_ensemble(params, 5, epsilon=eps, num_samples=50)
        for point in ensemble:
            dist = torch.norm(point[0]).item()
            assert dist <= eps + 1e-6
