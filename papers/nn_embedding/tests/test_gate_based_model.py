import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.lib.gate_based_model import (
    NeuralEmbeddingGateBasedModel,
    NeuralEmbeddingGateBasedKernel,
)
from papers.nn_embedding.utils.gate_based_embedding import EmbeddingCallable, QCNN


@pytest.fixture
def gate_based_model() -> NeuralEmbeddingGateBasedModel:
    classical_model = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
    )

    return NeuralEmbeddingGateBasedModel(
        num_qubits=8,
        classical_model=classical_model,
        quantum_embedding_layer=EmbeddingCallable().QuantumEmbedding1,
        quantum_classifier=QCNN,
        quantum_classifier_params_shape=(45,),
    )


@pytest.fixture
def gate_based_kernel() -> NeuralEmbeddingGateBasedKernel:
    classical_model = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
    )

    return NeuralEmbeddingGateBasedKernel(
        num_qubits=8,
        classical_model=classical_model,
        quantum_embedding_layer=EmbeddingCallable().QuantumEmbedding1,
    )


def test_gate_based_model_construction(gate_based_model):
    model = gate_based_model

    assert isinstance(model, NeuralEmbeddingGateBasedModel)
    assert isinstance(model.classical_encoder, nn.Module)
    assert isinstance(model.distance_circuit_layer, nn.Module)
    assert isinstance(model.complete_circuit_layer, nn.Module)
    assert callable(model.state_embedding_circuit)
    assert callable(model.complete_circuit_layer)
    assert callable(model.state_embedding_circuit)
    assert isinstance(model.embedding_training_model, nn.Module)
    assert isinstance(model.model, nn.Module)


def test_distance_layer_returns_one_score_per_pair(gate_based_model):
    model = gate_based_model

    x = torch.zeros((3, 16), dtype=torch.float32)
    scores = model.embedding_training_model(x)

    assert scores.shape == (3,)
    assert torch.allclose(scores, torch.ones(3, dtype=scores.dtype))


def test_embedding_training_model_forward_shape(gate_based_model):
    model = gate_based_model

    x = torch.zeros((4, 16), dtype=torch.float32)
    outputs = model.embedding_training_model(x)

    assert outputs.shape == (4,)


def test_trained_embedding_model_forward_shape(gate_based_model):
    model = gate_based_model

    x = torch.zeros((5, 8), dtype=torch.float32)
    outputs = model.model(x)

    assert outputs.shape == (5, 2)


def test_complete_circuit_layer_has_trainable_classifier_params(gate_based_model):
    model = gate_based_model

    params = list(model.complete_circuit_layer.parameters())

    assert len(params) == 1
    assert params[0].shape == torch.Size([45])


def test_state_embedding_circuit_returns_density_matrix(gate_based_model):
    model = gate_based_model

    x = torch.zeros(8, dtype=torch.float32)
    rho = model.state_embedding_circuit(x)

    assert rho.shape == (2**model.num_qubits, 2**model.num_qubits)
    assert torch.is_complex(rho)


def test_kernel_matrix(gate_based_kernel):
    kernel = gate_based_kernel

    x = torch.randn((5, 8), dtype=torch.float32)

    mat = kernel.compute_kernel_matrix(x)
    for i in range(5):
        for j in range(i + 1, 5):
            assert mat[i, j] == mat[j, i]
    assert mat.shape == (5, 5)
