import sys
from pathlib import Path

import merlin as ml
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.lib.merlin_based_model import (  # noqa: E402
    NeuralEmbeddingMerLinKernel,
    NeuralEmbeddingMerLinModel,
    create_basic_merlin_model,
)


@pytest.fixture
def merlin_model() -> NeuralEmbeddingMerLinModel:
    return create_basic_merlin_model()


@pytest.fixture
def merlin_kernel() -> NeuralEmbeddingMerLinKernel:
    # Quantum embedding
    circ = ml.CircuitBuilder(n_modes=8)
    circ.add_entangling_layer()
    embedder = ml.QuantumLayer(
        input_size=0,
        builder=circ,
        n_photons=4,
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )
    classical_model = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, sum([i.numel() for i in embedder.parameters()])),
    )

    return NeuralEmbeddingMerLinKernel(classical_model, embedder)


def test_create_basic_merlin_model(merlin_model):
    model = merlin_model

    assert isinstance(model, NeuralEmbeddingMerLinModel)
    assert isinstance(model.classical_encoder, nn.Module)
    assert isinstance(model.quantum_embedding_layer, nn.Module)
    assert isinstance(model.quantum_classifier, nn.Module)
    assert isinstance(model.similarity_layer, nn.Module)
    assert isinstance(model.embedding_training_model, nn.Module)
    assert isinstance(model.model, nn.Module)


def test_similarity_layer_returns_probs(merlin_model):
    model = merlin_model

    output_size = sum(
        [i.numel() for i in merlin_model.quantum_embedding_layer.parameters()]
    )

    x_1 = torch.zeros((3, output_size), dtype=torch.float32)
    x_2 = torch.zeros((3, output_size), dtype=torch.float32)

    scores = model.similarity_layer(x_1, x_2)

    assert scores.shape == (3, merlin_model.quantum_embedding_layer.output_size)


def test_embedding_training_model_forward_shape(merlin_model):
    model = merlin_model

    x = torch.zeros((4, 16), dtype=torch.float32)
    outputs = model.embedding_training_model(x)

    assert outputs.shape == (4,)


def test_trained_embedding_model_forward_shape(merlin_model):
    model = merlin_model

    x = torch.zeros((5, 8), dtype=torch.float32)
    outputs = model.model(x)

    assert outputs.shape == (5, 2)


def test_embedding_layer_is_frozen(merlin_model):
    model = merlin_model

    assert all(
        not param.requires_grad for param in model.quantum_embedding_layer.parameters()
    )
    assert all(
        not param.requires_grad
        for param in model.similarity_layer.fidelity_layer.parameters()
    )


def test_similarity_layer_target_index_matches_input_state(merlin_model):
    model = merlin_model

    target_state = tuple(model.similarity_layer.fidelity_layer.input_state)
    output_keys = list(model.similarity_layer.fidelity_layer.output_keys)

    assert output_keys[model.embedding_training_model.target_index] == target_state


def test_kernel_matrix(merlin_kernel):
    kernel = merlin_kernel

    x = torch.randn((5, 8), dtype=torch.float32)

    mat = kernel.compute_kernel_matrix(x)
    for i in range(5):
        for j in range(i + 1, 5):
            assert mat[i, j] == mat[j, i]
    assert mat.shape == (5, 5)
