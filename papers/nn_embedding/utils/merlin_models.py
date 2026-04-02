import merlin as ml
import torch.nn as nn
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.utils.utils import (
    randomize_trainable_parameters,
)


def create_merlin_fig_2_models():
    # Quantum embedding
    circ = ml.CircuitBuilder(n_modes=6)
    circ.add_entangling_layer()
    embedder = ml.QuantumLayer(
        input_size=0,
        builder=circ,
        n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )
    randomize_trainable_parameters(embedder)

    # Quantum classifier
    circ = ml.CircuitBuilder(n_modes=6)
    circ.add_entangling_layer()
    classifier = ml.QuantumLayer(
        builder=circ,
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
    )
    randomize_trainable_parameters(classifier)

    # PCA 8
    classical_model_8 = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, sum([i.numel() for i in embedder.parameters()])),
    )
    randomize_trainable_parameters(classical_model_8)

    # Full classical_model
    classical_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, sum([i.numel() for i in embedder.parameters()])),
    )
    randomize_trainable_parameters(classical_model)

    return embedder, classifier, classical_model_8, classical_model


def create_merlin_fig_3_models():
    # Quantum embedding
    circ = ml.CircuitBuilder(n_modes=10)
    circ.add_entangling_layer()
    embedder = ml.QuantumLayer(
        input_size=0,
        builder=circ,
        n_photons=5,
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )
    randomize_trainable_parameters(embedder)

    # Quantum classifier
    circ = ml.CircuitBuilder(n_modes=10)
    circ.add_entangling_layer()
    classifier = ml.QuantumLayer(
        builder=circ,
        n_photons=5,
        amplitude_encoding=True,
        measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
    )
    randomize_trainable_parameters(classifier)

    # PCA 8
    classical_model_8 = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, sum([i.numel() for i in embedder.parameters()])),
    )
    randomize_trainable_parameters(classical_model_8)

    # Full classical_model
    classical_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, sum([i.numel() for i in embedder.parameters()])),
    )
    randomize_trainable_parameters(classical_model)

    return embedder, classifier, classical_model_8, classical_model


def create_trainable_merlin_layer_fig_3(N_layer: int):
    circuit = ml.CircuitBuilder(n_modes=10)
    for _ in range(N_layer):
        circuit.add_entangling_layer()
        circuit.add_angle_encoding(modes=list(range(8)))
        circuit.add_entangling_layer()
    embedder = ml.QuantumLayer(
        input_size=8 * N_layer,
        builder=circuit,
        n_photons=5,
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )

    class BasicModelRepeatedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = embedder
            for param in self.embedder.parameters():
                param.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.reshape(x.size(0), -1)
            x = x.repeat((1, N_layer))

            return self.embedder(x)

    model = BasicModelRepeatedModel()

    randomize_trainable_parameters(model)

    return model


def create_merlin_fig_4_models():
    # Quantum embedding
    circ = ml.CircuitBuilder(n_modes=6)
    circ.add_entangling_layer()
    embedder = ml.QuantumLayer(
        input_size=0,
        builder=circ,
        n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )
    randomize_trainable_parameters(embedder)

    # Quantum classifier
    circ = ml.CircuitBuilder(n_modes=6)
    circ.add_entangling_layer()
    classifier = ml.QuantumLayer(
        builder=circ,
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
    )
    randomize_trainable_parameters(classifier)

    dim = sum([i.numel() for i in embedder.parameters()])

    # PCA 8
    classical_model = nn.Sequential(
        nn.Linear(dim, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, dim),
    )
    randomize_trainable_parameters(classical_model)

    return embedder, classifier, classical_model, dim


def create_merlin_fig_5_models():
    # Quantum embedding
    circ = ml.CircuitBuilder(n_modes=6)
    circ.add_entangling_layer()
    embedder = ml.QuantumLayer(
        input_size=0,
        builder=circ,
        n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )
    randomize_trainable_parameters(embedder)

    # PCA 8
    classical_model_4 = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, sum([i.numel() for i in embedder.parameters()])),
    )
    randomize_trainable_parameters(classical_model_4)

    # Full classical_model
    classical_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, sum([i.numel() for i in embedder.parameters()])),
    )
    randomize_trainable_parameters(classical_model)

    return (
        embedder,
        classical_model_4,
        classical_model,
        sum([i.numel() for i in embedder.parameters()]),
    )
