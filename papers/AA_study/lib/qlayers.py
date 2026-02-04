import merlin as ml
import perceval as pcvl
import numpy as np
import torch.nn as nn
from math import comb

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.photonic_QCNN.lib.src.merlin_pqcnn import HybridModel as PhotonicQCNN
from papers.AA_study.utils.utils import find_mode_photon_config


"""
Functions used to use photonics to check the final state convergence with a qubit vision
"""


# def angle_encoding_output_state_dual_rail(
#     num_features: int,
#     num_layers: int = 3,
#     measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
# ) -> ml.QuantumLayer:
#     input_state = [0] * num_features * 2
#     for i in range(num_features):
#         input_state[(i * 2) + 1] = 1
#     circuit = ml.CircuitBuilder(n_modes=num_features * 2)
#     circuit.add_angle_encoding(modes=[(2 * i) + 1 for i in range(num_features)])
#     return ml.QuantumLayer(
#         input_size=num_features,  # Follow the convention?
#         builder=circuit,
#         input_state=input_state,
#         n_photons=num_features,
#         measurement_strategy=measurement_strategy,
#         computation_space="dual_rail",
#     )


# def amplitude_encoding_output_state_dual_rail(
#     num_features: int,
#     num_layers: int = 3,
#     measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
# ) -> ml.QuantumLayer:
#     num_photons = int(np.ceil(np.log2(num_features)))
#     circuit = ml.CircuitBuilder(n_modes=2 * num_photons)
#     for _ in range(num_layers):
#         circuit.add_entangling_layer()
#     return ml.QuantumLayer(
#         builder=circuit,
#         amplitude_encoding=True,
#         n_photons=num_photons,
#         measurement_strategy=measurement_strategy,
#         computation_space="dual_rail",
#     )


"""Simple encoding layers"""


def angle_encoding_simple(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.PROBABILITIES,
    num_classes: int = 2,
) -> ml.QuantumLayer:
    input_state = [0] * num_features
    for i in range(num_features // 2):
        input_state[(i * 2) + 1] = 1
    circuit = ml.CircuitBuilder(n_modes=num_features)
    if num_features == 1:
        circuit.add_rotations(trainable=True)
    else:
        circuit.add_entangling_layer()
    circuit.add_angle_encoding()
    for _ in range(num_layers):
        if num_features == 1:
            circuit.add_rotations(trainable=True)
        else:
            circuit.add_entangling_layer()
    qlayer = ml.QuantumLayer(
        input_size=num_features,  # Follow the convention?
        builder=circuit,
        input_state=input_state,
        n_photons=num_features // 2,
        measurement_strategy=measurement_strategy,
    )
    return nn.Sequential(qlayer, ml.LexGrouping(qlayer.output_size, num_classes))


def amplitude_encoding_simple(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.PROBABILITIES,
    num_classes: int = 2,
) -> ml.QuantumLayer:
    n_modes, n_photons = find_mode_photon_config(num_features=num_features)
    circuit = ml.CircuitBuilder(n_modes=n_modes)
    for _ in range(num_layers):
        if n_modes == 1:
            circuit.add_rotations(trainable=True)
        else:
            circuit.add_entangling_layer()
    qlayer = ml.QuantumLayer(
        builder=circuit,
        amplitude_encoding=True,
        n_photons=n_photons,
        measurement_strategy=measurement_strategy,
    )
    return nn.Sequential(qlayer, ml.LexGrouping(qlayer.output_size, num_classes))


# class amplitude_encoding_simple(nn.Module):
#     def __init__(
#         self,
#         num_features: int,
#         num_layers: int = 3,
#         measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.PROBABILITIES,
#         num_classes: int = 2,
#     ):
#         super().__init__()

#         n_modes = int(np.ceil(np.log2(num_features)))
#         circuit = ml.CircuitBuilder(n_modes=n_modes)
#         for _ in range(num_layers):
#             if n_modes == 1:
#                 circuit.add_rotations(trainable=True)
#             else:
#                 circuit.add_entangling_layer()
#         qlayer = ml.QuantumLayer(
#             builder=circuit,
#             amplitude_encoding=True,
#             n_photons=n_modes // 2,
#             measurement_strategy=measurement_strategy,
#         )
#         self.model = nn.Sequential(
#             qlayer, ml.LexGrouping(qlayer.output_size, num_classes)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if len(x.shape) == 1:
#             return self.model(x / np.linalg.norm(x))
#         norm = torch.linalg.norm(x, dim=1, keepdim=True)
#         x = x / (norm + 1e-12)
#         return self.model(x)
