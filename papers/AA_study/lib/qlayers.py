import merlin as ml
import perceval as pcvl
import numpy as np
import torch
import torch.nn as nn


"""
Functions used to use photonics to check the final state convergence with a qubit vision
"""


def angle_encoding_dual_rail(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
) -> ml.QuantumLayer:
    input_state = [0] * num_features * 2
    for i in range(num_features):
        input_state[(i * 2) + 1] = 1
    circuit = ml.CircuitBuilder(n_modes=num_features * 2)
    circuit.add_angle_encoding(modes=[(2 * i) + 1 for i in range(num_features)])
    for _ in range(num_layers):
        circuit.add_entangling_layer()
    return ml.QuantumLayer(
        input_size=num_features,  # Follow the convention?
        builder=circuit,
        input_state=input_state,
        n_photons=num_features,
        measurement_strategy=measurement_strategy,
        computation_space="dual_rail",
    )


def amplitude_encoding_dual_rail(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
) -> ml.QuantumLayer:
    num_photons = int(np.ceil(np.log2(num_features)))
    circuit = ml.CircuitBuilder(n_modes=2 * num_photons)
    for _ in range(num_layers):
        circuit.add_entangling_layer()
    return ml.QuantumLayer(
        builder=circuit,
        amplitude_encoding=True,
        n_photons=num_photons,
        measurement_strategy=measurement_strategy,
        computation_space="dual_rail",
    )


"""Simple encoding layers"""


def angle_encoding_simple(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
) -> ml.QuantumLayer:
    input_state = [0] * num_features
    for i in range(num_features // 2):
        input_state[(i * 2) + 1] = 1
    circuit = ml.CircuitBuilder(n_modes=num_features)
    circuit.add_entangling_layer()
    circuit.add_angle_encoding(modes=[(2 * i) + 1 for i in range(num_features)])
    for _ in range(num_layers):
        circuit.add_entangling_layer()
    return ml.QuantumLayer(
        input_size=num_features,  # Follow the convention?
        builder=circuit,
        input_state=input_state,
        n_photons=num_features // 2,
        measurement_strategy=measurement_strategy,
    )


def amplitude_encoding_dual_rail(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
) -> ml.QuantumLayer:
    n_modes = int(np.ceil(np.log2(num_features)))
    circuit = ml.CircuitBuilder(n_modes=n_modes)
    for _ in range(num_layers):
        circuit.add_entangling_layer()
    return ml.QuantumLayer(
        builder=circuit,
        amplitude_encoding=True,
        n_photons=n_modes // 2,
        measurement_strategy=measurement_strategy,
    )


"""TODO QCNN model"""
