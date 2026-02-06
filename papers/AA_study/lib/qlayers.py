import merlin as ml
import torch.nn as nn

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.photonic_QCNN.lib.src.merlin_pqcnn import HybridModel as PhotonicQCNN
from papers.AA_study.utils.utils import find_mode_photon_config


def angle_encoding_simple(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.PROBABILITIES,
    num_classes: int = 2,
) -> ml.QuantumLayer:
    """
    Build a simple angle-encoding photonic quantum layer.

    Parameters
    ----------
    num_features : int
        Number of input features / modes.
    num_layers : int, optional
        Number of entangling/rotation layers.
    measurement_strategy : merlin.MeasurementStrategy, optional
        Measurement strategy for the quantum layer.
    num_classes : int, optional
        Number of output classes after lexicographic grouping.

    Returns
    -------
    torch.nn.Sequential
        Quantum layer followed by lexicographic grouping.
    """
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
    """
    Build a simple amplitude-encoding photonic quantum layer.

    Parameters
    ----------
    num_features : int
        Number of input features.
    num_layers : int, optional
        Number of entangling/rotation layers.
    measurement_strategy : merlin.MeasurementStrategy, optional
        Measurement strategy for the quantum layer.
    num_classes : int, optional
        Number of output classes after lexicographic grouping.

    Returns
    -------
    torch.nn.Sequential
        Quantum layer followed by lexicographic grouping.
    """
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
