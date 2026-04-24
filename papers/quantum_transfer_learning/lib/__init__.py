"""
Quantum Transfer Learning - Library
===================================

Implementation of "Transfer Learning in Hybrid Classical-Quantum Neural Networks"
by Mari et al. (2020).

Supports two backends:
    - MerLin (photonic): Primary implementation using QuantumLayer
    - PennyLane (qubit): Reference implementation for comparison

Modules:
    - circuits: Variational quantum circuit definitions (MerLin + PennyLane)
    - models: Hybrid classical-quantum model architectures
    - datasets: Data loading and preprocessing
    - training: Training loops and optimization
    - visualization: Plotting utilities
    - runner: Main entry point for MerLin CLI
"""

# Re-export ComputationSpace enum for easy access
from merlin import ComputationSpace

from .circuits import (
    DressedQuantumCircuit,
    MerLinDressedCircuit,
    MerLinQuantumLayer,
    MerLinSimpleLayer,
    PennyLaneDressedCircuit,
    PennyLaneQuantumLayer,
    ScaleLayer,
    VariationalCircuit,
)
from .datasets import (
    CIFAR10Binary,
    HymenopteraDataset,
    SpiralDataset,
    create_dataloaders,
)
from .models import (
    ClassicalBaseline,
    CQTransferModel,
    HybridModel,
    MerLinCQTransferModel,
    MerLinVQCModel,
    create_model,
)
from .runner import main, set_seed
from .training import Trainer, evaluate, train_epoch, train_model

__all__ = [
    # MerLin enums
    "ComputationSpace",
    # Circuits
    "VariationalCircuit",
    "DressedQuantumCircuit",
    "MerLinQuantumLayer",
    "MerLinDressedCircuit",
    "MerLinSimpleLayer",
    "PennyLaneQuantumLayer",
    "PennyLaneDressedCircuit",
    "ScaleLayer",
    # Models
    "HybridModel",
    "CQTransferModel",
    "MerLinCQTransferModel",
    "MerLinVQCModel",
    "ClassicalBaseline",
    "create_model",
    # Datasets
    "SpiralDataset",
    "HymenopteraDataset",
    "CIFAR10Binary",
    "create_dataloaders",
    # Training
    "train_epoch",
    "evaluate",
    "Trainer",
    "train_model",
    # Runner
    "main",
    "set_seed",
]

__version__ = "1.0.0"
