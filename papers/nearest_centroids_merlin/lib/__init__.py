"""Nearest centroid QML reproduced paper library package."""

from .classifier import MLQuantumNearestCentroid, QuantumNearestCentroid
from .defaults import default_config
from .synthetic_data import generate_paper_datasets, generate_synthetic_data

__all__ = [
    "default_config",
    "QuantumNearestCentroid",
    "MLQuantumNearestCentroid",
    "generate_synthetic_data",
    "generate_paper_datasets",
]
