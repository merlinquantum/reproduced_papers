"""
Utilities for generating figures from experiment results.

This package provides plotting utilities for visualizing:
- Training curves
- Attack results
- Robustness comparisons
- Noise vs adversarial comparisons
- Transfer attack matrices
- Model comparisons

Usage:
    # As a library
    from utils.plot_training import plot_training_from_results
    from utils.plot_attacks import plot_attack_from_results

    # As a CLI tool
    python utils/generate_figures.py results/ --all
"""

# Only import library functions, not CLI entry points
# This avoids RuntimeWarning when using `python -m utils.generate_figures`

from .plot_attacks import plot_attack_from_results, plot_robustness_from_results
from .plot_comparison import (
    plot_model_comparison_from_results,
    plot_noise_comparison_from_results,
    plot_transfer_from_results,
)
from .plot_training import plot_training_from_results

__all__ = [
    "plot_training_from_results",
    "plot_attack_from_results",
    "plot_robustness_from_results",
    "plot_noise_comparison_from_results",
    "plot_transfer_from_results",
    "plot_model_comparison_from_results",
]
