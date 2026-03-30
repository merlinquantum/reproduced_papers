"""
Quantum Adversarial Machine Learning - Library
===============================================

Implementation of "Quantum Adversarial Machine Learning"
by Lu et al. (2020).

This reproduction uses MerLin (photonic) as the primary backend,
implementing variational quantum circuits with beam splitter meshes
and phase shifter encoding.

Modules:
    - circuits: Variational quantum circuit classifiers (MerLin)
    - models: Quantum classifier architectures
    - datasets: Data loading (MNIST, Ising model)
    - attacks: Adversarial attack methods (FGSM, BIM, PGD, MIM)
    - defense: Adversarial training and defense strategies
    - training: Training loops and optimization
    - visualization: Plotting utilities
    - runner: Main entry point for experiments
"""

from .circuits import (
    PENNYLANE_AVAILABLE,
    MerLinAmplitudeClassifier,
    MerLinAmplitudeClassifierDirect,
    MerLinQuantumClassifier,
    ScaleLayer,
    create_classifier_circuit,
)

# Conditionally import PennyLane classes
if PENNYLANE_AVAILABLE:
    from .circuits import (
        PennyLaneHybridClassifier,
        PennyLaneQuantumClassifier,
        PennyLaneSimpleClassifier,
    )

from .attacks import (
    add_random_noise,
    bim_attack,
    compare_noise_vs_adversarial,
    compute_fidelity,
    evaluate_attack_success,
    evaluate_with_photon_loss,
    fgsm_attack,
    functional_attack,
    functional_fgsm_attack,
    generate_adversarial_examples,
    mim_attack,
    pgd_attack,
    run_noise_comparison_experiment,
    run_transfer_attack_experiment,
    transfer_attack,
)
from .datasets import (
    IsingDataset,
    MNISTBinaryDataset,
    MNISTMulticlassDataset,
    QAHDataset,
    SpiralDataset,
    TopologicalPhaseDataset,
    amplitude_encode,
    create_dataloaders,
)
from .defense import (
    AdversarialTrainer,
    adversarial_training_step,
    evaluate_robustness,
)
from .models import (
    ClassicalBaseline,
    ClassicalCNN,
    ClassicalFNN,
    HybridQuantumClassifier,
    QuantumClassifier,
    create_model,
)
from .runner import main, set_seed
from .training import (
    Trainer,
    evaluate,
    train_epoch,
    train_model,
)

__all__ = [
    # Circuits - Photonic
    "MerLinQuantumClassifier",
    "MerLinAmplitudeClassifier",
    "MerLinAmplitudeClassifierDirect",
    "create_classifier_circuit",
    "ScaleLayer",
    "PENNYLANE_AVAILABLE",
    # Circuits - Gate-based (conditional)
    "PennyLaneQuantumClassifier",
    "PennyLaneHybridClassifier",
    "PennyLaneSimpleClassifier",
    # Models
    "QuantumClassifier",
    "HybridQuantumClassifier",
    "ClassicalBaseline",
    "ClassicalCNN",
    "ClassicalFNN",
    "create_model",
    # Datasets
    "MNISTBinaryDataset",
    "MNISTMulticlassDataset",
    "IsingDataset",
    "QAHDataset",
    "TopologicalPhaseDataset",
    "SpiralDataset",
    "create_dataloaders",
    "amplitude_encode",
    # Attacks
    "fgsm_attack",
    "bim_attack",
    "pgd_attack",
    "mim_attack",
    "functional_attack",
    "functional_fgsm_attack",
    "transfer_attack",
    "run_transfer_attack_experiment",
    "generate_adversarial_examples",
    "compute_fidelity",
    "evaluate_attack_success",
    "add_random_noise",
    "evaluate_with_photon_loss",
    "compare_noise_vs_adversarial",
    "run_noise_comparison_experiment",
    # Defense
    "adversarial_training_step",
    "AdversarialTrainer",
    "evaluate_robustness",
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
