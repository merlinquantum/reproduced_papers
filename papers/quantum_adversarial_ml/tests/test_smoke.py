"""
Smoke Tests for Quantum Adversarial Machine Learning
====================================================

Quick validation tests for the reproduction.
Run with: pytest -q (from project directory)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add lib to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


class TestDatasets:
    """Test dataset generation and loading."""

    def test_mnist_binary_dataset(self):
        """Test binary MNIST dataset."""
        from lib.datasets import MNISTBinaryDataset

        dataset = MNISTBinaryDataset(
            root="data", train=True, download=True, digits=(1, 9), image_size=16
        )

        assert len(dataset) > 0

        x, label = dataset[0]
        assert x.shape == (256,)  # 16x16 = 256
        assert label in [0, 1]

    def test_mnist_multiclass_dataset(self):
        """Test multi-class MNIST dataset."""
        from lib.datasets import MNISTMulticlassDataset

        dataset = MNISTMulticlassDataset(
            root="data", train=True, download=True, digits=[1, 3, 7, 9], image_size=16
        )

        assert len(dataset) > 0

        x, label = dataset[0]
        assert x.shape == (256,)
        assert label in [0, 1, 2, 3]

    def test_ising_dataset(self):
        """Test Ising model dataset."""
        from lib.datasets import IsingDataset

        dataset = IsingDataset(
            n_spins=6,  # Small for testing
            n_samples=50,
            jx_range=(0.0, 2.0),
            seed=42,
        )

        assert len(dataset) == 50

        state, label = dataset[0]
        assert state.shape == (64,)  # 2^6 = 64
        assert label in [0, 1]

    def test_qah_dataset(self):
        """Test Quantum Anomalous Hall dataset."""
        from lib.datasets import QAHDataset

        dataset = QAHDataset(
            lattice_size=8,
            n_samples=20,
            momentum_resolution=10,  # Small for testing
            seed=42,
        )

        assert len(dataset) == 20

        image, label = dataset[0]
        assert image.shape == (100,)  # 10x10 = 100
        assert label in [0, 1]

        # Check 2D image retrieval
        img_2d = dataset.get_image_2d(0)
        assert img_2d.shape == (10, 10)

    def test_topological_phase_dataset(self):
        """Test simplified topological phase dataset."""
        from lib.datasets import TopologicalPhaseDataset

        dataset = TopologicalPhaseDataset(n_sites=10, n_samples=50, seed=42)

        assert len(dataset) == 50

        features, label = dataset[0]
        assert features.shape == (50,)  # n_k = 50
        assert label in [0, 1]

    def test_amplitude_encode(self):
        """Test amplitude encoding."""
        from lib.datasets import amplitude_encode

        x = torch.rand(5, 16)
        encoded = amplitude_encode(x)

        # Should be normalized
        norms = torch.norm(encoded, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_create_dataloaders(self):
        """Test dataloader creation."""
        from lib.datasets import create_dataloaders

        config = {"digits": [1, 9], "image_size": 16, "batch_size": 32}

        train_loader, test_loader = create_dataloaders("mnist_binary", config, seed=42)

        assert len(train_loader) > 0
        assert len(test_loader) > 0

        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] <= 32
        assert batch_x.shape[1] == 256


class TestCircuits:
    """Test MerLin quantum circuits."""

    def test_merlin_import(self):
        """Test MerLin can be imported."""
        import merlin as ml
        from merlin import QuantumLayer

        assert ml is not None
        assert QuantumLayer is not None

    def test_create_classifier_circuit(self):
        """Test circuit creation."""
        from lib.circuits import create_classifier_circuit

        circuit = create_classifier_circuit(n_modes=4, n_features=2, n_layers=2)

        assert circuit is not None
        assert circuit.m == 4

        # Check parameters
        params = circuit.get_parameters()
        param_names = [p.name for p in params]

        # Should have trainable params
        assert any(p.startswith("theta_") for p in param_names)

    def test_scale_layer(self):
        """Test ScaleLayer."""
        from lib.circuits import ScaleLayer

        # Learned scale
        layer = ScaleLayer(dim=4, scale_type="learned")
        x = torch.randn(3, 4)
        y = layer(x)
        assert y.shape == (3, 4)

        # Fixed scale
        layer_pi = ScaleLayer(dim=4, scale_type="pi")
        y_pi = layer_pi(x)
        assert y_pi.shape == (3, 4)

    def test_merlin_quantum_classifier(self):
        """Test MerLinQuantumClassifier."""
        from lib.circuits import MerLinQuantumClassifier

        model = MerLinQuantumClassifier(
            n_inputs=4,
            n_outputs=2,
            n_modes=4,
            n_photons=2,
            n_layers=2,
            computation_space="unbunched",
        )

        x = torch.randn(2, 4)
        output = model(x)

        assert output.shape == (2, 2)

    def test_pennylane_available_flag(self):
        """Test PennyLane availability flag."""
        from lib.circuits import PENNYLANE_AVAILABLE

        # Just check the flag exists and is boolean
        assert isinstance(PENNYLANE_AVAILABLE, bool)

    def test_pennylane_simple_classifier(self):
        """Test PennyLaneSimpleClassifier (skip if PennyLane unavailable)."""
        pytest.importorskip("pennylane")
        from lib.circuits import PennyLaneSimpleClassifier

        model = PennyLaneSimpleClassifier(
            n_inputs=4, n_outputs=2, n_layers=2, scale_type="pi"
        )

        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 2)

    def test_pennylane_hybrid_classifier(self):
        """Test PennyLaneHybridClassifier (skip if PennyLane unavailable)."""
        pytest.importorskip("pennylane")
        from lib.circuits import PennyLaneHybridClassifier

        model = PennyLaneHybridClassifier(
            input_dim=16, n_outputs=2, hidden_dims=[8], n_qubits=4, n_layers=2
        )

        x = torch.randn(2, 16)
        out = model(x)
        assert out.shape == (2, 2)


class TestModels:
    """Test model architectures."""

    def test_quantum_classifier(self):
        """Test QuantumClassifier."""
        from lib.models import QuantumClassifier

        model = QuantumClassifier(
            n_inputs=2, n_outputs=2, n_modes=4, n_photons=2, n_layers=2
        )

        x = torch.randn(2, 2)
        output = model(x)

        assert output.shape == (2, 2)

    def test_hybrid_quantum_classifier(self):
        """Test HybridQuantumClassifier."""
        from lib.models import HybridQuantumClassifier

        model = HybridQuantumClassifier(
            input_dim=256,
            n_outputs=2,
            hidden_dims=[64, 32],
            n_modes=6,
            n_photons=2,
            n_layers=2,
        )

        x = torch.randn(2, 256)
        output = model(x)

        assert output.shape == (2, 2)

    def test_classical_baseline(self):
        """Test ClassicalBaseline."""
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=256, n_outputs=2, hidden_dims=[64, 32])

        x = torch.randn(3, 256)
        output = model(x)

        assert output.shape == (3, 2)

    def test_classical_cnn(self):
        """Test ClassicalCNN for transfer attacks."""
        from lib.models import ClassicalCNN

        model = ClassicalCNN(
            input_channels=1, image_size=16, n_outputs=2, n_filters=[16, 32]
        )

        # Test with flattened input
        x = torch.randn(3, 256)  # 16x16 flattened
        output = model(x)
        assert output.shape == (3, 2)

        # Test with 2D input
        x_2d = torch.randn(3, 1, 16, 16)
        output_2d = model(x_2d)
        assert output_2d.shape == (3, 2)

    def test_classical_fnn(self):
        """Test ClassicalFNN for transfer attacks."""
        from lib.models import ClassicalFNN

        model = ClassicalFNN(input_dim=256, n_outputs=2, hidden_dims=[128, 64, 32])

        x = torch.randn(3, 256)
        output = model(x)

        assert output.shape == (3, 2)

    def test_create_model(self):
        """Test model factory."""
        from lib.models import create_model

        config = {
            "type": "hybrid_quantum",
            "input_dim": 256,
            "n_outputs": 2,
            "n_modes": 4,
            "n_photons": 2,
            "n_layers": 2,
        }

        model = create_model(config)
        x = torch.randn(2, 256)
        output = model(x)

        assert output.shape == (2, 2)

    def test_amplitude_classifier(self):
        """Test MerLinAmplitudeClassifier with proper amplitude encoding."""
        from math import comb

        from lib.circuits import MerLinAmplitudeClassifier

        model = MerLinAmplitudeClassifier(
            input_dim=64,
            n_outputs=2,
            n_modes=4,
            n_photons=2,
            n_layers=2,
            hidden_dims=[32, 16],
            computation_space="unbunched",  # Default
        )

        # Verify state space dimension calculation
        # For unbunched: C(n_modes, n_photons) = C(4, 2) = 6
        expected_fock_dim = comb(4, 2)  # 6 for unbunched
        assert model.fock_dim == expected_fock_dim

        # Test forward pass
        x = torch.randn(3, 64)
        output = model(x)
        assert output.shape == (3, 2)

    def test_create_amplitude_model(self):
        """Test create_model with amplitude encoding type."""
        from lib.models import create_model

        config = {
            "type": "amplitude_quantum",
            "input_dim": 64,
            "n_outputs": 2,
            "n_modes": 4,
            "n_photons": 2,
            "n_layers": 2,
            "hidden_dims": [32, 16],
            "computation_space": "unbunched",
        }
        model = create_model(config)
        assert model is not None

        x = torch.randn(2, 64)
        output = model(x)
        assert output.shape == (2, 2)


class TestAttacks:
    """Test adversarial attack methods."""

    def test_fgsm_attack(self):
        """Test FGSM attack."""
        from lib.attacks import fgsm_attack
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])
        x = torch.rand(5, 16)
        y = torch.randint(0, 2, (5,))

        x_adv = fgsm_attack(model, x, y, epsilon=0.1)

        assert x_adv.shape == x.shape
        # Adversarial should be different from original
        assert not torch.allclose(x_adv, x)

    def test_bim_attack(self):
        """Test BIM attack."""
        from lib.attacks import bim_attack
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])
        x = torch.rand(5, 16)
        y = torch.randint(0, 2, (5,))

        x_adv = bim_attack(model, x, y, epsilon=0.1, num_iter=3)

        assert x_adv.shape == x.shape

    def test_pgd_attack(self):
        """Test PGD attack."""
        from lib.attacks import pgd_attack
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])
        x = torch.rand(5, 16)
        y = torch.randint(0, 2, (5,))

        x_adv = pgd_attack(model, x, y, epsilon=0.1, num_iter=5)

        assert x_adv.shape == x.shape

    def test_mim_attack(self):
        """Test MIM attack."""
        from lib.attacks import mim_attack
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])
        x = torch.rand(5, 16)
        y = torch.randint(0, 2, (5,))

        x_adv = mim_attack(model, x, y, epsilon=0.1, num_iter=3)

        assert x_adv.shape == x.shape

    def test_functional_attack(self):
        """Test functional attack (local phase perturbations)."""
        from lib.attacks import functional_attack
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])
        x = torch.rand(5, 16)
        y = torch.randint(0, 2, (5,))

        x_adv = functional_attack(model, x, y, epsilon=0.5, num_iter=3)

        assert x_adv.shape == x.shape
        # Functional attack uses multiplicative perturbation (cos(delta))
        # Values should be in [0, 1] after clamp
        assert x_adv.min() >= 0
        assert x_adv.max() <= 1
        # Should produce valid output (not NaN or Inf)
        assert torch.isfinite(x_adv).all()

    def test_functional_fgsm_attack(self):
        """Test single-step functional attack."""
        from lib.attacks import functional_fgsm_attack
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])
        x = torch.rand(5, 16)
        y = torch.randint(0, 2, (5,))

        x_adv = functional_fgsm_attack(model, x, y, epsilon=0.5)

        assert x_adv.shape == x.shape

    def test_transfer_attack(self):
        """Test black-box transfer attack."""
        from lib.attacks import transfer_attack
        from lib.models import ClassicalBaseline, ClassicalFNN

        # Surrogate and target models
        surrogate = ClassicalFNN(input_dim=16, n_outputs=2, hidden_dims=[32, 16])
        target = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])

        # Simple dataloader
        X = torch.rand(50, 16)
        y = torch.randint(0, 2, (50,))
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        results = transfer_attack(
            surrogate_model=surrogate,
            target_model=target,
            dataloader=loader,
            attack_method="fgsm",
            epsilon=0.1,
        )

        assert "target_adversarial_accuracy" in results
        assert "transfer_rate" in results

    def test_compute_fidelity(self):
        """Test fidelity computation."""
        from lib.attacks import compute_fidelity

        x1 = torch.rand(5, 16)
        x2 = x1.clone()  # Identical

        fid = compute_fidelity(x1, x2)

        # Identical samples should have fidelity 1
        assert torch.allclose(fid, torch.ones(5), atol=1e-5)

    def test_add_random_noise(self):
        """Test random noise addition."""
        from lib.attacks import add_random_noise

        x = torch.rand(5, 16)

        # Uniform noise
        x_noisy = add_random_noise(x, epsilon=0.1, noise_type="uniform")
        assert x_noisy.shape == x.shape
        assert not torch.allclose(x_noisy, x)

        # Gaussian noise
        x_noisy = add_random_noise(x, epsilon=0.1, noise_type="gaussian")
        assert x_noisy.shape == x.shape

    def test_evaluate_with_photon_loss(self):
        """Test photon loss evaluation."""
        from lib.attacks import evaluate_with_photon_loss
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])

        # Simple dataloader
        X = torch.rand(50, 16)
        y = torch.randint(0, 2, (50,))
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        results = evaluate_with_photon_loss(model, loader, loss_rate=0.1)

        assert "accuracy" in results
        assert "loss_rate" in results
        assert 0 <= results["accuracy"] <= 1

    def test_compare_noise_vs_adversarial(self):
        """Test comprehensive noise comparison."""
        from lib.attacks import compare_noise_vs_adversarial
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])

        # Small dataloader for quick test
        X = torch.rand(30, 16)
        y = torch.randint(0, 2, (30,))
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        results = compare_noise_vs_adversarial(
            model=model,
            dataloader=loader,
            epsilon_values=[0.05, 0.1],
            attack_method="fgsm",
            num_iter=3,
        )

        assert "clean_accuracy" in results
        assert "adversarial" in results
        assert "random_uniform" in results
        assert "photon_loss" in results
        assert len(results["adversarial"]) == 2


class TestDefense:
    """Test defense mechanisms."""

    def test_adversarial_training_step(self):
        """Test adversarial training step."""
        import torch.optim as optim
        from lib.defense import adversarial_training_step
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(input_dim=16, n_outputs=2, hidden_dims=[8])
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.rand(10, 16)
        y = torch.randint(0, 2, (10,))

        loss, clean_acc, adv_acc = adversarial_training_step(
            model, x, y, optimizer, criterion, attack_method="fgsm", epsilon=0.1
        )

        assert loss > 0
        assert 0 <= clean_acc <= 1
        assert 0 <= adv_acc <= 1


class TestTraining:
    """Test training utilities."""

    def test_train_epoch(self):
        """Test single training epoch."""
        import torch.nn as nn
        import torch.optim as optim
        from lib.training import train_epoch
        from torch.utils.data import DataLoader, TensorDataset

        # Simple dataset
        X = torch.rand(100, 16)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)

        model = nn.Linear(16, 2)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        loss, acc = train_epoch(
            model, loader, optimizer, criterion, torch.device("cpu")
        )

        assert loss > 0
        assert 0 <= acc <= 1

    def test_evaluate(self):
        """Test evaluation."""
        import torch.nn as nn
        from lib.training import evaluate
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.rand(50, 16)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)

        model = nn.Linear(16, 2)
        criterion = nn.CrossEntropyLoss()

        loss, acc = evaluate(model, loader, criterion, torch.device("cpu"))

        assert loss > 0
        assert 0 <= acc <= 1


class TestRunner:
    """Test runner functionality."""

    def test_set_seed(self):
        """Test seed setting."""
        from lib.runner import set_seed

        set_seed(123)
        val1 = np.random.rand()

        set_seed(123)
        val2 = np.random.rand()

        assert val1 == val2


class TestExports:
    """Test that all exports are available."""

    def test_lib_exports(self):
        """Test main lib exports."""
        from lib import (
            # Defense
            ClassicalCNN,
            ClassicalFNN,
            MerLinAmplitudeClassifier,
            MerLinAmplitudeClassifierDirect,
            # Circuits
            MerLinQuantumClassifier,
            # Datasets
            QAHDataset,
            # Models
            add_random_noise,
            compare_noise_vs_adversarial,
            create_dataloaders,
            create_model,
            functional_attack,
            transfer_attack,
        )

        # Core quantum components
        assert MerLinQuantumClassifier is not None
        assert MerLinAmplitudeClassifier is not None
        assert MerLinAmplitudeClassifierDirect is not None

        # Classical baselines
        assert ClassicalCNN is not None
        assert ClassicalFNN is not None

        # Datasets
        assert QAHDataset is not None
        assert create_dataloaders is not None

        # Attacks
        assert functional_attack is not None
        assert transfer_attack is not None
        assert add_random_noise is not None
        assert compare_noise_vs_adversarial is not None

        # Factory
        assert create_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
