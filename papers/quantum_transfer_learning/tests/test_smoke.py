"""
Smoke Tests
===========

Quick validation tests for the quantum transfer learning reproduction.
Run with: pytest -q (from project directory)
"""

import json
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

    def test_spiral_generation(self):
        """Test spiral data generation."""
        from lib.datasets import generate_spiral_data

        X, y = generate_spiral_data(100, noise=0.0, seed=42)

        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert set(y) == {0, 1}
        assert np.abs(X).max() <= 1.0

    def test_spiral_dataset(self):
        """Test SpiralDataset class."""
        from lib.datasets import SpiralDataset

        dataset = SpiralDataset(n_samples=100, seed=42)

        assert len(dataset) == 100

        x, label = dataset[0]
        assert x.shape == (2,)
        assert label.item() in [0, 1]

    def test_spiral_dataset_attributes(self):
        """Test SpiralDataset has X and y attributes."""
        from lib.datasets import SpiralDataset

        dataset = SpiralDataset(n_samples=100, seed=42)

        assert hasattr(dataset, 'X')
        assert hasattr(dataset, 'y')
        assert dataset.X.shape == (100, 2)
        assert dataset.y.shape == (100,)


class TestClassicalModels:
    """Test classical model components."""

    def test_classical_baseline(self):
        """Test classical baseline model."""
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(
            n_inputs=2,
            n_outputs=2,
            hidden_sizes=[4],
            activation="tanh"
        )

        x = torch.randn(5, 2)
        output = model(x)

        assert output.shape == (5, 2)

    def test_classical_baseline_relu(self):
        """Test classical baseline with ReLU activation."""
        from lib.models import ClassicalBaseline

        model = ClassicalBaseline(
            n_inputs=2,
            n_outputs=2,
            hidden_sizes=[8, 4],
            activation="relu"
        )

        x = torch.randn(3, 2)
        output = model(x)

        assert output.shape == (3, 2)

    def test_scale_layer(self):
        """Test ScaleLayer."""
        from lib.circuits import ScaleLayer

        # Test learned scale
        layer = ScaleLayer(dim=4, scale_type="learned")
        x = torch.randn(3, 4)
        y = layer(x)
        assert y.shape == (3, 4)

        # Test fixed pi scale
        layer_pi = ScaleLayer(dim=4, scale_type="pi")
        y_pi = layer_pi(x)
        assert y_pi.shape == (3, 4)

        # Test 2pi scale
        layer_2pi = ScaleLayer(dim=4, scale_type="2pi")
        y_2pi = layer_2pi(x)
        assert y_2pi.shape == (3, 4)

        # Test unit scale
        layer_1 = ScaleLayer(dim=4, scale_type="1")
        y_1 = layer_1(x)
        assert y_1.shape == (3, 4)


class TestMerLinCircuits:
    """Test MerLin photonic circuits."""

    def test_merlin_import(self):
        """Test MerLin can be imported."""
        import merlin as ML
        from merlin import QuantumLayer, ComputationSpace

        assert ML is not None
        assert QuantumLayer is not None
        assert ComputationSpace is not None

    def test_circuit_creation(self):
        """Test Perceval circuit creation."""
        from lib.circuits import create_merlin_vqc_circuit

        circuit = create_merlin_vqc_circuit(n_modes=4, n_features=2)

        assert circuit is not None
        assert circuit.m == 4  # 4 modes

        # Check parameters exist
        params = circuit.get_parameters()
        param_names = [p.name for p in params]

        # Should have trainable params (theta_l*, theta_r*) and input params (x*)
        assert any(p.startswith("theta_l") for p in param_names)
        assert any(p.startswith("theta_r") for p in param_names)
        assert any(p.startswith("x") for p in param_names)

    def test_deep_circuit_creation(self):
        """Test deep variational circuit creation."""
        from lib.circuits import create_merlin_deep_circuit

        circuit = create_merlin_deep_circuit(n_modes=4, n_features=2, depth=3)

        assert circuit is not None
        assert circuit.m == 4

        # Verify parameter names are unique (the bug was duplicate x0, x1 names)
        params = circuit.get_parameters()
        param_names = [p.name for p in params]
        assert len(param_names) == len(set(param_names)), "Parameter names must be unique"

    def test_merlin_quantum_layer(self):
        """Test MerLinQuantumLayer."""
        from lib.circuits import MerLinQuantumLayer

        layer = MerLinQuantumLayer(
            n_modes=4,
            n_features=2,
            n_photons=2,
            q_depth=1,
            computation_space="unbunched",
            measurement_strategy="probabilities"
        )

        x = torch.randn(3, 2)
        output = layer(x)

        assert output.shape[0] == 3  # Batch size preserved
        assert output.shape[1] == layer.output_size

    def test_merlin_quantum_layer_deep(self):
        """Test MerLinQuantumLayer with depth > 1."""
        from lib.circuits import MerLinQuantumLayer

        layer = MerLinQuantumLayer(
            n_modes=4,
            n_features=2,
            n_photons=2,
            q_depth=2,
            computation_space="unbunched"
        )

        x = torch.randn(2, 2)
        output = layer(x)

        assert output.shape[0] == 2

    def test_merlin_dressed_circuit(self):
        """Test MerLinDressedCircuit."""
        from lib.circuits import MerLinDressedCircuit

        circuit = MerLinDressedCircuit(
            n_inputs=2,
            n_outputs=2,
            n_modes=4,
            n_photons=2,
            q_depth=1,
            computation_space="unbunched",
            scale_type="learned"
        )

        x = torch.randn(2, 2)
        output = circuit(x)

        assert output.shape == (2, 2)

    def test_merlin_simple_layer(self):
        """Test MerLinSimpleLayer using QuantumLayer.simple()."""
        from lib.circuits import MerLinSimpleLayer

        # Note: n_params must be >= 90 (the entangling layer minimum)
        # Values above 90 must differ by an even amount (each MZI adds 2 params)
        layer = MerLinSimpleLayer(
            input_size=4,
            n_params=90,  # Minimum valid value
        )

        x = torch.randn(2, 4)
        output = layer(x)

        assert output.shape[0] == 2
        assert output.shape[1] == layer.output_size


class TestPennyLaneCircuits:
    """Test PennyLane qubit circuits."""

    def test_pennylane_quantum_layer(self):
        """Test PennyLaneQuantumLayer."""
        from lib.circuits import PennyLaneQuantumLayer

        layer = PennyLaneQuantumLayer(n_qubits=4, q_depth=2)

        x = torch.randn(2, 4)
        output = layer(x)

        assert output.shape == (2, 4)
        # Expectation values should be in [-1, 1]
        assert output.abs().max() <= 1.0 + 1e-6

    def test_pennylane_quantum_layer_output_size(self):
        """Test PennyLaneQuantumLayer output_size property."""
        from lib.circuits import PennyLaneQuantumLayer

        layer = PennyLaneQuantumLayer(n_qubits=6, q_depth=3)
        assert layer.output_size == 6

    def test_pennylane_dressed_circuit(self):
        """Test PennyLaneDressedCircuit."""
        from lib.circuits import PennyLaneDressedCircuit

        circuit = PennyLaneDressedCircuit(
            n_inputs=2,
            n_outputs=2,
            n_qubits=4,
            q_depth=2
        )

        x = torch.randn(2, 2)
        output = circuit(x)

        assert output.shape == (2, 2)


class TestUnifiedInterface:
    """Test unified interface for both backends."""

    def test_dressed_circuit_merlin(self):
        """Test DressedQuantumCircuit with MerLin backend."""
        from lib.circuits import DressedQuantumCircuit

        circuit = DressedQuantumCircuit(
            n_inputs=2,
            n_outputs=2,
            n_qubits=4,
            q_depth=3,
            backend="merlin",
            n_photons=2,
            computation_space="unbunched",
            merlin_depth=1,
            scale_type="learned"
        )

        x = torch.randn(2, 2)
        output = circuit(x)

        assert output.shape == (2, 2)

    def test_dressed_circuit_pennylane(self):
        """Test DressedQuantumCircuit with PennyLane backend."""
        from lib.circuits import DressedQuantumCircuit

        circuit = DressedQuantumCircuit(
            n_inputs=2,
            n_outputs=2,
            n_qubits=4,
            q_depth=2,
            backend="pennylane"
        )

        x = torch.randn(2, 2)
        output = circuit(x)

        assert output.shape == (2, 2)

    def test_variational_circuit_merlin(self):
        """Test VariationalCircuit with MerLin backend."""
        from lib.circuits import VariationalCircuit

        circuit = VariationalCircuit(
            n_qubits=4,
            q_depth=2,
            backend="merlin",
            n_photons=2,
            computation_space="unbunched"
        )

        x = torch.randn(2, 4)
        output = circuit(x)

        assert output.shape[0] == 2
        assert output.shape[1] == circuit.output_size

    def test_variational_circuit_pennylane(self):
        """Test VariationalCircuit with PennyLane backend."""
        from lib.circuits import VariationalCircuit

        circuit = VariationalCircuit(
            n_qubits=4,
            q_depth=2,
            backend="pennylane"
        )

        x = torch.randn(2, 4)
        output = circuit(x)

        assert output.shape == (2, 4)


class TestHybridModels:
    """Test hybrid classical-quantum models."""

    def test_hybrid_model_merlin(self):
        """Test HybridModel with MerLin."""
        from lib.models import HybridModel

        model = HybridModel(
            n_inputs=2,
            n_outputs=2,
            n_qubits=4,
            q_depth=2,
            backend="merlin",
            n_photons=2,
            computation_space="unbunched"
        )

        x = torch.randn(2, 2)
        output = model(x)

        assert output.shape == (2, 2)

    def test_hybrid_model_pennylane(self):
        """Test HybridModel with PennyLane."""
        from lib.models import HybridModel

        model = HybridModel(
            n_inputs=2,
            n_outputs=2,
            n_qubits=4,
            q_depth=2,
            backend="pennylane"
        )

        x = torch.randn(2, 2)
        output = model(x)

        assert output.shape == (2, 2)

    def test_merlin_vqc_model(self):
        """Test MerLinVQCModel."""
        from lib.models import MerLinVQCModel

        model = MerLinVQCModel(
            n_inputs=2,
            n_outputs=2,
            n_modes=4,
            n_photons=2,
            computation_space="unbunched",
            scale_type="learned"
        )

        x = torch.randn(2, 2)
        output = model(x)

        assert output.shape == (2, 2)

    def test_model_factory_dressed_quantum(self):
        """Test model creation from config - dressed quantum."""
        from lib.models import create_model

        config = {
            "type": "dressed_quantum",
            "n_inputs": 2,
            "n_outputs": 2,
            "n_qubits": 4,
            "q_depth": 2,
            "n_photons": 2,
            "computation_space": "unbunched"
        }

        model = create_model(config, backend="merlin")

        x = torch.randn(2, 2)
        output = model(x)

        assert output.shape == (2, 2)

    def test_model_factory_merlin_vqc(self):
        """Test model creation from config - merlin_vqc."""
        from lib.models import create_model

        config = {
            "type": "merlin_vqc",
            "n_inputs": 2,
            "n_outputs": 2,
            "n_modes": 4,
            "n_photons": 2,
            "computation_space": "unbunched",
            "scale_type": "learned"
        }

        model = create_model(config, backend="merlin")

        x = torch.randn(2, 2)
        output = model(x)

        assert output.shape == (2, 2)

    def test_model_factory_classical(self):
        """Test model creation from config - classical."""
        from lib.models import create_model

        config = {
            "type": "classical",
            "n_inputs": 2,
            "n_outputs": 2,
            "hidden_sizes": [4, 4],
            "activation": "tanh"
        }

        model = create_model(config, backend="merlin")

        x = torch.randn(2, 2)
        output = model(x)

        assert output.shape == (2, 2)


class TestTraining:
    """Test training utilities."""

    def test_training_loop(self):
        """Test a minimal training loop."""
        import torch.nn as nn
        import torch.optim as optim
        from lib.datasets import create_dataloaders
        from lib.models import ClassicalBaseline
        from lib.training import evaluate, train_epoch

        # Minimal config
        config = {
            "n_samples": 100,
            "n_train": 80,
            "batch_size": 10
        }

        train_loader, test_loader = create_dataloaders("spiral", config, seed=42)

        model = ClassicalBaseline(n_inputs=2, n_outputs=2, hidden_sizes=[4])
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Single epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        assert train_loss > 0
        assert 0 <= train_acc <= 1
        assert test_loss > 0
        assert 0 <= test_acc <= 1

    def test_trainer_class(self):
        """Test Trainer class."""
        from lib.datasets import create_dataloaders
        from lib.models import ClassicalBaseline
        from lib.training import Trainer

        config = {
            "n_samples": 100,
            "n_train": 80,
            "batch_size": 10
        }

        train_loader, test_loader = create_dataloaders("spiral", config, seed=42)

        model = ClassicalBaseline(n_inputs=2, n_outputs=2, hidden_sizes=[4])
        device = torch.device("cpu")

        training_config = {
            "learning_rate": 0.01,
            "optimizer": "adam"
        }

        trainer = Trainer(model, train_loader, test_loader, training_config, device)
        results = trainer.train(epochs=2, verbose=False, save_best=False)

        assert "history" in results
        assert "best_accuracy" in results
        assert "final_accuracy" in results
        assert "total_time" in results
        assert len(results["history"]["train_loss"]) == 2

    def test_train_model_function(self):
        """Test train_model convenience function."""
        from lib.datasets import create_dataloaders
        from lib.models import ClassicalBaseline
        from lib.training import train_model

        config = {
            "n_samples": 100,
            "n_train": 80,
            "batch_size": 10
        }

        train_loader, test_loader = create_dataloaders("spiral", config, seed=42)

        model = ClassicalBaseline(n_inputs=2, n_outputs=2, hidden_sizes=[4])
        device = torch.device("cpu")

        training_config = {
            "epochs": 2,
            "learning_rate": 0.01
        }

        results = train_model(model, train_loader, test_loader, training_config, device)

        assert "best_accuracy" in results
        assert 0 <= results["best_accuracy"] <= 1


class TestRunner:
    """Test the main runner functionality."""

    def test_set_seed(self):
        """Test seed setting."""
        from lib.runner import set_seed

        set_seed(123)
        val1 = np.random.rand()

        set_seed(123)
        val2 = np.random.rand()

        assert val1 == val2

    def test_set_seed_torch(self):
        """Test seed setting affects torch."""
        from lib.runner import set_seed

        set_seed(456)
        t1 = torch.rand(5)

        set_seed(456)
        t2 = torch.rand(5)

        assert torch.allclose(t1, t2)


class TestIntegration:
    """Integration tests for full workflows."""

    def test_spiral_quantum_training(self):
        """Test full spiral classification with quantum model."""
        from lib.datasets import create_dataloaders
        from lib.models import HybridModel
        from lib.training import train_model
        from lib.runner import set_seed

        set_seed(42)

        config = {
            "n_samples": 50,
            "n_train": 40,
            "batch_size": 10
        }

        train_loader, test_loader = create_dataloaders("spiral", config, seed=42)

        model = HybridModel(
            n_inputs=2,
            n_outputs=2,
            n_qubits=4,
            q_depth=2,
            backend="merlin",
            n_photons=2,
            computation_space="unbunched"
        )

        training_config = {
            "epochs": 2,
            "learning_rate": 0.01
        }

        results = train_model(
            model, train_loader, test_loader,
            training_config, torch.device("cpu")
        )

        assert "best_accuracy" in results
        assert results["best_accuracy"] >= 0

    def test_spiral_pennylane_training(self):
        """Test full spiral classification with PennyLane model."""
        from lib.datasets import create_dataloaders
        from lib.models import HybridModel
        from lib.training import train_model
        from lib.runner import set_seed

        set_seed(42)

        config = {
            "n_samples": 50,
            "n_train": 40,
            "batch_size": 10
        }

        train_loader, test_loader = create_dataloaders("spiral", config, seed=42)

        model = HybridModel(
            n_inputs=2,
            n_outputs=2,
            n_qubits=4,
            q_depth=2,
            backend="pennylane"
        )

        training_config = {
            "epochs": 2,
            "learning_rate": 0.01
        }

        results = train_model(
            model, train_loader, test_loader,
            training_config, torch.device("cpu")
        )

        assert "best_accuracy" in results
        assert results["best_accuracy"] >= 0


class TestComputationSpaces:
    """Test different computation spaces."""

    def test_unbunched_space(self):
        """Test unbunched computation space."""
        from merlin import ComputationSpace

        space = ComputationSpace.coerce("unbunched")
        assert space is not None

    def test_fock_space(self):
        """Test Fock computation space."""
        from merlin import ComputationSpace

        space = ComputationSpace.coerce("fock")
        assert space is not None

    def test_merlin_layer_fock_space(self):
        """Test MerLinQuantumLayer with Fock space."""
        from lib.circuits import MerLinQuantumLayer

        layer = MerLinQuantumLayer(
            n_modes=4,
            n_features=2,
            n_photons=2,
            computation_space="fock"
        )

        x = torch.randn(2, 2)
        output = layer(x)

        assert output.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
