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

    def test_scale_layer(self):
        """Test ScaleLayer."""
        from lib.circuits import ScaleLayer

        # Test learned scale
        layer = ScaleLayer(dim=4, scale_type="learned")
        x = torch.randn(3, 4)
        y = layer(x)
        assert y.shape == (3, 4)

        # Test fixed scale
        layer_pi = ScaleLayer(dim=4, scale_type="pi")
        y_pi = layer_pi(x)
        assert y_pi.shape == (3, 4)


class TestMerLinCircuits:
    """Test MerLin photonic circuits."""

    def test_merlin_import(self):
        """Test MerLin can be imported."""
        import merlin as ML
        from merlin import QuantumLayer

        assert ML is not None
        assert QuantumLayer is not None

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

    def test_merlin_quantum_layer(self):
        """Test MerLinQuantumLayer."""
        from lib.circuits import MerLinQuantumLayer

        layer = MerLinQuantumLayer(
            n_modes=4,
            n_features=2,
            n_photons=2,
            computation_space="unbunched"
        )

        x = torch.randn(3, 2)
        output = layer(x)

        assert output.shape[0] == 3  # Batch size preserved
        assert output.shape[1] == layer.output_size

    def test_merlin_dressed_circuit(self):
        """Test MerLinDressedCircuit."""
        from lib.circuits import MerLinDressedCircuit

        circuit = MerLinDressedCircuit(
            n_inputs=2,
            n_outputs=2,
            n_modes=4,
            n_photons=2,
            computation_space="unbunched"
        )

        x = torch.randn(2, 2)
        output = circuit(x)

        assert output.shape == (2, 2)

    def test_merlin_simple_layer(self):
        """Test MerLinSimpleLayer using QuantumLayer.simple()."""
        from lib.circuits import MerLinSimpleLayer

        layer = MerLinSimpleLayer(
            input_size=4,
            n_params=50,
            computation_space="unbunched"
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
            computation_space="unbunched"
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

    def test_model_factory(self):
        """Test model creation from config."""
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

        model = ClassicalBaseline(n_inputs=2, n_outputs=2)
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

    def test_example_config(self, tmp_path):
        """Test running with example config."""
        from lib.runner import main, set_seed

        config_path = project_root / "configs" / "example.json"

        if not config_path.exists():
            pytest.skip("Example config not found")

        cfg = json.loads(config_path.read_text())
        cfg["outdir"] = str(tmp_path)

        set_seed(cfg.get("seed", 42))
        run_dir = Path(main(cfg))

        assert run_dir.exists()
        assert (run_dir / "summary_results.json").exists()
        assert (run_dir / "config_snapshot.json").exists()
        assert (run_dir / "done.txt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
