#!/usr/bin/env python
"""
Basic tests that run without GPU.

These tests verify core functionality that doesn't require GPU access.
Can be run on login nodes or any environment.

Usage:
    pytest tests/test_basic.py -v
"""

import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreImports:
    """Test core library imports (no GPU required)."""

    def test_numpy_import(self):
        """Test numpy import."""
        import numpy as np

        assert np.__version__ is not None

    def test_pandas_import(self):
        """Test pandas import."""
        import pandas as pd

        assert pd.__version__ is not None

    def test_sklearn_import(self):
        """Test scikit-learn import."""
        import sklearn

        assert sklearn.__version__ is not None

    def test_qiskit_import(self):
        """Test qiskit import."""
        import qiskit

        assert qiskit.__version__ is not None

    def test_matplotlib_import(self):
        """Test matplotlib import."""
        import matplotlib

        matplotlib.use("Agg")
        assert matplotlib.__version__ is not None


class TestQVEModule:
    """Test qve module imports and basic functionality."""

    def test_qve_import(self):
        """Test qve module import."""
        import qve

        assert qve is not None

    def test_set_seed_function(self):
        """Test set_seed function."""
        from qve import set_seed

        set_seed(42)  # Should not raise

    def test_make_bsp_function(self):
        """Test make_bsp circuit creation."""
        from qve import make_bsp

        qc = make_bsp(2)
        assert qc is not None
        assert qc.num_qubits == 2

    def test_build_qsvm_qc_function(self):
        """Test build_qsvm_qc function."""
        import numpy as np
        from qve import build_qsvm_qc, make_bsp

        n_qubits = 2
        bsp_qc = make_bsp(n_qubits)
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.2, 0.8])
        circuit = build_qsvm_qc(bsp_qc, n_qubits, x1, x2)
        assert circuit is not None


class TestQuantumCircuits:
    """Test Qiskit quantum circuit functionality."""

    def test_quantum_circuit_creation(self):
        """Test basic QuantumCircuit creation."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert qc.num_qubits == 2
        assert qc.depth() > 0

    def test_parameter_vector(self):
        """Test ParameterVector creation."""
        from qiskit.circuit import ParameterVector

        pv = ParameterVector("theta", 4)
        assert len(pv) == 4

    def test_parameterized_circuit(self):
        """Test parameterized circuit creation."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        pv = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.ry(pv[0], 0)
        qc.ry(pv[1], 1)
        assert qc.num_parameters == 2


class TestDataProcessing:
    """Test data processing utilities."""

    def test_data_prepare_cv(self):
        """Test data_prepare_cv function."""
        import numpy as np
        from qve import data_prepare_cv

        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(20, 100)
        X_test = np.random.randn(5, 100)

        n_qubits = 4
        data_train, data_test = data_prepare_cv(n_qubits, X_train, X_test)

        # Check dimensions
        assert data_train.shape == (20, n_qubits)
        assert data_test.shape == (5, n_qubits)

        # Check scaling to [-1, 1] (with floating point tolerance)
        assert data_train.min() >= -1.0 - 1e-10
        assert data_train.max() <= 1.0 + 1e-10

    def test_data_partition(self):
        """Test data_partition function."""
        import numpy as np
        from qve import data_partition

        # Create synthetic kernel matrix
        np.random.seed(42)
        K = np.random.randn(10, 10)

        # Partition should work (size=1, rank=0 for single process)
        result = data_partition(K, size=1, rank=0)
        assert result is not None


class TestSKLearnIntegration:
    """Test scikit-learn integration."""

    def test_svc_import(self):
        """Test SVC import."""
        from sklearn.svm import SVC

        svc = SVC(kernel="precomputed")
        assert svc is not None

    def test_train_test_split(self):
        """Test train_test_split."""
        import numpy as np
        from sklearn.model_selection import train_test_split

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_label_encoder(self):
        """Test LabelEncoder."""
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        labels = ["cat", "dog", "cat", "bird"]
        encoded = le.fit_transform(labels)
        assert len(encoded) == 4
        assert len(le.classes_) == 3

    def test_metrics(self):
        """Test classification metrics."""
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        assert 0 <= acc <= 1
        assert 0 <= f1 <= 1


if __name__ == "__main__":
    print("Running basic tests (no GPU required)...")
    print("=" * 60)
    pytest.main([__file__, "-v"])
