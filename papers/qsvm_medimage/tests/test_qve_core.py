#!/usr/bin/env python
"""
Tests for qve/core.py module.

Verifies that core functions work correctly and that recent changes
(hybrid kernel addition) don't break existing functionality.

Usage:
    pytest tests/test_qve_core.py -v
"""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreModuleSyntax:
    """Test that qve/core.py has no syntax errors."""

    def test_core_module_compiles(self):
        """Verify core.py has no syntax errors."""
        import py_compile

        core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "qve",
            "core.py",
        )
        # This will raise SyntaxError if there's a problem
        py_compile.compile(core_path, doraise=True)

    def test_core_module_imports(self):
        """Test that qve.core can be imported."""
        from qve import core

        assert core is not None

    def test_all_functions_importable(self):
        """Test all expected functions are importable from qve.core."""
        from qve.core import (
            build_qsvm_qc,
            get_hybrid_kernel_matrix,
            get_kernel_matrix,
            make_bsp,
        )

        # All imports should succeed
        assert make_bsp is not None
        assert build_qsvm_qc is not None
        assert get_kernel_matrix is not None
        assert get_hybrid_kernel_matrix is not None


class TestMakeBsp:
    """Test make_bsp quantum circuit creation."""

    def test_make_bsp_creates_circuit(self):
        """Test make_bsp creates a valid circuit."""
        from qve.core import make_bsp

        qc = make_bsp(2)
        assert qc is not None
        assert qc.num_qubits == 2

    def test_make_bsp_various_sizes(self):
        """Test make_bsp with different qubit counts."""
        from qve.core import make_bsp

        for n in [2, 3, 4, 5]:
            qc = make_bsp(n)
            assert qc.num_qubits == n

    def test_make_bsp_has_parameters(self):
        """Test make_bsp creates parameterized circuit."""
        from qve.core import make_bsp

        qc = make_bsp(3)
        # Should have 3 parameters (one per qubit)
        assert qc.num_parameters == 3


class TestBuildQsvmQc:
    """Test build_qsvm_qc function."""

    def test_build_qsvm_qc_basic(self):
        """Test basic kernel circuit construction."""
        from qve.core import build_qsvm_qc, make_bsp

        n_dim = 2
        bsp_qc = make_bsp(n_dim)
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.2, 0.8])
        circuit = build_qsvm_qc(bsp_qc, n_dim, x1, x2)
        assert circuit is not None
        assert circuit.num_qubits == n_dim

    def test_build_qsvm_qc_same_vectors(self):
        """Test kernel circuit with identical vectors (should give overlap=1)."""
        from qve.core import build_qsvm_qc, make_bsp

        n_dim = 2
        bsp_qc = make_bsp(n_dim)
        x = np.array([0.5, 0.3])
        circuit = build_qsvm_qc(bsp_qc, n_dim, x, x)
        assert circuit is not None


class TestDataPartition:
    """Test data_partition function for MPI-style distribution."""

    def test_single_process(self):
        """Test partition with single process (size=1, rank=0)."""
        from qve.core import data_partition

        indices = list(range(10))
        result = data_partition(indices, size=1, rank=0)
        assert result == indices  # Single process gets all data

    def test_two_processes_even_split(self):
        """Test even split between two processes."""
        from qve.core import data_partition

        indices = list(range(10))

        rank0 = data_partition(indices, size=2, rank=0)
        rank1 = data_partition(indices, size=2, rank=1)

        # Combined should equal original
        assert len(rank0) + len(rank1) == len(indices)
        # No overlap
        assert set(rank0).isdisjoint(set(rank1))

    def test_four_processes(self):
        """Test partition across four processes."""
        from qve.core import data_partition

        indices = list(range(100))

        all_parts = []
        for rank in range(4):
            part = data_partition(indices, size=4, rank=rank)
            all_parts.extend(part)

        # Should cover all indices
        assert sorted(all_parts) == indices


class TestGetKernelMatrix:
    """Test get_kernel_matrix function."""

    def test_kernel_matrix_shape(self):
        """Test kernel matrix has correct shape."""
        from qve.core import get_kernel_matrix

        data1 = np.random.randn(5, 2)
        data2 = np.random.randn(5, 2)

        # Simulate amplitude data for upper triangle
        indices = [(i + 1, j + 1) for i in range(5) for j in range(i, 5)]
        amp_data = [[np.random.rand() for _ in indices]]

        kernel = get_kernel_matrix(data1, data2, amp_data, indices, mode="train")
        assert kernel.shape == (5, 5)

    def test_kernel_matrix_train_mode_symmetric(self):
        """Test that train mode produces symmetric matrix."""
        from qve.core import get_kernel_matrix

        n = 4
        data = np.random.randn(n, 2)

        # Use upper triangle OFF-DIAGONAL indices only (1-indexed)
        # In practice, diagonal is added via np.diag(np.ones) in get_kernel_matrix
        indices = [(i + 1, j + 1) for i in range(n) for j in range(i + 1, n)]
        amp_data = [[0.5 for _ in indices]]

        kernel = get_kernel_matrix(data, data, amp_data, indices, mode="train")

        # Should be symmetric
        np.testing.assert_array_almost_equal(kernel, kernel.T)

        # Diagonal should be 1 (added by get_kernel_matrix via np.diag(np.ones))
        np.testing.assert_array_almost_equal(np.diag(kernel), np.ones(n))

    def test_kernel_matrix_test_mode(self):
        """Test kernel matrix in test mode (no symmetry operation)."""
        from qve.core import get_kernel_matrix

        data1 = np.random.randn(3, 2)  # test samples
        data2 = np.random.randn(5, 2)  # train samples

        indices = [(i + 1, j + 1) for i in range(3) for j in range(5)]
        amp_data = [[np.random.rand() for _ in indices]]

        kernel = get_kernel_matrix(data1, data2, amp_data, indices, mode=None)
        assert kernel.shape == (3, 5)


class TestGetHybridKernelMatrix:
    """Test get_hybrid_kernel_matrix function."""

    def test_hybrid_kernel_alpha_zero(self):
        """Test alpha=0 gives pure classical kernel."""
        from qve.core import get_hybrid_kernel_matrix

        classical = np.array([[1, 0.5], [0.5, 1]])
        quantum = np.array([[1, 0.3], [0.3, 1]])

        hybrid = get_hybrid_kernel_matrix(classical, quantum, alpha=0.0)
        np.testing.assert_array_almost_equal(hybrid, classical)

    def test_hybrid_kernel_alpha_one(self):
        """Test alpha=1 gives pure quantum kernel."""
        from qve.core import get_hybrid_kernel_matrix

        classical = np.array([[1, 0.5], [0.5, 1]])
        quantum = np.array([[1, 0.3], [0.3, 1]])

        hybrid = get_hybrid_kernel_matrix(classical, quantum, alpha=1.0)
        np.testing.assert_array_almost_equal(hybrid, quantum)

    def test_hybrid_kernel_alpha_half(self):
        """Test alpha=0.5 gives equal mix."""
        from qve.core import get_hybrid_kernel_matrix

        classical = np.array([[1, 0.5], [0.5, 1]])
        quantum = np.array([[1, 0.3], [0.3, 1]])

        hybrid = get_hybrid_kernel_matrix(classical, quantum, alpha=0.5)
        expected = 0.5 * quantum + 0.5 * classical
        np.testing.assert_array_almost_equal(hybrid, expected)

    def test_hybrid_kernel_preserves_shape(self):
        """Test hybrid kernel preserves input shape."""
        from qve.core import get_hybrid_kernel_matrix

        classical = np.random.randn(10, 10)
        quantum = np.random.randn(10, 10)

        hybrid = get_hybrid_kernel_matrix(classical, quantum, alpha=0.7)
        assert hybrid.shape == classical.shape

    def test_hybrid_kernel_non_square(self):
        """Test hybrid kernel with non-square matrices (test kernel)."""
        from qve.core import get_hybrid_kernel_matrix

        classical = np.random.randn(5, 10)  # 5 test, 10 train
        quantum = np.random.randn(5, 10)

        hybrid = get_hybrid_kernel_matrix(classical, quantum, alpha=0.3)
        assert hybrid.shape == (5, 10)


class TestSinCosCaching:
    """Test sin_cos cached function."""

    def test_sin_cos_values(self):
        """Test sin_cos returns correct values."""
        from qve.core import sin_cos

        s, c = sin_cos(0.0)
        np.testing.assert_almost_equal(s, 0.0)
        np.testing.assert_almost_equal(c, 1.0)

        s, c = sin_cos(np.pi / 2)
        np.testing.assert_almost_equal(s, 1.0)
        np.testing.assert_almost_equal(c, 0.0, decimal=10)


class TestGetFromD1D2:
    """Test get_from_d1 and get_from_d2 cached functions."""

    def test_get_from_d1_structure(self):
        """Test get_from_d1 returns correct structure."""
        from qve.core import get_from_d1

        s, c, z_g = get_from_d1(0.5)
        assert isinstance(s, float)
        assert isinstance(c, float)
        assert len(z_g) == 2
        assert len(z_g[0]) == 2

    def test_get_from_d2_structure(self):
        """Test get_from_d2 returns correct structure."""
        from qve.core import get_from_d2

        s, c, z_gd = get_from_d2(0.5)
        assert isinstance(s, float)
        assert isinstance(c, float)
        assert len(z_gd) == 2
        assert len(z_gd[0]) == 2


class TestBackwardsCompatibility:
    """Test that existing functionality is not broken."""

    def test_original_exports_available(self):
        """Test retained qve exports are still available."""
        from qve import (
            build_qsvm_qc,
            data_prepare_cv,
            get_kernel_matrix,
            make_bsp,
            set_seed,
        )

        # All should be importable
        assert all(
            [set_seed, data_prepare_cv, make_bsp, build_qsvm_qc, get_kernel_matrix]
        )

    def test_new_hybrid_export_available(self):
        """Test new hybrid function is exported."""
        from qve import get_hybrid_kernel_matrix

        assert get_hybrid_kernel_matrix is not None

    def test_get_kernel_matrix_unchanged_behavior(self):
        """Test get_kernel_matrix behaves the same as before."""
        from qve.core import get_kernel_matrix

        # This tests the exact behavior expected
        # Use upper triangle OFF-DIAGONAL indices only (1-indexed)
        # Diagonal is automatically added as 1.0 by get_kernel_matrix
        data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        # Upper triangle off-diagonal indices (1-indexed as in original)
        indices = [(1, 2), (1, 3), (2, 3)]
        amp_values = [0.8, 0.6, 0.7]
        amp_data = [amp_values]

        kernel = get_kernel_matrix(data, data, amp_data, indices, mode="train")

        # Check specific values
        assert kernel[0, 0] == 1.0  # Diagonal (added by np.diag(ones))
        assert kernel[0, 1] == 0.8  # Off-diagonal
        assert kernel[1, 0] == 0.8  # Symmetric
        assert kernel[0, 2] == 0.6  # Off-diagonal
        assert kernel[1, 2] == 0.7  # Off-diagonal
        assert kernel.shape == (3, 3)


if __name__ == "__main__":
    print("Running qve/core.py tests...")
    print("=" * 60)
    pytest.main([__file__, "-v"])
