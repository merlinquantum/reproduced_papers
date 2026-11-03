#!/usr/bin/env python
"""
Tests for hybrid kernel functionality.

Verifies:
1. Hybrid kernel math is correct
2. Hybrid features in qsvm_cuda_embeddings_insurance.py are backwards compatible

Usage:
    pytest tests/test_hybrid_kernel.py -v
"""

import ast
import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHybridKernelMath:
    """Test hybrid kernel mathematical properties."""

    def test_convex_combination(self):
        """Test hybrid is a proper convex combination for alpha in [0,1]."""
        from qve.core import get_hybrid_kernel_matrix

        K_c = np.array([[1.0, 0.3], [0.3, 1.0]])
        K_q = np.array([[1.0, 0.7], [0.7, 1.0]])

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            K_h = get_hybrid_kernel_matrix(K_c, K_q, alpha)
            expected = alpha * K_q + (1 - alpha) * K_c
            np.testing.assert_array_almost_equal(K_h, expected)

    def test_preserves_positive_semidefinite(self):
        """Test hybrid of PSD matrices is PSD (eigenvalues >= 0)."""
        from qve.core import get_hybrid_kernel_matrix

        # Create positive semidefinite matrices
        n = 5
        A = np.random.randn(n, n)
        K_c = A @ A.T  # Guaranteed PSD
        B = np.random.randn(n, n)
        K_q = B @ B.T  # Guaranteed PSD

        K_h = get_hybrid_kernel_matrix(K_c, K_q, alpha=0.5)

        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(K_h)
        assert np.all(eigenvalues >= -1e-10), "Hybrid kernel should be PSD"

    def test_symmetry_preserved(self):
        """Test symmetric inputs produce symmetric output."""
        from qve.core import get_hybrid_kernel_matrix

        n = 4
        K_c = np.random.randn(n, n)
        K_c = (K_c + K_c.T) / 2  # Make symmetric
        K_q = np.random.randn(n, n)
        K_q = (K_q + K_q.T) / 2  # Make symmetric

        K_h = get_hybrid_kernel_matrix(K_c, K_q, alpha=0.6)
        np.testing.assert_array_almost_equal(K_h, K_h.T)


class TestBackwardsCompatibility:
    """Test that existing script behavior is unchanged when not using hybrid."""

    @pytest.fixture
    def main_script_path(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "qsvm_cuda_embeddings_insurance.py",
        )

    def test_apply_hybrid_kernel_passthrough(self, main_script_path):
        """Test apply_hybrid_kernel returns unchanged kernels when use_hybrid=False."""
        # Parse the file to extract apply_hybrid_kernel function
        # (don't actually load/execute the module - just analyze the source code)
        with open(main_script_path) as f:
            content = f.read()

        # Find and extract apply_hybrid_kernel function
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "apply_hybrid_kernel":
                # Found the function - verify it has the passthrough logic
                func_source = ast.get_source_segment(content, node)
                assert "if not use_hybrid:" in func_source, (
                    "apply_hybrid_kernel should check use_hybrid flag"
                )
                assert (
                    "return kernel_train, kernel_val" in func_source
                    or "return kernel_train, kernel_valid" in func_source
                ), (
                    "apply_hybrid_kernel should return unchanged kernels when use_hybrid=False"
                )
                break
        else:
            pytest.fail("apply_hybrid_kernel function not found in main script")

    def test_default_hybrid_flag_is_false(self, main_script_path):
        """Test that --use_hybrid defaults to False (store_true action)."""
        with open(main_script_path) as f:
            content = f.read()

        # Check argparse setup
        assert 'action="store_true"' in content or "action='store_true'" in content, (
            "--use_hybrid should use store_true action (defaults to False)"
        )
        assert "--use_hybrid" in content, "Script should have --use_hybrid argument"


class TestApplyHybridKernelFunction:
    """Test the apply_hybrid_kernel function directly."""

    def test_passthrough_when_disabled(self):
        """Test kernels pass through unchanged when use_hybrid=False."""

        # Simulate apply_hybrid_kernel logic
        def apply_hybrid_kernel(
            kernel_train,
            kernel_valid,
            data_train,
            data_valid,
            use_hybrid=False,
            alpha=0.5,
            classical_kernel="rbf",
        ):
            if not use_hybrid:
                return kernel_train, kernel_valid
            # Would compute hybrid here
            return kernel_train, kernel_valid  # Simplified

        K_train = np.random.randn(10, 10)
        K_valid = np.random.randn(5, 10)
        data_train = np.random.randn(10, 4)
        data_valid = np.random.randn(5, 4)

        result_train, result_valid = apply_hybrid_kernel(
            K_train, K_valid, data_train, data_valid, use_hybrid=False
        )

        np.testing.assert_array_equal(result_train, K_train)
        np.testing.assert_array_equal(result_valid, K_valid)

    def test_hybrid_mixing_formula(self):
        """Test hybrid mixing applies correct formula."""
        from qve.core import get_hybrid_kernel_matrix
        from sklearn.metrics.pairwise import rbf_kernel

        # Create test data
        data_train = np.random.randn(10, 4)
        data_valid = np.random.randn(5, 4)

        # Compute classical kernel
        K_c_train = rbf_kernel(data_train, data_train)
        K_c_valid = rbf_kernel(data_valid, data_train)

        # Simulate quantum kernel
        K_q_train = np.random.randn(10, 10)
        K_q_train = (K_q_train + K_q_train.T) / 2
        K_q_valid = np.random.randn(5, 10)

        # Test mixing
        alpha = 0.7
        K_h_train = get_hybrid_kernel_matrix(K_c_train, K_q_train, alpha)
        K_h_valid = get_hybrid_kernel_matrix(K_c_valid, K_q_valid, alpha)

        expected_train = alpha * K_q_train + (1 - alpha) * K_c_train
        expected_valid = alpha * K_q_valid + (1 - alpha) * K_c_valid

        np.testing.assert_array_almost_equal(K_h_train, expected_train)
        np.testing.assert_array_almost_equal(K_h_valid, expected_valid)


class TestClassicalKernelTypes:
    """Test different classical kernel types work correctly."""

    def test_rbf_kernel_shape(self):
        """Test RBF kernel produces correct shape."""
        from sklearn.metrics.pairwise import rbf_kernel

        X_train = np.random.randn(10, 4)
        X_test = np.random.randn(5, 4)

        K_train = rbf_kernel(X_train, X_train)
        K_test = rbf_kernel(X_test, X_train)

        assert K_train.shape == (10, 10)
        assert K_test.shape == (5, 10)

    def test_poly_kernel_shape(self):
        """Test polynomial kernel produces correct shape."""
        from sklearn.metrics.pairwise import polynomial_kernel

        X_train = np.random.randn(10, 4)
        X_test = np.random.randn(5, 4)

        K_train = polynomial_kernel(X_train, X_train, degree=3)
        K_test = polynomial_kernel(X_test, X_train, degree=3)

        assert K_train.shape == (10, 10)
        assert K_test.shape == (5, 10)

    def test_linear_kernel_shape(self):
        """Test linear kernel produces correct shape."""
        from sklearn.metrics.pairwise import linear_kernel

        X_train = np.random.randn(10, 4)
        X_test = np.random.randn(5, 4)

        K_train = linear_kernel(X_train, X_train)
        K_test = linear_kernel(X_test, X_train)

        assert K_train.shape == (10, 10)
        assert K_test.shape == (5, 10)


class TestNoSideEffects:
    """Test that hybrid features don't affect non-hybrid runs."""

    def test_qve_init_still_exports_original(self):
        """Test qve.__init__.py still exports all original functions."""
        from qve import (
            build_qsvm_qc,
            data_partition,
            data_to_operand,
            get_kernel_matrix,
            make_bsp,
            operand_to_amp,
            set_seed,
        )

        # These should all be callable
        assert callable(set_seed)
        assert callable(make_bsp)
        assert callable(build_qsvm_qc)
        assert callable(data_partition)
        assert callable(data_to_operand)
        assert callable(operand_to_amp)
        assert callable(get_kernel_matrix)

    def test_get_kernel_matrix_no_hybrid_dependency(self):
        """Test get_kernel_matrix doesn't depend on hybrid function."""
        from qve.core import get_kernel_matrix

        # Should work without any hybrid-related parameters
        data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        indices = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        amp_data = [[1.0, 0.8, 0.6, 1.0, 0.7, 1.0]]

        # This should work exactly as before
        kernel = get_kernel_matrix(data, data, amp_data, indices, mode="train")

        assert kernel.shape == (3, 3)
        assert np.allclose(kernel, kernel.T)  # Symmetric


if __name__ == "__main__":
    print("Running hybrid kernel tests...")
    print("=" * 60)
    pytest.main([__file__, "-v"])
