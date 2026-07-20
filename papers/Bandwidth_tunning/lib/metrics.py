import merlin
import torch
from .kernels import FidelityKernel, ProjectedFidelityKernel


def matrix_sqrt(A):
    """Compute the matrix square root of a symmetric positive semi-definite matrix."""
    # Eigendecomposition (A = V * L * V^T)
    L, V = torch.linalg.eigh(A)
    # Numerical precision can sometimes produce very slightly negative eigenvalues
    # (e.g. -1e-15). Clamp them to zero.
    L = torch.clamp(L, min=0.0)
    # Reconstruct: V * sqrt(L) * V^T
    return V @ torch.diag(torch.sqrt(L)) @ V.T


def calculate_g(K1, K2):
    """Compute g(K1, K2) with lambda = 0."""
    # 1. Compute the inverse of K2
    # Using pseudo-inverse (pinv) is safer than inv()
    # because a kernel matrix (K2) can have a near-zero determinant.
    K2_inv = torch.linalg.pinv(K2)
    # 2. Compute the matrix square root of K1
    sqrt_K1 = matrix_sqrt(K1)
    # 3. Central matrix product: sqrt(K1) @ K2^-1 @ sqrt(K1)
    inner_matrix = sqrt_K1 @ K2_inv @ sqrt_K1
    # 4. Spectral norm (largest singular value, equivalent to ord=2)
    spectral_norm = torch.linalg.matrix_norm(inner_matrix, ord=2)
    # 5. Final square root of the formula
    g = torch.sqrt(spectral_norm)
    return g


def calculate_eta_max(K):
    """
    Compute the normalized largest eigenvalue of the kernel matrix K.
    The normalization is by the trace (sum of eigenvalues), matching the 
    trace-normalized kernel Gram matrices defined in the paper where tr(K) = N.
    """
    L, V = torch.linalg.eigh(K)
    # Clamp negative eigenvalues to zero (numerical precision artifacts)
    L = torch.clamp(L, min=0.0)
    # Normalize by trace
    trace_K = torch.sum(L)
    eta_max = L[-1] / trace_K if trace_K > 0 else L[-1]
    return eta_max


def calculate_kernel_distance_F(K_C, K_Q):
    """
    Compute F(K_C, K_Q), the relative Frobenius distance between two kernel matrices.
    K_C : PyTorch tensor representing the classical kernel matrix.
    K_Q : PyTorch tensor representing the quantum kernel matrix.
    """
    # 1. Compute the numerator: Frobenius norm of the difference
    numerateur = torch.linalg.matrix_norm(K_C - K_Q, ord="fro")

    # 2. Compute the denominator: Frobenius norm of K_Q
    denominateur = torch.linalg.matrix_norm(K_Q, ord="fro")

    # 3. Final ratio
    F = numerateur / denominateur

    return F


def RBF(X_train):
    # Compute the RBF kernel
    distances = torch.cdist(X_train, X_train, p=2)
    distances_carré = distances**2
    K_rbf = torch.exp(-distances_carré)
    return K_rbf


def RBF_2(X_train):
    # Compute the RBF kernel of order 2
    distances = torch.cdist(X_train, X_train, p=2)
    z = distances**2
    K_rbf_order_2 = 1.0 - z + 0.5 * (z**2)
    return K_rbf_order_2

def fidelity_kernel(feature_map, X_train, X_test = None):
    _fidelity_kernel = FidelityKernel(
        feature_map=feature_map,
        input_state=[
            1 - (i % 2) for i in range(X_train.shape[1] + 1)
        ],  # alternating photons for n_modes
        computation_space=merlin.ComputationSpace.FOCK,
    )

    if X_test is None:
        return _fidelity_kernel(X_train)
    else:
        return _fidelity_kernel(X_test, X_train)

def projected_fidelity_kernel(feature_map, X_train, X_test = None):
    _projected_fidelity_kernel = ProjectedFidelityKernel(
        feature_map=feature_map,
        input_state=[
            1 - (i % 2) for i in range(X_train.shape[1] + 1)
        ],  # alternating photons for n_modes
        computation_space=merlin.ComputationSpace.FOCK,
    )

    if X_test is None:
        return _projected_fidelity_kernel(X_train)
    else:
        return _projected_fidelity_kernel(X_test, X_train)
