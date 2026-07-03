"""Wasserstein diagnostic: Table I (input-space W1) and Fig 1 (trace dist vs W1).

Eq. (7): D_tr(rho+, rho-) <= kappa_F * W1(P+, P-).  Datasets whose class-conditional input
distributions are close in 1-Wasserstein distance admit little class separation from any
embedding in the family, so embedding search saturates.

Uses scipy's linear_sum_assignment for exact EMD without threading issues.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .circuits import zz_feature_states


def wasserstein1_l1(
    X_pos: np.ndarray, X_neg: np.ndarray, max_per_class: int = 500, seed: int = 0
) -> float:
    """Empirical 1-Wasserstein distance with L1 ground metric between two point clouds.

    Uses scipy's linear_sum_assignment for exact EMD (Hungarian algorithm). This is
    a polynomial algorithm. It is ok since all the points have the same points but different distances.
    The problem becomes an assignement problem.
    No threading issues, pure Python implementation.
    """
    rng = np.random.default_rng(seed)
    if len(X_pos) > max_per_class:
        X_pos = X_pos[rng.choice(len(X_pos), max_per_class, replace=False)]
    if len(X_neg) > max_per_class:
        X_neg = X_neg[rng.choice(len(X_neg), max_per_class, replace=False)]

    # Compute L1 cost matrix
    M = np.abs(X_pos[:, np.newaxis, :] - X_neg[np.newaxis, :, :]).sum(axis=2)

    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(M)

    # Compute transport cost
    transport_cost = M[row_ind, col_ind].sum()

    # Normalize by number of samples (average cost per sample)
    return float(transport_cost / max(len(X_pos), len(X_neg)))


def dataset_wasserstein(X: np.ndarray, y: np.ndarray, **kw) -> float:
    """Compute average pairwise 1-Wasserstein distance between all class pairs.

    For binary: returns W1(pos, neg).
    For multiclass: returns mean W1 over all (class_i, class_j) pairs.
    """
    classes = np.unique(y)

    # Binary case (backward compatible)
    if len(classes) == 2:
        y_pos, y_neg = classes[0], classes[1]
        return wasserstein1_l1(X[y == y_pos], X[y == y_neg], **kw)

    # Multiclass: average pairwise distances
    w1_distances = []
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1 :]:
            w1 = wasserstein1_l1(X[y == c1], X[y == c2], **kw)
            w1_distances.append(w1)

    return float(np.mean(w1_distances))


def trace_distance_states(states_pos: torch.Tensor, states_neg: torch.Tensor) -> float:
    """D_tr between two class-conditional mixture states rho+- = mean |psi><psi|.

    rho is (2^n, 2^n); D_tr = 0.5 * sum |eigvals(rho+ - rho-)|.
    """
    rho_p = (states_pos.t() @ states_pos.conj()) / states_pos.shape[0]
    rho_n = (states_neg.t() @ states_neg.conj()) / states_neg.shape[0]
    diff = rho_p - rho_n
    eig = torch.linalg.eigvalsh(diff)
    return 0.5 * eig.abs().sum().item()


def fig1_curve(
    n_qubits: int = 4,
    n_layers: int = 1,
    n_per_class: int = 60,
    shifts=None,
    sigma: float = 0.3,
    seed: int = 0,
):
    """Fig 1: trace distance vs input W1 for a ZZ feature map.

    Two Gaussian clouds in input space; one is fixed, the other translated to sweep W1.
    Returns (w1_values, trace_distances).
    """
    if shifts is None:
        shifts = np.linspace(0.0, 1.2, 9)
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, sigma, size=(n_per_class, n_qubits))
    w1s, tds = [], []
    for sh in shifts:
        Xp = base
        Xn = rng.normal(0.0, sigma, size=(n_per_class, n_qubits)) + sh
        # rescale jointly into a sensible angle range [0, 2pi]
        allX = np.vstack([Xp, Xn])
        lo, hi = allX.min(0), allX.max(0)
        rng_span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        Xp_s = (Xp - lo) / rng_span * (2 * np.pi)
        Xn_s = (Xn - lo) / rng_span * (2 * np.pi)
        w1 = wasserstein1_l1(Xp_s, Xn_s, seed=seed)
        sp = zz_feature_states(torch.tensor(Xp_s), n_qubits, n_layers=n_layers)
        sn = zz_feature_states(torch.tensor(Xn_s), n_qubits, n_layers=n_layers)
        td = trace_distance_states(sp, sn)
        w1s.append(w1)
        tds.append(td)
    return np.array(w1s), np.array(tds)
