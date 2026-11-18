"""Exact small-size references for the MMD estimator (verification V1, V2).

Everything here is brute force and only meant for m, n small enough to
enumerate all outputs and all Walsh operators (m <= ~12, n <= ~4).
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def permanent(A: np.ndarray) -> complex:
    """Permanent of a square matrix by Ryser's formula, O(2^n n^2)."""
    n = A.shape[0]
    if n == 0:
        return 1.0 + 0j
    total = 0j
    for subset_size in range(1, n + 1):
        sign = (-1) ** subset_size
        for columns in itertools.combinations(range(n), subset_size):
            total += sign * np.prod(A[:, columns].sum(axis=1))
    return (-1) ** n * total


def fock_states(m: int, n: int) -> list[tuple[int, ...]]:
    """All occupations of n photons in m modes (collisions included)."""
    states = []
    for positions in itertools.combinations_with_replacement(range(m), n):
        occupation = [0] * m
        for mode in positions:
            occupation[mode] += 1
        states.append(tuple(occupation))
    return states


def output_distribution(
    U: np.ndarray, input_modes, distinguishable: bool = False
) -> dict:
    """Exact output distribution {occupation: probability} for single photons in ``input_modes``.

    Indistinguishable photons: |Perm(U[rows, S])|^2 / prod(t!) (paper Eq. (1)).
    Distinguishable photons: Perm(|U[rows, S]|^2) / prod(t!).
    """
    U = np.asarray(U)
    m, n = U.shape[0], len(input_modes)
    distribution = {}
    for occupation in fock_states(m, n):
        rows = [mode for mode, count in enumerate(occupation) for _ in range(count)]
        sub = U[np.ix_(rows, list(input_modes))]
        norm = math.prod(math.factorial(c) for c in occupation)
        if distinguishable:
            prob = permanent(np.abs(sub) ** 2).real / norm
        else:
            prob = abs(permanent(sub)) ** 2 / norm
        distribution[occupation] = prob
    return distribution


def empirical_distribution(X: np.ndarray) -> dict:
    """Empirical distribution {occupation: frequency} of a sample array."""
    states, counts = np.unique(
        np.asarray(X, dtype=np.int64), axis=0, return_counts=True
    )
    return {
        tuple(int(v) for v in s): c / counts.sum()
        for s, c in zip(states, counts, strict=True)
    }


def kernel_mmd(p: dict, q: dict, kernel) -> float:
    """MMD^2 = sum_{x,y} (p - q)(x) (p - q)(y) K(x, y) between two finite distributions."""
    support = sorted(set(p) | set(q))
    diff = np.array([p.get(s, 0.0) - q.get(s, 0.0) for s in support])
    points = np.array(support, dtype=np.float64)
    gram = np.array([[kernel(a, b) for b in points] for a in points])
    return float(diff @ gram @ diff)


def gaussian_kernel(sigma: float):
    """Gaussian kernel exp(-||x - y||^2 / (2 sigma^2)) as a callable."""
    return lambda a, b: math.exp(-float(np.sum((a - b) ** 2)) / (2.0 * sigma**2))


def walsh_mmd(p: dict, q: dict, operator_probability) -> float:
    """MMD^2 = sum_k P(k) (<W_k>_p - <W_k>_q)^2, enumerating all k in {0,1}^m.

    ``operator_probability(k)`` returns P(k). On collision-free distributions
    this equals ``kernel_mmd`` for the kernel whose Walsh transform is P
    (Prop. 1-2); with collisions it is the MMD of the parity features x mod 2.
    """
    m = len(next(iter(p)))
    total = 0.0
    for bits in itertools.product((0, 1), repeat=m):
        k = np.array(bits)
        weight = operator_probability(k)
        if weight == 0.0:
            continue
        ep = sum(prob * (-1) ** int(k @ np.array(s)) for s, prob in p.items())
        eq = sum(prob * (-1) ** int(k @ np.array(s)) for s, prob in q.items())
        total += weight * (ep - eq) ** 2
    return float(total)


def gaussian_operator_probability(sigma: float):
    """P_sigma(k) = (1 - p)^{m - |k|} p^{|k|}, p = (1 - exp(-1/(2 sigma^2))) / 2."""
    p = (1.0 - math.exp(-1.0 / (2.0 * sigma**2))) / 2.0
    return lambda k: (1.0 - p) ** (len(k) - int(k.sum())) * p ** int(k.sum())


def merlin_output_distribution(U: np.ndarray, input_modes) -> dict:
    """Exact Fock-space output distribution from the MerLin fork (independent oracle).

    Uses ``merlin.QuantumLayer`` on a fixed ``Unitary`` component with the
    FOCK computation space, in float64.
    """
    import merlin as ml
    import perceval as pcvl
    import torch

    m = U.shape[0]
    input_state = [1 if mode in set(input_modes) else 0 for mode in range(m)]
    circuit = pcvl.Circuit(m).add(
        0, pcvl.components.Unitary(pcvl.Matrix(np.asarray(U)))
    )
    layer = ml.QuantumLayer(
        circuit=circuit,
        input_state=input_state,
        measurement_strategy=ml.MeasurementStrategy.probs(ml.ComputationSpace.FOCK),
        dtype=torch.float64,
    )
    with torch.no_grad():
        probabilities = layer().reshape(-1).cpu().numpy()
    return {
        tuple(int(v) for v in key): float(prob)
        for key, prob in zip(layer.output_keys, probabilities, strict=True)
    }
