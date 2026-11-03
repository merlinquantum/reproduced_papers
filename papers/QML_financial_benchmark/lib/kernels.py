"""Quantum fidelity kernels for QSVR (§III-E2, Eq. 16-19, Fig. 9).

Both feature maps are simulated exactly; kernels are statevector overlaps
``|<psi(x_i)|psi(x_j)>|^(2 beta)``.

A data-independent unitary applied after the last data encoding cancels in a
fidelity kernel. Two consequences:

- Eq. 18's fixed random unitary ``U_rnd`` never changes the amplitude kernel,
  which equals ``|<x_i/|x_i|, x_j/|x_j|>|^(2 beta)`` exactly
  (``tests/test_kernels.py`` checks this against an explicit unitary).
- For the depth ``L`` of Eq. 17 to matter, the data must be re-uploaded
  between the ``U_rot`` blocks, as ``(U_rot R_Y(x))^L`` implies. This module
  prepares ``R_Y(x) (U_rot R_Y(x))^L |0>``, so ``L = 0`` is the product-state
  kernel ``prod_k cos^2((x_ik - x_jk) / 2)``. ``U_rot`` is RY-RZ on every
  qubit with fixed random angles followed by CZ entanglers, drawn once from
  ``rotation_seed``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import statevector as sv
from .models import angle_encode


def angle_feature_states(
    features: np.ndarray,
    layers: int,
    topology: str,
    angle_scale: float,
    rotation_seed: int,
) -> torch.Tensor:
    """Statevectors of the angle-encoded feature map, shape ``(n, 2**d)``, complex128.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n, d)`` inputs scaled to ``[0, 1]``; one qubit per feature.
    layers : int
        Number of ``U_rot`` blocks, each followed by re-uploading ``R_Y(x)``.
    topology : str
        CZ pattern inside ``U_rot``: ``"ring"`` (the paper's choice) or ``"chain"``.
    angle_scale : float
        Feature-to-angle multiplier.
    rotation_seed : int
        Seed of the fixed random ``U_rot`` angles.
    """
    angles = torch.as_tensor(features, dtype=torch.float64) * angle_scale
    n_qubits = angles.shape[1]
    generator = torch.Generator().manual_seed(rotation_seed)
    rotation_angles = (
        torch.rand(layers, n_qubits, 2, generator=generator, dtype=torch.float64)
        * 2
        * math.pi
    )
    phase = sv.cz_phase(
        n_qubits, sv.entangling_pairs(n_qubits, topology), torch.float64, "cpu"
    )
    state = angle_encode(angles)
    for layer in range(layers):
        for qubit in range(n_qubits):
            ry_angle, rz_angle = rotation_angles[layer, qubit]
            state = sv.apply_single_qubit(
                state, sv.rz_matrix(rz_angle) @ sv.ry_matrix(ry_angle), qubit
            )
        state = state * phase
        for qubit in range(n_qubits):
            state = sv.apply_single_qubit(state, sv.ry_matrix(angles[:, qubit]), qubit)
    return sv.flatten(state)


def fidelity_kernel(
    states_a: torch.Tensor, states_b: torch.Tensor, beta: int = 1
) -> np.ndarray:
    """``|<a_i|b_j>|^(2 beta)`` for all pairs (Eq. 16 and Eq. 19)."""
    overlaps = states_a @ states_b.conj().transpose(0, 1)
    return (overlaps.abs() ** (2 * beta)).numpy()


def amplitude_kernel(
    features_a: np.ndarray, features_b: np.ndarray, beta: int, eps: float = 1e-12
) -> np.ndarray:
    """Amplitude-encoding fidelity kernel ``|<x_i_hat, x_j_hat>|^(2 beta)`` (Eq. 18-19).

    Inputs are zero-padded to the next power of two, as amplitude encoding on
    ``ceil(log2 d)`` qubits requires.
    """
    width = 2 ** math.ceil(math.log2(features_a.shape[1]))
    pad = ((0, 0), (0, width - features_a.shape[1]))
    a, b = np.pad(features_a, pad), np.pad(features_b, pad)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.abs(a @ b.T) ** (2 * beta)
