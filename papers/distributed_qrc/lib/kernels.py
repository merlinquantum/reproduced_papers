"""Fidelity ("quantum") kernel of the paper's ridge output layer (Sec. II-B, Fig. 3, Eqs. 6-8).

Feature map ``Phi`` on n qubits: ``B = 2`` structural blocks, each made of ``L``
layers of ``[H on every qubit, then P(2 x_i) on qubit i]``, with
``L = ceil(d / n)`` for d input features (Fig. 3c). Feature ``l * n + q`` drives
qubit q in layer l; a missing feature (d not a multiple of n) leaves ``P(0) = I``.
Both blocks re-encode the same features.

The circuit has no entangling gate, so ``|Phi(x)>`` is a product of single-qubit
states and the fidelity kernel factorizes exactly:

    K(x, x') = prod_q |<phi_q(x)|phi_q(x')>|^2 = prod_q (1 + b_q(x) . b_q(x')) / 2,

with ``b_q`` the Bloch vector of qubit q (our derivation; the paper gives only the
circuit and Eq. 6). This is the exact statevector value of Eq. 6 (checked against
Qiskit in ``tests/test_kernels.py``); it costs O(d) per point for the Bloch vectors,
then O(n) per pair.

Noisy kernel (compute-uncompute circuit ``Phi(x')^dagger Phi(x)`` measured in Z,
probability of all zeros): with the depolarizing model of :mod:`lib.noise`, each
of the ``2 * B * L`` H gates per qubit shrinks the Bloch vector by
``1 - 2 r_1q`` (depolarizing commutes with rotations; P gates are virtual), and a
readout flip p maps ``z`` to ``(1 - 2p) z``. Hence

    K_noisy(x, x') = prod_q (1 + c (b_q(x) . b_q(x'))) / 2,
    c = (1 - 2p) (1 - 2 r_1q)^(2 B L).

Finite shots (optional, gap diagnosis): each kernel entry K is replaced by an
estimate from S shots of the compute-uncompute circuit, using the normal
approximation of the binomial, ``K + sqrt(K (1 - K) / S) * N(0, 1)`` clipped to
[0, 1]. The training Gram matrix stays symmetric (one estimate per pair).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .noise import NATIVE_1Q_PER_H, DeviceNoise

KERNEL_BLOCKS = 2

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def n_layers(n_features: int, n_qubits: int) -> int:
    return math.ceil(n_features / n_qubits)


def bloch_vectors(
    features: np.ndarray, n_qubits: int, blocks: int = KERNEL_BLOCKS
) -> np.ndarray:
    """Bloch vectors ``(B, n_qubits, 3)`` of each qubit after the Fig. 3 feature map."""

    n_samples, n_features = features.shape
    layers = n_layers(n_features, n_qubits)
    padded = np.zeros((n_samples, layers * n_qubits))
    padded[:, :n_features] = features
    angles = 2.0 * padded.reshape(n_samples, layers, n_qubits)  # P(2 x)

    psi = np.zeros((n_samples, n_qubits, 2), dtype=np.complex128)
    psi[..., 0] = 1.0
    for _ in range(blocks):
        for layer in range(layers):
            psi = psi @ _H.T
            psi[..., 1] *= np.exp(1j * angles[:, layer, :])
    a, b = psi[..., 0], psi[..., 1]
    cross = np.conj(a) * b
    return np.stack(
        [2 * cross.real, 2 * cross.imag, np.abs(a) ** 2 - np.abs(b) ** 2], axis=-1
    )


def contraction_factor(
    noise: DeviceNoise, n_features: int, n_qubits: int, blocks: int = KERNEL_BLOCKS
) -> float:
    """``c`` of the module docstring; 1 for the ideal kernel."""

    h_gates = 2 * blocks * n_layers(n_features, n_qubits) * NATIVE_1Q_PER_H
    return (1 - 2 * noise.readout_error) * noise.bloch_shrink_1q**h_gates


@dataclass
class ShotNoise:
    """Finite-shot estimation of kernel entries."""

    shots: int
    generator: torch.Generator

    def sample(self, block: torch.Tensor) -> torch.Tensor:
        p = block.clamp(0.0, 1.0)
        noise = torch.randn(
            block.shape,
            generator=self.generator,
            dtype=block.dtype,
            device=block.device,
        )
        return (p + torch.sqrt(p * (1.0 - p) / self.shots) * noise).clamp_(0.0, 1.0)


def kernel_block(rows: torch.Tensor, cols: torch.Tensor, factor: float) -> torch.Tensor:
    """Kernel between two sets of Bloch vectors ``(n_a, q, 3)`` and ``(n_b, q, 3)``."""

    block = torch.ones(
        rows.shape[0], cols.shape[0], dtype=rows.dtype, device=rows.device
    )
    for q in range(rows.shape[1]):
        block *= 0.5 * (1.0 + factor * (rows[:, q, :] @ cols[:, q, :].T))
    return block


def gram_matrix(
    bloch: torch.Tensor,
    factor: float,
    chunk_size: int = 4096,
    shot_noise: ShotNoise | None = None,
) -> torch.Tensor:
    """Symmetric training Gram matrix, built in row chunks."""

    n = bloch.shape[0]
    gram = torch.empty(n, n, dtype=bloch.dtype, device=bloch.device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = kernel_block(bloch[start:stop], bloch, factor)
        if shot_noise is not None:
            block = shot_noise.sample(block)
        gram[start:stop] = block
        if shot_noise is not None:
            # One estimate per pair: mirror the already-sampled upper triangle.
            gram[start:stop, :start] = gram[:start, start:stop].T
            diagonal = gram[start:stop, start:stop]
            diagonal.copy_(torch.triu(diagonal) + torch.triu(diagonal, 1).T)
    return gram


def kernel_times_vector(
    rows: torch.Tensor,
    cols: torch.Tensor,
    factor: float,
    vector: torch.Tensor,
    chunk_size: int = 4096,
    shot_noise: ShotNoise | None = None,
) -> torch.Tensor:
    """``K(rows, cols) @ vector`` without materializing the full cross kernel."""

    out = torch.empty(rows.shape[0], dtype=rows.dtype, device=rows.device)
    for start in range(0, rows.shape[0], chunk_size):
        block = kernel_block(rows[start : start + chunk_size], cols, factor)
        if shot_noise is not None:
            block = shot_noise.sample(block)
        out[start : start + chunk_size] = block @ vector
    return out
