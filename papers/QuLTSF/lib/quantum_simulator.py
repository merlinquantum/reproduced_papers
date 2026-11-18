"""Batched statevector simulation of the QuLTSF hidden quantum layer.

The upstream model runs this circuit through PennyLane ``default.qubit``
(``models/QuLTSF.py``)::

    qml.AmplitudeEmbedding(features, wires=range(N), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(N))   # weights: (K, N, 3)
    return [qml.expval(qml.PauliZ(i)) for i in range(N)]

This module computes the same function directly in PyTorch so that a batch of
``B * 21`` circuits runs as a handful of tensor operations on CPU or GPU, with
exact gradients by backpropagation (equivalent to PennyLane's
``diff_method="best"`` on a simulator). ``tests/test_quantum_simulator.py``
checks it against PennyLane.

Conventions (PennyLane's):
- basis index ``b`` of the ``2**N`` amplitudes is big-endian: wire 0 is the
  most significant bit;
- ``Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)``;
- layer ``l`` applies ``Rot`` on every wire, then ``CNOT(i, (i + r_l) % N)``
  for ``i = 0..N-1`` in order, with range ``r_l = (l mod (N - 1)) + 1``;
- ``<Z_i> = sum_b |psi_b|^2 (1 - 2 bit_i(b))``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

CIRCUIT_MODES = ("trainable", "frozen", "identity")


def _wire_bits(num_qubits: int) -> torch.Tensor:
    """Return ``bits[b, w]``, the value of wire ``w`` in basis state ``b``."""
    basis = torch.arange(2**num_qubits)
    shifts = torch.arange(num_qubits - 1, -1, -1)
    return (basis[:, None] >> shifts[None, :]) & 1


def default_ranges(num_qubits: int, num_layers: int) -> list[int]:
    """Return PennyLane's default CNOT ranges for StronglyEntanglingLayers."""
    if num_qubits == 1:
        return [0] * num_layers
    return [(layer % (num_qubits - 1)) + 1 for layer in range(num_layers)]


def cnot_ring_permutation(num_qubits: int, cnot_range: int) -> torch.Tensor:
    """Return the gather index of one layer's CNOT ring.

    After the ring, ``new_state[b] = old_state[index[b]]``. Each CNOT ``g`` is
    the involution ``f_g`` flipping the target bit when the control bit is set,
    so ``index = f_1(f_2(...f_k(b)))``: the gates are applied to the index in
    reverse order.

    Parameters
    ----------
    num_qubits : int
        Number of wires N.
    cnot_range : int
        Distance between control and target wires.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[2**N]``.
    """
    index = torch.arange(2**num_qubits)
    if num_qubits == 1:
        return index
    for control in reversed(range(num_qubits)):
        target = (control + cnot_range) % num_qubits
        control_bit = (index >> (num_qubits - 1 - control)) & 1
        index = index ^ (control_bit << (num_qubits - 1 - target))
    return index


def rot_matrices(weights: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """Return ``Rot(phi, theta, omega)`` matrices for a ``[..., 3]`` weight tensor.

    Parameters
    ----------
    weights : torch.Tensor
        Angles ``(phi, theta, omega)`` in the last dimension.
    complex_dtype : torch.dtype
        Complex dtype of the result.

    Returns
    -------
    torch.Tensor
        Unitaries of shape ``[..., 2, 2]``.
    """
    phi, theta, omega = weights.unbind(-1)
    cos = torch.cos(theta / 2).to(complex_dtype)
    sin = torch.sin(theta / 2).to(complex_dtype)
    plus = torch.exp(-0.5j * (phi + omega).to(complex_dtype))
    minus = torch.exp(-0.5j * (phi - omega).to(complex_dtype))
    row0 = torch.stack([plus * cos, -minus.conj() * sin], dim=-1)
    row1 = torch.stack([minus * sin, plus.conj() * cos], dim=-1)
    return torch.stack([row0, row1], dim=-2)


class AmplitudeSELayer(nn.Module):
    """Amplitude embedding, StronglyEntanglingLayers and PauliZ readout.

    Parameters
    ----------
    num_qubits : int
        Number of qubits N; the layer takes ``2**N`` features.
    num_layers : int
        Number of entangling layers K.
    circuit : str
        ``"trainable"`` (the paper), ``"frozen"`` (random weights drawn at
        init, not trained) or ``"identity"`` (no gates: PauliZ read directly
        from the embedded state; a classical control). Default value is
        ``"trainable"``.
    dtype : torch.dtype
        Real dtype of inputs and outputs; the state uses the matching complex
        dtype. Default value is ``torch.float32``.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        circuit: str = "trainable",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if circuit not in CIRCUIT_MODES:
            raise ValueError(f"circuit must be one of {CIRCUIT_MODES}, got {circuit!r}")
        self.num_qubits = int(num_qubits)
        self.num_layers = int(num_layers)
        self.circuit = circuit
        self.real_dtype = dtype
        self.complex_dtype = (
            torch.complex128 if dtype == torch.float64 else torch.complex64
        )

        # Same shape and initialisation as PennyLane's TorchLayer default
        # (uniform on [0, 2 pi), drawn in float32 and then cast), drawn even for
        # "identity" so that the surrounding classical layers get the same
        # random draws in every mode.
        weights = torch.empty(self.num_layers, self.num_qubits, 3, dtype=torch.float32)
        nn.init.uniform_(weights, b=2 * math.pi)
        weights = weights.to(dtype)
        if circuit == "trainable":
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer("weights", weights)

        bits = _wire_bits(self.num_qubits)
        self.register_buffer("z_signs", (1 - 2 * bits).to(dtype), persistent=False)
        permutations = torch.stack(
            [
                cnot_ring_permutation(self.num_qubits, cnot_range)
                for cnot_range in default_ranges(self.num_qubits, self.num_layers)
            ]
        )
        self.register_buffer("cnot_permutations", permutations, persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``<Z_i>`` for each row of ``features``.

        Parameters
        ----------
        features : torch.Tensor
            Real tensor ``[batch, 2**N]``; each row is L2-normalised as in
            ``AmplitudeEmbedding(normalize=True)``.

        Returns
        -------
        torch.Tensor
            Real tensor ``[batch, N]``.
        """
        dim = 2**self.num_qubits
        if features.shape[-1] != dim:
            raise ValueError(f"Expected {dim} features, got {features.shape[-1]}")
        amplitudes = features / torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        if self.circuit == "identity":
            return (amplitudes * amplitudes) @ self.z_signs

        batch = amplitudes.shape[0]
        state = amplitudes.to(self.complex_dtype)
        unitaries = rot_matrices(self.weights, self.complex_dtype)
        for layer in range(self.num_layers):
            for wire in range(self.num_qubits):
                view = state.reshape(
                    batch, 2**wire, 2, 2 ** (self.num_qubits - wire - 1)
                )
                view = torch.einsum("ab,pibj->piaj", unitaries[layer, wire], view)
                state = view.reshape(batch, dim)
            state = state[:, self.cnot_permutations[layer]]
        probabilities = state.real**2 + state.imag**2
        return probabilities @ self.z_signs


__all__ = [
    "AmplitudeSELayer",
    "CIRCUIT_MODES",
    "cnot_ring_permutation",
    "default_ranges",
    "rot_matrices",
]
