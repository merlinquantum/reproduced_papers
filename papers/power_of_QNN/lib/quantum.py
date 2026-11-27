"""Batched statevector simulation of the paper's qubit models (Abbas et al., Sec. 2, App. A).

Every model maps an input batch ``x`` of shape ``(B, S)`` and a parameter batch
``theta`` of shape ``(B, d)`` to class probabilities ``(B, 2)``. Each row uses its
own parameter vector, so one backward pass of ``probabilities[:, k].sum()`` gives
the per-sample gradients that the Fisher information needs.

Conventions, chosen to match Qiskit 0.23 (the authors' framework, upstream
``effective_dimension`` v1.0.0):

- Qubit ``q`` is tensor axis ``q`` of the reshaped state ``(B, 2, ..., 2)``. Basis
  ordering differs from Qiskit's little-endian labels, but the parity readout is
  invariant under qubit relabelling of basis strings.
- ``ZFeatureMap``: per repetition, H on every qubit, then ``P(2 x_i)``.
- ``ZZFeatureMap``: per repetition, H on every qubit, ``P(2 x_i)``, then for every
  entangled pair ``(i, j)``: ``CX(i, j) P(2 (pi - x_i)(pi - x_j)) on j, CX(i, j)``.
  Qiskit's ``PauliFeatureMap`` uses ``alpha = 2`` and repeats the Hadamard layer in
  every repetition; App. A of the paper says the Hadamards are applied once.
  ``hadamard_every_rep`` selects between the two (V-00 decides which reproduces
  the cached arrays).
- ``RealAmplitudes(reps=D)``: RY layer, then ``D`` times (CX over the entangled
  pairs in order, RY layer), so ``d = (D + 1) S``. Qiskit 0.23 defaulted to
  ``entanglement="full"``: pairs ``(i, j)``, ``i < j``, in lexicographic order.
- Parity readout: class 0 = even Hamming weight of the measured bit string,
  class 1 = odd.
"""

from __future__ import annotations

import itertools
import math

import torch

FEATURE_MAPS = ("zz", "z", "ry_angle")
ENTANGLEMENTS = ("full", "linear")


def entangled_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    """Return the ordered ``(control, target)`` pairs of a Qiskit entanglement map.

    Parameters
    ----------
    n_qubits : int
        Number of qubits ``S``.
    entanglement : str
        ``"full"`` (every pair ``i < j``) or ``"linear"`` (``(i, i + 1)``).

    Returns
    -------
    list of tuple of int
        Pairs in the order Qiskit applies them.

    Raises
    ------
    ValueError
        If ``entanglement`` is not supported.
    """
    if entanglement == "full":
        return list(itertools.combinations(range(n_qubits), 2))
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    raise ValueError(
        f"Unsupported entanglement {entanglement!r}; expected one of {ENTANGLEMENTS}"
    )


class StatevectorBatch:
    """Batch of ``n_qubits`` pure states with autograd-friendly gate applications.

    Parameters
    ----------
    batch_size : int
        Number of states ``B``.
    n_qubits : int
        Number of qubits ``S``.
    device : torch.device or str
        Device holding the amplitudes.
    real_dtype : torch.dtype
        ``torch.float64`` or ``torch.float32``; amplitudes use the matching
        complex dtype.
    """

    def __init__(self, batch_size: int, n_qubits: int, device, real_dtype: torch.dtype):
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )
        amplitudes = torch.zeros(
            batch_size, 2**n_qubits, dtype=complex_dtype, device=device
        )
        amplitudes[:, 0] = 1.0
        self.amplitudes = amplitudes

    def _split(self, qubit: int) -> tuple[torch.Tensor, torch.Tensor]:
        view = self.amplitudes.reshape(self.batch_size, 2**qubit, 2, -1)
        return view[:, :, 0, :], view[:, :, 1, :]

    def _merge(self, zero: torch.Tensor, one: torch.Tensor) -> None:
        self.amplitudes = torch.stack((zero, one), dim=2).reshape(self.batch_size, -1)

    def hadamard(self, qubit: int) -> None:
        """Apply H to ``qubit``."""
        zero, one = self._split(qubit)
        scale = 1.0 / math.sqrt(2.0)
        self._merge(scale * (zero + one), scale * (zero - one))

    def phase(self, qubit: int, angle: torch.Tensor) -> None:
        """Apply ``P(angle) = diag(1, exp(i angle))`` with a per-sample angle ``(B,)``."""
        zero, one = self._split(qubit)
        factor = torch.exp(1j * angle.to(self.amplitudes.dtype)).reshape(-1, 1, 1)
        self._merge(zero, factor * one)

    def ry(self, qubit: int, angle: torch.Tensor) -> None:
        """Apply ``RY(angle)`` with a per-sample angle ``(B,)``."""
        zero, one = self._split(qubit)
        cos = torch.cos(angle / 2).reshape(-1, 1, 1).to(self.amplitudes.dtype)
        sin = torch.sin(angle / 2).reshape(-1, 1, 1).to(self.amplitudes.dtype)
        self._merge(cos * zero - sin * one, sin * zero + cos * one)

    def cx(self, control: int, target: int) -> None:
        """Apply CNOT with ``control`` and ``target`` qubits."""
        state = self.amplitudes.reshape([self.batch_size] + [2] * self.n_qubits)
        control_on = state.select(control + 1, 1)
        target_axis = target + 1 if target < control else target
        flipped = control_on.flip(target_axis)
        state = torch.stack((state.select(control + 1, 0), flipped), dim=control + 1)
        self.amplitudes = state.reshape(self.batch_size, -1)

    def parity_probabilities(self) -> torch.Tensor:
        """Return ``(B, 2)`` probabilities of even and odd measured parity."""
        probabilities = self.amplitudes.abs() ** 2
        odd_mask = _odd_parity_mask(self.n_qubits, probabilities.device)
        odd = probabilities[:, odd_mask].sum(dim=1)
        even = probabilities[:, ~odd_mask].sum(dim=1)
        return torch.stack((even, odd), dim=1)


def _odd_parity_mask(n_qubits: int, device) -> torch.Tensor:
    indices = torch.arange(2**n_qubits, device=device)
    weights = torch.zeros_like(indices)
    for bit in range(n_qubits):
        weights = weights + ((indices >> bit) & 1)
    return (weights % 2).bool()


class QubitQNN:
    """Feature map, RealAmplitudes ansatz, and parity readout (paper Fig. 1).

    Parameters
    ----------
    n_qubits : int
        Number of qubits ``S``, equal to the input size ``s_in``.
    feature_map : str
        ``"zz"`` (the paper's quantum neural network), ``"z"`` (the easy quantum
        model in the released code), or ``"ry_angle"`` (the easy model of footnote 6).
    feature_map_reps : int
        Feature-map depth (2 for the QNN, 1 for the easy model).
    ansatz_reps : int
        RealAmplitudes depth ``D``; the model has ``d = (D + 1) S`` parameters.
    feature_entanglement : str
        Entanglement map of the ZZ feature map. Default value is ``"full"``.
    ansatz_entanglement : str
        Entanglement map of the ansatz. Default value is ``"full"``.
    hadamard_every_rep : bool
        Repeat the Hadamard layer in every feature-map repetition (Qiskit) or only
        in the first (paper App. A). Default value is True.
    """

    output_size = 2

    def __init__(
        self,
        n_qubits: int,
        feature_map: str,
        feature_map_reps: int,
        ansatz_reps: int,
        feature_entanglement: str = "full",
        ansatz_entanglement: str = "full",
        hadamard_every_rep: bool = True,
    ):
        if feature_map not in FEATURE_MAPS:
            raise ValueError(
                f"Unsupported feature map {feature_map!r}; expected one of {FEATURE_MAPS}"
            )
        self.n_qubits = n_qubits
        self.input_size = n_qubits
        self.feature_map = feature_map
        self.feature_map_reps = feature_map_reps
        self.ansatz_reps = ansatz_reps
        self.feature_pairs = entangled_pairs(n_qubits, feature_entanglement)
        self.ansatz_pairs = entangled_pairs(n_qubits, ansatz_entanglement)
        self.hadamard_every_rep = hadamard_every_rep
        self.d = (ansatz_reps + 1) * n_qubits

    def _encode(self, state: StatevectorBatch, x: torch.Tensor) -> None:
        for repetition in range(self.feature_map_reps):
            if self.feature_map == "ry_angle":
                for qubit in range(self.n_qubits):
                    state.ry(qubit, x[:, qubit])
                continue
            if repetition == 0 or self.hadamard_every_rep:
                for qubit in range(self.n_qubits):
                    state.hadamard(qubit)
            for qubit in range(self.n_qubits):
                state.phase(qubit, 2.0 * x[:, qubit])
            if self.feature_map == "zz":
                for control, target in self.feature_pairs:
                    state.cx(control, target)
                    state.phase(
                        target,
                        2.0 * (math.pi - x[:, control]) * (math.pi - x[:, target]),
                    )
                    state.cx(control, target)

    def _ansatz(self, state: StatevectorBatch, theta: torch.Tensor) -> None:
        for qubit in range(self.n_qubits):
            state.ry(qubit, theta[:, qubit])
        for layer in range(1, self.ansatz_reps + 1):
            for control, target in self.ansatz_pairs:
                state.cx(control, target)
            offset = layer * self.n_qubits
            for qubit in range(self.n_qubits):
                state.ry(qubit, theta[:, offset + qubit])

    def probabilities(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the parity class probabilities ``(B, 2)``.

        Parameters
        ----------
        x : torch.Tensor
            Inputs ``(B, S)``.
        theta : torch.Tensor
            Ansatz parameters ``(B, d)``, one row per sample.

        Returns
        -------
        torch.Tensor
            ``[p(even), p(odd)]`` per sample.
        """
        if x.shape[1] != self.n_qubits or theta.shape[1] != self.d:
            raise ValueError(
                f"Expected x (B, {self.n_qubits}) and theta (B, {self.d}); got {tuple(x.shape)}, {tuple(theta.shape)}"
            )
        state = StatevectorBatch(x.shape[0], self.n_qubits, x.device, x.dtype)
        self._encode(state, x)
        self._ansatz(state, theta)
        return state.parity_probabilities()
