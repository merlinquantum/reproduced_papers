"""Torch statevector simulator for the paper's qubit models (Figs. 3-5).

The paper's numerics use PennyLane ``default.qubit``. This module re-implements
the same circuits with the same gate conventions so that gradients come from
torch autograd:

- the state has shape ``(batch, 2, ..., 2)``, one axis per wire, and wire 0 is
  the most significant qubit, as in PennyLane;
- ``RX(t) = exp(-i t X / 2)``, so an ``RX`` encoding has generator ``X / 2``
  with eigenvalues ``+-1/2`` and frequency spectrum ``{-1, 0, 1}`` (paper
  Section III.A);
- ``Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)`` (PennyLane ``Rot``);
- ``StronglyEntanglingLayers`` and ``BasicEntanglerLayers`` follow PennyLane's
  default CNOT patterns.
"""

from __future__ import annotations

import torch
from torch import nn

ANSATZ_NAMES = ("strongly", "basic", "none")


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype matching a real dtype.

    Parameters
    ----------
    real_dtype : torch.dtype
        ``torch.float32`` or ``torch.float64``.

    Returns
    -------
    torch.dtype
        ``torch.complex64`` or ``torch.complex128``.

    Raises
    ------
    ValueError
        If ``real_dtype`` is not a supported floating dtype.
    """
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    raise ValueError(f"Unsupported real dtype for statevector simulation: {real_dtype}")


def zero_state(
    n_qubits: int, batch_size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return ``|0...0>`` repeated ``batch_size`` times.

    Parameters
    ----------
    n_qubits : int
        Number of wires.
    batch_size : int
        Number of independent states (one per data point).
    dtype : torch.dtype
        Complex dtype of the state.
    device : torch.device
        Device holding the state.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size, 2, ..., 2)``.
    """
    state = torch.zeros((batch_size,) + (2,) * n_qubits, dtype=dtype, device=device)
    state.view(batch_size, -1)[:, 0] = 1.0
    return state


def _two_by_two(a, b, c, d) -> torch.Tensor:
    """Stack four entries (scalars or equal-shape tensors) into ``(..., 2, 2)``."""
    return torch.stack(
        (torch.stack((a, b), dim=-1), torch.stack((c, d), dim=-1)), dim=-2
    )


def rx_matrix(angle: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return ``RX(angle) = exp(-i angle X / 2)`` with shape ``angle.shape + (2, 2)``."""
    cos = torch.cos(angle / 2).to(dtype)
    minus_i_sin = -1j * torch.sin(angle / 2).to(dtype)
    return _two_by_two(cos, minus_i_sin, minus_i_sin, cos)


def ry_matrix(angle: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return ``RY(angle) = exp(-i angle Y / 2)`` with shape ``angle.shape + (2, 2)``."""
    cos = torch.cos(angle / 2).to(dtype)
    sin = torch.sin(angle / 2).to(dtype)
    return _two_by_two(cos, -sin, sin, cos)


def rz_matrix(angle: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return ``RZ(angle) = exp(-i angle Z / 2)`` with shape ``angle.shape + (2, 2)``."""
    phase = torch.exp(-0.5j * angle.to(dtype))
    zero = torch.zeros_like(phase)
    return _two_by_two(phase, zero, zero, phase.conj())


def rot_matrix(
    phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Return PennyLane ``Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)``."""
    return rz_matrix(omega, dtype) @ ry_matrix(theta, dtype) @ rz_matrix(phi, dtype)


def apply_single_qubit_gate(
    state: torch.Tensor, gate: torch.Tensor, wire: int
) -> torch.Tensor:
    """Apply a one-qubit gate to ``wire``.

    Parameters
    ----------
    state : torch.Tensor
        State of shape ``(batch, 2, ..., 2)``.
    gate : torch.Tensor
        Either one ``(2, 2)`` matrix shared by the batch, or ``(batch, 2, 2)``
        matrices, one per batch entry (used for data-encoding gates).
    wire : int
        Target wire.

    Returns
    -------
    torch.Tensor
        New state with the same shape.
    """
    axis = wire + 1
    moved = torch.movedim(state, axis, -1)
    moved_shape = moved.shape
    flat = moved.reshape(moved_shape[0], -1, 2)
    if gate.dim() == 2:
        updated = flat @ gate.transpose(-1, -2)
    else:
        updated = torch.einsum("brj,bij->bri", flat, gate)
    return torch.movedim(updated.reshape(moved_shape), -1, axis)


def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """Apply ``CNOT(control -> target)``.

    Parameters
    ----------
    state : torch.Tensor
        State of shape ``(batch, 2, ..., 2)``.
    control : int
        Control wire.
    target : int
        Target wire, different from ``control``.

    Returns
    -------
    torch.Tensor
        New state with the same shape.
    """
    if control == target:
        raise ValueError("CNOT control and target must differ")
    control_axis = control + 1
    target_axis = target + 1
    # Selecting the control axis removes it, which shifts later axes by one.
    target_axis_in_slice = (
        target_axis - 1 if target_axis > control_axis else target_axis
    )
    control_off = state.select(control_axis, 0)
    control_on = state.select(control_axis, 1).flip(target_axis_in_slice)
    return torch.stack((control_off, control_on), dim=control_axis)


def expval_z(state: torch.Tensor, wire: int) -> torch.Tensor:
    """Return ``<Z>`` on ``wire`` for every batch entry, shape ``(batch,)``."""
    probabilities = state.real**2 + state.imag**2
    batch_size = state.shape[0]
    prob_zero = probabilities.select(wire + 1, 0).reshape(batch_size, -1).sum(dim=-1)
    prob_one = probabilities.select(wire + 1, 1).reshape(batch_size, -1).sum(dim=-1)
    return prob_zero - prob_one


def strongly_entangling_layers(
    state: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Apply PennyLane ``StronglyEntanglingLayers`` (default ranges, CNOT imprimitive).

    Layer ``l`` applies ``Rot`` on every wire, then ``CNOT(i, (i + r_l) mod n)``
    for every wire ``i`` with range ``r_l = (l mod (n - 1)) + 1`` when ``n > 1``.

    Parameters
    ----------
    state : torch.Tensor
        State of shape ``(batch, 2, ..., 2)`` with ``n`` wires.
    weights : torch.Tensor
        Rotation angles of shape ``(layers, n, 3)``.

    Returns
    -------
    torch.Tensor
        New state.
    """
    n_qubits = state.dim() - 1
    layers, weight_wires, _ = weights.shape
    if weight_wires != n_qubits:
        raise ValueError(f"weights cover {weight_wires} wires, state has {n_qubits}")
    for layer in range(layers):
        for wire in range(n_qubits):
            gate = rot_matrix(
                weights[layer, wire, 0],
                weights[layer, wire, 1],
                weights[layer, wire, 2],
                state.dtype,
            )
            state = apply_single_qubit_gate(state, gate, wire)
        if n_qubits > 1:
            cnot_range = (layer % (n_qubits - 1)) + 1
            for wire in range(n_qubits):
                state = apply_cnot(state, wire, (wire + cnot_range) % n_qubits)
    return state


def basic_entangler_layers(state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply PennyLane ``BasicEntanglerLayers`` with ``RX`` rotations.

    Layer ``l`` applies ``RX`` on every wire, then a CNOT chain: one CNOT for
    two wires, a closed ring ``CNOT(i, i + 1 mod n)`` for more than two.

    Parameters
    ----------
    state : torch.Tensor
        State of shape ``(batch, 2, ..., 2)`` with ``n`` wires.
    weights : torch.Tensor
        Rotation angles of shape ``(layers, n)``.

    Returns
    -------
    torch.Tensor
        New state.
    """
    n_qubits = state.dim() - 1
    layers, weight_wires = weights.shape
    if weight_wires != n_qubits:
        raise ValueError(f"weights cover {weight_wires} wires, state has {n_qubits}")
    for layer in range(layers):
        for wire in range(n_qubits):
            state = apply_single_qubit_gate(
                state, rx_matrix(weights[layer, wire], state.dtype), wire
            )
        if n_qubits == 2:
            state = apply_cnot(state, 0, 1)
        elif n_qubits > 2:
            for wire in range(n_qubits):
                state = apply_cnot(state, wire, (wire + 1) % n_qubits)
    return state


READOUTS = ("z0", "trainable_linear")


def _build_readout(readout: str, n_qubits: int, dtype: torch.dtype) -> nn.Linear | None:
    """Return the trainable readout for ``"trainable_linear"``, or None for the paper's ``<Z_0>``.

    ``"trainable_linear"`` is a trainable observable diagonal in the computational
    basis: a linear map over all ``2^n`` outcome probabilities (paper Section V,
    practical insight 5). It matches the photonic adaptation's trainable readout
    over click patterns.
    """
    if readout not in READOUTS:
        raise ValueError(f"Unknown readout {readout!r}; expected one of {READOUTS}")
    if readout == "z0":
        return None
    return nn.Linear(2**n_qubits, 1, dtype=dtype)


def _measure(state: torch.Tensor, readout_layer: nn.Linear | None) -> torch.Tensor:
    """Apply the readout: ``<Z_0>``, or the trainable linear map over basis probabilities."""
    if readout_layer is None:
        return expval_z(state, 0)
    probabilities = (state.real**2 + state.imag**2).reshape(state.shape[0], -1)
    return readout_layer(probabilities).squeeze(-1)


def _reset_readout(readout_layer: nn.Linear | None) -> None:
    """Zero-initialise a trainable readout, so the model starts at f = 0."""
    if readout_layer is not None:
        with torch.no_grad():
            readout_layer.weight.zero_()
            readout_layer.bias.zero_()


def _uniform_angles(
    shape, high: float, generator: torch.Generator, dtype
) -> torch.Tensor:
    return torch.rand(shape, generator=generator, dtype=dtype) * high


class SerialPauliModel(nn.Module):
    """Single-qubit model with the ``RX`` encoding repeated in sequence (Fig. 2a).

    ``f(x) = <0| U(x)^dagger Z U(x) |0>`` with
    ``U(x) = W^(r+1) RX(s x) W^(r) ... RX(s x) W^(1)`` and every ``W`` a
    PennyLane ``Rot`` (an arbitrary single-qubit unitary up to phase), matching
    ``serial_quantum_model`` in the reference notebook. The spectrum is
    ``{-r, ..., r}`` in units of ``s`` (paper Section III.B).

    Parameters
    ----------
    repetitions : int
        Number of encoding gates ``r = L``.
    encoding_scaling : float
        Classical input scaling ``s`` inside ``RX(s x)``.
    readout : str
        ``"z0"`` (paper) or ``"trainable_linear"`` (Phase 4 amendment).
    dtype : torch.dtype
        Real dtype of the trainable angles.
    """

    def __init__(
        self,
        repetitions: int,
        encoding_scaling: float,
        readout: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        self.repetitions = repetitions
        self.encoding_scaling = float(encoding_scaling)
        self.weights = nn.Parameter(torch.zeros(repetitions + 1, 3, dtype=dtype))
        self.readout_layer = _build_readout(readout, 1, dtype)

    def reset_parameters(self, generator: torch.Generator, high: float) -> None:
        """Draw every angle from ``U[0, high)`` (reference code: ``high = 2 pi``); zero the readout."""
        with torch.no_grad():
            self.weights.copy_(
                _uniform_angles(self.weights.shape, high, generator, self.weights.dtype)
            )
        _reset_readout(self.readout_layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on inputs of shape ``(batch,)``; returns ``(batch,)``."""
        complex_dtype = complex_dtype_for(self.weights.dtype)
        state = zero_state(1, inputs.shape[0], complex_dtype, inputs.device)
        encoding = rx_matrix(self.encoding_scaling * inputs, complex_dtype)
        for layer in range(self.repetitions):
            trainable = rot_matrix(*self.weights[layer], complex_dtype)
            state = apply_single_qubit_gate(state, trainable, 0)
            state = apply_single_qubit_gate(state, encoding, 0)
        final = rot_matrix(*self.weights[self.repetitions], complex_dtype)
        state = apply_single_qubit_gate(state, final, 0)
        return _measure(state, self.readout_layer)


class ParallelPauliModel(nn.Module):
    """``L = 1`` model with the ``RX`` encoding repeated on ``r`` wires (Fig. 2b).

    ``f(x) = <0| U(x)^dagger Z_0 U(x) |0>`` with ``U(x) = W^(2) S(x) W^(1)`` and
    ``S(x) = RX(s x)`` on every wire, matching ``parallel_quantum_model`` in the
    reference notebook. ``W`` is an ansatz instead of an arbitrary unitary:
    ``"strongly"`` (Circuit A, Figs. 4-5), ``"basic"`` (Circuit B, Fig. 5) or
    ``"none"`` (identity, the notebook's "no unitary" option).

    Parameters
    ----------
    n_qubits : int
        Number of parallel encodings ``r``.
    encoding_scaling : float
        Classical input scaling ``s`` inside ``RX(s x)``.
    ansatz : str
        One of ``ANSATZ_NAMES``.
    ansatz_layers : int
        Number of ansatz layers inside each trainable block.
    readout : str
        ``"z0"`` (paper) or ``"trainable_linear"`` (Phase 4 amendment).
    dtype : torch.dtype
        Real dtype of the trainable angles.
    """

    def __init__(
        self,
        n_qubits: int,
        encoding_scaling: float,
        ansatz: str,
        ansatz_layers: int,
        readout: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if ansatz not in ANSATZ_NAMES:
            raise ValueError(
                f"Unknown ansatz {ansatz!r}; expected one of {ANSATZ_NAMES}"
            )
        self.n_qubits = n_qubits
        self.encoding_scaling = float(encoding_scaling)
        self.ansatz = ansatz
        self.ansatz_layers = ansatz_layers
        if ansatz == "strongly":
            shape = (2, ansatz_layers, n_qubits, 3)
        elif ansatz == "basic":
            shape = (2, ansatz_layers, n_qubits)
        else:
            shape = (0,)
        self.weights = nn.Parameter(torch.zeros(shape, dtype=dtype))
        self.readout_layer = _build_readout(readout, n_qubits, dtype)

    def reset_parameters(self, generator: torch.Generator, high: float) -> None:
        """Draw every angle from ``U[0, high)`` (reference code: ``high = 2 pi``); zero the readout."""
        with torch.no_grad():
            self.weights.copy_(
                _uniform_angles(self.weights.shape, high, generator, self.weights.dtype)
            )
        _reset_readout(self.readout_layer)

    def _trainable_block(self, state: torch.Tensor, block: int) -> torch.Tensor:
        if self.ansatz == "strongly":
            return strongly_entangling_layers(state, self.weights[block])
        if self.ansatz == "basic":
            return basic_entangler_layers(state, self.weights[block])
        return state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on inputs of shape ``(batch,)``; returns ``(batch,)``."""
        complex_dtype = complex_dtype_for(self.weights.dtype)
        state = zero_state(self.n_qubits, inputs.shape[0], complex_dtype, inputs.device)
        state = self._trainable_block(state, 0)
        encoding = rx_matrix(self.encoding_scaling * inputs, complex_dtype)
        for wire in range(self.n_qubits):
            state = apply_single_qubit_gate(state, encoding, wire)
        state = self._trainable_block(state, 1)
        return _measure(state, self.readout_layer)
