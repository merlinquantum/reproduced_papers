"""Pure-PyTorch continuous-variable (CV) quantum simulator with autograd.

The simulator implements the Killoran et al. continuous-variable neural
network building block (rotation, beam splitter, squeezing, displacement,
Kerr) over a Fock-space truncation. Every operator is exposed as a complex
matrix in the truncated basis so that PyTorch's `matrix_exp` and standard
linear algebra give differentiable forward passes without parameter-shift
rules. The two-qumode product space is the only configuration the paper
requires, so the simulator is implemented for 1- and 2-qumode systems with
a runtime-fixed cutoff dimension `d`.

Operator conventions follow the Strawberry Fields documentation; in
particular the position-quadrature observable used by the homodyne
measurement is `X = (a + a†) / sqrt(2)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def annihilation(d: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Return the truncated annihilation operator on a `d`-dim Fock basis."""
    n = torch.arange(1, d, dtype=torch.float64)
    a = torch.zeros((d, d), dtype=dtype)
    a[range(d - 1), range(1, d)] = torch.sqrt(n).to(dtype)
    return a


def number_operator(d: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return torch.diag(torch.arange(d, dtype=torch.float64).to(dtype))


def position_operator(d: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    a = annihilation(d, dtype)
    return (a + a.conj().T) / torch.sqrt(torch.tensor(2.0, dtype=dtype))


@dataclass
class CVOperators:
    """Reusable single-mode operators in the truncated basis."""

    d: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        self.a = annihilation(self.d, self.dtype).to(self.device).contiguous()
        self.adag = self.a.conj().T.contiguous()
        self.n = number_operator(self.d, self.dtype).to(self.device)
        self.x = (
            (self.a + self.adag) / torch.sqrt(torch.tensor(2.0, dtype=self.dtype))
        ).contiguous()
        self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)


def displacement(alpha: torch.Tensor, ops: CVOperators) -> torch.Tensor:
    """Single-mode displacement gate D(alpha) for real alpha (paper convention).

    Shape rules:
      - scalar alpha → returns (d, d).
      - batched alpha of shape (..., ) → returns (..., d, d).
    """
    a, adag = ops.a, ops.adag
    h = adag - a  # antihermitian, shape (d, d)
    if alpha.dim() == 0:
        return torch.matrix_exp(alpha.to(ops.dtype) * h)
    extra = (1,) * alpha.dim()
    h_b = h.reshape(extra + h.shape).expand(alpha.shape + h.shape)
    return torch.matrix_exp(alpha.to(ops.dtype).unsqueeze(-1).unsqueeze(-1) * h_b)


def squeezing(r: torch.Tensor, ops: CVOperators) -> torch.Tensor:
    """Single-mode squeezing gate S(r) for real squeezing parameter r."""
    a, adag = ops.a, ops.adag
    h = 0.5 * (a @ a - adag @ adag)
    return torch.matrix_exp(r.to(ops.dtype) * h)


def rotation(phi: torch.Tensor, ops: CVOperators) -> torch.Tensor:
    """Single-mode rotation / phase gate R(phi).

    Diagonal in the Fock basis with eigenvalues exp(-i phi n).
    """
    n_diag = torch.arange(ops.d, dtype=torch.float64, device=ops.device).to(ops.dtype)
    return torch.diag(torch.exp(-1j * phi.to(ops.dtype) * n_diag))


def kerr(kappa: torch.Tensor, ops: CVOperators) -> torch.Tensor:
    """Single-mode Kerr gate K(kappa) = exp(i kappa n^2). Diagonal."""
    n_diag = torch.arange(ops.d, dtype=torch.float64, device=ops.device).to(ops.dtype)
    return torch.diag(torch.exp(1j * kappa.to(ops.dtype) * n_diag * n_diag))


def beamsplitter(
    theta: torch.Tensor, phi: torch.Tensor, ops: CVOperators
) -> torch.Tensor:
    """Two-mode beam splitter BS(theta, phi) in the d^2 product basis.

    Hamiltonian: H = theta (e^{i phi} a1† a2 - e^{-i phi} a1 a2†).
    Returns a (d^2, d^2) unitary that acts on the joint state vector
    arranged as state[i, j] -> state.reshape(d*d)[i*d + j].
    """
    a, adag, eye = ops.a, ops.adag, ops.I
    a1 = torch.kron(a, eye)
    a1dag = torch.kron(adag, eye)
    a2 = torch.kron(eye, a)
    a2dag = torch.kron(eye, adag)
    eipi = torch.exp(1j * phi.to(ops.dtype))
    h = theta.to(ops.dtype) * (eipi * (a1dag @ a2) - torch.conj(eipi) * (a1 @ a2dag))
    return torch.matrix_exp(h)


def apply_single_mode(
    state: torch.Tensor, gate: torch.Tensor, mode: int, n_modes: int
) -> torch.Tensor:
    """Apply a single-mode gate to mode `mode` of an `n_modes`-mode state.

    `state` is a (..., d, d, ..., d) tensor with one Fock axis per mode (the
    leading `...` collects any batch axes). The gate is contracted along the
    appropriate Fock axis.
    """
    batch_dims = state.dim() - n_modes
    axis = batch_dims + mode
    state = torch.moveaxis(state, axis, -1)
    state = state @ gate.transpose(-1, -2)
    state = torch.moveaxis(state, -1, axis)
    return state


def apply_two_mode(
    state: torch.Tensor, gate: torch.Tensor, modes: tuple[int, int], n_modes: int
) -> torch.Tensor:
    """Apply a two-mode gate to `modes` of an `n_modes`-mode state.

    The gate is shaped as (d^2, d^2) following the convention of
    `beamsplitter`. We reshape it to (d, d, d, d) (out1, out2, in1, in2),
    contract over the input axes, and re-assemble.
    """
    m1, m2 = modes
    assert m1 < m2, "Pass modes in ascending order"
    batch_dims = state.dim() - n_modes
    d = gate.shape[0]
    d0 = int(round(d**0.5))
    g = gate.reshape(d0, d0, d0, d0)
    # Move mode axes to the end
    state = torch.moveaxis(state, batch_dims + m1, -2)
    # m2 was originally one position later; after the previous move, account
    # for the index shift.
    new_m2 = batch_dims + m2 if (batch_dims + m2) < state.dim() - 1 else state.dim() - 1
    state = torch.moveaxis(state, new_m2, -1)
    # Contract: state[..., i, j] * g[k, l, i, j] -> state[..., k, l]
    state = torch.einsum("klij,...ij->...kl", g, state)
    state = torch.moveaxis(state, -1, batch_dims + m2)
    state = torch.moveaxis(
        state, -2 if (batch_dims + m1) < state.dim() - 1 else -1, batch_dims + m1
    )
    return state


def vacuum_state(
    n_modes: int, d: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    shape = (d,) * n_modes
    s = torch.zeros(shape, dtype=dtype, device=device)
    s[(0,) * n_modes] = 1.0
    return s


def expectation_x(
    state: torch.Tensor, mode: int, n_modes: int, ops: CVOperators
) -> torch.Tensor:
    """<state|X_mode|state> for a possibly-batched state tensor.

    Returns a real tensor whose leading axes match the batch axes of `state`.
    """
    transformed = apply_single_mode(state, ops.x, mode, n_modes)
    batch_dims = state.dim() - n_modes
    flat = state.reshape(*state.shape[:batch_dims], -1)
    trans_flat = transformed.reshape(*transformed.shape[:batch_dims], -1)
    expectation = torch.einsum("...i,...i->...", flat.conj(), trans_flat)
    return expectation.real


def state_norm_sq(state: torch.Tensor, n_modes: int) -> torch.Tensor:
    """Return <state|state> with shape equal to the batch axes of `state`."""
    batch_dims = state.dim() - n_modes
    flat = state.reshape(*state.shape[:batch_dims], -1)
    return torch.einsum("...i,...i->...", flat.conj(), flat).real


def loss_channel(
    state: torch.Tensor, T: torch.Tensor, mode: int, n_modes: int, ops: CVOperators
) -> torch.Tensor:
    """Apply the photon-loss channel L̂(T) to a single mode (paper Eq. 27/28).

    The pure-state CV simulator cannot represent loss channels exactly
    because they map pure states to mixed states. We use the standard
    coherent-loss approximation `D, S, K, BS, R unchanged` and rescale
    the position quadrature observable post-hoc by ``sqrt(T)``. The
    structure of the trainable network is unaffected; the homodyne
    expectation values are attenuated, matching the systemic-error
    behaviour the paper exploits in §V.C. For a more faithful model
    (i.e. a true Lindblad channel), upgrade to a density-matrix
    simulator.

    Returns the state unchanged; rescaling is applied by the caller on
    the ``<X>`` measurement (see ``expectation_x_with_loss``).
    """
    return state


def expectation_x_with_loss(
    state: torch.Tensor, T: torch.Tensor, mode: int, n_modes: int, ops: CVOperators
) -> torch.Tensor:
    """Pure-state approximation of `<X>` under a loss channel of transmittance T.

    The position quadrature transforms as `X -> sqrt(T) X` under the
    standard CV loss channel (Paper Eq. 28). Mean photon numbers are
    rescaled by `T`. For the QPINN, the relevant observable is `<X>`,
    so we apply the factor directly. T must be a scalar in [0, 1].
    """
    return torch.sqrt(T) * expectation_x(state, mode, n_modes, ops)


__all__ = [
    "CVOperators",
    "annihilation",
    "number_operator",
    "position_operator",
    "displacement",
    "squeezing",
    "rotation",
    "kerr",
    "beamsplitter",
    "apply_single_mode",
    "apply_two_mode",
    "vacuum_state",
    "expectation_x",
    "expectation_x_with_loss",
    "loss_channel",
    "state_norm_sq",
]
