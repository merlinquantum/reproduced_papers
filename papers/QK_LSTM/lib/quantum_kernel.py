"""Fidelity quantum kernel of QK-LSTM (arXiv:2412.08851, Sec. III-B3).

The kernel is ``k(v_t, v_j) = |<0|U^dagger(v_j) U(v_t)|0>|^2`` (Eq. 13), computed
here by exact statevector simulation in PyTorch so that gradients flow to the
encoded angles by autograd. On an ideal simulator this gives the same
derivative as the parameter-shift rule of Eq. 15-16; only the estimator
differs.

Conventions
-----------
- Qubit 0 is the most significant bit of the computational-basis index, so a
  state of ``n`` qubits is stored as a complex tensor of shape ``(B, 2**n)``.
- Two circuit layouts are available because the paper's text and Fig. 2(d)
  disagree (LOG.md, disagreement D2):

  ``"text"``   Eq. 9-12: ``H`` layer, then ``R_y(theta_k) R_z(phi_k)`` on each
               qubit, then the chain ``CNOT(k, k+1)``.
  ``"figure"`` Fig. 2(d): ``H`` layer, ``R_z`` and ``R_y`` encodings, the CNOT
               chain, then the ``R_z`` encoding re-applied after the chain.

  In the ``"text"`` layout the CNOT chain is the same data-independent unitary
  on both sides of the overlap, so it cancels and the kernel factorises into a
  product of single-qubit fidelities. ``tests/test_quantum_kernel.py`` checks
  this identity.
"""

from __future__ import annotations

import math

import torch

CIRCUIT_LAYOUTS = ("text", "figure")
ANGLE_MAPS = ("identity", "arctan", "tanh_pi")


def encoding_angles(
    z: torch.Tensor, angle_map: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map compressed features to the rotation angles ``(theta_k, phi_k)``.

    The paper only says the angles are "functions of the components of v"
    (Eq. 10), so the map is a documented choice (LOG.md, Reproducibility
    Assessment).

    Parameters
    ----------
    z : torch.Tensor
        Real tensor of shape ``(B, n_qubits)``, one component per qubit.
    angle_map : str
        ``"identity"``: ``theta = phi = z``.
        ``"arctan"``: ``theta = arctan(z)``, ``phi = arctan(z**2)``, as in the
        QLSTM encoding of ``papers/variational_qml_ts_benchmark``.
        ``"tanh_pi"``: ``theta = phi = pi * tanh(z)``.

    Returns
    -------
    tuple of torch.Tensor
        ``(theta, phi)``, each of shape ``(B, n_qubits)``.

    Raises
    ------
    ValueError
        If ``angle_map`` is unknown.
    """
    if angle_map == "identity":
        return z, z
    if angle_map == "arctan":
        return torch.atan(z), torch.atan(z * z)
    if angle_map == "tanh_pi":
        angle = math.pi * torch.tanh(z)
        return angle, angle
    raise ValueError(f"Unknown angle_map {angle_map!r}; expected one of {ANGLE_MAPS}")


def _ry(theta: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    row0 = torch.stack([cos, -sin], dim=-1)
    row1 = torch.stack([sin, cos], dim=-1)
    return torch.stack([row0, row1], dim=-2).to(complex_dtype)


def _rz(phi: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    phase = torch.exp(-0.5j * phi.to(complex_dtype))
    zero = torch.zeros_like(phase)
    row0 = torch.stack([phase, zero], dim=-1)
    row1 = torch.stack([zero, phase.conj()], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _apply_single_qubit(
    state: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int
) -> torch.Tensor:
    """Apply a batch of 2x2 gates ``(B, 2, 2)`` to one qubit of ``(B, 2**n)`` states."""
    batch = state.shape[0]
    tensor = state.reshape(batch, *([2] * n_qubits))
    tensor = torch.movedim(tensor, qubit + 1, -1)
    tensor = torch.einsum("bij,b...j->b...i", gate, tensor)
    tensor = torch.movedim(tensor, -1, qubit + 1)
    return tensor.reshape(batch, 2**n_qubits)


def _cnot_chain_permutation(n_qubits: int) -> torch.Tensor:
    """Basis permutation of ``CNOT(0,1) CNOT(1,2) ... CNOT(n-2, n-1)`` (Eq. 11).

    The gates act in order ``CNOT(0,1)`` first. ``perm[b]`` is the input basis
    index whose amplitude ends at output index ``b``.
    """
    dim = 2**n_qubits
    source = torch.arange(dim)
    for control in range(n_qubits - 1):
        target = control + 1
        control_bit = (source >> (n_qubits - 1 - control)) & 1
        source = source ^ (control_bit << (n_qubits - 1 - target))
    # `source` maps each input index to its output index; invert it.
    permutation = torch.empty(dim, dtype=torch.long)
    permutation[source] = torch.arange(dim)
    return permutation


def feature_map_states(
    z: torch.Tensor,
    *,
    layout: str = "text",
    angle_map: str = "identity",
) -> torch.Tensor:
    """Prepare the feature-map states ``|phi(v)> = U(v)|0...0>``.

    Parameters
    ----------
    z : torch.Tensor
        Real tensor ``(B, n_qubits)`` of compressed features (one per qubit).
    layout : str
        ``"text"`` or ``"figure"``; see the module docstring. Default value is
        ``"text"``.
    angle_map : str
        Angle map passed to :func:`encoding_angles`. Default value is
        ``"identity"``.

    Returns
    -------
    torch.Tensor
        Complex tensor ``(B, 2**n_qubits)`` of normalised states.

    Raises
    ------
    ValueError
        If ``layout`` is unknown.
    """
    if layout not in CIRCUIT_LAYOUTS:
        raise ValueError(
            f"Unknown circuit layout {layout!r}; expected one of {CIRCUIT_LAYOUTS}"
        )
    batch, n_qubits = z.shape
    complex_dtype = torch.complex128 if z.dtype == torch.float64 else torch.complex64
    theta, phi = encoding_angles(z, angle_map)

    # H^{\otimes n}|0> is the uniform superposition (Eq. 9).
    dim = 2**n_qubits
    state = torch.full(
        (batch, dim), 1.0 / math.sqrt(dim), dtype=complex_dtype, device=z.device
    )
    for qubit in range(n_qubits):
        # Eq. 10 writes R_y(theta) R_z(phi): R_z acts first.
        state = _apply_single_qubit(
            state, _rz(phi[:, qubit], complex_dtype), qubit, n_qubits
        )
        state = _apply_single_qubit(
            state, _ry(theta[:, qubit], complex_dtype), qubit, n_qubits
        )
    state = state[:, _cnot_chain_permutation(n_qubits).to(z.device)]
    if layout == "figure":
        for qubit in range(n_qubits):
            state = _apply_single_qubit(
                state, _rz(phi[:, qubit], complex_dtype), qubit, n_qubits
            )
    return state


def fidelity_kernel(
    states: torch.Tensor, reference_states: torch.Tensor
) -> torch.Tensor:
    """Fidelity kernel matrix ``|<phi_j|phi_t>|^2`` (Eq. 7 and 13).

    Parameters
    ----------
    states : torch.Tensor
        Complex ``(B, D)`` feature-map states of the current inputs.
    reference_states : torch.Tensor
        Complex ``(N, D)`` feature-map states of the reference vectors.

    Returns
    -------
    torch.Tensor
        Real ``(B, N)`` kernel values in ``[0, 1]``.
    """
    overlaps = states @ reference_states.conj().transpose(0, 1)
    return overlaps.real**2 + overlaps.imag**2


def quantum_kernel(
    z: torch.Tensor,
    z_reference: torch.Tensor,
    *,
    layout: str = "text",
    angle_map: str = "identity",
) -> torch.Tensor:
    """Quantum fidelity kernel between compressed inputs and references.

    Parameters
    ----------
    z : torch.Tensor
        ``(B, n_qubits)`` compressed current inputs.
    z_reference : torch.Tensor
        ``(N, n_qubits)`` compressed reference vectors.
    layout : str
        Circuit layout. Default value is ``"text"``.
    angle_map : str
        Angle map. Default value is ``"identity"``.

    Returns
    -------
    torch.Tensor
        ``(B, N)`` kernel matrix.
    """
    states = feature_map_states(z, layout=layout, angle_map=angle_map)
    reference_states = feature_map_states(
        z_reference, layout=layout, angle_map=angle_map
    )
    return fidelity_kernel(states, reference_states)


def rbf_kernel(
    z: torch.Tensor, z_reference: torch.Tensor, *, gamma: float
) -> torch.Tensor:
    """Classical Gaussian kernel ``exp(-gamma ||z - z_j||^2)`` for the B3 ablation.

    Parameters
    ----------
    z : torch.Tensor
        ``(B, d)`` compressed current inputs.
    z_reference : torch.Tensor
        ``(N, d)`` compressed reference vectors.
    gamma : float
        Kernel bandwidth.

    Returns
    -------
    torch.Tensor
        ``(B, N)`` kernel matrix in ``(0, 1]``.
    """
    squared_distance = torch.cdist(z, z_reference).pow(2)
    return torch.exp(-gamma * squared_distance)
