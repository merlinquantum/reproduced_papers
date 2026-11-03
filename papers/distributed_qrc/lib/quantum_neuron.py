"""Exact simulation of the paper's 4-qubit "quantum neuron" (Sec. II-A, Fig. 2).

Circuit of one neuron with fixed parameters ``w = (w0, w1, w2, w3)``:

* feature map: ``RX(x_q)`` on qubit ``q`` (angle encoding, Fig. 2a);
* ansatz: ``B = 2`` structural blocks, each ``RX(w_j)`` on every qubit, then for
  ``j = 1, 2, 3`` the sequence ``CX(j-1 -> j)``, ``RZ(w[j-1] - w[j])`` on qubit
  ``j``, ``CX(j-1 -> j)`` (Fig. 2b, a rendered Qiskit circuit; it is followed
  instead of Eq. 1, whose angle factors differ);
* output: the expectation of a Pauli-Z observable (Eq. 2; which qubits is not
  stated; ``z_all`` is our choice, see `docs/reproduction_details.md`).

Qiskit conventions are used: ``RX(t) = exp(-i t X / 2)``, ``RZ(t) = exp(-i t Z / 2)``.
Tensor axes are ordered qubit 0 .. qubit 3.

The ansatz, its noise channels, the readout error and the observable do not
depend on the data. For each neuron they are therefore folded once into a
16 x 16 matrix ``M`` with ``<O> = sum_ij M[i, j] rho_in[i, j]``, where
``rho_in`` is the (noisy) encoded product state. Evaluating a batch of windows
then costs one tensor contraction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .noise import IDEAL, NATIVE_1Q_PER_CX_QUBIT, NATIVE_1Q_PER_RX, DeviceNoise

N_QUBITS = 4
DIM = 2**N_QUBITS
ANSATZ_BLOCKS = 2

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULIS = (_I2, _X, _Y, _Z)
_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
).reshape(2, 2, 2, 2)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(np.complex128)


# --- batched density-matrix gates; rho has shape (K, 2,2,2,2, 2,2,2,2) ---------


def _apply_1q(rho: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    row, col = 1 + qubit, 1 + N_QUBITS + qubit
    rho = np.moveaxis(np.tensordot(gate, rho, axes=([1], [row])), 0, row)
    return np.moveaxis(np.tensordot(rho, gate.conj(), axes=([col], [1])), -1, col)


def _apply_2q(rho: np.ndarray, gate: np.ndarray, q1: int, q2: int) -> np.ndarray:
    rows = (1 + q1, 1 + q2)
    cols = (1 + N_QUBITS + q1, 1 + N_QUBITS + q2)
    rho = np.moveaxis(np.tensordot(gate, rho, axes=([2, 3], list(rows))), (0, 1), rows)
    rho = np.tensordot(rho, gate.conj(), axes=(list(cols), [2, 3]))
    return np.moveaxis(rho, (-2, -1), cols)


def _depolarize_1q(rho: np.ndarray, qubit: int, shrink: float) -> np.ndarray:
    """``rho -> shrink rho + (1 - shrink) I/2 (x) Tr_q rho`` via the Pauli twirl."""

    if shrink == 1.0:
        return rho
    twirl = sum(_apply_1q(rho, pauli, qubit) for pauli in _PAULIS) / 4
    return shrink * rho + (1 - shrink) * twirl


def _depolarize_2q(rho: np.ndarray, q1: int, q2: int, lam: float) -> np.ndarray:
    if lam == 1.0:
        return rho
    twirl = np.zeros_like(rho)
    for p1 in _PAULIS:
        partial = _apply_1q(rho, p1, q1)
        for p2 in _PAULIS:
            twirl += _apply_1q(partial, p2, q2)
    return lam * rho + (1 - lam) * twirl / 16


def apply_ansatz_channel(
    rho: np.ndarray, weights: Sequence[float], noise: DeviceNoise
) -> np.ndarray:
    """Apply the (noisy) Fig. 2b ansatz to a batch of 4-qubit density matrices."""

    shrink_rx = noise.bloch_shrink_1q**NATIVE_1Q_PER_RX
    shrink_cx_1q = noise.bloch_shrink_1q**NATIVE_1Q_PER_CX_QUBIT
    lam_2q = noise.depolarizing_2q

    def noisy_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
        state = _apply_2q(state, _CX, control, target)
        state = _depolarize_2q(state, control, target, lam_2q)
        state = _depolarize_1q(state, control, shrink_cx_1q)
        return _depolarize_1q(state, target, shrink_cx_1q)

    for _ in range(ANSATZ_BLOCKS):
        for qubit in range(N_QUBITS):
            rho = _apply_1q(rho, rx(weights[qubit]), qubit)
            rho = _depolarize_1q(rho, qubit, shrink_rx)
        for target in range(1, N_QUBITS):
            control = target - 1
            rho = noisy_cx(rho, control, target)
            # RZ is a virtual gate on IBM and IonQ hardware: no error channel.
            rho = _apply_1q(rho, rz(weights[control] - weights[target]), target)
            rho = noisy_cx(rho, control, target)
    return rho


def observable_diagonal(observable: str, readout_error: float) -> np.ndarray:
    """Diagonal of the measured observable, including symmetric readout bit flips.

    ``"z_all"`` is ``Z (x) Z (x) Z (x) Z`` (the EstimatorQNN default), ``"z0"`` is Z
    on qubit 0. A bit flip with probability p maps each measured Z to ``(1-2p) Z``.
    """

    bits = (np.arange(DIM)[:, None] >> (N_QUBITS - 1 - np.arange(N_QUBITS))) & 1
    signs = 1 - 2 * bits  # (16, 4): Z eigenvalue of each qubit, qubit 0 first
    damping = 1 - 2 * readout_error
    if observable == "z_all":
        return np.prod(signs, axis=1) * damping**N_QUBITS
    if observable == "z0":
        return signs[:, 0] * damping
    raise ValueError(f"observable must be 'z_all' or 'z0', got {observable!r}")


def neuron_functional(
    weights: Sequence[float], noise: DeviceNoise, observable: str
) -> np.ndarray:
    """Return ``M`` (16 x 16) with ``<O> = sum_ij M[i,j] rho_in[i,j]`` for this neuron."""

    basis = np.zeros((DIM * DIM, DIM, DIM), dtype=np.complex128)
    basis[
        np.arange(DIM * DIM),
        np.repeat(np.arange(DIM), DIM),
        np.tile(np.arange(DIM), DIM),
    ] = 1
    rho = basis.reshape((DIM * DIM,) + (2,) * (2 * N_QUBITS))
    out = apply_ansatz_channel(rho, weights, noise).reshape(DIM * DIM, DIM, DIM)
    diag = observable_diagonal(observable, noise.readout_error)
    values = np.einsum("kii,i->k", out, diag)
    return values.reshape(DIM, DIM)


def encoded_qubit_states(angles: np.ndarray, noise: DeviceNoise) -> np.ndarray:
    """Single-qubit density matrices after (noisy) ``RX(angle)|0>``; shape ``angles.shape + (2, 2)``."""

    # RX(theta)|0> = c|0> - i s|1>, with c = cos(theta/2), s = sin(theta/2).
    # rho = |psi><psi|, i.e. rho_ij = psi_i * conj(psi_j):
    #   rho_00 = c^2, rho_11 = s^2, rho_01 = c * conj(-i s) = i c s, rho_10 = -i c s.
    # The encoding acts on each qubit separately, so the neuron's input state is
    # the product of these 2x2 matrices (contracted in neuron_expectations).
    c, s = np.cos(angles / 2), np.sin(angles / 2)
    rho = np.empty(angles.shape + (2, 2), dtype=np.complex128)
    rho[..., 0, 0] = c * c
    rho[..., 1, 1] = s * s
    rho[..., 0, 1] = 1j * c * s
    rho[..., 1, 0] = -1j * c * s
    # Noisy device: depolarizing error of the native gates of RX shrinks the
    # Bloch vector, rho -> p rho + (1 - p) I/2 (a mixed state, hence a density matrix).
    shrink = noise.bloch_shrink_1q**NATIVE_1Q_PER_RX
    if shrink != 1.0:
        rho = shrink * rho + (1 - shrink) * 0.5 * _I2
    return rho


def neuron_expectations(
    functionals: np.ndarray,
    angles: np.ndarray,
    noise: DeviceNoise = IDEAL,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Expectation values of N neurons on a batch of encodings.

    Parameters
    ----------
    functionals : numpy.ndarray
        ``(N, 16, 16)`` matrices from :func:`neuron_functional`.
    angles : numpy.ndarray
        ``(B, N, 4)`` RX angles, one row of 4 per neuron and sample.
    noise : DeviceNoise
        Device whose single-qubit error applies to the encoding RX gates.
        Default value is ``IDEAL``.
    chunk_size : int
        Number of samples contracted at once, to bound memory. Default value is 4096.

    Returns
    -------
    numpy.ndarray
        ``(B, N)`` real expectation values.
    """

    n_neurons = functionals.shape[0]
    tensor = functionals.reshape((n_neurons,) + (2,) * (2 * N_QUBITS))
    outputs = np.empty(angles.shape[:2], dtype=np.float64)
    for start in range(0, angles.shape[0], chunk_size):
        rho = encoded_qubit_states(angles[start : start + chunk_size], noise)
        # Contract qubit 3, 2, 1, 0 in turn: M[n, i0..i3, j0..j3] rho_q[b, n, i_q, j_q].
        t = np.einsum("nabcdefgh,zndh->znabcefg", tensor, rho[:, :, 3])
        t = np.einsum("znabcefg,zncg->znabef", t, rho[:, :, 2])
        t = np.einsum("znabef,znbf->znae", t, rho[:, :, 1])
        t = np.einsum("znae,znae->zn", t, rho[:, :, 0])
        outputs[start : start + chunk_size] = t.real
    return outputs
