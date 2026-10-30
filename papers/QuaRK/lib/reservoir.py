"""QuaRK quantum reservoir as exact density-matrix evolution (paper Sec. 3.1).

Each sub-reservoir applies, per time step,

    rho <- E_lambda( W R_y(theta(z)) rho R_y(theta(z))^dag W^dag ),
    E_lambda(rho) = lambda * rho + (1 - lambda) |+><+|^{n},

with ``z = x @ Pi`` (Gaussian JL projection), ``theta(z) = pi (1 - 1e-6) tanh(z)``
and the Ising ring unitary ``W = R_x R_z R_zz`` (R_zz applied first). The authors
simulate the SWAP dilation of ``E_lambda`` (2n + 1 qubits, Fig. 2) in Qiskit Aer;
here the channel is applied in closed form on the n-qubit density matrix, which
gives the same reduced state (checked in ``tests/test_reservoir.py``).

Qubit convention: as in Qiskit, qubit ``q`` is bit ``q`` of the basis index, so a
multi-qubit operator is ``kron(op_{n-1}, ..., op_0)``. Feature columns therefore
follow the authors' ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANGLE_SCALE = np.pi * (1.0 - 1e-6)


def ring_edges(num_qubits: int) -> list[tuple[int, int]]:
    """Ring edges ``(0,1), (1,2), ..., (n-1,0)`` in the authors' order."""
    edges = [(q, q + 1) for q in range(num_qubits - 1)]
    if num_qubits > 2:
        edges.append((num_qubits - 1, 0))
    return edges


def kron_qubits(ops_by_qubit: list[np.ndarray]) -> np.ndarray:
    """Tensor single-qubit operators given for qubits ``0..n-1`` (qubit 0 least significant)."""
    result = ops_by_qubit[-1]
    for op in reversed(ops_by_qubit[:-1]):
        result = np.kron(result, op)
    return result


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def ising_unitary(num_qubits, zz_angles, x_angles, z_angles) -> np.ndarray:
    """Fixed reservoir unitary ``W`` of Eq. (8): R_zz on the ring, then R_z, then R_x."""
    dim = 2**num_qubits
    bits = (np.arange(dim)[:, None] >> np.arange(num_qubits)[None, :]) & 1
    spins = 1 - 2 * bits  # Z eigenvalue of each qubit in each basis state
    phase = np.zeros(dim)
    for angle, (q1, q2) in zip(zz_angles, ring_edges(num_qubits)):
        phase += -0.5 * angle * spins[:, q1] * spins[:, q2]
    zz_layer = np.diag(np.exp(1j * phase))
    z_layer = kron_qubits([_rz(a) for a in z_angles])
    x_layer = kron_qubits([_rx(a) for a in x_angles])
    return x_layer @ z_layer @ zz_layer


def batched_ry_layer(angles: np.ndarray) -> np.ndarray:
    """``kron`` of per-qubit R_y gates for a batch of angle vectors of shape ``(B, n)``."""
    cos, sin = np.cos(angles / 2), np.sin(angles / 2)
    gates = np.stack(
        [np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)], axis=-2
    )  # (B, n, 2, 2)
    result = gates[:, -1]
    for q in range(angles.shape[1] - 2, -1, -1):
        b, a1, a2 = result.shape
        result = np.einsum("bij,bkl->bikjl", result, gates[:, q]).reshape(
            b, 2 * a1, 2 * a2
        )
    return result


@dataclass
class ReservoirParameters:
    """Random reservoir parameters, drawn as in the authors' circuit factory.

    Attributes
    ----------
    projection : numpy.ndarray
        JL matrix of shape ``(d, n)`` with entries ``N(0, 1/n)``; ``z = x @ projection``.
    zz_angles, x_angles, z_angles : numpy.ndarray
        Angles of shape ``(R, n)``, uniform in ``(-pi, pi)`` (one zz angle per ring edge).
    contractions : numpy.ndarray
        ``lambda_r`` of shape ``(R,)``: ``lambda_1`` fixed, the others uniform in ``(eps, 1 - eps)``.
    """

    projection: np.ndarray
    zz_angles: np.ndarray
    x_angles: np.ndarray
    z_angles: np.ndarray
    contractions: np.ndarray


def draw_reservoir_parameters(
    *, input_dim, num_qubits, num_reservoirs, first_contraction, seed, eps=1e-8
) -> ReservoirParameters:
    """Draw the JL projection and the sub-reservoir parameters.

    The projection and the reservoir angles use two separate ``default_rng(seed)``
    streams, as in ``RingQRConfig`` and ``set_reservoirs_parameterizationSWAP``.
    """
    projection = np.random.default_rng(seed).normal(
        0.0, 1.0 / np.sqrt(num_qubits), size=(input_dim, num_qubits)
    )
    rng = np.random.default_rng(seed)
    num_edges = len(ring_edges(num_qubits))
    zz_angles = rng.uniform(-np.pi, np.pi, size=(num_reservoirs, num_edges))
    x_angles = rng.uniform(-np.pi, np.pi, size=(num_reservoirs, num_qubits))
    z_angles = rng.uniform(-np.pi, np.pi, size=(num_reservoirs, num_qubits))
    contractions = np.empty(num_reservoirs)
    contractions[0] = first_contraction
    contractions[1:] = rng.uniform(eps, 1 - eps, size=num_reservoirs - 1)
    return ReservoirParameters(projection, zz_angles, x_angles, z_angles, contractions)


def contraction_channel(rho: np.ndarray, contraction: float) -> np.ndarray:
    """Reset-rate channel ``lambda * rho + (1 - lambda) |+><+|`` (Eq. 9), batched."""
    dim = rho.shape[-1]
    return contraction * rho + (1.0 - contraction) / dim


def reservoir_states(
    windows: np.ndarray, params: ReservoirParameters, *, chunk_size: int = 2000
) -> np.ndarray:
    """Final density matrices of every window in every sub-reservoir.

    Parameters
    ----------
    windows : numpy.ndarray
        Windows of shape ``(N, w, d)``, oldest point first.
    params : ReservoirParameters
        Projection and sub-reservoir parameters.
    chunk_size : int
        Windows processed together (memory bound). Default value is 2000.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(N, R, 2**n, 2**n)``.
    """
    num_windows, window, _ = windows.shape
    num_qubits = params.projection.shape[1]
    num_reservoirs = params.contractions.shape[0]
    dim = 2**num_qubits
    unitaries = [
        ising_unitary(
            num_qubits, params.zz_angles[r], params.x_angles[r], params.z_angles[r]
        )
        for r in range(num_reservoirs)
    ]
    angles = ANGLE_SCALE * np.tanh(windows @ params.projection)  # (N, w, n)
    states = np.empty((num_windows, num_reservoirs, dim, dim), dtype=complex)
    for start in range(0, num_windows, chunk_size):
        stop = min(start + chunk_size, num_windows)
        injections = [batched_ry_layer(angles[start:stop, t]) for t in range(window)]
        for r in range(num_reservoirs):
            rho = np.full((stop - start, dim, dim), 1.0 / dim, dtype=complex)
            for t in range(window):
                step = unitaries[r] @ injections[t]
                rho = step @ rho @ step.conj().transpose(0, 2, 1)
                rho = contraction_channel(rho, params.contractions[r])
            states[start:stop, r] = rho
    return states
