"""Clements-layout meshes of phase-only Mach-Zehnder interferometers.

Paper Sec. 4.1 trains "photonic circuits arranged in a Clements scheme beam
splitter configuration, initialised at a Haar-random unitary". Here every
trainable parameter is a phase shift, so the generalised parameter-shift rule
(paper Eq. 143) applies with frequency bound ``R = observable order``.

Block convention (paper convention ``a_i^+ -> sum_j U_ij a_j^+``): an MZI on
modes ``(m, m + 1)`` has the 2x2 matrix

    MZI(delta, phi) = B @ diag(e^{i delta}, 1) @ B @ diag(e^{i phi}, 1),
    B = [[1, i], [i, 1]] / sqrt(2)

and the mesh unitary is ``U = diag(e^{i omega}) @ T_1 @ T_2 @ ... @ T_K`` with
``K = M (M - 1) / 2`` blocks in Clements order. ``diag(e^{i omega})`` acts on
the input modes of a Fock state and only adds a global phase, so it is kept
for exact reconstruction but never trained.

Haar initialisation decomposes a Haar unitary with the Clements nulling
procedure (Clements et al., Optica 3, 1460 (2016)): elements are nulled
alternately by right-multiplying inverse blocks and left-multiplying blocks,
then the leftover diagonal is commuted through the left blocks with
``MZI(delta, phi)^+ D = D' MZI(delta, phi')``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import perceval as pcvl

SPLITTER = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)


def mzi_matrix(delta: float, phi: float) -> np.ndarray:
    """2x2 matrix of one phase-only MZI (module docstring convention)."""
    return (
        SPLITTER
        @ np.diag([np.exp(1j * delta), 1.0])
        @ SPLITTER
        @ np.diag([np.exp(1j * phi), 1.0])
    )


@dataclass
class ClementsMesh:
    """A Clements mesh: block modes, phases and the untrained input phases.

    Attributes
    ----------
    modes : int
        Number of modes ``M``.
    block_modes : numpy.ndarray
        Upper mode ``m`` of each block, shape ``(K,)``; the block acts on
        ``(m, m + 1)``.
    internal : numpy.ndarray
        Internal phases ``delta``, shape ``(K,)``.
    external : numpy.ndarray
        External phases ``phi``, shape ``(K,)``.
    input_phases : numpy.ndarray
        Diagonal phases ``omega``, shape ``(M,)``.
    """

    modes: int
    block_modes: np.ndarray
    internal: np.ndarray
    external: np.ndarray
    input_phases: np.ndarray

    @property
    def parameter_count(self) -> int:
        """Number of trainable phases, ``M (M - 1)``."""
        return 2 * len(self.block_modes)

    def parameters(self) -> np.ndarray:
        """Trainable phases as ``[delta_1, phi_1, delta_2, phi_2, ...]``."""
        return np.stack([self.internal, self.external], axis=1).reshape(-1)

    def with_parameters(self, parameters: np.ndarray) -> ClementsMesh:
        """Copy of the mesh with new trainable phases (same layout)."""
        pairs = np.asarray(parameters, dtype=np.float64).reshape(-1, 2)
        return ClementsMesh(
            self.modes, self.block_modes, pairs[:, 0], pairs[:, 1], self.input_phases
        )

    def unitary(self) -> np.ndarray:
        """Mesh unitary in the paper convention."""
        total = np.diag(np.exp(1j * self.input_phases))
        for mode, delta, phi in zip(self.block_modes, self.internal, self.external):
            block = np.eye(self.modes, dtype=np.complex128)
            block[mode : mode + 2, mode : mode + 2] = mzi_matrix(delta, phi)
            total = total @ block
        return total


def _null_right(matrix: np.ndarray, row: int, column: int) -> tuple[float, float]:
    """Phases of the block on columns ``(column, column+1)`` nulling ``matrix[row, column]``
    when right-multiplied by its inverse."""
    left, right = matrix[row, column], matrix[row, column + 1]
    if abs(left) < 1e-300:
        return np.pi, 0.0
    delta = 2.0 * np.arctan2(abs(right), abs(left))
    phi = float(np.angle(left / right)) - np.pi if abs(right) > 1e-300 else 0.0
    return delta, phi


def _null_left(matrix: np.ndarray, row: int, column: int) -> tuple[float, float]:
    """Phases of the block on rows ``(row-1, row)`` nulling ``matrix[row, column]``
    when left-multiplied."""
    upper, lower = matrix[row - 1, column], matrix[row, column]
    if abs(lower) < 1e-300:
        return 0.0, 0.0
    delta = 2.0 * np.arctan2(abs(upper), abs(lower))
    phi = float(np.angle(lower / upper)) if abs(upper) > 1e-300 else 0.0
    return delta, phi


def _embed(modes: int, mode: int, delta: float, phi: float) -> np.ndarray:
    block = np.eye(modes, dtype=np.complex128)
    block[mode : mode + 2, mode : mode + 2] = mzi_matrix(delta, phi)
    return block


def decompose(unitary: np.ndarray) -> ClementsMesh:
    """Exact Clements decomposition of ``unitary`` into the mesh convention.

    Parameters
    ----------
    unitary : numpy.ndarray
        ``(M, M)`` unitary.

    Returns
    -------
    ClementsMesh
        Mesh with ``mesh.unitary() == unitary`` up to rounding.
    """
    modes = unitary.shape[0]
    work = np.array(unitary, dtype=np.complex128)
    right_blocks: list[tuple[int, float, float]] = []
    left_blocks: list[tuple[int, float, float]] = []
    for step, diagonal in enumerate(range(modes - 2, -1, -1)):
        if step % 2 == 0:
            for offset in reversed(range(modes - 1 - diagonal)):
                row, column = diagonal + offset + 1, offset
                delta, phi = _null_right(work, row, column)
                right_blocks.append((column, delta, phi))
                work = work @ _embed(modes, column, delta, phi).conj().T
        else:
            for offset in range(modes - 1 - diagonal):
                row, column = diagonal + offset + 1, offset
                delta, phi = _null_left(work, row, column)
                left_blocks.append((row - 1, delta, phi))
                work = _embed(modes, row - 1, delta, phi) @ work
    # Now L_k ... L_1 U R_1^+ ... R_q^+ = D, so U = L_1^+ ... L_k^+ D R_q ... R_1.
    diagonal_phases = np.diag(work).copy()
    commuted: list[tuple[int, float, float]] = []
    for mode, delta, phi in reversed(left_blocks):
        first, second = diagonal_phases[mode], diagonal_phases[mode + 1]
        new_phi = float(np.angle(first / second))
        diagonal_phases[mode] = -np.exp(-1j * (delta + phi)) * second
        diagonal_phases[mode + 1] = -np.exp(-1j * delta) * second
        commuted.append((mode, delta, new_phi))
    # U = D' M'_1 ... M'_k R_q ... R_1, with M'_1 produced last.
    ordered = list(reversed(commuted)) + list(reversed(right_blocks))
    return ClementsMesh(
        modes=modes,
        block_modes=np.array([mode for mode, _, _ in ordered], dtype=np.int64),
        internal=np.array([delta for _, delta, _ in ordered]),
        external=np.array([phi for _, _, phi in ordered]),
        input_phases=np.angle(diagonal_phases),
    )


def parameter_name(index: int) -> str:
    """Zero-padded Perceval parameter name, so name order equals index order."""
    return f"phi_{index:05d}"


def perceval_circuit(mesh: ClementsMesh) -> pcvl.Circuit:
    """Perceval circuit of the trainable part of ``mesh`` with named phases.

    Perceval composes ``a // b`` as "``a`` first" and represents a circuit by the
    transpose of the paper-convention matrix, so the paper product
    ``T_1 ... T_K`` (with the input-phase diagonal dropped) becomes the
    Perceval sequence ``T_1^T, ..., T_K^T``, and each
    ``MZI^T = diag(e^{i phi}) B diag(e^{i delta}) B`` is "B, PS(delta), B,
    PS(phi)" in time order. Parameter ``2k`` is ``delta_k`` and ``2k + 1`` is
    ``phi_k``.
    """
    circuit = pcvl.Circuit(mesh.modes)
    for block_index, mode in enumerate(mesh.block_modes):
        mode = int(mode)
        circuit.add(mode, pcvl.BS())
        circuit.add(mode, pcvl.PS(pcvl.P(parameter_name(2 * block_index))))
        circuit.add(mode, pcvl.BS())
        circuit.add(mode, pcvl.PS(pcvl.P(parameter_name(2 * block_index + 1))))
    return circuit


def loop_circuit_blocks(modes: int, loops: int) -> np.ndarray:
    """Upper modes of the beam splitters of a cascaded single-delay loop processor.

    A time-bin loop with a one-bin delay couples bin ``i`` with bin ``i + 1`` in
    sequence, i.e. a chain of beam splitters on ``(0, 1), (1, 2), ...``. ORCA's
    PT-2 has two such loops (paper Sec. 4.2, Fig. 8); both delays are assumed to
    be one bin (not stated in the paper).
    """
    return np.tile(np.arange(modes - 1), loops)


def beam_splitter_matrix(theta: float) -> np.ndarray:
    """``[[cos theta, i sin theta], [i sin theta, cos theta]]``: integer-spectrum angle."""
    return np.array(
        [[np.cos(theta), 1j * np.sin(theta)], [1j * np.sin(theta), np.cos(theta)]]
    )


def loop_unitary(modes: int, block_modes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Paper-convention unitary of a beam-splitter chain applied in list order.

    The first beam splitter in the list acts first on the photons, so with
    ``a_i^+ -> sum_j U_ij a_j^+`` the product is ``B_1 B_2 ... B_K``.
    """
    total = np.eye(modes, dtype=np.complex128)
    for mode, theta in zip(block_modes, angles):
        block = np.eye(modes, dtype=np.complex128)
        block[mode : mode + 2, mode : mode + 2] = beam_splitter_matrix(theta)
        total = total @ block
    return total


def perceval_loop_circuit(modes: int, block_modes: np.ndarray) -> pcvl.Circuit:
    """Perceval circuit of a beam-splitter chain with named angles.

    Perceval's ``BS(theta)`` has ``cos(theta / 2)`` entries, so callers pass
    ``2 * angle`` (``PhotonicExpectation(parameter_scale=2)``) to match
    :func:`beam_splitter_matrix`. Beam-splitter matrices are symmetric, so the
    Perceval sequence equals the list order of :func:`loop_unitary`.
    """
    circuit = pcvl.Circuit(modes)
    for index, mode in enumerate(block_modes):
        circuit.add(int(mode), pcvl.BS(theta=pcvl.P(parameter_name(index))))
    return circuit
