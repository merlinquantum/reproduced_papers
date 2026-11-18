"""Quantum embeddings E1/E2/E3, the Heisenberg QNN, and per-dataset quantum banks.

Paper mapping (arXiv:2011.01938, App. J.1 and J.4; journal SI Section 12):

* E1, separable rotations: ``|x> = prod_j exp(-i X_j x_j) |0^n>``, i.e. RX(2 x_j).
* E2, IQP-style: ``|x> = U_Z(x) H^n U_Z(x) H^n |0^n>``. The printed ``U_Z`` lacks
  the imaginary unit (AMB-1); we use ``exp(i (sum_j x_j Z_j +
  c * sum_{j<j'} x_j x_j' Z_j Z_j'))`` with ``c = pair_coefficient`` (2 reproduces
  the paper's unordered double sum over j != j').
* E3, Hamiltonian evolution: ``T`` Trotter steps of a 1D Heisenberg chain with
  couplings ``x_j`` for time ``t``, applied to a fixed Haar-random product state on
  ``n + 1`` qubits (AMB-2; bonds j = 0..n-1 on an open chain).
* QNN labels: ``f(x) = <Z_1>`` after ``T = t = 10`` Trotter steps of a Heisenberg
  chain with couplings ``J_j ~ N(0, 1)`` on the embedding's register (AMB-3).

A "quantum bank" holds what all downstream analyses need for one embedding and
one data subsample: PQK features (single-qubit Pauli expectations), the
fidelity quantum kernel matrix, and QNN labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from . import simulator

LOGGER = logging.getLogger(__name__)

EMBEDDINGS = ("E1", "E2", "E3")


@dataclass(frozen=True)
class EmbeddingConventions:
    """Inferred circuit conventions (see LOG.md, Reproducibility Assessment).

    Attributes
    ----------
    e2_pair_coefficient : float
        Coefficient on each ``x_j x_j' Z_j Z_j'`` pair with j < j' in E2 (AMB-1).
    e2_phase_sign : float
        Sign of the imaginary unit in ``U_Z`` (AMB-1); +1 or -1.
    e3_register : str
        ``"n_plus_one"`` (paper equation) or ``"n"`` (journal caption) (AMB-2).
    e3_time : str
        ``"n_over_3"`` (paper) or ``"n_plus_one_over_3"`` (TFQ tutorial) (AMB-2).
    e3_trotter_steps : int
        Trotter steps of E3 (paper: 20).
    qnn_trotter_steps : int
        Trotter steps of the labelling QNN (paper: 10).
    qnn_total_time : float
        Total evolution time of the labelling QNN (paper: 10).
    qnn_boundary : str
        ``"open"`` or ``"periodic"`` chain for the QNN (AMB-3).
    qnn_observable_qubit : int
        Qubit whose Z expectation is the label (paper: Z_1, i.e. index 0).
    input_scale : float
        Factor applied to the standardized features before they enter the
        quantum embedding (paper text implies 1.0; the Source Data dimensions
        suggest a smaller effective scale, AMB-17). Classical models always see
        the unscaled standardized features.
    """

    e2_pair_coefficient: float
    e2_phase_sign: float
    e3_register: str
    e3_time: str
    e3_trotter_steps: int
    qnn_trotter_steps: int
    qnn_total_time: float
    qnn_boundary: str
    qnn_observable_qubit: int
    input_scale: float

    @classmethod
    def from_config(cls, conventions_cfg: dict) -> EmbeddingConventions:
        """Build conventions from the ``conventions`` config block.

        Raises
        ------
        KeyError
            If a required convention is missing.
        ValueError
            If a categorical convention has an unknown value.
        """
        conventions = cls(
            e2_pair_coefficient=float(conventions_cfg["e2_pair_coefficient"]),
            e2_phase_sign=float(conventions_cfg["e2_phase_sign"]),
            e3_register=str(conventions_cfg["e3_register"]),
            e3_time=str(conventions_cfg["e3_time"]),
            e3_trotter_steps=int(conventions_cfg["e3_trotter_steps"]),
            qnn_trotter_steps=int(conventions_cfg["qnn_trotter_steps"]),
            qnn_total_time=float(conventions_cfg["qnn_total_time"]),
            qnn_boundary=str(conventions_cfg["qnn_boundary"]),
            qnn_observable_qubit=int(conventions_cfg["qnn_observable_qubit"]),
            input_scale=float(conventions_cfg["input_scale"]),
        )
        if conventions.e3_register not in {"n_plus_one", "n"}:
            raise ValueError(f"Unknown e3_register: {conventions.e3_register}")
        if conventions.e3_time not in {"n_over_3", "n_plus_one_over_3"}:
            raise ValueError(f"Unknown e3_time: {conventions.e3_time}")
        if conventions.qnn_boundary not in {"open", "periodic"}:
            raise ValueError(f"Unknown qnn_boundary: {conventions.qnn_boundary}")
        if conventions.e2_phase_sign not in {1.0, -1.0}:
            raise ValueError("e2_phase_sign must be +1 or -1")
        if conventions.input_scale <= 0.0:
            raise ValueError("input_scale must be positive")
        return conventions


def register_size(
    embedding: str, n_features: int, conventions: EmbeddingConventions
) -> int:
    """Return the number of qubits used by ``embedding`` for ``n_features`` inputs."""
    if embedding in {"E1", "E2"}:
        return n_features
    if embedding == "E3":
        return n_features + 1 if conventions.e3_register == "n_plus_one" else n_features
    raise ValueError(f"Unknown embedding: {embedding}")


def sample_haar_product_state(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one Haar-random single-qubit state per qubit, shape ``(n_qubits, 2)``."""
    gaussian = rng.standard_normal((n_qubits, 2)) + 1j * rng.standard_normal(
        (n_qubits, 2)
    )
    return gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)


def _qnn_bonds(n_qubits: int, boundary: str) -> list[tuple[int, int]]:
    bonds = [(j, j + 1) for j in range(n_qubits - 1)]
    if boundary == "periodic" and n_qubits > 2:
        bonds.append((n_qubits - 1, 0))
    return bonds


def qnn_coupling_count(n_qubits: int, conventions: EmbeddingConventions) -> int:
    """Number of random couplings J_j the labelling QNN needs."""
    return len(_qnn_bonds(n_qubits, conventions.qnn_boundary))


def embed(
    embedding: str,
    features: torch.Tensor,
    conventions: EmbeddingConventions,
    *,
    haar_product_state: np.ndarray | None,
    dtype,
) -> torch.Tensor:
    """Prepare embedded states for a batch of feature vectors.

    Parameters
    ----------
    embedding : str
        ``"E1"``, ``"E2"`` or ``"E3"``.
    features : torch.Tensor
        Real tensor of shape ``(batch, n_features)`` on the target device.
    conventions : EmbeddingConventions
        Circuit conventions.
    haar_product_state : numpy.ndarray or None
        Fixed Haar-random product state for E3, shape ``(n_qubits, 2)``. Required
        for E3 and ignored otherwise.
    dtype : torch.dtype
        Complex dtype of the states.

    Returns
    -------
    torch.Tensor
        Batched states.
    """
    batch_size, n_features = features.shape
    device = features.device
    if embedding == "E1":
        # exp(-i x X)|0> = cos(x)|0> - i sin(x)|1>
        single = torch.stack(
            (torch.cos(features).to(dtype), -1j * torch.sin(features).to(dtype)), dim=-1
        )
        return simulator.product_state(single)

    if embedding == "E2":
        z_table = simulator.z_eigenvalue_table(
            n_features, dtype=features.dtype, device=device
        )
        linear_term = features @ z_table.T  # s = sum_j x_j z_j, shape (batch, 2**n)
        squared_norm = (features**2).sum(dim=1, keepdim=True)
        # sum_{j<j'} x_j x_j' z_j z_j' = (s^2 - sum_j x_j^2) / 2
        pair_term = 0.5 * (linear_term**2 - squared_norm)
        phases = conventions.e2_phase_sign * (
            linear_term + conventions.e2_pair_coefficient * pair_term
        )
        states = simulator.zero_state(
            batch_size, n_features, dtype=dtype, device=device
        )
        states = simulator.apply_hadamard_layer(states)
        states = simulator.apply_diagonal_phase(states, phases)
        states = simulator.apply_hadamard_layer(states)
        return simulator.apply_diagonal_phase(states, phases)

    if embedding == "E3":
        if haar_product_state is None:
            raise ValueError("E3 requires a Haar-random product state")
        n_qubits = register_size("E3", n_features, conventions)
        if haar_product_state.shape != (n_qubits, 2):
            raise ValueError(
                f"Haar state shape {haar_product_state.shape} does not match {n_qubits} qubits"
            )
        initial = torch.as_tensor(haar_product_state, dtype=dtype, device=device)
        states = simulator.product_state(
            initial.unsqueeze(0).expand(batch_size, -1, -1)
        )
        time_numerator = (
            n_features if conventions.e3_time == "n_over_3" else n_features + 1
        )
        step_scale = (time_numerator / 3.0) / conventions.e3_trotter_steps
        # Bond j couples qubits j and j+1 with strength x_j; with the n-qubit
        # register only the first n-1 features have a bond (AMB-2 variant).
        bond_count = min(n_features, n_qubits - 1)
        for _ in range(conventions.e3_trotter_steps):
            for bond in range(bond_count):
                states = simulator.apply_heisenberg_bond(
                    states, bond, bond + 1, step_scale * features[:, bond]
                )
        return states

    raise ValueError(f"Unknown embedding: {embedding}")


def apply_qnn(
    states: torch.Tensor, couplings: np.ndarray, conventions: EmbeddingConventions
) -> torch.Tensor:
    """Evolve states under the labelling QNN ``U_QNN`` (App. J.4, Dataset (Q, E*)).

    Parameters
    ----------
    states : torch.Tensor
        Batched embedded states.
    couplings : numpy.ndarray
        Couplings ``J_j``, one per bond.
    conventions : EmbeddingConventions
        QNN conventions.
    """
    n_qubits = states.ndim - 1
    bonds = _qnn_bonds(n_qubits, conventions.qnn_boundary)
    if len(couplings) != len(bonds):
        raise ValueError(f"Expected {len(bonds)} couplings, got {len(couplings)}")
    step_time = conventions.qnn_total_time / conventions.qnn_trotter_steps
    batch_size = states.shape[0]
    real_dtype = states.real.dtype
    for _ in range(conventions.qnn_trotter_steps):
        for (qubit_a, qubit_b), coupling in zip(bonds, couplings):
            theta = torch.full(
                (batch_size,),
                step_time * float(coupling),
                dtype=real_dtype,
                device=states.device,
            )
            states = simulator.apply_heisenberg_bond(states, qubit_a, qubit_b, theta)
    return states


def e1_pauli_features(features: np.ndarray) -> np.ndarray:
    """Exact E1 single-qubit Pauli expectations, shape ``(N, n, 3)`` (X, Y, Z).

    For ``cos(x)|0> - i sin(x)|1>``: <X> = 0, <Y> = -sin(2x), <Z> = cos(2x).
    """
    return np.stack(
        (np.zeros_like(features), -np.sin(2 * features), np.cos(2 * features)), axis=-1
    )


def e1_quantum_kernel(features: np.ndarray) -> np.ndarray:
    """Exact E1 fidelity kernel ``prod_j cos^2(x_ij - x_kj)``."""
    differences = features[:, None, :] - features[None, :, :]
    return np.prod(np.cos(differences) ** 2, axis=-1)


@dataclass
class QuantumBank:
    """Quantum quantities for one embedding and one data subsample.

    Attributes
    ----------
    pauli_features : numpy.ndarray
        Shape ``(N, n_qubits, 3)``.
    quantum_kernel : numpy.ndarray or None
        Fidelity kernel ``|<x_i|x_j>|^2``, shape ``(N, N)``; None if not requested.
    qnn_labels : numpy.ndarray or None
        ``<Z_1>`` after the QNN, shape ``(N,)``; None if not requested.
    n_qubits : int
        Register size.
    """

    pauli_features: np.ndarray
    quantum_kernel: np.ndarray | None
    qnn_labels: np.ndarray | None
    n_qubits: int


def _batch_size_for(
    n_qubits: int, n_points: int, memory_budget_mb: float, bytes_per_amp: int
) -> int:
    # Each gate creates a few temporaries of the batch size, so budget 4 copies.
    per_state = (2**n_qubits) * bytes_per_amp * 4
    return max(1, min(n_points, int(memory_budget_mb * 2**20 // per_state)))


def compute_quantum_bank(
    embedding: str,
    features: np.ndarray,
    conventions: EmbeddingConventions,
    *,
    haar_product_state: np.ndarray | None,
    qnn_couplings: np.ndarray | None,
    need_quantum_kernel: bool,
    device: str,
    complex_dtype,
    batch_memory_mb: float,
    e1_analytic: bool = True,
) -> QuantumBank:
    """Compute PQK features, the fidelity kernel and QNN labels for one subsample.

    Parameters
    ----------
    embedding : str
        Embedding name.
    features : numpy.ndarray
        Standardized PCA features, shape ``(N, n_features)``; multiplied by
        ``conventions.input_scale`` before embedding.
    conventions : EmbeddingConventions
        Circuit conventions.
    haar_product_state : numpy.ndarray or None
        Fixed E3 initial product state.
    qnn_couplings : numpy.ndarray or None
        QNN couplings; when None no QNN labels are computed.
    need_quantum_kernel : bool
        Whether to build the fidelity kernel (needs all states in memory).
    device : str
        Torch device for simulation.
    complex_dtype : torch.dtype
        Complex dtype for simulation.
    batch_memory_mb : float
        Memory budget per simulated batch.
    e1_analytic : bool
        Use closed forms for E1 kernels and features. Default value is True.

    Returns
    -------
    QuantumBank
        The computed quantities.
    """
    features = conventions.input_scale * features
    n_points, n_features = features.shape
    n_qubits = register_size(embedding, n_features, conventions)
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    bytes_per_amp = 16 if complex_dtype == torch.complex128 else 8

    analytic_e1 = embedding == "E1" and e1_analytic
    need_states = qnn_couplings is not None or not analytic_e1
    pauli_features = e1_pauli_features(features) if analytic_e1 else None
    quantum_kernel = (
        e1_quantum_kernel(features) if (analytic_e1 and need_quantum_kernel) else None
    )

    qnn_labels = None
    if need_states:
        batch_size = _batch_size_for(n_qubits, n_points, batch_memory_mb, bytes_per_amp)
        store_states = need_quantum_kernel and not analytic_e1
        stored = (
            torch.empty((n_points, 2**n_qubits), dtype=complex_dtype, device=device)
            if store_states
            else None
        )
        feature_chunks, label_chunks = [], []
        feature_tensor = torch.as_tensor(features, dtype=real_dtype, device=device)
        for start in range(0, n_points, batch_size):
            stop = min(start + batch_size, n_points)
            states = embed(
                embedding,
                feature_tensor[start:stop],
                conventions,
                haar_product_state=haar_product_state,
                dtype=complex_dtype,
            )
            if not analytic_e1:
                feature_chunks.append(
                    simulator.single_qubit_pauli_expectations(states).cpu().numpy()
                )
            if stored is not None:
                stored[start:stop] = states.reshape(stop - start, -1)
            if qnn_couplings is not None:
                evolved = apply_qnn(states, qnn_couplings, conventions)
                label_chunks.append(
                    simulator.z_expectation(evolved, conventions.qnn_observable_qubit)
                    .cpu()
                    .numpy()
                )
            LOGGER.debug(
                "Simulated %s states %d-%d of %d (%d qubits)",
                embedding,
                start,
                stop,
                n_points,
                n_qubits,
            )
        if not analytic_e1:
            pauli_features = np.concatenate(feature_chunks, axis=0)
        if qnn_couplings is not None:
            qnn_labels = np.concatenate(label_chunks).astype(np.float64)
        if stored is not None:
            quantum_kernel = _fidelity_gram(stored)

    return QuantumBank(
        pauli_features=np.asarray(pauli_features, dtype=np.float64),
        quantum_kernel=quantum_kernel,
        qnn_labels=qnn_labels,
        n_qubits=n_qubits,
    )


def _fidelity_gram(states: torch.Tensor, row_block: int = 256) -> np.ndarray:
    """Return ``|<psi_i|psi_j>|^2`` for stored flat states, computed in row blocks."""
    n_points = states.shape[0]
    gram = np.empty((n_points, n_points), dtype=np.float64)
    conjugated = states.conj()
    for start in range(0, n_points, row_block):
        stop = min(start + row_block, n_points)
        overlaps = conjugated[start:stop] @ states.T
        gram[start:stop] = (overlaps.abs() ** 2).double().cpu().numpy()
    # Pure states: the diagonal is exactly one; remove rounding drift.
    np.fill_diagonal(gram, 1.0)
    return 0.5 * (gram + gram.T)


def complex_dtype_for(precision_label: str | None):
    """Map a top-level float dtype label to the simulation complex dtype."""
    if precision_label in (None, "float64", "auto"):
        return torch.complex128
    if precision_label == "float32":
        return torch.complex64
    raise ValueError(f"Unsupported simulation precision: {precision_label}")


def statevector_bytes(n_qubits: int, complex_dtype) -> int:
    """Bytes of one statevector."""
    return (2**n_qubits) * (16 if complex_dtype == torch.complex128 else 8)


__all__ = [
    "EMBEDDINGS",
    "EmbeddingConventions",
    "QuantumBank",
    "apply_qnn",
    "complex_dtype_for",
    "compute_quantum_bank",
    "e1_pauli_features",
    "e1_quantum_kernel",
    "embed",
    "qnn_coupling_count",
    "register_size",
    "sample_haar_product_state",
    "statevector_bytes",
]
