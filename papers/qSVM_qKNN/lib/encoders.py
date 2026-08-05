"""Quantum and photonic encoders for the Sodar qSVM/qKNN reproduction."""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pennylane as qml

Encoding_type = Literal["angle", "amplitude", "zz"]
HybridReadout_type = Literal["pauli_z", "state_real", "state_real_imag"]
PhotonicComputationSpace_type = Literal["FOCK", "UNBUNCHED"]


def _effective_n_jobs(
    n_jobs: int | None,
    *,
    total_rows: int,
    min_rows: int,
) -> int:
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs <= 1 or total_rows < int(min_rows):
        return 1
    return max(1, min(n_jobs, total_rows))


def _row_chunk_size(
    total_rows: int,
    n_jobs: int,
    chunk_size: int | None,
) -> int:
    if chunk_size is not None and int(chunk_size) > 0:
        return int(chunk_size)
    return max(1, math.ceil(total_rows / max(1, 4 * n_jobs)))


def _row_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + chunk_size, total_rows))
        for start in range(0, total_rows, chunk_size)
    ]


def parallel_encode_rows(
    X: np.ndarray,
    worker: Callable[[tuple[int, np.ndarray, dict[str, Any]]], tuple[int, np.ndarray]],
    worker_kwargs: dict[str, Any],
    *,
    n_jobs: int | None = 1,
    chunk_size: int | None = None,
    min_rows: int = 64,
) -> np.ndarray:
    """Apply a chunk encoder with optional process-level parallelism.

    Parameters
    ----------
    X : np.ndarray
        Input rows to encode.
    worker : callable
        Top-level worker function receiving ``(start, X_chunk, kwargs)`` and
        returning ``(start, encoded_chunk)``. Workers should encode the whole
        chunk in one vectorized backend call whenever the backend supports
        batched inputs.
    worker_kwargs : dict
        Keyword payload copied to each worker.
    n_jobs : int | None, optional
        Number of worker processes. ``-1`` uses all available CPU cores.
        Default value is 1.
    chunk_size : int | None, optional
        Number of rows per worker task. If omitted, a moderate automatic size
        is selected. Default value is None.
    min_rows : int, optional
        Minimum row count before spawning workers. Default value is 64.

    Returns
    -------
    np.ndarray
        Encoded rows in the original order.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Parallel row encoders expect a 2D input matrix.")
    jobs = _effective_n_jobs(n_jobs, total_rows=X.shape[0], min_rows=min_rows)
    size = _row_chunk_size(X.shape[0], jobs, chunk_size)
    chunks = _row_chunks(X.shape[0], size)
    tasks = [(start, X[start:stop], dict(worker_kwargs)) for start, stop in chunks]
    if jobs == 1 or len(tasks) == 1:
        parts = [worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            parts = list(pool.map(worker, tasks))
    parts.sort(key=lambda item: item[0])
    return np.vstack([part for _, part in parts])


def _as_batched_state_array(output: Any, expected_rows: int) -> np.ndarray:
    """Convert PennyLane batched ``qml.state`` output to ``(rows, state_dim)``."""

    states = np.asarray(output, dtype=np.complex128)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.shape[0] != expected_rows and states.shape[-1] == expected_rows:
        states = np.moveaxis(states, -1, 0)
    if states.shape[0] != expected_rows:
        raise ValueError(
            "Batched state output has unexpected shape "
            f"{states.shape}; expected first dimension {expected_rows}."
        )
    return states


def _as_batched_expval_array(output: Any, expected_rows: int) -> np.ndarray:
    """Convert PennyLane batched expvals to ``(rows, n_observables)``."""

    features = np.asarray(output, dtype=np.float64)
    if features.ndim == 1:
        if expected_rows == 1:
            return features.reshape(1, -1)
        features = features.reshape(expected_rows, 1)
    if features.shape[0] != expected_rows and features.shape[-1] == expected_rows:
        features = np.moveaxis(features, -1, 0)
    if features.shape[0] != expected_rows:
        raise ValueError(
            "Batched expectation output has unexpected shape "
            f"{features.shape}; expected first dimension {expected_rows}."
        )
    return features.astype(np.float64, copy=False)


def _all_zero_probabilities(output: Any, expected_rows: int) -> np.ndarray:
    """Extract all-zero probabilities from batched ``qml.probs`` output."""

    probs = np.asarray(output, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if probs.shape[0] != expected_rows and probs.shape[-1] == expected_rows:
        probs = np.moveaxis(probs, -1, 0)
    if probs.shape[0] != expected_rows:
        raise ValueError(
            "Batched probability output has unexpected shape "
            f"{probs.shape}; expected first dimension {expected_rows}."
        )
    return probs[:, 0]


def parallel_kernel_matrix(
    A: np.ndarray,
    B: np.ndarray,
    worker: Callable[
        [tuple[int, np.ndarray, np.ndarray, dict[str, Any]]],
        tuple[int, np.ndarray],
    ],
    worker_kwargs: dict[str, Any],
    *,
    n_jobs: int | None = 1,
    chunk_size: int | None = None,
    min_rows: int = 64,
    symmetric: bool = False,
    lower_triangle: bool = False,
) -> np.ndarray:
    """Compute a row-chunked kernel matrix with optional multiprocessing.

    Parameters
    ----------
    A : np.ndarray
        Left input matrix.
    B : np.ndarray
        Right input matrix.
    worker : callable
        Top-level worker receiving ``(start, A_chunk, B, kwargs)``.
    worker_kwargs : dict
        Keyword payload copied to each worker.
    n_jobs : int | None, optional
        Number of worker processes. ``-1`` uses all available CPU cores.
        Default value is 1.
    chunk_size : int | None, optional
        Number of left rows per worker task. Default value is None.
    min_rows : int, optional
        Minimum row count before spawning workers. Default value is 64.
    symmetric : bool, optional
        Whether the output is a square training Gram matrix. Default value is
        False.
    lower_triangle : bool, optional
        Whether workers only fill the strict lower triangle and diagonal for a
        symmetric kernel. Default value is False.

    Returns
    -------
    np.ndarray
        Kernel matrix with shape ``(A.shape[0], B.shape[0])``.
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Parallel kernel inputs must be 2D arrays.")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Kernel feature dimensions differ: {A.shape[1]} for A, {B.shape[1]} for B."
        )
    if symmetric and A.shape[0] != B.shape[0]:
        raise ValueError("Symmetric kernels require square A/B row dimensions.")

    jobs = _effective_n_jobs(n_jobs, total_rows=A.shape[0], min_rows=min_rows)
    size = _row_chunk_size(A.shape[0], jobs, chunk_size)
    tasks = [
        (start, A[start:stop], B, dict(worker_kwargs))
        for start, stop in _row_chunks(A.shape[0], size)
    ]
    if jobs == 1 or len(tasks) == 1:
        parts = [worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            parts = list(pool.map(worker, tasks))

    matrix = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
    for start, block in parts:
        matrix[start : start + block.shape[0], :] = block
    if symmetric:
        if lower_triangle:
            matrix = matrix + matrix.T
        else:
            matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 1.0)
    return matrix


def nearest_psd_kernel(matrix: np.ndarray) -> np.ndarray:
    """Project a square kernel matrix to a PSD correlation-like matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square kernel matrix to symmetrise and project.

    Returns
    -------
    np.ndarray
        Symmetric positive-semidefinite matrix with unit diagonal when the
        diagonal is numerically positive.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PSD projection requires a square matrix.")
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 0.0, None)
    psd = (eigvecs * eigvals) @ eigvecs.T
    psd = 0.5 * (psd + psd.T)
    diag = np.diag(psd)
    if np.all(diag > 1e-12):
        scale = 1.0 / np.sqrt(diag)
        psd = scale[:, None] * psd * scale[None, :]
    np.fill_diagonal(psd, 1.0)
    return psd


def required_qubits(encoding: Encoding_type, n_features: int) -> int:
    """Compute the number of wires required by an encoding.

    Parameters
    ----------
    encoding : {"angle", "amplitude", "zz"}
        Quantum encoding or feature map.
    n_features : int
        Number of classical input features.

    Returns
    -------
    int
        Required qubit/wire count.
    """

    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if encoding in {"angle", "zz"}:
        return n_features
    if encoding == "amplitude":
        return int(np.ceil(np.log2(n_features)))
    raise ValueError(f"Unsupported encoding {encoding!r}.")


def _normalise_vector(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= 1e-12:
        return x
    return x / norm


def _normalise_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=X.copy(), where=norms > 1e-12)


def _entangling_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entanglement == "reverse_linear":
        return [(i, i - 1) for i in range(n_qubits - 1, 0, -1)]
    raise ValueError(
        "Supported entanglement values are: 'full', 'linear', 'reverse_linear'."
    )


def qiskit_like_zz_feature_map(
    inputs,
    n_qubits: int,
    *,
    reps: int = 1,
    entanglement: str = "full",
    alpha: float = 2.0,
) -> None:
    """Apply a PennyLane ZZFeatureMap compatible with Qiskit's convention.

    Parameters
    ----------
    inputs : array-like
        Input angles, one per qubit.
    n_qubits : int
        Number of qubits and input features.
    reps : int, optional
        Number of repeated feature-map blocks. Default value is 1.
    entanglement : {"full", "linear", "reverse_linear"}, optional
        Pairing pattern for ZZ terms. Default value is "full".
    alpha : float, optional
        Global feature-map scale. Default value is 2.0.
    """

    if reps <= 0:
        raise ValueError("ZZ feature-map reps must be positive.")
    pairs = _entangling_pairs(n_qubits, entanglement)

    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.PhaseShift(alpha * inputs[..., i], wires=i)

        for i, j in pairs:
            qml.CNOT(wires=[i, j])
            qml.PhaseShift(
                alpha * (math.pi - inputs[..., i]) * (math.pi - inputs[..., j]),
                wires=j,
            )
            qml.CNOT(wires=[i, j])


def _apply_gate_based_encoding(
    x,
    *,
    encoding: Encoding_type,
    wires,
    n_qubits: int,
    angle_rotation: str,
    zz_reps: int,
    zz_entanglement: str,
    zz_alpha: float,
) -> None:
    if encoding == "angle":
        qml.AngleEmbedding(x, wires=wires, rotation=angle_rotation)
        return
    if encoding == "amplitude":
        qml.AmplitudeEmbedding(x, wires=wires, pad_with=0.0, normalize=True)
        return
    if encoding == "zz":
        qiskit_like_zz_feature_map(
            x,
            n_qubits,
            reps=zz_reps,
            entanglement=zz_entanglement,
            alpha=zz_alpha,
        )
        return
    raise ValueError(f"Unsupported encoding {encoding!r}.")


def _qsvm_fidelity_kernel_qnode(
    encoding: Encoding_type,
    n_qubits: int,
    *,
    device_name: str,
    angle_rotation: str,
    zz_reps: int,
    zz_entanglement: str,
    zz_alpha: float,
):
    dev = qml.device(device_name, wires=n_qubits)
    wires = range(n_qubits)

    def apply_embedding(x):
        _apply_gate_based_encoding(
            x,
            encoding=encoding,
            wires=wires,
            n_qubits=n_qubits,
            angle_rotation=angle_rotation,
            zz_reps=zz_reps,
            zz_entanglement=zz_entanglement,
            zz_alpha=zz_alpha,
        )

    @qml.qnode(dev)
    def kernel_circuit(a, b):
        apply_embedding(a)
        qml.adjoint(apply_embedding)(b)
        return qml.probs(wires=wires)

    return kernel_circuit


@dataclass
class GateFidelityKernelEncoder:
    """Reusable PennyLane fidelity-kernel encoder.

    Parameters
    ----------
    n_features : int
        Number of input features before encoding.
    encoding : {"angle", "amplitude", "zz"}
        Gate-based feature map.
    device_name : str, optional
        PennyLane device name. Default value is "lightning.qubit".
    angle_rotation : str, optional
        Axis used for angle encoding. Default value is "X".
    zz_reps : int, optional
        Number of ZZ feature-map repetitions. Default value is 1.
    zz_entanglement : str, optional
        ZZ feature-map entanglement pattern. Default value is "full".
    zz_alpha : float, optional
        Global ZZ feature-map scale. Default value is 2.0.
    n_jobs : int | None, optional
        Number of process workers for CPU row chunks. Default value is 1.
    parallel_chunk_size : int | None, optional
        Rows per worker task or serial batch. Default value is None.
    parallel_min_rows : int, optional
        Minimum rows before spawning process workers. Default value is 64.
    """

    n_features: int
    encoding: Encoding_type
    device_name: str = "lightning.qubit"
    angle_rotation: str = "X"
    zz_reps: int = 1
    zz_entanglement: str = "full"
    zz_alpha: float = 2.0
    n_jobs: int | None = 1
    parallel_chunk_size: int | None = None
    parallel_min_rows: int = 64

    def __post_init__(self) -> None:
        """Prepare the PennyLane device and QNode once for serial inference."""

        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")
        self.n_qubits = required_qubits(self.encoding, int(self.n_features))
        self.circuit = _qsvm_fidelity_kernel_qnode(
            self.encoding,
            self.n_qubits,
            device_name=self.device_name,
            angle_rotation=self.angle_rotation,
            zz_reps=int(self.zz_reps),
            zz_entanglement=self.zz_entanglement,
            zz_alpha=float(self.zz_alpha),
        )

    def kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute a fidelity kernel while reusing the prepared QNode."""

        A = self._validate_inputs(A)
        B = self._validate_inputs(B)
        symmetric = A.shape == B.shape and np.array_equal(A, B)
        jobs = _effective_n_jobs(
            self.n_jobs,
            total_rows=A.shape[0],
            min_rows=int(self.parallel_min_rows),
        )
        if jobs > 1:
            return parallel_kernel_matrix(
                A,
                B,
                _qsvm_fidelity_kernel_chunk_worker,
                self._worker_kwargs(symmetric),
                n_jobs=jobs,
                chunk_size=self.parallel_chunk_size,
                min_rows=int(self.parallel_min_rows),
                symmetric=symmetric,
                lower_triangle=symmetric,
            )

        size = _row_chunk_size(A.shape[0], 1, self.parallel_chunk_size)
        matrix = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
        for start, stop in _row_chunks(A.shape[0], size):
            block = self._kernel_block(A[start:stop], B, start, symmetric)
            matrix[start : start + block.shape[0], :] = block
        if symmetric:
            matrix = matrix + matrix.T
            np.fill_diagonal(matrix, 1.0)
        return matrix

    def _kernel_block(
        self,
        A_chunk: np.ndarray,
        B: np.ndarray,
        start: int,
        symmetric: bool,
    ) -> np.ndarray:
        block = np.empty((A_chunk.shape[0], B.shape[0]), dtype=np.float64)
        if symmetric:
            block.fill(0.0)
            for local_i, a in enumerate(A_chunk):
                global_i = start + local_i
                left = B[: global_i + 1]
                block[local_i, : global_i + 1] = _all_zero_probabilities(
                    self.circuit(a, left),
                    expected_rows=left.shape[0],
                )
            return block
        for i, a in enumerate(A_chunk):
            block[i, :] = _all_zero_probabilities(
                self.circuit(a, B),
                expected_rows=B.shape[0],
            )
        return block

    def _worker_kwargs(self, symmetric: bool) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "device_name": self.device_name,
            "angle_rotation": self.angle_rotation,
            "zz_reps": int(self.zz_reps),
            "zz_entanglement": self.zz_entanglement,
            "zz_alpha": float(self.zz_alpha),
            "symmetric": symmetric,
        }

    def _validate_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Gate fidelity-kernel inputs must be 2D arrays.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Gate fidelity-kernel encoder expected {self.n_features} "
                f"features, got {X.shape[1]}."
            )
        return X


def _qsvm_fidelity_kernel_chunk_worker(
    task: tuple[int, np.ndarray, np.ndarray, dict[str, Any]],
) -> tuple[int, np.ndarray]:
    start, A_chunk, B, kwargs = task
    encoder = GateFidelityKernelEncoder(
        n_features=A_chunk.shape[1],
        encoding=kwargs["encoding"],
        device_name=kwargs["device_name"],
        angle_rotation=kwargs["angle_rotation"],
        zz_reps=int(kwargs["zz_reps"]),
        zz_entanglement=kwargs["zz_entanglement"],
        zz_alpha=float(kwargs["zz_alpha"]),
        n_jobs=1,
    )
    symmetric = bool(kwargs.get("symmetric", False))
    return start, encoder._kernel_block(A_chunk, B, start, symmetric)


def encoder_qsvm_fidelity_kernel(
    A: np.ndarray,
    B: np.ndarray,
    *,
    encoding: Encoding_type,
    device_name: str = "lightning.qubit",
    angle_rotation: str = "X",
    zz_reps: int = 1,
    zz_entanglement: str = "full",
    zz_alpha: float = 2.0,
    n_jobs: int | None = 1,
    parallel_chunk_size: int | None = None,
    parallel_min_rows: int = 64,
) -> np.ndarray:
    """Compute a PennyLane QSVM fidelity-kernel Gram matrix.

    Parameters
    ----------
    A : np.ndarray
        Left feature matrix.
    B : np.ndarray
        Right feature matrix.
    encoding : {"angle", "amplitude", "zz"}
        Encoding applied before the inverse embedding.
    device_name : str, optional
        PennyLane device name. Default value is "lightning.qubit".
    angle_rotation : str, optional
        Axis used for angle encoding. Default value is "X".
    zz_reps : int, optional
        Number of ZZ feature-map repetitions. Default value is 1.
    zz_entanglement : str, optional
        ZZ feature-map entanglement pattern. Default value is "full".
    zz_alpha : float, optional
        Global ZZ feature-map scale. Default value is 2.0.
    n_jobs : int | None, optional
        Number of CPU worker processes used for row chunks. ``-1`` uses all
        available cores. Default value is 1.
    parallel_chunk_size : int | None, optional
        Number of left rows per worker task. Default value is None.
    parallel_min_rows : int, optional
        Minimum row count before spawning workers. Default value is 64.

    Returns
    -------
    np.ndarray
        Matrix whose entries are the probability of the all-zero state after
        applying U(a) followed by U(b)^dagger.
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    encoder = GateFidelityKernelEncoder(
        n_features=A.shape[1],
        encoding=encoding,
        device_name=device_name,
        angle_rotation=angle_rotation,
        zz_reps=int(zz_reps),
        zz_entanglement=zz_entanglement,
        zz_alpha=float(zz_alpha),
        n_jobs=n_jobs,
        parallel_chunk_size=parallel_chunk_size,
        parallel_min_rows=parallel_min_rows,
    )
    return encoder.kernel_matrix(A, B)


def _qknn_state_qnode(
    encoding: Encoding_type,
    n_qubits: int,
    *,
    device_name: str,
    angle_rotation: str,
    entangle: bool,
    zz_reps: int,
    zz_entanglement: str,
    zz_alpha: float,
):
    dev = qml.device(device_name, wires=n_qubits)
    wires = range(n_qubits)

    @qml.qnode(dev)
    def state_circuit(x):
        _apply_gate_based_encoding(
            x,
            encoding=encoding,
            wires=wires,
            n_qubits=n_qubits,
            angle_rotation=angle_rotation,
            zz_reps=zz_reps,
            zz_entanglement=zz_entanglement,
            zz_alpha=zz_alpha,
        )
        if entangle and n_qubits > 1:
            for _ in range(2):
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.RY(np.pi / 4, wires=n_qubits - 1)
        return qml.state()

    return state_circuit


@dataclass
class GateStateVectorEncoder:
    """Reusable PennyLane encoder returning full state vectors.

    Parameters
    ----------
    n_features : int
        Number of input features before encoding.
    encoding : {"angle", "amplitude", "zz"}
        Gate-based feature map.
    device_name : str, optional
        PennyLane device name. Default value is "lightning.qubit".
    angle_rotation : str, optional
        Axis used for angle encoding. Default value is "X".
    normalize_inputs : bool, optional
        Whether to L2-normalize each row before encoding. Default value is
        False.
    entangle : bool, optional
        Whether to add the authors' QKNN entangling layer. Default value is
        False.
    zz_reps : int, optional
        Number of ZZ feature-map repetitions. Default value is 1.
    zz_entanglement : str, optional
        ZZ feature-map entanglement pattern. Default value is "full".
    zz_alpha : float, optional
        Global ZZ feature-map scale. Default value is 2.0.
    n_jobs : int | None, optional
        Number of process workers for CPU row chunks. Default value is 1.
    parallel_chunk_size : int | None, optional
        Rows per worker task or serial batch. Default value is None.
    parallel_min_rows : int, optional
        Minimum rows before spawning process workers. Default value is 64.
    """

    n_features: int
    encoding: Encoding_type
    device_name: str = "lightning.qubit"
    angle_rotation: str = "X"
    normalize_inputs: bool = False
    entangle: bool = False
    zz_reps: int = 1
    zz_entanglement: str = "full"
    zz_alpha: float = 2.0
    n_jobs: int | None = 1
    parallel_chunk_size: int | None = None
    parallel_min_rows: int = 64

    def __post_init__(self) -> None:
        """Prepare the PennyLane state-vector QNode once."""

        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")
        self.n_qubits = required_qubits(self.encoding, int(self.n_features))
        self.circuit = _qknn_state_qnode(
            self.encoding,
            self.n_qubits,
            device_name=self.device_name,
            angle_rotation=self.angle_rotation,
            entangle=bool(self.entangle),
            zz_reps=int(self.zz_reps),
            zz_entanglement=self.zz_entanglement,
            zz_alpha=float(self.zz_alpha),
        )

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode rows while reusing the prepared QNode."""

        X = self._validate_inputs(X)
        jobs = _effective_n_jobs(
            self.n_jobs,
            total_rows=X.shape[0],
            min_rows=int(self.parallel_min_rows),
        )
        if jobs > 1:
            return parallel_encode_rows(
                X,
                _qknn_state_vectors_chunk_worker,
                self._worker_kwargs(),
                n_jobs=jobs,
                chunk_size=self.parallel_chunk_size,
                min_rows=int(self.parallel_min_rows),
            )
        size = _row_chunk_size(X.shape[0], 1, self.parallel_chunk_size)
        parts = []
        for start, stop in _row_chunks(X.shape[0], size):
            parts.append(self._encode_chunk(X[start:stop]))
        return np.vstack(parts)

    def _encode_chunk(self, X_chunk: np.ndarray) -> np.ndarray:
        encoded = _normalise_rows(X_chunk) if self.normalize_inputs else X_chunk
        return _as_batched_state_array(
            self.circuit(encoded),
            expected_rows=encoded.shape[0],
        )

    def _worker_kwargs(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "device_name": self.device_name,
            "angle_rotation": self.angle_rotation,
            "normalize_inputs": bool(self.normalize_inputs),
            "entangle": bool(self.entangle),
            "zz_reps": int(self.zz_reps),
            "zz_entanglement": self.zz_entanglement,
            "zz_alpha": float(self.zz_alpha),
        }

    def _validate_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Gate state-vector inputs must be a 2D array.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Gate state-vector encoder expected {self.n_features} "
                f"features, got {X.shape[1]}."
            )
        return X


def _qknn_state_vectors_chunk_worker(
    task: tuple[int, np.ndarray, dict[str, Any]],
) -> tuple[int, np.ndarray]:
    start, X_chunk, kwargs = task
    encoder = GateStateVectorEncoder(
        n_features=X_chunk.shape[1],
        encoding=kwargs["encoding"],
        device_name=kwargs["device_name"],
        angle_rotation=kwargs["angle_rotation"],
        normalize_inputs=bool(kwargs["normalize_inputs"]),
        entangle=bool(kwargs["entangle"]),
        zz_reps=int(kwargs["zz_reps"]),
        zz_entanglement=kwargs["zz_entanglement"],
        zz_alpha=float(kwargs["zz_alpha"]),
        n_jobs=1,
    )
    return start, encoder._encode_chunk(X_chunk)


def encoder_qknn_state_vectors(
    X: np.ndarray,
    *,
    encoding: Encoding_type,
    device_name: str = "lightning.qubit",
    angle_rotation: str = "X",
    normalize_inputs: bool = False,
    entangle: bool = False,
    zz_reps: int = 1,
    zz_entanglement: str = "full",
    zz_alpha: float = 2.0,
    n_jobs: int | None = 1,
    parallel_chunk_size: int | None = None,
    parallel_min_rows: int = 64,
) -> np.ndarray:
    """Encode QKNN samples as state vectors.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    encoding : {"angle", "amplitude", "zz"}
        Quantum encoding.
    device_name : str, optional
        PennyLane device name. Default value is "lightning.qubit".
    angle_rotation : str, optional
        Axis used for angle encoding. Default value is "X".
    normalize_inputs : bool, optional
        Whether to L2-normalize each row before encoding. Default value is
        False.
    entangle : bool, optional
        Whether to add the two CNOT/RY layers from the authors' QKNN feature
        circuit. Default value is False.
    zz_reps : int, optional
        Number of ZZ feature-map repetitions. Default value is 1.
    zz_entanglement : str, optional
        ZZ feature-map entanglement pattern. Default value is "full".
    zz_alpha : float, optional
        Global ZZ feature-map scale. Default value is 2.0.
    n_jobs : int | None, optional
        Number of CPU worker processes used for row chunks. ``-1`` uses all
        available cores. Default value is 1.
    parallel_chunk_size : int | None, optional
        Number of rows per worker task. Default value is None.
    parallel_min_rows : int, optional
        Minimum row count before spawning workers. Default value is 64.

    Returns
    -------
    np.ndarray
        Complex state vectors, one per row.
    """

    X = np.asarray(X, dtype=np.float64)
    encoder = GateStateVectorEncoder(
        n_features=X.shape[1],
        encoding=encoding,
        device_name=device_name,
        angle_rotation=angle_rotation,
        normalize_inputs=bool(normalize_inputs),
        entangle=bool(entangle),
        zz_reps=int(zz_reps),
        zz_entanglement=zz_entanglement,
        zz_alpha=float(zz_alpha),
        n_jobs=n_jobs,
        parallel_chunk_size=parallel_chunk_size,
        parallel_min_rows=parallel_min_rows,
    )
    return encoder.encode(X)


def encoder_qknn_distances(
    left_states: np.ndarray,
    right_states: np.ndarray,
) -> np.ndarray:
    """Compute QKNN fidelity distances between encoded state vectors.

    Parameters
    ----------
    left_states : np.ndarray
        Encoded state vectors to compare.
    right_states : np.ndarray
        Reference encoded state vectors.

    Returns
    -------
    np.ndarray
        Pairwise distances ``1 - |<left|right>|^2``.
    """

    return 1.0 - state_fidelity_kernel(left_states, right_states)


def state_fidelity_kernel(
    left_states: np.ndarray,
    right_states: np.ndarray,
) -> np.ndarray:
    """Compute pairwise state fidelities from explicit amplitudes.

    Parameters
    ----------
    left_states : np.ndarray
        Complex amplitudes for the left encoded states.
    right_states : np.ndarray
        Complex amplitudes for the reference encoded states.

    Returns
    -------
    np.ndarray
        Pairwise fidelities ``|<left|right>|^2`` clipped to ``[0, 1]`` for
        numerical stability.
    """

    left_states = np.asarray(left_states, dtype=np.complex128)
    right_states = np.asarray(right_states, dtype=np.complex128)
    if left_states.ndim != 2 or right_states.ndim != 2:
        raise ValueError("State-fidelity inputs must be 2D arrays.")
    if left_states.shape[1] != right_states.shape[1]:
        raise ValueError(
            "State dimensions differ: "
            f"{left_states.shape[1]} for left states, "
            f"{right_states.shape[1]} for right states."
        )
    overlaps = left_states @ np.conjugate(right_states.T)
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)


@dataclass
class GateHybridFeatureEncoder:
    """Reusable PennyLane encoder returning explicit hybrid features.

    Parameters
    ----------
    n_features : int
        Number of input features before encoding.
    encoding : {"angle", "amplitude", "zz"}
        Gate-based feature map.
    device_name : str, optional
        PennyLane device name. Default value is "lightning.qubit".
    angle_rotation : str, optional
        Axis used for angle encoding. Default value is "Y".
    readout : {"pauli_z", "state_real", "state_real_imag"}, optional
        Feature readout applied after the encoding. Default value is
        "pauli_z".
    zz_reps : int, optional
        Number of ZZ feature-map repetitions. Default value is 1.
    zz_entanglement : str, optional
        ZZ feature-map entanglement pattern. Default value is "full".
    zz_alpha : float, optional
        Global ZZ feature-map scale. Default value is 2.0.
    n_jobs : int | None, optional
        Number of process workers for CPU row chunks. Default value is 1.
    parallel_chunk_size : int | None, optional
        Rows per worker task or serial batch. Default value is None.
    parallel_min_rows : int, optional
        Minimum rows before spawning process workers. Default value is 64.
    """

    n_features: int
    encoding: Encoding_type
    device_name: str = "lightning.qubit"
    angle_rotation: str = "Y"
    readout: HybridReadout_type = "pauli_z"
    zz_reps: int = 1
    zz_entanglement: str = "full"
    zz_alpha: float = 2.0
    n_jobs: int | None = 1
    parallel_chunk_size: int | None = None
    parallel_min_rows: int = 64

    def __post_init__(self) -> None:
        """Prepare the PennyLane hybrid-feature QNode once."""

        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")
        if self.readout not in {"pauli_z", "state_real", "state_real_imag"}:
            raise ValueError(
                "Hybrid readout must be 'pauli_z', 'state_real', or 'state_real_imag'."
            )
        self.n_qubits = required_qubits(self.encoding, int(self.n_features))
        self.wires = range(self.n_qubits)
        dev = qml.device(self.device_name, wires=self.n_qubits)

        if self.readout == "pauli_z":

            @qml.qnode(dev)
            def circuit(row):
                _apply_gate_based_encoding(
                    row,
                    encoding=self.encoding,
                    wires=self.wires,
                    n_qubits=self.n_qubits,
                    angle_rotation=self.angle_rotation,
                    zz_reps=int(self.zz_reps),
                    zz_entanglement=self.zz_entanglement,
                    zz_alpha=float(self.zz_alpha),
                )
                return [qml.expval(qml.PauliZ(wire)) for wire in self.wires]

        else:

            @qml.qnode(dev)
            def circuit(row):
                _apply_gate_based_encoding(
                    row,
                    encoding=self.encoding,
                    wires=self.wires,
                    n_qubits=self.n_qubits,
                    angle_rotation=self.angle_rotation,
                    zz_reps=int(self.zz_reps),
                    zz_entanglement=self.zz_entanglement,
                    zz_alpha=float(self.zz_alpha),
                )
                return qml.state()

        self.circuit = circuit

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode rows while reusing the prepared QNode."""

        X = self._validate_inputs(X)
        jobs = _effective_n_jobs(
            self.n_jobs,
            total_rows=X.shape[0],
            min_rows=int(self.parallel_min_rows),
        )
        if jobs > 1:
            return parallel_encode_rows(
                X,
                _hybrid_features_chunk_worker,
                self._worker_kwargs(),
                n_jobs=jobs,
                chunk_size=self.parallel_chunk_size,
                min_rows=int(self.parallel_min_rows),
            )
        size = _row_chunk_size(X.shape[0], 1, self.parallel_chunk_size)
        parts = []
        for start, stop in _row_chunks(X.shape[0], size):
            parts.append(self._encode_chunk(X[start:stop]))
        return np.vstack(parts)

    def _encode_chunk(self, X_chunk: np.ndarray) -> np.ndarray:
        if self.readout == "pauli_z":
            return _as_batched_expval_array(
                self.circuit(X_chunk),
                expected_rows=X_chunk.shape[0],
            )
        states = _as_batched_state_array(
            self.circuit(X_chunk),
            expected_rows=X_chunk.shape[0],
        )
        if self.readout == "state_real_imag":
            return np.concatenate([np.real(states), np.imag(states)], axis=1)
        return np.real(states).astype(np.float64)

    def _worker_kwargs(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "device_name": self.device_name,
            "angle_rotation": self.angle_rotation,
            "readout": self.readout,
            "zz_reps": int(self.zz_reps),
            "zz_entanglement": self.zz_entanglement,
            "zz_alpha": float(self.zz_alpha),
        }

    def _validate_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Hybrid feature inputs must be a 2D array.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Hybrid feature encoder expected {self.n_features} "
                f"features, got {X.shape[1]}."
            )
        return X


def _hybrid_features_chunk_worker(
    task: tuple[int, np.ndarray, dict[str, Any]],
) -> tuple[int, np.ndarray]:
    start, X_chunk, kwargs = task
    encoder = GateHybridFeatureEncoder(
        n_features=X_chunk.shape[1],
        encoding=kwargs["encoding"],
        device_name=kwargs["device_name"],
        angle_rotation=kwargs["angle_rotation"],
        readout=kwargs["readout"],
        zz_reps=int(kwargs["zz_reps"]),
        zz_entanglement=kwargs["zz_entanglement"],
        zz_alpha=float(kwargs["zz_alpha"]),
        n_jobs=1,
    )
    return start, encoder._encode_chunk(X_chunk)


def encoder_hybrid_features(
    X: np.ndarray,
    *,
    encoding: Encoding_type,
    device_name: str = "lightning.qubit",
    angle_rotation: str = "Y",
    readout: HybridReadout_type = "pauli_z",
    zz_reps: int = 1,
    zz_entanglement: str = "full",
    zz_alpha: float = 2.0,
    n_jobs: int | None = 1,
    parallel_chunk_size: int | None = None,
    parallel_min_rows: int = 64,
) -> np.ndarray:
    """Apply hybrid quantum feature maps used before classical SVM/KNN.

    The default readout returns Pauli-Z expectations for each wire for all
    encodings. Statevector readouts are kept as explicit alternatives for
    reproducing the authors' amplitude-hybrid convention or exploratory
    analyses.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    encoding : {"angle", "amplitude", "zz"}
        Quantum encoding.
    device_name : str, optional
        PennyLane device name. Default value is "lightning.qubit".
    angle_rotation : str, optional
        Axis used for angle encoding. Default value is "Y".
    readout : {"pauli_z", "state_real", "state_real_imag"}, optional
        Feature readout applied after the encoding. Default value is
        "pauli_z".
    zz_reps : int, optional
        Number of ZZ feature-map repetitions. Default value is 1.
    zz_entanglement : str, optional
        ZZ feature-map entanglement pattern. Default value is "full".
    zz_alpha : float, optional
        Global ZZ feature-map scale. Default value is 2.0.
    n_jobs : int | None, optional
        Number of CPU worker processes used for row chunks. ``-1`` uses all
        available cores. Default value is 1.
    parallel_chunk_size : int | None, optional
        Number of rows per worker task. Default value is None.
    parallel_min_rows : int, optional
        Minimum row count before spawning workers. Default value is 64.

    Returns
    -------
    np.ndarray
        Quantum-transformed feature matrix.
    """

    X = np.asarray(X, dtype=np.float64)
    if readout not in {"pauli_z", "state_real", "state_real_imag"}:
        raise ValueError(
            "Hybrid readout must be 'pauli_z', 'state_real', or 'state_real_imag'."
        )
    encoder = GateHybridFeatureEncoder(
        n_features=X.shape[1],
        encoding=encoding,
        device_name=device_name,
        angle_rotation=angle_rotation,
        readout=readout,
        zz_reps=int(zz_reps),
        zz_entanglement=zz_entanglement,
        zz_alpha=float(zz_alpha),
        n_jobs=n_jobs,
        parallel_chunk_size=parallel_chunk_size,
        parallel_min_rows=parallel_min_rows,
    )
    return encoder.encode(X)


def fock_output_size(n_modes: int, n_photons: int) -> int:
    """Return the number of Fock outcomes for ``n_photons`` in ``n_modes``.

    Parameters
    ----------
    n_modes : int
        Number of optical modes.
    n_photons : int
        Number of input photons.

    Returns
    -------
    int
        Number of weak compositions of ``n_photons`` into ``n_modes`` modes.
    """

    if n_modes <= 0:
        raise ValueError("n_modes must be positive.")
    if n_photons <= 0:
        raise ValueError("n_photons must be positive.")
    return math.comb(n_modes + n_photons - 1, n_photons)


def unbunched_output_size(n_modes: int, n_photons: int) -> int:
    """Return the number of unbunched outcomes for photons in modes.

    Parameters
    ----------
    n_modes : int
        Number of optical modes.
    n_photons : int
        Number of input photons.

    Returns
    -------
    int
        Number of collision-free photon patterns.
    """

    if n_modes <= 0:
        raise ValueError("n_modes must be positive.")
    if n_photons <= 0:
        raise ValueError("n_photons must be positive.")
    if n_photons > n_modes:
        raise ValueError(
            "UNBUNCHED computation space requires n_photons <= n_modes; got "
            f"{n_photons} photons and {n_modes} modes."
        )
    return math.comb(n_modes, n_photons)


def normalize_photonic_computation_space(
    computation_space: str,
) -> PhotonicComputationSpace_type:
    """Normalize a configured MerLin computation-space name."""

    value = str(computation_space).upper()
    if value in {"FOCK", "UNBUNCHED"}:
        return value  # type: ignore[return-value]
    raise ValueError("photonic_computation_space must be 'FOCK' or 'UNBUNCHED'.")


def photonic_output_size(
    n_modes: int,
    n_photons: int,
    computation_space: str = "FOCK",
) -> int:
    """Return the MerLin output dimension for the configured space."""

    space = normalize_photonic_computation_space(computation_space)
    if space == "FOCK":
        return fock_output_size(n_modes, n_photons)
    return unbunched_output_size(n_modes, n_photons)


def minimal_fock_modes(n_features: int, n_photons: int) -> int:
    """Return the smallest mode count whose Fock basis fits the features.

    Parameters
    ----------
    n_features : int
        Number of scalar features to embed as amplitudes.
    n_photons : int
        Number of photons defining the Fock basis.

    Returns
    -------
    int
        Smallest ``n_modes`` satisfying
        ``fock_output_size(n_modes, n_photons) >= n_features``.
    """

    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if n_photons <= 0:
        raise ValueError("n_photons must be positive.")
    n_modes = 1
    while fock_output_size(n_modes, n_photons) < n_features:
        n_modes += 1
    return n_modes


def minimal_photonic_modes(
    n_features: int,
    n_photons: int,
    computation_space: str = "FOCK",
) -> int:
    """Return the smallest mode count whose output basis fits the features."""

    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if n_photons <= 0:
        raise ValueError("n_photons must be positive.")
    n_modes = max(1, n_photons)
    while photonic_output_size(n_modes, n_photons, computation_space) < n_features:
        n_modes += 1
    return n_modes


def _torch_dtype(torch, dtype: str):
    dtype = str(dtype).lower()
    if dtype in {"float64", "double", "torch.float64"}:
        return torch.float64
    if dtype in {"float32", "float", "torch.float32"}:
        return torch.float32
    raise ValueError("Photonic dtype must be 'float64' or 'float32'.")


def _merlin_computation_space(ml, computation_space: str):
    space = normalize_photonic_computation_space(computation_space)
    return getattr(ml.ComputationSpace, space)


def _fixed_haar_unitary_circuit(n_modes: int, seed: int):
    import perceval as pcvl
    from perceval import random_seed

    random_seed(int(seed))
    unitary = pcvl.Matrix.random_unitary(n_modes)
    return pcvl.Circuit(n_modes) // pcvl.Unitary(unitary)


def _fixed_reservoir_circuit(n_modes: int, seed: int):
    import perceval as pcvl
    from perceval import random_seed

    random_seed(int(seed))
    unitary = pcvl.Matrix.random_unitary(n_modes)
    interferometer_1 = pcvl.Unitary(unitary)
    interferometer_2 = interferometer_1.copy()

    phase_circuit = pcvl.Circuit(n_modes)
    for mode in range(n_modes):
        phase_circuit.add(mode, pcvl.PS(pcvl.P(f"px{mode + 1}")))

    return interferometer_1 // phase_circuit // interferometer_2


def evenly_spaced_fock_state(n_modes: int, n_photons: int) -> list[int]:
    """Create the deterministic input state used by the QORC notebook.

    Parameters
    ----------
    n_modes : int
        Number of optical modes.
    n_photons : int
        Number of photons to inject.

    Returns
    -------
    list[int]
        Occupation list with photons spread from the first to the last mode.

    Raises
    ------
    ValueError
        If there are more photons than modes.
    """

    if n_photons > n_modes:
        raise ValueError(
            "The current reservoir input convention supports at most one "
            f"photon per occupied mode; got {n_photons} photons and {n_modes} modes."
        )
    step = (n_modes - 1) / (n_photons - 1) if n_photons > 1 else 0
    input_state = [0] * n_modes
    for photon_index in range(n_photons):
        mode_index = int(round(photon_index * step))
        input_state[mode_index] = 1
    return input_state


def _clear_empty_postselection(experiment) -> bool:
    postselect = getattr(experiment, "post_select_fn", None)
    if postselect is None:
        return False
    if getattr(postselect, "has_condition", True):
        return False
    # MerLin 0.3.2 validates FidelityKernel experiments with
    # `post_select_fn is not None`, while Perceval represents "no
    # post-selection" as an empty PostSelect object. Normalize only that empty
    # case so real post-selection still fails loudly.
    experiment._postselect = None
    return True


@dataclass
class PhotonicReservoirEncoder:
    """Fixed MerLin photonic reservoir used as an explicit feature map.

    Parameters
    ----------
    n_modes : int, optional
        Number of optical modes and input angles. Default value is 4.
    n_photons : int, optional
        Number of photons in the Fock input state. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype, either "float64" or "float32". Default value is
        "float64".
    phase_scale : float, optional
        Multiplicative factor applied to inputs before phase shifters. Default
        value is pi.
    batch_size : int, optional
        Number of samples encoded per MerLin call. Default value is 32.
    """

    n_modes: int = 4
    n_photons: int = 2
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    phase_scale: float = math.pi
    batch_size: int = 32
    computation_space: PhotonicComputationSpace_type = "FOCK"

    def __post_init__(self) -> None:
        """Build the fixed QORC-style MerLin reservoir."""

        import merlin as ml
        import torch

        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")
        if self.n_photons <= 0:
            raise ValueError("n_photons must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        torch_dtype = _torch_dtype(torch, self.dtype)
        torch_device = torch.device(self.device)

        circuit = _fixed_reservoir_circuit(self.n_modes, self.seed)
        measurement_strategy = ml.MeasurementStrategy.probs(
            computation_space=_merlin_computation_space(ml, self.computation_space)
        )
        self.input_state = evenly_spaced_fock_state(self.n_modes, self.n_photons)
        self.layer = ml.QuantumLayer(
            input_size=self.n_modes,
            circuit=circuit,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=self.input_state,
            n_photons=self.n_photons,
            measurement_strategy=measurement_strategy,
            device=torch_device,
            dtype=torch_dtype,
        )
        self.layer.eval()
        trainable_count = sum(1 for _ in self.layer.parameters())
        if trainable_count != 0:
            raise RuntimeError(
                "Photonic reservoir must be fixed; MerLin reported "
                f"{trainable_count} trainable parameters."
            )
        self.output_size = int(self.layer.output_size)
        expected_output_size = photonic_output_size(
            self.n_modes,
            self.n_photons,
            self.computation_space,
        )
        if self.output_size != expected_output_size:
            raise RuntimeError(
                "Unexpected MerLin photonic output size: "
                f"got {self.output_size}, expected {expected_output_size}."
            )
        self.torch = torch
        self.torch_device = torch_device
        self.torch_dtype = torch_dtype

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode samples into Fock-probability reservoir features.

        Parameters
        ----------
        X : np.ndarray
            Input matrix with exactly ``n_modes`` columns and values already
            scaled to the configured phase-input range.

        Returns
        -------
        np.ndarray
            Fock probability features with shape ``(n_samples, output_size)``.
        """

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Photonic inputs must be a 2D array.")
        if X.shape[1] != self.n_modes:
            raise ValueError(
                f"Photonic reservoir expected {self.n_modes} modes, got "
                f"{X.shape[1]} features."
            )

        outputs = []
        with self.torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                batch = X[start : start + self.batch_size] * float(self.phase_scale)
                tensor = self.torch.as_tensor(
                    batch,
                    dtype=self.torch_dtype,
                    device=self.torch_device,
                )
                encoded = self.layer(tensor).detach().cpu().numpy()
                outputs.append(np.asarray(encoded, dtype=np.float64))
        return np.vstack(outputs)


@dataclass
class PhotonicReservoirStateEncoder:
    """Fixed MerLin angle reservoir returning Fock-state amplitudes.

    Parameters
    ----------
    n_modes : int, optional
        Number of optical modes and input angles. Default value is 4.
    n_photons : int, optional
        Number of photons in the Fock input state. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype, either "float64" or "float32". Default value is
        "float64".
    phase_scale : float, optional
        Multiplicative factor applied to inputs before phase shifters. Default
        value is pi.
    batch_size : int, optional
        Number of samples encoded per MerLin call. Default value is 32.
    """

    n_modes: int = 4
    n_photons: int = 2
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    phase_scale: float = math.pi
    batch_size: int = 32
    computation_space: PhotonicComputationSpace_type = "FOCK"

    def __post_init__(self) -> None:
        """Build the fixed QORC-style MerLin reservoir with amplitude readout."""

        import merlin as ml
        import torch

        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")
        if self.n_photons <= 0:
            raise ValueError("n_photons must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        torch_dtype = _torch_dtype(torch, self.dtype)
        torch_device = torch.device(self.device)

        circuit = _fixed_reservoir_circuit(self.n_modes, self.seed)
        measurement_strategy = ml.MeasurementStrategy.amplitudes(
            computation_space=_merlin_computation_space(ml, self.computation_space)
        )
        self.input_state = evenly_spaced_fock_state(self.n_modes, self.n_photons)
        self.layer = ml.QuantumLayer(
            input_size=self.n_modes,
            circuit=circuit,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=self.input_state,
            n_photons=self.n_photons,
            measurement_strategy=measurement_strategy,
            device=torch_device,
            dtype=torch_dtype,
        )
        self.layer.eval()
        trainable_count = sum(1 for _ in self.layer.parameters())
        if trainable_count != 0:
            raise RuntimeError(
                "Photonic reservoir must be fixed; MerLin reported "
                f"{trainable_count} trainable parameters."
            )
        self.output_size = int(self.layer.output_size)
        expected_output_size = photonic_output_size(
            self.n_modes,
            self.n_photons,
            self.computation_space,
        )
        if self.output_size != expected_output_size:
            raise RuntimeError(
                "Unexpected MerLin photonic output size: "
                f"got {self.output_size}, expected {expected_output_size}."
            )
        self.torch = torch
        self.torch_device = torch_device
        self.torch_dtype = torch_dtype

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode samples into complex Fock amplitudes.

        Parameters
        ----------
        X : np.ndarray
            Input matrix with exactly ``n_modes`` columns and values already
            scaled to the configured phase-input range.

        Returns
        -------
        np.ndarray
            Complex Fock amplitudes with shape ``(n_samples, output_size)``.
        """

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Photonic inputs must be a 2D array.")
        if X.shape[1] != self.n_modes:
            raise ValueError(
                f"Photonic reservoir expected {self.n_modes} modes, got "
                f"{X.shape[1]} features."
            )

        outputs = []
        with self.torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                batch = X[start : start + self.batch_size] * float(self.phase_scale)
                tensor = self.torch.as_tensor(
                    batch,
                    dtype=self.torch_dtype,
                    device=self.torch_device,
                )
                encoded = self.layer(tensor).detach().cpu().numpy()
                encoded = np.atleast_2d(np.asarray(encoded, dtype=np.complex128))
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                if np.any(norms <= 1e-12):
                    raise RuntimeError(
                        "MerLin returned an empty photonic amplitude state."
                    )
                outputs.append(encoded / norms)
        return np.vstack(outputs)


@dataclass
class PhotonicAmplitudeReservoirEncoder:
    """Fixed MerLin amplitude encoder followed by a Haar interferometer.

    Parameters
    ----------
    n_features : int
        Number of input features before padding to the Fock basis size.
    n_modes : int
        Number of optical modes. The associated Fock basis must contain at
        least ``n_features`` states.
    n_photons : int, optional
        Number of photons defining the Fock basis. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype, either "float64" or "float32". Default value is
        "float64".
    batch_size : int, optional
        Number of samples encoded per MerLin call. Default value is 32.
    """

    n_features: int
    n_modes: int
    n_photons: int = 2
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    batch_size: int = 32
    computation_space: PhotonicComputationSpace_type = "FOCK"

    def __post_init__(self) -> None:
        """Build the fixed MerLin amplitude-encoded reservoir."""

        import merlin as ml
        import torch

        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")
        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")
        if self.n_photons <= 0:
            raise ValueError("n_photons must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.computation_space = normalize_photonic_computation_space(
            self.computation_space
        )
        self.input_size = photonic_output_size(
            self.n_modes,
            self.n_photons,
            self.computation_space,
        )
        if self.input_size < self.n_features:
            raise ValueError(
                "Photonic amplitude encoding requires a computation-space basis "
                "at least as "
                f"large as the input feature vector; got basis size "
                f"{self.input_size} for {self.n_features} features."
            )

        torch_dtype = _torch_dtype(torch, self.dtype)
        torch_device = torch.device(self.device)
        circuit = _fixed_haar_unitary_circuit(self.n_modes, self.seed)
        measurement_strategy = ml.MeasurementStrategy.probs(
            computation_space=_merlin_computation_space(ml, self.computation_space)
        )
        self.layer = ml.QuantumLayer(
            circuit=circuit,
            trainable_parameters=[],
            n_photons=self.n_photons,
            measurement_strategy=measurement_strategy,
            device=torch_device,
            dtype=torch_dtype,
        )
        self.layer.eval()
        trainable_count = sum(1 for _ in self.layer.parameters())
        if trainable_count != 0:
            raise RuntimeError(
                "Photonic amplitude reservoir must be fixed; MerLin reported "
                f"{trainable_count} trainable parameters."
            )
        self.output_size = int(self.layer.output_size)
        expected_output_size = photonic_output_size(
            self.n_modes,
            self.n_photons,
            self.computation_space,
        )
        if self.output_size != expected_output_size:
            raise RuntimeError(
                "Unexpected MerLin photonic output size: "
                f"got {self.output_size}, expected {expected_output_size}."
            )
        self.torch = torch
        self.torch_device = torch_device
        self.torch_dtype = torch_dtype
        self.torch_complex_dtype = (
            torch.complex128 if torch_dtype == torch.float64 else torch.complex64
        )

    def _amplitude_input_from_batch(self, batch: np.ndarray):
        padded = np.zeros((batch.shape[0], self.input_size), dtype=np.float64)
        padded[:, : self.n_features] = batch
        norms = np.linalg.norm(padded, axis=1, keepdims=True)
        zero_rows = np.squeeze(norms <= 1e-12, axis=1)
        if np.any(zero_rows):
            padded[zero_rows, 0] = 1.0
            norms[zero_rows, 0] = 1.0
        amplitudes = padded / norms
        tensor = self.torch.as_tensor(
            amplitudes,
            dtype=self.torch_dtype,
            device=self.torch_device,
        )
        if self.computation_space == "UNBUNCHED":
            return tensor.to(dtype=self.torch_complex_dtype)

        from merlin.core.state_vector import StateVector

        return StateVector.from_tensor(
            tensor,
            n_modes=int(self.n_modes),
            n_photons=int(self.n_photons),
        ).normalize()

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode samples into Fock probabilities from amplitude states.

        Parameters
        ----------
        X : np.ndarray
            Input matrix with ``n_features`` columns. Rows are padded to the
            Fock basis size and L2-normalized before being passed to MerLin.

        Returns
        -------
        np.ndarray
            Fock probability features with shape ``(n_samples, output_size)``.
        """

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Photonic amplitude inputs must be a 2D array.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Photonic amplitude encoder expected {self.n_features} "
                f"features, got {X.shape[1]}."
            )

        outputs = []
        with self.torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                batch = X[start : start + self.batch_size]
                state = self._amplitude_input_from_batch(batch)
                encoded = self.layer(state).detach().cpu().numpy()
                encoded = np.atleast_2d(np.asarray(encoded, dtype=np.float64))
                row_sums = encoded.sum(axis=1, keepdims=True)
                if np.any(row_sums <= 1e-12):
                    raise RuntimeError(
                        "MerLin returned an empty photonic amplitude "
                        "probability distribution."
                    )
                outputs.append(encoded / row_sums)
        return np.vstack(outputs)


@dataclass
class PhotonicFidelityKernelEncoder:
    """Fixed MerLin photonic reservoir used as a fidelity kernel feature map.

    Parameters
    ----------
    n_modes : int, optional
        Number of optical modes and input angles. Default value is 4.
    n_photons : int, optional
        Number of photons in the Fock input state. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype. Default value is "float64".
    phase_scale : float, optional
        Multiplicative factor applied to inputs before phase shifters. Default
        value is pi.
    shots : int, optional
        Pseudo-sampling shots used by MerLin's FidelityKernel. A value of 0
        means exact probabilities. Default value is 0.
    sampling_method : str, optional
        MerLin pseudo-sampling method. Default value is "multinomial".
    force_psd : bool, optional
        Whether the final train Gram matrix should be projected to a PSD
        matrix. In serial mode this is delegated to MerLin; in parallel mode it
        is applied after chunk reassembly. Default value is False.
    n_jobs : int | None, optional
        Number of CPU worker processes used for row chunks. ``-1`` uses all
        available cores. Default value is 1.
    parallel_chunk_size : int | None, optional
        Number of left rows per worker task. Default value is None.
    parallel_min_rows : int, optional
        Minimum row count before spawning workers. Default value is 64.
    """

    n_modes: int = 4
    n_photons: int = 2
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    phase_scale: float = math.pi
    shots: int = 0
    sampling_method: str = "multinomial"
    force_psd: bool = False
    computation_space: PhotonicComputationSpace_type = "FOCK"
    n_jobs: int | None = 1
    parallel_chunk_size: int | None = None
    parallel_min_rows: int = 64

    def __post_init__(self) -> None:
        """Build the fixed reservoir as a MerLin fidelity-kernel feature map."""

        import merlin as ml
        import torch

        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")
        if self.n_photons <= 0:
            raise ValueError("n_photons must be positive.")

        torch_dtype = _torch_dtype(torch, self.dtype)
        torch_device = torch.device(self.device)
        circuit = _fixed_reservoir_circuit(self.n_modes, self.seed)
        self.feature_map = ml.FeatureMap(
            circuit=circuit,
            input_size=self.n_modes,
            input_parameters=["px"],
            trainable_parameters=[],
            dtype=torch_dtype,
            device=torch_device,
        )
        self.cleared_empty_postselection = _clear_empty_postselection(
            self.feature_map.experiment
        )
        self.input_state = evenly_spaced_fock_state(self.n_modes, self.n_photons)
        self.n_jobs = _effective_n_jobs(
            self.n_jobs,
            total_rows=max(1, int(self.parallel_min_rows)),
            min_rows=int(self.parallel_min_rows),
        )
        self.merlin_force_psd = bool(self.force_psd) and int(self.n_jobs) == 1
        self.kernel = ml.FidelityKernel(
            feature_map=self.feature_map,
            input_state=self.input_state,
            shots=int(self.shots),
            sampling_method=self.sampling_method,
            computation_space=_merlin_computation_space(ml, self.computation_space),
            force_psd=bool(self.merlin_force_psd),
            device=torch_device,
            dtype=torch_dtype,
        )
        self.torch = torch
        self.torch_device = torch_device
        self.torch_dtype = torch_dtype

    def kernel_matrix(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        """Compute MerLin fidelity-kernel Gram matrices for angle inputs.

        Parameters
        ----------
        A : np.ndarray
            Left input matrix with one phase feature per optical mode.
        B : np.ndarray | None, optional
            Optional right input matrix. If omitted, a training Gram matrix is
            computed. Default value is None.

        Returns
        -------
        np.ndarray
            Fidelity-kernel matrix.
        """

        A_raw = self._validate_inputs(A)
        is_training_kernel = B is None
        B_raw = A_raw if B is None else self._validate_inputs(B)
        if int(self.n_jobs) > 1 and A_raw.shape[0] >= int(self.parallel_min_rows):
            matrix = parallel_kernel_matrix(
                A_raw,
                B_raw,
                _photonic_fidelity_kernel_chunk_worker,
                {
                    "n_modes": int(self.n_modes),
                    "n_photons": int(self.n_photons),
                    "seed": int(self.seed),
                    "device": self.device,
                    "dtype": self.dtype,
                    "phase_scale": float(self.phase_scale),
                    "shots": int(self.shots),
                    "sampling_method": self.sampling_method,
                    "computation_space": self.computation_space,
                },
                n_jobs=int(self.n_jobs),
                chunk_size=self.parallel_chunk_size,
                min_rows=int(self.parallel_min_rows),
                symmetric=is_training_kernel,
                lower_triangle=False,
            )
            if is_training_kernel and bool(self.force_psd):
                return nearest_psd_kernel(matrix)
            return matrix

        A = self._prepare_inputs(A_raw)
        B = None if is_training_kernel else self._prepare_inputs(B_raw)
        with self.torch.no_grad():
            if B is None:
                matrix = self.kernel(A)
            else:
                matrix = self.kernel(A, B)
        matrix_np = np.asarray(matrix.detach().cpu().numpy(), dtype=np.float64)
        if is_training_kernel and bool(self.force_psd) and not self.merlin_force_psd:
            return nearest_psd_kernel(matrix_np)
        return matrix_np

    def _validate_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Photonic kernel inputs must be a 2D array.")
        if X.shape[1] != self.n_modes:
            raise ValueError(
                f"Photonic fidelity kernel expected {self.n_modes} modes, got "
                f"{X.shape[1]} features."
            )
        return X

    def _prepare_inputs(self, X: np.ndarray) -> np.ndarray:
        return self._validate_inputs(X) * float(self.phase_scale)


def _photonic_fidelity_kernel_chunk_worker(
    task: tuple[int, np.ndarray, np.ndarray, dict[str, Any]],
) -> tuple[int, np.ndarray]:
    start, A_chunk, B, kwargs = task
    encoder = PhotonicFidelityKernelEncoder(
        n_modes=int(kwargs["n_modes"]),
        n_photons=int(kwargs["n_photons"]),
        seed=int(kwargs["seed"]),
        device=kwargs["device"],
        dtype=kwargs["dtype"],
        phase_scale=float(kwargs["phase_scale"]),
        shots=int(kwargs["shots"]),
        sampling_method=kwargs["sampling_method"],
        computation_space=kwargs["computation_space"],
        force_psd=False,
        n_jobs=1,
    )
    return start, encoder.kernel_matrix(A_chunk, B)


def encoder_photonic(
    X: np.ndarray,
    *,
    n_modes: int = 4,
    n_photons: int = 2,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float64",
    phase_scale: float = math.pi,
    batch_size: int = 32,
    computation_space: PhotonicComputationSpace_type = "FOCK",
) -> tuple[np.ndarray, PhotonicReservoirEncoder]:
    """Encode samples with a fixed MerLin photonic reservoir.

    Parameters
    ----------
    X : np.ndarray
        Input matrix with one phase feature per optical mode.
    n_modes : int, optional
        Number of optical modes. Default value is 4.
    n_photons : int, optional
        Number of Fock input photons. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype. Default value is "float64".
    phase_scale : float, optional
        Multiplicative factor applied before phase shifters. Default value is
        pi.
    batch_size : int, optional
        Number of samples encoded per MerLin call. Default value is 32.

    Returns
    -------
    tuple[np.ndarray, PhotonicReservoirEncoder]
        Encoded Fock-probability features and the fitted fixed encoder.
    """

    encoder = PhotonicReservoirEncoder(
        n_modes=n_modes,
        n_photons=n_photons,
        seed=seed,
        device=device,
        dtype=dtype,
        phase_scale=phase_scale,
        batch_size=batch_size,
        computation_space=normalize_photonic_computation_space(computation_space),
    )
    return encoder.encode(X), encoder


def encoder_photonic_amplitude(
    X: np.ndarray,
    *,
    n_modes: int | None = None,
    n_photons: int = 2,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float64",
    batch_size: int = 32,
    computation_space: PhotonicComputationSpace_type = "FOCK",
) -> tuple[np.ndarray, PhotonicAmplitudeReservoirEncoder]:
    """Encode samples with MerLin amplitude encoding and a fixed unitary.

    Parameters
    ----------
    X : np.ndarray
        Input matrix. The number of columns must be no larger than the Fock
        basis size implied by ``n_modes`` and ``n_photons``.
    n_modes : int | None, optional
        Number of optical modes. If omitted, the smallest mode count whose Fock
        basis can contain the input feature vector is selected. Default value
        is None.
    n_photons : int, optional
        Number of photons defining the Fock basis. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype. Default value is "float64".
    batch_size : int, optional
        Number of samples encoded per MerLin call. Default value is 32.

    Returns
    -------
    tuple[np.ndarray, PhotonicAmplitudeReservoirEncoder]
        Encoded Fock-probability features and the fitted fixed encoder.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Photonic amplitude inputs must be a 2D array.")
    resolved_modes = (
        max(
            int(n_photons),
            minimal_photonic_modes(
                X.shape[1],
                int(n_photons),
                computation_space,
            ),
        )
        if n_modes is None
        else int(n_modes)
    )
    encoder = PhotonicAmplitudeReservoirEncoder(
        n_features=X.shape[1],
        n_modes=resolved_modes,
        n_photons=n_photons,
        seed=seed,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        computation_space=normalize_photonic_computation_space(computation_space),
    )
    return encoder.encode(X), encoder


def encoder_photonic_state_amplitudes(
    X: np.ndarray,
    *,
    n_modes: int = 4,
    n_photons: int = 2,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float64",
    phase_scale: float = math.pi,
    batch_size: int = 32,
    computation_space: PhotonicComputationSpace_type = "FOCK",
) -> tuple[np.ndarray, PhotonicReservoirStateEncoder]:
    """Encode samples as explicit MerLin Fock amplitudes.

    Parameters
    ----------
    X : np.ndarray
        Input matrix with one phase feature per optical mode.
    n_modes : int, optional
        Number of optical modes. Default value is 4.
    n_photons : int, optional
        Number of Fock input photons. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype. Default value is "float64".
    phase_scale : float, optional
        Multiplicative factor applied before phase shifters. Default value is
        pi.
    batch_size : int, optional
        Number of samples encoded per MerLin call. Default value is 32.

    Returns
    -------
    tuple[np.ndarray, PhotonicReservoirStateEncoder]
        Complex Fock amplitudes and the fixed encoder.
    """

    encoder = PhotonicReservoirStateEncoder(
        n_modes=n_modes,
        n_photons=n_photons,
        seed=seed,
        device=device,
        dtype=dtype,
        phase_scale=phase_scale,
        batch_size=batch_size,
        computation_space=normalize_photonic_computation_space(computation_space),
    )
    return encoder.encode(X), encoder


def encoder_photonic_fidelity_kernel(
    A: np.ndarray,
    B: np.ndarray | None = None,
    *,
    n_modes: int = 4,
    n_photons: int = 2,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float64",
    phase_scale: float = math.pi,
    shots: int = 0,
    sampling_method: str = "multinomial",
    force_psd: bool = False,
    computation_space: PhotonicComputationSpace_type = "FOCK",
    n_jobs: int | None = 1,
    parallel_chunk_size: int | None = None,
    parallel_min_rows: int = 64,
) -> tuple[np.ndarray, PhotonicFidelityKernelEncoder]:
    """Encode samples through MerLin's photonic fidelity kernel.

    Parameters
    ----------
    A : np.ndarray
        Left input matrix with one phase feature per optical mode.
    B : np.ndarray | None, optional
        Optional right input matrix. If omitted, a training Gram matrix is
        computed. Default value is None.
    n_modes : int, optional
        Number of optical modes. Default value is 4.
    n_photons : int, optional
        Number of Fock input photons. Default value is 2.
    seed : int, optional
        Perceval random seed for the Haar interferometer. Default value is 42.
    device : str, optional
        Torch device used by MerLin. Default value is "cpu".
    dtype : str, optional
        Torch floating dtype. Default value is "float64".
    phase_scale : float, optional
        Multiplicative factor applied before phase shifters. Default value is
        pi.
    shots : int, optional
        Pseudo-sampling shots used by MerLin's FidelityKernel. A value of 0
        means exact probabilities. Default value is 0.
    sampling_method : str, optional
        MerLin pseudo-sampling method. Default value is "multinomial".
    force_psd : bool, optional
        Whether the final train Gram matrix should be projected to a PSD
        matrix. In serial mode this is delegated to MerLin; in parallel mode it
        is applied after chunk reassembly. Default value is False.
    n_jobs : int | None, optional
        Number of CPU worker processes used for row chunks. ``-1`` uses all
        available cores. Default value is 1.
    parallel_chunk_size : int | None, optional
        Number of left rows per worker task. Default value is None.
    parallel_min_rows : int, optional
        Minimum row count before spawning workers. Default value is 64.

    Returns
    -------
    tuple[np.ndarray, PhotonicFidelityKernelEncoder]
        Fidelity-kernel matrix and the fixed kernel encoder.
    """

    encoder = PhotonicFidelityKernelEncoder(
        n_modes=n_modes,
        n_photons=n_photons,
        seed=seed,
        device=device,
        dtype=dtype,
        phase_scale=phase_scale,
        shots=shots,
        sampling_method=sampling_method,
        force_psd=force_psd,
        computation_space=normalize_photonic_computation_space(computation_space),
        n_jobs=n_jobs,
        parallel_chunk_size=parallel_chunk_size,
        parallel_min_rows=parallel_min_rows,
    )
    return encoder.kernel_matrix(A, B), encoder


def explicit_dot_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the explicit linear Gram matrix ``A @ B.T``.

    Parameters
    ----------
    A : np.ndarray
        Left feature matrix.
    B : np.ndarray
        Right feature matrix.

    Returns
    -------
    np.ndarray
        Explicit dot-product matrix.
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Dot-product inputs must be 2D arrays.")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: {A.shape[1]} for A, {B.shape[1]} for B."
        )
    return A @ B.T


def euclidean_distances_from_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances using explicit dot products.

    Parameters
    ----------
    A : np.ndarray
        Left feature matrix.
    B : np.ndarray
        Right feature matrix.

    Returns
    -------
    np.ndarray
        Pairwise Euclidean distance matrix.
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    dot = explicit_dot_product(A, B)
    a_norm = np.sum(A * A, axis=1)[:, None]
    b_norm = np.sum(B * B, axis=1)[None, :]
    squared = np.maximum(a_norm + b_norm - 2.0 * dot, 0.0)
    return np.sqrt(squared)
