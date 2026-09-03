"""Exact simulation of the paper's 10-qubit transverse-field-Ising quantum reservoir.

Reservoir Hamiltonian (paper Eq. 12, ``Qreservoir`` in Time_series.jl)::

    H = sum_{i<j} J_ij X_i X_j + v * sum_i Z_i,   v = 1

Each forecasting step encodes ``n1`` features on fresh input qubits with
``RY(pi * x)`` (paper Eq. 14; the reference code passes ``para .* pi`` to
``RyGate``, and ``Data.CSV`` features live in ``[-1, 1]``, so rotation angles
span ``[-pi, pi]``), evolves the whole register for time ``tau``, then discards
the input qubits by a partial trace (paper Eq. 16/18). After the final encoding
the register evolves for ``tau / virtual_nodes`` at a time and ``<Z_j>`` is read
out after each sub-step, so ``virtual_nodes = 1`` reproduces QR1 and
``virtual_nodes = 2`` reproduces QR2's ``{tau/2, tau}`` ensemble (paper Fig. 3).

Complexity note
---------------
The joint state is always ``rho_hidden (x) |psi_input><psi_input|`` with a *pure*
input factor, so its rank never exceeds ``2 ** n_hidden``. Propagating that many
state vectors instead of a full ``2^n x 2^n`` density matrix is exact and reduces
the per-step cost from ``O(4^n)`` to ``O(rank * 4^n / 2^n)``; a full 816-month
QR1 pass takes a few seconds on CPU rather than tens of minutes.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Discard rank-1 components whose weight is numerically zero.
_RANK_TOLERANCE = 1e-12
# Cap on the working-set size of the batched SVD, in complex64 elements.
_SVD_ELEMENT_BUDGET = 24_000_000


def build_hamiltonian(couplings: np.ndarray, n_qubits: int, field: float = 1.0) -> np.ndarray:
    """Dense matrix of ``sum_{i<j} J_ij X_i X_j + field * sum_i Z_i``.

    Qubit 0 is the most significant bit of the computational basis index.

    Parameters
    ----------
    couplings : numpy.ndarray, shape (n_qubits, n_qubits)
        Symmetric ``J`` matrix; only the strict upper triangle is read.
    n_qubits : int
        Reservoir size.
    field : float
        Transverse field strength ``v``. Default value is 1.0.

    Returns
    -------
    numpy.ndarray, shape (2**n_qubits, 2**n_qubits)
        Real symmetric Hamiltonian (``X X`` couplings are real off-diagonal).
    """
    dim = 1 << n_qubits
    index = np.arange(dim)
    weights = np.arange(n_qubits - 1, -1, -1)
    bits = (index[:, None] >> weights[None, :]) & 1
    hamiltonian = np.zeros((dim, dim))
    hamiltonian[index, index] = field * np.sum(1.0 - 2.0 * bits, axis=1)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coupling = couplings[i, j]
            if coupling == 0.0:
                continue
            partner = index ^ (1 << (n_qubits - 1 - i)) ^ (1 << (n_qubits - 1 - j))
            hamiltonian[index, partner] += coupling
    return hamiltonian


def evolution_operator(hamiltonian: np.ndarray, tau: float) -> np.ndarray:
    """``exp(-i * tau * H)`` via exact diagonalisation (paper Appendix B)."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phased = eigenvectors * np.exp(-1j * tau * eigenvalues)[None, :]
    return phased @ eigenvectors.conj().T


def encode_inputs(features: np.ndarray) -> np.ndarray:
    """Product state ``prod_j RY(pi * x_j) |0>`` for a batch of feature vectors.

    Parameters
    ----------
    features : numpy.ndarray, shape (batch, n_input)
        Feature values; qubit 0 corresponds to column 0.

    Returns
    -------
    numpy.ndarray, shape (batch, 2**n_input), complex64
        Amplitudes with qubit 0 as the most significant bit.
    """
    batch, n_input = features.shape
    half_angle = 0.5 * np.pi * features
    amplitudes = np.stack([np.cos(half_angle), np.sin(half_angle)], axis=-1)
    state = amplitudes[:, 0, :]
    for qubit in range(1, n_input):
        state = (state[:, :, None] * amplitudes[:, qubit, None, :]).reshape(batch, -1)
    return state.astype(np.complex64)


def _pauli_z_signs(n_qubits: int) -> np.ndarray:
    """``+1 / -1`` sign table of ``Z_q`` on every computational basis state."""
    dim = 1 << n_qubits
    index = np.arange(dim)
    weights = np.arange(n_qubits - 1, -1, -1)
    return (1.0 - 2.0 * ((index[:, None] >> weights[None, :]) & 1)).astype(np.float32)


def _reduce_hidden(joint: np.ndarray, weights: np.ndarray, dim_input: int, dim_hidden: int):
    """Partial-trace the input register, returning a fresh rank decomposition.

    Parameters
    ----------
    joint : numpy.ndarray, shape (batch, rank, dim_input * dim_hidden)
        State vectors of the joint register (input qubits most significant).
    weights : numpy.ndarray, shape (batch, rank)
        Non-negative mixture weights.
    dim_input, dim_hidden : int
        Hilbert-space dimensions of the two registers.

    Returns
    -------
    tuple of numpy.ndarray
        ``(vectors, weights)`` for the reduced hidden state, with rank equal to
        ``min(rank * dim_input, dim_hidden)``.
    """
    batch, rank = joint.shape[:2]
    scaled = joint.reshape(batch, rank, dim_input, dim_hidden) * np.sqrt(
        np.maximum(weights, 0.0)
    )[:, :, None, None].astype(np.complex64)
    # With S = conj(scaled) we get S^dagger S = rho_hidden exactly, so the SVD
    # S = U diag(sigma) W^dagger gives rho_hidden = W diag(sigma^2) W^dagger.
    # The eigenvectors are the *columns* of W, i.e. the conjugated rows of
    # W^dagger; dropping that conjugation silently evolves the conjugate state.
    stacked = scaled.reshape(batch, rank * dim_input, dim_hidden).conj()
    _, singular, right = np.linalg.svd(stacked, full_matrices=False)
    return np.ascontiguousarray(right.conj()), (singular ** 2).astype(np.float32)


class QuantumReservoir:
    """Fixed transverse-field Ising reservoir with a cached time-evolution operator.

    Parameters
    ----------
    couplings : numpy.ndarray, shape (n_qubits, n_qubits)
        Symmetric ``J`` matrix; only the strict upper triangle is read.
    n_input : int
        Number of feature-carrying input qubits ``n1``.
    n_qubits : int
        Total reservoir size ``n1 + n2``. Default value is 10.
    tau : float
        Evolution time per encoding step. Default value is 1.0.
    virtual_nodes : int
        Readouts taken during the final evolution: 1 for QR1, 2 for QR2.
        Default value is 1.
    field : float
        Transverse field ``v``. Default value is 1.0.

    Raises
    ------
    ValueError
        If ``n_input`` exceeds ``n_qubits``.
    """

    def __init__(
        self,
        couplings: np.ndarray,
        n_input: int,
        *,
        n_qubits: int = 10,
        tau: float = 1.0,
        virtual_nodes: int = 1,
        field: float = 1.0,
    ) -> None:
        if n_input > n_qubits:
            raise ValueError(f"n_input={n_input} exceeds n_qubits={n_qubits}")
        self.n_input = n_input
        self.n_qubits = n_qubits
        self.n_hidden = n_qubits - n_input
        self.dim_input = 1 << n_input
        self.dim_hidden = 1 << self.n_hidden
        self.virtual_nodes = virtual_nodes
        self.n_readout = n_qubits * virtual_nodes

        hamiltonian = build_hamiltonian(couplings, n_qubits, field)
        self._full_step = evolution_operator(hamiltonian, tau).astype(np.complex64)
        self._sub_step = evolution_operator(hamiltonian, tau / virtual_nodes).astype(
            np.complex64
        )
        self._signs = _pauli_z_signs(n_qubits)

    def evaluate(self, windows: np.ndarray, batch: int | None = None) -> np.ndarray:
        """Read out ``<Z_j>`` for a batch of complete input windows.

        Parameters
        ----------
        windows : numpy.ndarray, shape (batch, n_lags, n_input)
            ``windows[b, 0]`` is the oldest lag of window ``b``.
        batch : int or None
            Chunk size for the batched linear algebra. Chosen from a memory
            budget when omitted. Default value is None.

        Returns
        -------
        numpy.ndarray, shape (batch, n_qubits * virtual_nodes)
        """
        n_windows, n_lags, _ = windows.shape
        max_rank = min(self.dim_hidden, self.dim_input ** max(n_lags - 1, 1))
        if batch is None:
            batch = max(
                1,
                int(
                    _SVD_ELEMENT_BUDGET
                    / max(max_rank * self.dim_input * self.dim_hidden, 1)
                ),
            )
        out = np.zeros((n_windows, self.n_readout), dtype=np.float64)
        for start in range(0, n_windows, batch):
            stop = min(n_windows, start + batch)
            out[start:stop] = self._evaluate_chunk(windows[start:stop])
        return out

    def _evaluate_chunk(self, windows: np.ndarray) -> np.ndarray:
        size, n_lags, _ = windows.shape
        hidden = np.zeros((size, 1, self.dim_hidden), dtype=np.complex64)
        hidden[:, 0, 0] = 1.0
        weights = np.ones((size, 1), dtype=np.float32)
        out = np.empty((size, self.n_readout))

        for lag in range(n_lags):
            encoded = encode_inputs(windows[:, lag, :])
            rank = hidden.shape[1]
            # Input qubits occupy the most significant block of the joint index.
            joint = (encoded[:, None, :, None] * hidden[:, :, None, :]).reshape(
                size, rank, -1
            )
            if lag < n_lags - 1:
                joint = joint @ self._full_step.T
                hidden, weights = _reduce_hidden(
                    joint, weights, self.dim_input, self.dim_hidden
                )
            else:
                for node in range(self.virtual_nodes):
                    joint = joint @ self._sub_step.T
                    probabilities = np.abs(joint) ** 2
                    out[:, node * self.n_qubits:(node + 1) * self.n_qubits] = np.einsum(
                        "trd,tr,dq->tq", probabilities, weights, self._signs
                    )
        return out


def reservoir_readout(
    inputs: np.ndarray,
    couplings: np.ndarray,
    *,
    n_qubits: int = 10,
    tau: float = 1.0,
    virtual_nodes: int = 1,
    field: float = 1.0,
) -> np.ndarray:
    """Pauli-Z readout matrix of the quantum reservoir for a whole time series.

    Parameters
    ----------
    inputs : numpy.ndarray, shape (T, n_lags, n_input)
        Lagged feature tensor from :func:`lib.data.build_lagged_inputs`;
        ``inputs[t, 0]`` is the oldest lag.
    couplings : numpy.ndarray, shape (n_qubits, n_qubits)
        Fixed reservoir coupling matrix.
    n_qubits, tau, virtual_nodes, field
        Reservoir settings; see :class:`QuantumReservoir`.

    Returns
    -------
    numpy.ndarray, shape (T, n_qubits * virtual_nodes)
        ``<Z_j>`` per qubit and virtual node. The first ``n_lags`` rows are left
        at zero because the reference implementation never evaluates them.
    """
    n_lags = inputs.shape[1]
    reservoir = QuantumReservoir(
        couplings,
        inputs.shape[2],
        n_qubits=n_qubits,
        tau=tau,
        virtual_nodes=virtual_nodes,
        field=field,
    )
    readout = np.zeros((inputs.shape[0], reservoir.n_readout))
    readout[n_lags:] = reservoir.evaluate(inputs[n_lags:])
    return readout


def rolling_ridge_forecast(
    readout: np.ndarray,
    target: np.ndarray,
    train_slices,
    predict_index,
    delta: float = 1e-8,
) -> np.ndarray:
    """Rolling-window ridge readout, re-estimated at every forecast origin.

    Implements paper Eq. 23 exactly as the reference code does: no intercept
    term, and ``delta`` present only to keep the Gram matrix invertible.

    Parameters
    ----------
    readout : numpy.ndarray, shape (T, n_features)
        Reservoir readout matrix (or any feature matrix).
    target : numpy.ndarray, shape (T,)
        Normalised realized volatility.
    train_slices : sequence of (int, int)
        Half-open training ranges, one per forecast.
    predict_index : sequence of int
        Row forecast by each window.
    delta : float
        Ridge regulariser. Default value is 1e-8.

    Returns
    -------
    numpy.ndarray, shape (len(predict_index),)
        One-step-ahead forecasts on the normalised target scale.
    """
    n_features = readout.shape[1]
    eye = np.eye(n_features)
    forecasts = np.empty(len(predict_index))
    for position, ((low, high), row) in enumerate(zip(train_slices, predict_index)):
        design = readout[low:high].T
        gram = design @ design.T + delta * eye
        weights = np.linalg.solve(gram, design @ target[low:high])
        forecasts[position] = weights @ readout[row]
    return forecasts


def ridge_weight_paths(
    readout: np.ndarray, target: np.ndarray, train_slices, delta: float = 1e-8
) -> np.ndarray:
    """Ridge readout weights for every rolling training window.

    Returns
    -------
    numpy.ndarray, shape (len(train_slices), n_features)
    """
    n_features = readout.shape[1]
    eye = np.eye(n_features)
    weights = np.empty((len(train_slices), n_features))
    for position, (low, high) in enumerate(train_slices):
        design = readout[low:high].T
        gram = design @ design.T + delta * eye
        weights[position] = np.linalg.solve(gram, design @ target[low:high])
    return weights


def closed_loop_forecast(
    inputs: np.ndarray,
    target: np.ndarray,
    couplings: np.ndarray,
    train_slices,
    origin_index,
    *,
    rv_column: int,
    horizon: int,
    n_qubits: int = 10,
    tau: float = 1.0,
    virtual_nodes: int = 1,
    delta: float = 1e-8,
    readout: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-step closed-loop forecasts (paper Section IV C).

    The readout is trained once per origin on ground-truth open-loop features,
    then applied ``horizon`` times. At each feedback step the newest ``RV`` input
    is replaced by the model's own prediction while exogenous features keep their
    ground-truth values, exactly as the paper specifies.

    Parameters
    ----------
    inputs : numpy.ndarray, shape (T, n_lags, n_input)
        Ground-truth lagged feature tensor.
    target : numpy.ndarray, shape (T,)
        Normalised realized volatility.
    couplings : numpy.ndarray
        Reservoir coupling matrix.
    train_slices : sequence of (int, int)
        Training range for each forecast path.
    origin_index : sequence of int
        First row predicted by each path; the path ends at
        ``origin + horizon - 1``.
    rv_column : int
        Index of the ``RV`` feature inside the input vector.
    horizon : int
        Number of steps ``S``. ``horizon = 1`` reduces to the open-loop case.
    n_qubits, tau, virtual_nodes, delta
        Reservoir and readout settings; see :class:`QuantumReservoir`.
    readout : numpy.ndarray or None
        Pre-computed open-loop readout matrix used for training. Recomputed when
        omitted. Default value is None.

    Returns
    -------
    numpy.ndarray, shape (len(origin_index),)
        Forecasts of ``target[origin + horizon - 1]`` on the normalised scale.
    """
    n_lags, n_input = inputs.shape[1], inputs.shape[2]
    reservoir = QuantumReservoir(
        couplings,
        n_input,
        n_qubits=n_qubits,
        tau=tau,
        virtual_nodes=virtual_nodes,
    )
    if readout is None:
        readout = np.zeros((inputs.shape[0], reservoir.n_readout))
        readout[n_lags:] = reservoir.evaluate(inputs[n_lags:])

    weights = ridge_weight_paths(readout, target, train_slices, delta)
    origins = np.asarray(origin_index)
    windows = inputs[origins].copy()          # (n_paths, n_lags, n_input)
    predictions = np.zeros(len(origins))

    for step in range(horizon):
        if step > 0:
            rows = np.clip(origins + step, 0, inputs.shape[0] - 1)
            windows = np.roll(windows, -1, axis=1)
            windows[:, -1, :] = inputs[rows][:, -1, :]
            windows[:, -1, rv_column] = predictions
        features = reservoir.evaluate(windows)
        predictions = np.einsum("pf,pf->p", weights, features)
    return predictions
