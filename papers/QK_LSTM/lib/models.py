"""QK-LSTM variants and classical baselines (LOG.md, Fair Baseline Plan).

Trainable models map a window ``x`` of shape ``(B, L, F)`` to a scaled AQI
prediction of shape ``(B, 1)`` from the last hidden state, through the scalar
head ``Linear(hidden, 1)`` implied by the paper's parameter counts.

Kernel-gated cells (``QKLSTMCell``):

``gate_mode="per_gate"`` (Q-A, paper Eq. 6a-6f)
    ``v_t = [h_{t-1}; x_t]`` is compressed by ``Linear(F + H -> n_qubits)``.
    For every gate ``g`` the pre-activation is
    ``sum_j alpha_j^{(g)} k(v_t, v_j) + b_g`` with ``alpha^{(g)}`` of shape
    ``(N, H)``.

``gate_mode="shared"`` (Q-B, count-matched to Table II)
    Same compression and kernel against ``N = 4`` references, then a single
    ``Linear(N -> H)`` pre-activation ``a`` shared by all gates:
    ``f = i = o = sigmoid(a)``, ``C~ = tanh(a)``. With F=11, H=16 and 4 qubits
    this has exactly 209 trainable parameters, the only architecture we found
    that matches both the paper's 209 and the parent paper's 183 (LOG.md,
    parameter-count reverse engineering).

The reference vectors ``v_j = [0; x_j]`` are fixed training-set inputs with a
zero hidden part. They are buffers, not parameters, because neither parameter
count leaves room for learned references. The kernel has no trainable circuit
parameters for the same reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .quantum_kernel import quantum_kernel, rbf_kernel

KERNEL_MODEL_TYPES = ("qk_lstm", "rbf_lstm", "qk_lstm_merlin")
TORCH_MODEL_TYPES = (*KERNEL_MODEL_TYPES, "lstm")
CLOSED_FORM_MODEL_TYPES = ("persistence", "ridge")


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters of a module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class QKLSTMCell(nn.Module):
    """One kernel-gated LSTM cell (Eq. 6a-6f).

    Parameters
    ----------
    input_size : int
        Number of input features ``F``.
    hidden_size : int
        Hidden size ``H``.
    n_qubits : int
        Compressed dimension, one feature per qubit.
    reference_inputs : torch.Tensor
        ``(N, F)`` scaled training inputs ``x_j`` defining ``v_j = [0; x_j]``.
    gate_mode : str
        ``"per_gate"`` (Q-A) or ``"shared"`` (Q-B).
    kernel : str
        ``"quantum"`` (qubit fidelity kernel), ``"rbf"`` (classical ablation B3)
        or ``"photonic"`` (MerLin fidelity kernel; Phase 4 adaptation).
    circuit_layout : str
        Quantum circuit layout, ``"text"`` or ``"figure"``.
    angle_map : str
        Quantum angle map, see :func:`lib.quantum_kernel.encoding_angles`.
    rbf_gamma : float
        RBF bandwidth for ``kernel="rbf"``.
    photonic : dict | None
        ``input_state``, ``mesh_seeds`` and ``encoding_scale`` for
        ``kernel="photonic"``. Default value is None.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int,
        reference_inputs: torch.Tensor,
        *,
        gate_mode: str,
        kernel: str,
        circuit_layout: str,
        angle_map: str,
        rbf_gamma: float,
        photonic: dict | None = None,
    ):
        super().__init__()
        if gate_mode not in ("per_gate", "shared"):
            raise ValueError(f"Unknown gate_mode {gate_mode!r}")
        if kernel not in ("quantum", "rbf", "photonic"):
            raise ValueError(f"Unknown kernel {kernel!r}")
        self.hidden_size = hidden_size
        self.gate_mode = gate_mode
        self.kernel = kernel
        self.circuit_layout = circuit_layout
        self.angle_map = angle_map
        self.rbf_gamma = rbf_gamma

        n_references = reference_inputs.shape[0]
        reference_vectors = torch.cat(
            [torch.zeros(n_references, hidden_size), reference_inputs.float()], dim=1
        )
        self.register_buffer("reference_vectors", reference_vectors)
        # v_t = [h_{t-1}; x_t] (Eq. 5) -> one feature per qubit.
        self.compression = nn.Linear(hidden_size + input_size, n_qubits)
        if gate_mode == "per_gate":
            # alpha^{(g)}_j is a vector over hidden units so each gate stays H-dimensional.
            self.alpha = nn.Parameter(0.1 * torch.randn(4, n_references, hidden_size))
            self.gate_bias = nn.Parameter(torch.zeros(4, hidden_size))
        else:
            self.shared_gate = nn.Linear(n_references, hidden_size)
        if kernel == "photonic":
            if photonic is None:
                raise ValueError("kernel='photonic' needs the photonic configuration")
            # Imported here so the qubit/RBF paths never depend on MerLin.
            from .qk_lstm_merlin import PhotonicFidelityKernel

            self.photonic_kernel = PhotonicFidelityKernel(
                n_qubits,
                photonic["input_state"],
                photonic["mesh_seeds"],
                encoding_scale=float(photonic["encoding_scale"]),
            )

    def kernel_features(self, v: torch.Tensor) -> torch.Tensor:
        """Kernel values ``k(v_t, v_j)`` of shape ``(B, N)``."""
        z = self.compression(v)
        z_reference = self.compression(self.reference_vectors)
        if self.kernel == "quantum":
            return quantum_kernel(
                z, z_reference, layout=self.circuit_layout, angle_map=self.angle_map
            )
        if self.kernel == "photonic":
            return self.photonic_kernel(z, z_reference)
        return rbf_kernel(z, z_reference, gamma=self.rbf_gamma)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        kernel_values = self.kernel_features(torch.cat([h_prev, x_t], dim=1))
        if self.gate_mode == "per_gate":
            pre = (
                torch.einsum("bn,gnh->gbh", kernel_values, self.alpha)
                + self.gate_bias[:, None, :]
            )
            forget, input_gate, output = (torch.sigmoid(pre[k]) for k in (0, 1, 3))
            candidate = torch.tanh(pre[2])
        else:
            shared = self.shared_gate(kernel_values)
            forget = input_gate = output = torch.sigmoid(shared)
            candidate = torch.tanh(shared)
        cell = forget * c_prev + input_gate * candidate  # Eq. 6d
        hidden = output * torch.tanh(cell)  # Eq. 6f
        return hidden, cell


class KernelLSTMRegressor(nn.Module):
    """Unrolls :class:`QKLSTMCell` over a window and applies the scalar head."""

    def __init__(self, cell: QKLSTMCell):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        hidden = x.new_zeros(batch, self.cell.hidden_size)
        cell_state = x.new_zeros(batch, self.cell.hidden_size)
        for step in range(x.shape[1]):
            hidden, cell_state = self.cell(x[:, step], (hidden, cell_state))
        return self.head(hidden)


class LSTMRegressor(nn.Module):
    """Classical baseline ``nn.LSTM(F, H)`` + ``Linear(H, 1)``.

    ``torch.nn.LSTM`` keeps two bias vectors per gate; with F=11 and H=16 this
    gives exactly the paper's 1,873 parameters, and H=3 gives the 196-parameter
    matched baseline B2.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        return self.head(outputs[:, -1])


def build_torch_model(
    model_cfg: dict[str, Any], input_size: int, reference_inputs: torch.Tensor | None
) -> nn.Module:
    """Build a trainable model from the ``model`` config section.

    Parameters
    ----------
    model_cfg : dict
        ``model`` section with ``type`` and ``params``.
    input_size : int
        Number of input features.
    reference_inputs : torch.Tensor | None
        ``(N, F)`` reference inputs for kernel models; ignored otherwise.

    Returns
    -------
    torch.nn.Module
        The model.

    Raises
    ------
    ValueError
        If the model type is not trainable by gradient descent.
    """
    model_type = model_cfg["type"]
    params = model_cfg["params"]
    if model_type == "lstm":
        return LSTMRegressor(input_size, int(params["hidden_size"]))
    if model_type in KERNEL_MODEL_TYPES:
        if reference_inputs is None:
            raise ValueError("Kernel models need reference inputs")
        cell = QKLSTMCell(
            input_size,
            int(params["hidden_size"]),
            int(params["n_qubits"]),
            reference_inputs,
            gate_mode=params["gate_mode"],
            kernel={
                "qk_lstm": "quantum",
                "rbf_lstm": "rbf",
                "qk_lstm_merlin": "photonic",
            }[model_type],
            circuit_layout=params.get("circuit_layout", "text"),
            angle_map=params.get("angle_map", "identity"),
            rbf_gamma=float(params.get("rbf_gamma", 1.0)),
            photonic=params.get("photonic"),
        )
        return KernelLSTMRegressor(cell)
    raise ValueError(
        f"Model type {model_type!r} is not a torch model; expected one of {TORCH_MODEL_TYPES}"
    )


def sample_reference_inputs(
    train_features: np.ndarray, n_references: int, seed: int
) -> torch.Tensor:
    """Draw ``n_references`` distinct training-day input vectors (seeded)."""
    if n_references > train_features.shape[0]:
        raise ValueError("More references than training days")
    rng = np.random.default_rng(seed)
    rows = rng.choice(train_features.shape[0], size=n_references, replace=False)
    return torch.from_numpy(np.asarray(train_features[np.sort(rows)], dtype=np.float32))


class RidgeWindowRegressor:
    """Closed-form ridge regression on the flattened ``L x F`` window (B0b).

    The intercept is not penalised. Parameters: ``L * F + 1`` (34 for L=3, F=11).
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None
        self.intercept = 0.0

    @property
    def n_parameters(self) -> int:
        return 0 if self.weights is None else int(self.weights.size + 1)

    def fit(self, windows: np.ndarray, targets: np.ndarray) -> RidgeWindowRegressor:
        design = windows.reshape(windows.shape[0], -1).astype(np.float64)
        targets = targets.reshape(-1).astype(np.float64)
        design_mean, target_mean = design.mean(axis=0), targets.mean()
        centred = design - design_mean
        gram = centred.T @ centred + self.alpha * np.eye(centred.shape[1])
        self.weights = np.linalg.solve(gram, centred.T @ (targets - target_mean))
        self.intercept = float(target_mean - design_mean @ self.weights)
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("RidgeWindowRegressor is not fitted")
        return windows.reshape(windows.shape[0], -1) @ self.weights + self.intercept
