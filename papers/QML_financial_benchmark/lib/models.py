"""Quantum and classical models of arXiv:2601.03802.

Every model returns one logit per sample, so all classifiers share
``BCEWithLogitsLoss`` and the same AUC evaluation.

Rotation angles are initialised uniformly in ``[-pi, pi)``, the TorchQuantum
default for trainable rotation gates (the paper's simulator, §IV-A).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import statevector as sv


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class VariationalAnsatz(nn.Module):
    """``layers`` repetitions of RZ-RY-RZ on every qubit followed by a CZ entangler (Fig. 5).

    Parameters: ``3 * n_qubits * layers``.
    """

    def __init__(self, n_qubits: int, layers: int, topology: str = "chain"):
        super().__init__()
        self.n_qubits = n_qubits
        self.angles = nn.Parameter(
            torch.empty(layers, n_qubits, 3).uniform_(-math.pi, math.pi)
        )
        self.pairs = sv.entangling_pairs(n_qubits, topology)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        phase = (
            sv.cz_phase(self.n_qubits, self.pairs, self.angles.dtype, state.device)
            if self.pairs
            else None
        )
        for layer_angles in self.angles:
            for qubit in range(self.n_qubits):
                first, second, third = layer_angles[qubit]
                gate = sv.rz_matrix(third) @ sv.ry_matrix(second) @ sv.rz_matrix(first)
                state = sv.apply_single_qubit(state, gate, qubit)
            if phase is not None:
                state = state * phase
        return state


def angle_encode(angles: torch.Tensor) -> torch.Tensor:
    """``prod_k RY(angles[:, k]) |0...0>`` for a batch of angle vectors."""
    batch, n_qubits = angles.shape
    state = sv.zero_state(batch, n_qubits, angles.dtype, angles.device)
    for qubit in range(n_qubits):
        state = sv.apply_single_qubit(state, sv.ry_matrix(angles[:, qubit]), qubit)
    return state


class QNNClassifier(nn.Module):
    """The four QNN families of §III-C3: {fully quantum, hybrid} x {single, multi-qubit readout}.

    Parameters
    ----------
    n_features : int
        Input dimension ``d``.
    n_qubits : int
        Register size ``q``.
    layers : int
        Ansatz depth ``L``.
    hybrid : bool
        If True, a classical ``Linear(d, k)`` precedes encoding, where ``k`` is
        ``q`` for angle encoding and ``2**q`` for amplitude encoding (reading (i)
        of R5). If False, ``d`` must equal ``k``.
    multi_qubit_readout : bool
        If True, ``<Z>`` of all qubits feeds ``Linear(q, 1)``; otherwise
        ``(1 + <Z_last>) / 2`` of the last qubit is the probability (Fig. 5).
    encoding : str
        ``"angle"`` (one RY per qubit) or ``"amplitude"``.
    angle_scale : float
        Multiplier applied to ``[0, 1]``-scaled features before RY encoding in
        non-hybrid angle models (assumption: ``pi``, so inputs span a half turn).
        Hybrid models feed the linear outputs directly as angles.
    topology : str
        CZ entangler pattern, ``"chain"`` or ``"ring"``. Default value is ``"chain"``.
    """

    def __init__(
        self,
        n_features: int,
        n_qubits: int,
        layers: int,
        hybrid: bool,
        multi_qubit_readout: bool,
        encoding: str,
        angle_scale: float,
        topology: str = "chain",
    ):
        super().__init__()
        if encoding not in {"angle", "amplitude"}:
            raise ValueError(f"Unknown encoding {encoding!r}")
        encoded_dim = n_qubits if encoding == "angle" else 2**n_qubits
        if not hybrid and n_features != encoded_dim:
            raise ValueError(
                f"Non-hybrid {encoding} QNN on {n_qubits} qubits needs {encoded_dim} features, got {n_features}"
            )
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.angle_scale = angle_scale
        self.hybrid = hybrid
        self.preprocess = nn.Linear(n_features, encoded_dim) if hybrid else None
        self.ansatz = VariationalAnsatz(n_qubits, layers, topology)
        self.readout = nn.Linear(n_qubits, 1) if multi_qubit_readout else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.preprocess(inputs) if self.hybrid else inputs
        if self.encoding == "angle":
            angles = encoded if self.hybrid else self.angle_scale * encoded
            state = angle_encode(angles)
        else:
            state = sv.amplitude_state(encoded, self.n_qubits)
        expectations = sv.expectation_z(self.ansatz(state)).to(inputs.dtype)
        if self.readout is not None:
            return self.readout(expectations).squeeze(-1)
        probability = ((1.0 + expectations[:, -1]) / 2.0).clamp(1e-6, 1 - 1e-6)
        return torch.log(probability) - torch.log1p(-probability)


class MLPClassifier(nn.Module):
    """Shallow ReLU network ``[d - h1 - (h2) - 1]`` (§III-C3 ANN baselines)."""

    def __init__(self, n_features: int, hidden_widths: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        width_in = n_features
        for width in hidden_widths:
            layers += [nn.Linear(width_in, width), nn.ReLU()]
            width_in = width
        layers.append(nn.Linear(width_in, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class VQCBlock(nn.Module):
    """One QLSTM gate circuit: per-qubit affine input map, RY encoding, ansatz, all-qubit ``<Z>``.

    Parameters: ``2 * q`` (input scale and bias) ``+ 3 * q * layers``.
    """

    def __init__(self, n_qubits: int, layers: int, topology: str = "chain"):
        super().__init__()
        self.input_scale = nn.Parameter(torch.ones(n_qubits))
        self.input_bias = nn.Parameter(torch.zeros(n_qubits))
        self.ansatz = VariationalAnsatz(n_qubits, layers, topology)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        state = angle_encode(self.input_scale * embedded + self.input_bias)
        return sv.expectation_z(self.ansatz(state)).to(embedded.dtype)


class QLSTMClassifier(nn.Module):
    """QLSTM of Chen et al. (2020) as used in §III-D3, ending in a direction logit.

    Layout reconstructed from the Table XII parameter counts (decision R12):
    a shared ``Linear(d + q, q)`` embeds ``[x_t, h_{t-1}]``; six VQC blocks give
    the forget, input, candidate and output gates, the new hidden state
    ``h_t = VQC5(o * tanh(c_t))`` and the step output ``y_t = VQC6(o * tanh(c_t))``;
    ``Linear(q, 1)`` maps the last ``y_T`` to the logit. Total parameters
    ``q^2 + (d + 2) q + 12 q + 18 q L + 1``, which gives 233, 476, 386 and 656
    for the four Table XII configurations with ``d = 4``.
    """

    def __init__(
        self, n_features: int, n_qubits: int, layers: int, topology: str = "chain"
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.embedding = nn.Linear(n_features + n_qubits, n_qubits)
        self.circuits = nn.ModuleList(
            VQCBlock(n_qubits, layers, topology) for _ in range(6)
        )
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = sequences.shape
        hidden = sequences.new_zeros(batch, self.n_qubits)
        cell = sequences.new_zeros(batch, self.n_qubits)
        output = hidden
        (
            forget_vqc,
            input_vqc,
            candidate_vqc,
            output_vqc,
            hidden_vqc,
            step_output_vqc,
        ) = self.circuits
        for step in range(steps):
            embedded = self.embedding(torch.cat([sequences[:, step], hidden], dim=1))
            forget_gate = torch.sigmoid(forget_vqc(embedded))
            input_gate = torch.sigmoid(input_vqc(embedded))
            candidate = torch.tanh(candidate_vqc(embedded))
            output_gate = torch.sigmoid(output_vqc(embedded))
            cell = forget_gate * cell + input_gate * candidate
            gated = output_gate * torch.tanh(cell)
            hidden = hidden_vqc(gated)
            output = step_output_vqc(gated)
        return self.head(output).squeeze(-1)


class LSTMClassifier(nn.Module):
    """Stacked ``nn.LSTM`` plus ``Linear(h, 1)`` on the last step.

    Matches the Table XII counts exactly, e.g. ``h=4, layers=2 -> 325`` and
    ``h=5, layers=6 -> 1426``.
    """

    def __init__(self, n_features: int, hidden_size: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers=layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequences)
        return self.head(outputs[:, -1]).squeeze(-1)
