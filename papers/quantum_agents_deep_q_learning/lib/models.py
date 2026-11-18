"""Q-function approximators with a leading agent dimension.

Every parameter tensor has shape ``(A, ...)``: agent ``a`` owns slice ``[a]``
and never shares parameters with another agent, so training ``A`` agents in
one process is mathematically identical to ``A`` separate runs (up to
floating-point summation order).

* ``PQCQFunction``: the paper's PQC (Fig. 1, eqs. 10-12, 19).
* ``MLPQFunction``: the classical DQN network (Sec. 5.2.2, Table 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from . import circuits
from .config_access import require, require_choice

ANSATZE = ("ry_rz_cz_ring", "rz_ry_cz_all_final_rotation")
STATE_ENCODINGS_PQC = ("continuous_rx", "basis_state")
STATE_ENCODINGS_MLP = ("raw", "basis_bits")
READOUTS = ("cartpole_zz_pairs", "single_z_per_qubit")
ENCODING_ACTIVATIONS = ("arctan", "tanh")


@dataclass
class ParameterGroup:
    """A set of trainable tensors sharing one optimiser configuration.

    ``learning_rate_key`` names an entry of ``training.learning_rates`` and
    ``amsgrad`` selects the AMSGrad variant (reference code: AMSGrad for the
    variational / NN weights, plain Adam for input and output weights).
    """

    name: str
    tensors: list[nn.Parameter]
    learning_rate_key: str
    amsgrad: bool


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """complex64 for float32 (and lower) precision, complex128 for float64."""
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _uniform(
    shape: tuple[int, ...], low: float, high: float, seeds: list[int], dtype
) -> torch.Tensor:
    """Per-agent uniform initialisation: agent ``a`` draws from its own seed."""
    slices = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(int(seed))
        slices.append(
            low + (high - low) * torch.rand(shape, generator=generator, dtype=dtype)
        )
    return torch.stack(slices)


class PQCQFunction(nn.Module):
    """Parametrised-circuit Q-function, one independent circuit per agent.

    Circuit (paper Fig. 1, ``ansatz="ry_rz_cz_ring"``), repeated ``n_layers``
    times: optional encoding ``RX(x'_i)`` on each qubit, then ``RY(theta)``,
    ``RZ(theta)`` on each qubit, then CZ on the ring (0,1), (1,2), (2,3),
    (3,0). Encoding happens in every layer with data re-uploading, otherwise
    only before the first layer (paper Sec. 3.1, reference code).

    Continuous encoding (eq. 10): ``x' = arctan(x * w_d)``; the reference
    code uses ``tanh`` instead (selectable). One input weight per qubit and
    per encoding layer, which reproduces the paper's parameter counts
    (5 layers: 40 rotations + 20 input weights + 2 output weights = 62).

    Readout (eqs. 12, 19): ``Q(s, a) = (<O_a> + 1) / 2 * w_{o_a} * factor``.
    With ``train_output_weights=False`` the weights stay 1, which gives the
    fixed ranges ``[0, factor]`` of Fig. 5.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``model`` section is read).
    seeds : list of int
        One seed per agent; agent ``a`` initialises its angles from
        ``seeds[a]``.
    n_actions : int
        Number of Q-values per state.
    real_dtype : torch.dtype
        Real precision; complex precision follows (``complex_dtype_for``).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        seeds: list[int],
        n_actions: int,
        real_dtype: torch.dtype,
    ):
        super().__init__()
        self.n_agents = len(seeds)
        self.n_qubits = int(require(cfg, "model.n_qubits"))
        self.n_layers = int(require(cfg, "model.n_layers"))
        self.ansatz = require_choice(cfg, "model.ansatz", ANSATZE)
        self.state_encoding = require_choice(
            cfg, "model.state_encoding", STATE_ENCODINGS_PQC
        )
        self.readout = require_choice(cfg, "model.readout", READOUTS)
        self.output_factor = float(require(cfg, "model.output_factor"))
        self.real_dtype = real_dtype
        self.complex_dtype = complex_dtype_for(real_dtype)

        if self.state_encoding == "continuous_rx":
            self.encoding_activation = require_choice(
                cfg, "model.encoding_activation", ENCODING_ACTIVATIONS
            )
            self.data_reuploading = bool(require(cfg, "model.data_reuploading"))
            train_input_weights = bool(require(cfg, "model.train_input_weights"))
        else:
            self.encoding_activation = None
            self.data_reuploading = False
            train_input_weights = False
            if bool(require(cfg, "model.data_reuploading")) or bool(
                require(cfg, "model.train_input_weights")
            ):
                raise ValueError(
                    "basis_state encoding supports neither re-uploading nor input weights"
                )
        train_output_weights = bool(require(cfg, "model.train_output_weights"))

        rotation_layers = self.n_layers + (
            1 if self.ansatz == "rz_ry_cz_all_final_rotation" else 0
        )
        # Variational angles ~ U[low, high]. Reference code: U[0, pi] for Cart Pole; Keras
        # RandomUniform default U[-0.05, 0.05] for Frozen Lake, which reproduces the paper's
        # initial MAE of 0.54 in Fig. 3b (LOG.md). Input and output weights start at 1.
        theta_low = float(require(cfg, "model.theta_init.low"))
        theta_high = float(require(cfg, "model.theta_init.high"))
        theta = _uniform(
            (rotation_layers, self.n_qubits, 2),
            theta_low,
            theta_high,
            seeds,
            real_dtype,
        )
        self.theta = nn.Parameter(theta)

        self.n_encoding_layers = 0
        if self.state_encoding == "continuous_rx":
            self.n_encoding_layers = self.n_layers if self.data_reuploading else 1
        input_weights = torch.ones(
            self.n_agents,
            max(self.n_encoding_layers, 1),
            self.n_qubits,
            dtype=real_dtype,
        )
        output_weights = torch.ones(self.n_agents, n_actions, dtype=real_dtype)
        self.input_weights = nn.Parameter(
            input_weights, requires_grad=train_input_weights
        )
        self.output_weights = nn.Parameter(
            output_weights, requires_grad=train_output_weights
        )

        pairs = (
            circuits.ring_pairs(self.n_qubits)
            if self.ansatz == "ry_rz_cz_ring"
            else circuits.all_pairs(self.n_qubits)
        )
        self.register_buffer(
            "cz_diagonal",
            circuits.cz_phase_diagonal(self.n_qubits, pairs).to(self.complex_dtype),
        )
        diagonals = circuits.readout_diagonals(self.readout, self.n_qubits).to(
            real_dtype
        )
        if diagonals.shape[1] != n_actions:
            raise ValueError(
                f"Readout {self.readout} gives {diagonals.shape[1]} values, environment has {n_actions} actions"
            )
        self.register_buffer("readout_diagonal", diagonals)

    def parameter_groups(self) -> list[ParameterGroup]:
        """Trainable tensors grouped by learning rate (paper: eta, eta_wd, eta_wo)."""
        groups = [ParameterGroup("variational", [self.theta], "main", amsgrad=True)]
        if self.input_weights.requires_grad and self.n_encoding_layers > 0:
            groups.append(
                ParameterGroup(
                    "input_weights",
                    [self.input_weights],
                    "input_weights",
                    amsgrad=False,
                )
            )
        if self.output_weights.requires_grad:
            groups.append(
                ParameterGroup(
                    "output_weights",
                    [self.output_weights],
                    "output_weights",
                    amsgrad=False,
                )
            )
        return groups

    def count_trainable_parameters(self) -> int:
        """Trainable parameters of one agent (the paper's '#parameters')."""
        count = self.theta[0].numel()
        if self.input_weights.requires_grad and self.n_encoding_layers > 0:
            count += self.input_weights[0].numel()
        if self.output_weights.requires_grad:
            count += self.output_weights[0].numel()
        return count

    def _variational_operators(self, layer: int) -> torch.Tensor:
        """Per-agent single-qubit operators ``(A, n, 2, 2)`` of one rotation layer."""
        first_angles = self.theta[:, layer, :, 0]
        second_angles = self.theta[:, layer, :, 1]
        if self.ansatz == "ry_rz_cz_ring":
            # RY then RZ in time order (Fig. 1): operator RZ @ RY.
            return circuits.rz_matrices(
                second_angles, self.complex_dtype
            ) @ circuits.ry_matrices(first_angles, self.complex_dtype)
        # Reference Frozen Lake code: RZ then RY in time order: operator RY @ RZ.
        return circuits.ry_matrices(
            second_angles, self.complex_dtype
        ) @ circuits.rz_matrices(first_angles, self.complex_dtype)

    def _encoding_angles(self, inputs: torch.Tensor, layer: int) -> torch.Tensor:
        """Encoding angles ``(A, B, n)`` for encoding layer ``layer``."""
        scaled = inputs * self.input_weights[:, layer, None, :]
        if self.encoding_activation == "arctan":
            return torch.atan(scaled)
        return torch.tanh(scaled)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Q-values for a batch of observations per agent.

        Parameters
        ----------
        observations : torch.Tensor
            ``(A, B, 4)`` Cart Pole variables, or ``(A, B, 1)`` Frozen Lake
            state indices.

        Returns
        -------
        torch.Tensor
            ``(A, B, n_actions)`` Q-values.
        """
        n_agents, batch = observations.shape[0], observations.shape[1]
        if self.state_encoding == "basis_state":
            state = circuits.basis_states(
                observations[..., 0].round(), self.n_qubits, self.complex_dtype
            )
        else:
            state = circuits.zero_states(
                (n_agents, batch),
                self.n_qubits,
                self.complex_dtype,
                observations.device,
            )
            inputs = observations.to(self.real_dtype)

        for layer in range(self.n_layers):
            variational = self._variational_operators(layer)
            encode_now = self.state_encoding == "continuous_rx" and (
                self.data_reuploading or layer == 0
            )
            if encode_now:
                encoding_layer = layer if self.data_reuploading else 0
                encoding = circuits.rx_matrices(
                    self._encoding_angles(inputs, encoding_layer), self.complex_dtype
                )
                per_sample = variational[:, None] @ encoding
                state = circuits.apply_operator(state, circuits.kron_qubits(per_sample))
            else:
                state = circuits.apply_operator(
                    state, circuits.kron_qubits(variational)[:, None]
                )
            state = state * self.cz_diagonal

        if self.ansatz == "rz_ry_cz_all_final_rotation":
            final = self._variational_operators(self.n_layers)
            state = circuits.apply_operator(state, circuits.kron_qubits(final)[:, None])

        expectation = circuits.expectations(state, self.readout_diagonal)
        return (
            (expectation + 1.0)
            / 2.0
            * self.output_weights[:, None, :]
            * self.output_factor
        )


class MLPQFunction(nn.Module):
    """Fully connected DQN network, one independent network per agent.

    ReLU hidden layers, linear output, Keras default initialisation (Glorot
    uniform weights, zero biases), as in the reference code.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``model.hidden_units``,
        ``model.state_encoding``).
    seeds : list of int
        One seed per agent.
    n_actions : int
        Output size.
    real_dtype : torch.dtype
        Parameter precision.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        seeds: list[int],
        n_actions: int,
        real_dtype: torch.dtype,
    ):
        super().__init__()
        self.n_agents = len(seeds)
        self.state_encoding = require_choice(
            cfg, "model.state_encoding", STATE_ENCODINGS_MLP
        )
        hidden_units = [int(units) for units in require(cfg, "model.hidden_units")]
        self.real_dtype = real_dtype
        sizes = [4, *hidden_units, n_actions]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer_index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            layer_seeds = [seed * 1000 + layer_index for seed in seeds]
            self.weights.append(
                nn.Parameter(
                    _uniform((fan_in, fan_out), -limit, limit, layer_seeds, real_dtype)
                )
            )
            self.biases.append(
                nn.Parameter(torch.zeros(self.n_agents, 1, fan_out, dtype=real_dtype))
            )
        self.register_buffer("bit_shifts", torch.arange(3, -1, -1))

    def parameter_groups(self) -> list[ParameterGroup]:
        """All weights share eta and the reference code's AMSGrad Adam."""
        return [
            ParameterGroup(
                "network", [*self.weights, *self.biases], "main", amsgrad=True
            )
        ]

    def count_trainable_parameters(self) -> int:
        """Trainable parameters of one agent."""
        return sum(tensor[0].numel() for tensor in [*self.weights, *self.biases])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Q-values ``(A, B, n_actions)`` from ``(A, B, 4)`` or ``(A, B, 1)`` inputs."""
        if self.state_encoding == "basis_bits":
            indices = observations[..., 0].round().long()
            hidden = ((indices[..., None] >> self.bit_shifts) & 1).to(self.real_dtype)
        else:
            hidden = observations.to(self.real_dtype)
        last = len(self.weights) - 1
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.baddbmm(bias, hidden, weight)
            if layer_index < last:
                hidden = torch.relu(hidden)
        return hidden


def build_q_function(
    cfg: dict[str, Any], seeds: list[int], n_actions: int, real_dtype: torch.dtype
) -> nn.Module:
    """Instantiate the configured Q-function (``model.kind``: ``pqc``, ``mlp`` or ``merlin``).

    ``merlin`` is the Phase 4 photonic adaptation (``lib/qfunction_merlin.py``),
    imported lazily so the qubit and NN paths do not depend on MerLin.
    """
    kind = require_choice(cfg, "model.kind", ("pqc", "mlp", "merlin"))
    if kind == "pqc":
        return PQCQFunction(cfg, seeds, n_actions, real_dtype)
    if kind == "merlin":
        from .qfunction_merlin import MerlinQFunction

        return MerlinQFunction(cfg, seeds, n_actions, real_dtype)
    return MLPQFunction(cfg, seeds, n_actions, real_dtype)
