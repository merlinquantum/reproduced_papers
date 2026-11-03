"""Fixed-frequency (FF) and trainable-frequency (TF) QNNs from the paper.

Feature map (paper Eq. 1 and Sec. V): every encoding qubit ``m`` applies
``R_y(gamma_m * w_m * x_d)`` where ``x_d`` is the input coordinate assigned to
that qubit, ``gamma_m`` is the fixed generator coefficient (simple: 1, tower: m,
exponential: 2**(m-1)) and ``w_m`` depends on the frequency mode:

- ``fixed``        : ``w_m = 1`` (FF model, the paper's baseline);
- ``trainable``    : ``w_m = theta_{F,m}``, one parameter per encoding qubit,
  initialised at 1 (TF model, ``G_theta = sum_m theta_m Y_m / 2``);
- ``global_scale`` : ``w_m = s_d``, one trainable scale per input dimension,
  initialised at 1 (FF+S fairness control from LOG.md, B2).

Ansatz (paper App. B, Fig. 6): per layer ``R_x R_y R_x`` on every qubit with
independent angles, then a ring of CNOTs (digital HEA, Fig. 6a) or CZs (sDAQC
HEA, Fig. 6b). Output is ``<sum_m Z_m>``.

Layouts
-------
``toy``         : FM, then ``n_layers`` ansatz layers (Sec. V, App. A Eq. A1).
``ns_parallel`` : A, FM, ``n_layers - 2`` A, FM (re-upload, shared theta_F), A
                  (Sec. VI, Fig. 4); 2 qubits per coordinate (x, y, t).
``ns_serial``   : FM_x, A, FM_y, A, FM_t, ``n_mid`` A, re-upload of the same
                  serial FM, remaining A (App. C); every qubit encodes every
                  coordinate. The split is fixed so the ansatz parameter count
                  matches the paper's 804 for N = 4 (LOG.md, A12).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .statevector import (
    apply_cnot,
    apply_cz,
    apply_single_qubit,
    dense_operator,
    magnetization_diagonal,
    rotation_matrix,
    total_magnetization,
    y_eigenbasis,
    zero_state,
)

FEATURE_MAP_KINDS = ("simple", "tower", "exponential")
FREQUENCY_MODES = ("fixed", "trainable", "global_scale")
ENTANGLERS = ("cnot", "cz")
LAYOUTS = ("toy", "ns_parallel", "ns_serial")


def generator_coefficients(kind: str, n_encoding_qubits: int) -> list[float]:
    """Return the fixed generator coefficients ``gamma_m`` (paper Sec. V).

    Parameters
    ----------
    kind : str
        ``"simple"`` (gamma_m = 1), ``"tower"`` (gamma_m = m) or
        ``"exponential"`` (gamma_m = 2**(m-1)), with m = 1..N.
    n_encoding_qubits : int
        Number of qubits encoding one input coordinate.

    Returns
    -------
    list of float
        Coefficients in qubit order.
    """
    if kind == "simple":
        return [1.0] * n_encoding_qubits
    if kind == "tower":
        return [float(m) for m in range(1, n_encoding_qubits + 1)]
    if kind == "exponential":
        return [float(2 ** (m - 1)) for m in range(1, n_encoding_qubits + 1)]
    raise ValueError(
        f"Unknown feature map kind {kind!r}; expected one of {FEATURE_MAP_KINDS}."
    )


def ring_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Return the ring entangler order of paper Fig. 6.

    Even bonds (0-1, 2-3, ...) first, then odd bonds (1-2, 3-4, ...) and the
    wrap-around bond (N-1, 0), as drawn for N = 4 (CNOT) and N = 6 (CZ). The
    first index is the CNOT control.
    """
    if n_qubits < 2:
        return []
    if n_qubits == 2:
        return [(0, 1)]
    even = [(q, q + 1) for q in range(0, n_qubits - 1, 2)]
    odd = [(q, q + 1) for q in range(1, n_qubits - 1, 2)]
    return even + odd + [(n_qubits - 1, 0)]


class TrainableFrequencyQNN(nn.Module):
    """QNN with a fixed, trainable or globally scaled frequency feature map.

    Parameters
    ----------
    n_qubits : int
        Number of qubits N.
    n_inputs : int
        Number of input coordinates (1 for the toy task, 3 for (x, y, t)).
    layout : str
        One of ``"toy"``, ``"ns_parallel"``, ``"ns_serial"``.
    n_layers : int
        Total number of ansatz layers in the circuit.
    entangler : str
        ``"cnot"`` or ``"cz"``.
    feature_map : str
        ``"simple"``, ``"tower"`` or ``"exponential"``.
    frequency_mode : str
        ``"fixed"``, ``"trainable"`` or ``"global_scale"``.
    frequency_init : float
        Initial value of every theta_F or global scale, and the constant
        multiplier ``w`` of a fixed-frequency model. The paper uses 1.0 (Sec. V);
        other values are diagnosis controls (LOG.md, e4b / e5b sweeps).
    dtype : torch.dtype
        Real dtype of the parameters (float64 recommended).
    generator : torch.Generator
        Seeded generator for the ansatz initialisation. Ansatz angles are drawn
        uniformly in [0, 2 pi) (assumption A3). Because theta_F is not drawn
        from it, FF and TF models built with the same seed share their ansatz
        initialisation (paired seeds, LOG.md Fair Baseline Plan).
    """

    def __init__(
        self,
        *,
        n_qubits: int,
        n_inputs: int,
        layout: str,
        n_layers: int,
        entangler: str,
        feature_map: str,
        frequency_mode: str,
        frequency_init: float,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout {layout!r}; expected one of {LAYOUTS}.")
        if entangler not in ENTANGLERS:
            raise ValueError(
                f"Unknown entangler {entangler!r}; expected one of {ENTANGLERS}."
            )
        if frequency_mode not in FREQUENCY_MODES:
            raise ValueError(
                f"Unknown frequency_mode {frequency_mode!r}; expected one of {FREQUENCY_MODES}."
            )
        self.n_qubits = n_qubits
        self.n_inputs = n_inputs
        self.layout = layout
        self.n_layers = n_layers
        self.entangler = entangler
        self.feature_map = feature_map
        self.frequency_mode = frequency_mode
        self.real_dtype = dtype
        self.complex_dtype = (
            torch.complex128 if dtype == torch.float64 else torch.complex64
        )

        self.encoding_input_index, gamma = self._encoding_assignment()
        self.register_buffer("gamma", torch.tensor(gamma, dtype=dtype))
        self.register_buffer(
            "encoding_input_index_tensor", torch.tensor(self.encoding_input_index)
        )
        self.n_encoding_qubits = len(gamma)

        # Ansatz angles are drawn first so the draw order does not depend on
        # the frequency mode.
        initial_angles = torch.rand(
            (n_layers, n_qubits, 3), generator=generator, dtype=dtype
        ) * (2 * math.pi)
        self.ansatz_angles = nn.Parameter(initial_angles)

        if frequency_mode == "trainable":
            self.frequency_weights = nn.Parameter(
                torch.full((self.n_encoding_qubits,), frequency_init, dtype=dtype)
            )
        elif frequency_mode == "global_scale":
            self.input_scales = nn.Parameter(
                torch.full((n_inputs,), frequency_init, dtype=dtype)
            )
        else:
            self.register_buffer(
                "fixed_weight", torch.tensor(frequency_init, dtype=dtype)
            )

        self.ring = ring_pairs(n_qubits)
        self._program = self._build_program()

    def _encoding_assignment(self) -> tuple[list[int], list[float]]:
        """Map each encoding gate to an input coordinate and its gamma_m."""
        if self.layout == "toy":
            if self.n_inputs != 1:
                raise ValueError("The toy layout encodes a single input coordinate.")
            return [0] * self.n_qubits, generator_coefficients(
                self.feature_map, self.n_qubits
            )
        if self.layout == "ns_parallel":
            if self.n_qubits % self.n_inputs != 0:
                raise ValueError("ns_parallel needs n_qubits divisible by n_inputs.")
            per_input = self.n_qubits // self.n_inputs
            input_index = [d for d in range(self.n_inputs) for _ in range(per_input)]
            gamma = generator_coefficients(self.feature_map, per_input) * self.n_inputs
            return input_index, gamma
        input_index = [d for d in range(self.n_inputs) for _ in range(self.n_qubits)]
        gamma = generator_coefficients(self.feature_map, self.n_qubits) * self.n_inputs
        return input_index, gamma

    def _build_program(self) -> list[tuple[str, int]]:
        """Return the ordered list of ("fm", block) / ("ansatz", layer) steps."""
        layers = iter(range(self.n_layers))
        program: list[tuple[str, int]] = []

        def ansatz(count: int) -> None:
            for _ in range(count):
                program.append(("ansatz", next(layers)))

        if self.layout == "toy":
            program.append(("fm", -1))
            ansatz(self.n_layers)
        elif self.layout == "ns_parallel":
            if self.n_layers < 3:
                raise ValueError("ns_parallel needs at least 3 ansatz layers.")
            ansatz(1)
            program.append(("fm", -1))
            ansatz(self.n_layers - 2)
            program.append(("fm", -1))
            ansatz(1)
        else:
            separators = self.n_inputs - 1
            remaining = self.n_layers - 2 * separators
            if remaining < 2:
                raise ValueError(
                    "ns_serial needs more ansatz layers than serial separators."
                )
            first_half = (remaining + 1) // 2
            for block in range(2):
                for d in range(self.n_inputs):
                    program.append(("fm", d))
                    if d < self.n_inputs - 1:
                        ansatz(1)
                ansatz(first_half if block == 0 else remaining - first_half)
        if next(layers, None) is not None:
            raise RuntimeError("Circuit program did not consume every ansatz layer.")
        return program

    def encoding_frequencies(self) -> torch.Tensor:
        """Return the effective per-gate multipliers ``gamma_m * w_m``."""
        if self.frequency_mode == "trainable":
            return self.gamma * self.frequency_weights
        if self.frequency_mode == "global_scale":
            scales = self.input_scales[self.encoding_input_index_tensor]
            return self.gamma * scales
        return self.gamma * self.fixed_weight

    def n_frequency_parameters(self) -> int:
        """Number of trainable feature-map parameters (|theta_F| or scales)."""
        if self.frequency_mode == "trainable":
            return self.frequency_weights.numel()
        if self.frequency_mode == "global_scale":
            return self.input_scales.numel()
        return 0

    def n_ansatz_parameters(self) -> int:
        """Number of trainable ansatz angles |theta_A|."""
        return self.ansatz_angles.numel()

    def _apply_feature_map(
        self,
        state: torch.Tensor,
        inputs: torch.Tensor,
        multipliers: torch.Tensor,
        block: int,
    ) -> torch.Tensor:
        for gate_index, input_index in enumerate(self.encoding_input_index):
            if block >= 0 and input_index != block:
                continue
            qubit = gate_index % self.n_qubits
            angle = multipliers[gate_index] * inputs[:, input_index]
            state = apply_single_qubit(state, rotation_matrix("y", angle), qubit)
        return state

    def _apply_ansatz_layer(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        angles = self.ansatz_angles[layer]
        for qubit in range(self.n_qubits):
            state = apply_single_qubit(
                state, rotation_matrix("x", angles[qubit, 0]), qubit
            )
            state = apply_single_qubit(
                state, rotation_matrix("y", angles[qubit, 1]), qubit
            )
            state = apply_single_qubit(
                state, rotation_matrix("x", angles[qubit, 2]), qubit
            )
        for first, second in self.ring:
            if self.entangler == "cnot":
                state = apply_cnot(state, first, second)
            else:
                state = apply_cz(state, first, second, self.n_qubits)
        return state

    def _check_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.dim() != 2 or inputs.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected inputs of shape (batch, {self.n_inputs}), got {tuple(inputs.shape)}."
            )

    def forward_gatewise(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reference evaluation applying every gate to the batch one by one.

        Mathematically identical to :meth:`forward`; kept to test the fused path.
        """
        self._check_inputs(inputs)
        multipliers = self.encoding_frequencies()
        state = zero_state(
            inputs.shape[0], self.n_qubits, self.complex_dtype, inputs.device
        )
        for step, index in self._program:
            if step == "fm":
                state = self._apply_feature_map(state, inputs, multipliers, index)
            else:
                state = self._apply_ansatz_layer(state, index)
        return total_magnetization(state, self.n_qubits)

    def _entangler_matrix(self) -> torch.Tensor:
        def apply_ring(state: torch.Tensor) -> torch.Tensor:
            for first, second in self.ring:
                if self.entangler == "cnot":
                    state = apply_cnot(state, first, second)
                else:
                    state = apply_cz(state, first, second, self.n_qubits)
            return state

        return dense_operator(
            apply_ring, self.n_qubits, self.complex_dtype, self.ansatz_angles.device
        )

    def _segment_unitary(
        self, layers: list[int], entangler: torch.Tensor
    ) -> torch.Tensor:
        """Return the dense unitary of consecutive ansatz layers (first layer acts first)."""
        unitary = None
        for layer in layers:
            angles = self.ansatz_angles[layer]
            # Per qubit R_x(c) R_y(b) R_x(a): the first R_x acts first (paper Fig. 6).
            single_qubit = (
                rotation_matrix("x", angles[:, 2])
                @ rotation_matrix("y", angles[:, 1])
                @ rotation_matrix("x", angles[:, 0])
            )
            # Qubit 0 is the most significant index, matching the flattened state.
            product = single_qubit[0]
            for qubit in range(1, self.n_qubits):
                product = torch.kron(product, single_qubit[qubit])
            layer_unitary = entangler @ product
            unitary = layer_unitary if unitary is None else layer_unitary @ unitary
        return unitary

    def _fused_program(self) -> list[tuple[str, object]]:
        """Group consecutive ansatz layers into segments."""
        fused: list[tuple[str, object]] = []
        for step, index in self._program:
            if step == "ansatz" and fused and fused[-1][0] == "segment":
                fused[-1][1].append(index)
            elif step == "ansatz":
                fused.append(("segment", [index]))
            else:
                fused.append(("fm", index))
        return fused

    def _block_phases(
        self,
        inputs: torch.Tensor,
        multipliers: torch.Tensor,
        block: int,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``exp(-i phi)`` of one feature-map block in the Y eigenbasis.

        ``phi_k = sum_m s_{k,q(m)} gamma_m w_m x_{d(m)} / 2`` over the gates m of
        the block, where q(m) is the gate's qubit and s the Y eigenvalue sign.
        """
        gate_signs = []
        gate_angles = []
        for gate_index, input_index in enumerate(self.encoding_input_index):
            if block >= 0 and input_index != block:
                continue
            gate_signs.append(signs[:, gate_index % self.n_qubits])
            gate_angles.append(multipliers[gate_index] * inputs[:, input_index])
        sign_matrix = torch.stack(gate_signs, dim=1).to(inputs.dtype)
        angles = torch.stack(gate_angles, dim=1)
        phases = angles @ sign_matrix.transpose(0, 1) / 2
        return torch.polar(torch.ones_like(phases), -phases)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate ``<sum_m Z_m>`` for a batch of inputs.

        Every feature-map block is a product of R_y gates, hence diagonal in the
        fixed product eigenbasis W of Pauli Y. The circuit is therefore evaluated
        in that basis: input-independent ansatz layers are fused into dense
        ``W^dagger U W`` matrices, a leading segment acts on ``|0...0>`` once, a
        trailing one is folded into the observable, and each encoding block is a
        per-sample elementwise phase. This keeps the nested autograd graph of the
        Navier-Stokes loss small. The result equals :meth:`forward_gatewise` to
        numerical precision (tested).

        Parameters
        ----------
        inputs : torch.Tensor
            Real tensor of shape ``(batch, n_inputs)``.

        Returns
        -------
        torch.Tensor
            Model outputs of shape ``(batch,)``.
        """
        self._check_inputs(inputs)
        batch_size = inputs.shape[0]
        dim = 2**self.n_qubits
        device = inputs.device
        multipliers = self.encoding_frequencies()
        entangler = self._entangler_matrix()
        eigenbasis, signs = y_eigenbasis(self.n_qubits, self.complex_dtype, device)
        eigenbasis_dagger = eigenbasis.conj().transpose(0, 1)
        program = self._fused_program()
        observable = torch.diag(
            magnetization_diagonal(self.n_qubits, self.real_dtype, device).to(
                self.complex_dtype
            )
        )

        if program[-1][0] == "segment":
            trailing = self._segment_unitary(program[-1][1], entangler)
            observable = trailing.conj().transpose(0, 1) @ observable @ trailing
            program = program[:-1]
        observable_eigen = eigenbasis_dagger @ observable @ eigenbasis

        initial = torch.zeros(dim, dtype=self.complex_dtype, device=device)
        initial[0] = 1.0
        if program[0][0] == "segment":
            initial = self._segment_unitary(program[0][1], entangler)[:, 0]
            program = program[1:]
        state = (eigenbasis_dagger @ initial).expand(batch_size, dim)

        for step, payload in program:
            if step == "fm":
                state = state * self._block_phases(inputs, multipliers, payload, signs)
            else:
                segment = (
                    eigenbasis_dagger
                    @ self._segment_unitary(payload, entangler)
                    @ eigenbasis
                )
                state = state @ segment.transpose(0, 1)
        return ((state.conj() @ observable_eigen) * state).sum(dim=1).real
