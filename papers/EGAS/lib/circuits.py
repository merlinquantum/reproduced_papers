"""Token pool, sequence-to-circuit translation, and the ZZ feature-map baseline.

A candidate embedding is a length-D sequence of depth-one subcircuit *tokens* drawn from a
fixed pool C (Appendix A).  Each token is a tuple

    (gate_type, qubit_index, data_index, coeff_r)

* gate_type in {RX, RY, RZ, H, I, CNOT, MultiRZ}
* qubit_index in {0..n-1}
* data_index in {0..n-1}   (feature x_i injected into parameterized gates)
* coeff_r    in {0.1,0.3,0.5,0.7,1.0}

For a parameterized gate the data-dependent angle is ``phi(x) = r * x[data_index]``; with the
optional bias-refinement MLP it becomes ``phi(x) = r * x[data_index] + b_omega(x)``.

ASSUMPTIONS (documented in LOG.md, underspecified in paper):
* two-qubit gates (CNOT, MultiRZ) act on (q, (q+1) mod n) — nearest-neighbour ring.
* non-parameterized gates (H, I) ignore data_index/coeff; we fix them to 0 in the vocabulary
  so each (gate, qubit) maps to a single token.
"""

from __future__ import annotations

import torch

from .statevec import apply_gate, init_state

PARAM_1Q = ("RX", "RY", "RZ")
NONPARAM_1Q = ("H", "I")
COEFFS = (0.1, 0.3, 0.5, 0.7, 1.0)


def build_token_pool(n_qubits: int):
    """Enumerate the full token pool C. Returns list of (gate, q, data_idx, r)."""
    tokens = []
    for q in range(n_qubits):
        for d in range(n_qubits):
            for r in COEFFS:
                for g in PARAM_1Q:
                    tokens.append((g, q, d, r))
    for q in range(n_qubits):
        for g in NONPARAM_1Q:
            tokens.append((g, q, 0, 0.0))
    for q in range(n_qubits):  # CNOT (q, q+1)
        tokens.append(("CNOT", q, 0, 0.0))
    for q in range(n_qubits):  # MultiRZ (q, q+1) parameterized
        for d in range(n_qubits):
            for r in COEFFS:
                tokens.append(("MultiRZ", q, d, r))
    return tokens


def embed_states(sequence, X: torch.Tensor, n_qubits: int, bias=None) -> torch.Tensor:
    """Apply a token sequence to a batch of inputs X (B, n) -> statevectors (B, 2**n).

    `sequence` is a list of token tuples (gate, q, data_idx, r).
    `bias` (optional) is a callable X -> (B,) additive angle offset for parameterized gates.
    """
    batch = X.shape[0]
    device = X.device
    state = init_state(batch, n_qubits, device=device)
    b_off = bias(X) if bias is not None else None  # (B,) shared additive offset
    bias_index = 0
    for gate, q, d, r in sequence:
        if gate in PARAM_1Q:
            angle = r * X[:, d]
            if b_off is not None:
                angle = angle + b_off[:, bias_index]
                bias_index += 1
            state = apply_gate(state, n_qubits, gate, (q,), angle)
        elif gate == "MultiRZ":
            angle = r * X[:, d]
            if b_off is not None:
                angle = angle + b_off[:, bias_index]
                bias_index += 1
            qn = (q + 1) % n_qubits
            state = apply_gate(state, n_qubits, gate, (q, qn), angle)
        elif gate == "CNOT":
            qn = (q + 1) % n_qubits
            state = apply_gate(state, n_qubits, gate, (q, qn))
        else:  # H, I
            state = apply_gate(state, n_qubits, gate, (q,))
    return state


def zz_feature_states(
    X: torch.Tensor, n_qubits: int, n_layers: int = 1
) -> torch.Tensor:
    """ZZ feature map (Eq. A1), repeated n_layers times. Returns statevectors (B, 2**n).

    U_ZZ(x) = exp(i sum_j theta_j Z_j + i sum_j theta_{j,j+1} Z_j Z_{j+1}) H^{x n},
      theta_j(x) = x_j,  theta_{j,j+1}(x) = (pi - x_j)(pi - x_{j+1}).
    exp(i a Z) = RZ(-2a);  exp(i a Z Z) = MultiRZ(-2a).
    """
    import math

    batch = X.shape[0]
    device = X.device
    state = init_state(batch, n_qubits, device=device)
    pi = math.pi
    for _ in range(n_layers):
        for q in range(n_qubits):
            state = apply_gate(state, n_qubits, "H", (q,))
        for q in range(n_qubits):
            state = apply_gate(state, n_qubits, "RZ", (q,), -2.0 * X[:, q])
        for q in range(n_qubits - 1):
            theta = (pi - X[:, q]) * (pi - X[:, q + 1])
            state = apply_gate(state, n_qubits, "MultiRZ", (q, q + 1), -2.0 * theta)
    return state
