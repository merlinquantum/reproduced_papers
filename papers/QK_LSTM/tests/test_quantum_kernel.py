from __future__ import annotations

import math

import pytest
import torch
from lib.quantum_kernel import (
    feature_map_states,
    fidelity_kernel,
    quantum_kernel,
    rbf_kernel,
)


def _dense_reference_state(z: torch.Tensor) -> torch.Tensor:
    """Text-layout state built from explicit 16x16 matrices (independent of the fast path)."""
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    identity = torch.eye(2, dtype=torch.complex128)

    def ry(theta):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.complex128)

    def rz(phi):
        return torch.diag(
            torch.tensor(
                [
                    complex(math.cos(phi / 2), -math.sin(phi / 2)),
                    complex(math.cos(phi / 2), math.sin(phi / 2)),
                ],
                dtype=torch.complex128,
            )
        )

    def on_qubit(gate, qubit, n=4):
        out = torch.ones(1, 1, dtype=torch.complex128)
        for q in range(n):
            out = torch.kron(out, gate if q == qubit else identity)
        return out

    def cnot(control, target, n=4):
        dim = 2**n
        matrix = torch.zeros(dim, dim, dtype=torch.complex128)
        for basis in range(dim):
            bits = [(basis >> (n - 1 - q)) & 1 for q in range(n)]
            if bits[control]:
                bits[target] ^= 1
            out = sum(bit << (n - 1 - q) for q, bit in enumerate(bits))
            matrix[out, basis] = 1
        return matrix

    state = torch.zeros(16, dtype=torch.complex128)
    state[0] = 1
    for q in range(4):
        state = on_qubit(hadamard, q) @ state
    for q in range(4):
        state = on_qubit(ry(float(z[q])) @ rz(float(z[q])), q) @ state
    for k in range(3):
        state = cnot(k, k + 1) @ state
    return state


def test_states_match_dense_matrix_simulation():
    z = torch.tensor([[0.3, -1.2, 2.0, 0.7]], dtype=torch.float64)
    fast = feature_map_states(z, layout="text", angle_map="identity")[0]
    assert torch.allclose(fast, _dense_reference_state(z[0]), atol=1e-12)


@pytest.mark.parametrize("layout", ["text", "figure"])
@pytest.mark.parametrize("angle_map", ["identity", "arctan", "tanh_pi"])
def test_kernel_is_a_fidelity(layout, angle_map):
    torch.manual_seed(0)
    z = torch.randn(5, 4, dtype=torch.float64)
    states = feature_map_states(z, layout=layout, angle_map=angle_map)
    assert torch.allclose(
        states.abs().pow(2).sum(dim=1), torch.ones(5, dtype=torch.float64)
    )
    gram = fidelity_kernel(states, states)
    assert torch.allclose(torch.diagonal(gram), torch.ones(5, dtype=torch.float64))
    assert torch.allclose(gram, gram.T, atol=1e-12)
    assert gram.min() >= 0 and gram.max() <= 1 + 1e-12


def test_text_layout_kernel_factorises_into_single_qubit_fidelities():
    # The CNOT chain is data independent and cancels in <phi_j|phi_t> (LOG.md, D2).
    torch.manual_seed(1)
    z, z_ref = (
        torch.randn(3, 4, dtype=torch.float64),
        torch.randn(2, 4, dtype=torch.float64),
    )
    full = quantum_kernel(z, z_ref, layout="text")
    product = torch.ones_like(full)
    for qubit in range(4):
        product = product * quantum_kernel(
            z[:, qubit : qubit + 1], z_ref[:, qubit : qubit + 1], layout="text"
        )
    assert torch.allclose(full, product, atol=1e-12)


def test_figure_layout_kernel_does_not_factorise():
    torch.manual_seed(2)
    z, z_ref = (
        torch.randn(3, 4, dtype=torch.float64),
        torch.randn(2, 4, dtype=torch.float64),
    )
    full = quantum_kernel(z, z_ref, layout="figure")
    product = torch.ones_like(full)
    for qubit in range(4):
        product = product * quantum_kernel(
            z[:, qubit : qubit + 1], z_ref[:, qubit : qubit + 1], layout="figure"
        )
    assert not torch.allclose(full, product, atol=1e-6)


@pytest.mark.parametrize("layout", ["text", "figure"])
def test_autograd_matches_parameter_shift(layout):
    # Shift the angle of one rotation gate by +-pi/2: dk/dtheta = [k(+) - k(-)] / 2.
    # With angle_map="identity" every z_k feeds both R_z and R_y (and, in the
    # figure layout, a second R_z), so the shift rule is applied per gate and summed.
    z = torch.tensor([[0.4, -0.9, 1.3, 0.2]], dtype=torch.float64, requires_grad=True)
    z_ref = torch.tensor([[1.1, 0.3, -0.5, 0.8]], dtype=torch.float64)
    quantum_kernel(z, z_ref, layout=layout)[0, 0].backward()
    autograd = z.grad[0].clone()

    states_ref = feature_map_states(z_ref, layout=layout)

    def kernel_with_gate_shift(qubit, gate, shift):
        states_t = _states_with_shift(z.detach(), qubit, gate, shift, layout)
        return fidelity_kernel(states_t, states_ref)[0, 0]

    gates = ["rz", "ry", "rz2"] if layout == "figure" else ["rz", "ry"]
    for qubit in range(4):
        shift_gradient = sum(
            (
                kernel_with_gate_shift(qubit, gate, math.pi / 2)
                - kernel_with_gate_shift(qubit, gate, -math.pi / 2)
            )
            / 2
            for gate in gates
        )
        assert math.isclose(
            float(shift_gradient), float(autograd[qubit]), rel_tol=1e-9, abs_tol=1e-12
        )


def _states_with_shift(z, qubit, gate, shift, layout):
    """Re-implement the feature map with one gate's angle shifted."""
    from lib.quantum_kernel import (
        _apply_single_qubit,
        _cnot_chain_permutation,
        _ry,
        _rz,
    )

    n = z.shape[1]
    state = torch.full((1, 2**n), 1.0 / math.sqrt(2**n), dtype=torch.complex128)
    for q in range(n):
        phi = z[:, q] + (shift if (q == qubit and gate == "rz") else 0.0)
        theta = z[:, q] + (shift if (q == qubit and gate == "ry") else 0.0)
        state = _apply_single_qubit(state, _rz(phi, torch.complex128), q, n)
        state = _apply_single_qubit(state, _ry(theta, torch.complex128), q, n)
    state = state[:, _cnot_chain_permutation(n)]
    if layout == "figure":
        for q in range(n):
            phi = z[:, q] + (shift if (q == qubit and gate == "rz2") else 0.0)
            state = _apply_single_qubit(state, _rz(phi, torch.complex128), q, n)
    return state


def test_rbf_kernel_values():
    z = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    gram = rbf_kernel(z, z, gamma=1.0)
    assert torch.allclose(
        gram, torch.tensor([[1.0, math.exp(-1.0)], [math.exp(-1.0), 1.0]])
    )
