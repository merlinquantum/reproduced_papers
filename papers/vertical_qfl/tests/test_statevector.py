"""The PyTorch simulator against an independent dense 256x256 numpy reference."""

from __future__ import annotations

import numpy as np
import torch
from common import PROJECT_DIR  # noqa: F401  (puts the project on sys.path)
from lib.model import ReuploadingCircuit

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)


def rot(pauli, angle):
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * pauli


def on_wire(gate, wire, n=8):
    ops = [I2] * n
    ops[wire] = gate
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def cz_ring(n=8):
    diag = np.ones(2**n)
    for index in range(2**n):
        bits = [(index >> (n - 1 - w)) & 1 for w in range(n)]
        diag[index] = (-1) ** sum(bits[w] * bits[(w + 1) % n] for w in range(n))
    return np.diag(diag).astype(complex)


def reference_client(angles, x, n_blocks, depth):
    """Dense-matrix version of the client circuit described in LOG.md."""
    ring = cz_ring()
    state = np.zeros(256, dtype=complex)
    state[0] = 1
    for block in range(n_blocks):
        for layer in range(depth):
            index = block * depth + layer
            for wire in range(8):
                state = on_wire(rot(X, angles[index, 0, wire]), wire) @ state
            for wire in range(8):
                state = on_wire(rot(Y, angles[index, 1, wire]), wire) @ state
            last = block == n_blocks - 1 and layer == depth - 1
            if not last:
                state = ring @ state
        if block < n_blocks - 1:
            pauli = X if block % 2 == 0 else Y
            for wire in range(8):
                state = on_wire(rot(pauli, x[block * 8 + wire]), wire) @ state
            state = ring @ state
    return np.array(
        [np.real(state.conj() @ on_wire(Z, wire) @ state) for wire in range(8)]
    )


def test_client_matches_dense_reference():
    torch.manual_seed(0)
    n_blocks, depth = 3, 2
    circuit = ReuploadingCircuit(
        "client", n_blocks, depth, dtype=torch.float64
    ).double()
    x = torch.rand(1, 32, dtype=torch.float64) * np.pi / 4
    got = circuit(x).detach().numpy()[0]
    want = reference_client(
        circuit.angles.detach().numpy(), x.numpy()[0], n_blocks, depth
    )
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_server_readout_matches_dense_reference_for_x_and_z():
    torch.manual_seed(1)
    circuit = ReuploadingCircuit("server_narrow", 2, 1, dtype=torch.float64).double()
    x = torch.rand(1, 8, dtype=torch.float64)
    scores = circuit(x).detach().numpy()[0]
    assert scores.shape == (10,)
    ring = cz_ring()
    angles = circuit.angles.detach().numpy()
    state = np.zeros(256, dtype=complex)
    state[0] = 1
    for wire in range(8):
        state = on_wire(rot(X, angles[0, 0, wire]), wire) @ state
    for wire in range(8):
        state = on_wire(rot(Y, angles[0, 1, wire]), wire) @ state
    state = ring @ state
    for wire in range(8):
        state = on_wire(rot(X, x.numpy()[0][wire]), wire) @ state
    state = ring @ state
    for wire in range(8):
        state = on_wire(rot(X, angles[1, 0, wire]), wire) @ state
    for wire in range(8):
        state = on_wire(rot(Y, angles[1, 1, wire]), wire) @ state
    want = [
        np.real(state.conj() @ on_wire(P, w) @ state) for P in (Z, X) for w in range(5)
    ]
    np.testing.assert_allclose(scores, want, atol=1e-10)


def test_parameter_counts_match_paper():
    from lib.model import VerticalQuantumFederatedModel

    model = VerticalQuantumFederatedModel(list(range(8)), "server_wide", True)
    counts = model.trainable_parameter_counts()
    assert counts["server"] == 8000 and counts["client_0"] == 8000
    assert counts["total"] == 9 * 8000
