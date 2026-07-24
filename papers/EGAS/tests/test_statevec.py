"""Fast, network-free tests for the quantum-statevector engine and EGAS energy."""

from __future__ import annotations

import math

import numpy as np
import torch
from lib.circuits import build_token_pool, embed_states, zz_feature_states
from lib.egas import pairwise_energy
from lib.statevec import fidelity_matrix


def test_states_are_normalised():
    n = 4
    X = torch.tensor(np.random.default_rng(0).uniform(0, 2 * math.pi, size=(6, n)))
    st = zz_feature_states(X, n)
    norms = (st.abs() ** 2).sum(1).real.numpy()
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_fidelity_is_one_on_identical_inputs():
    n = 4
    seq = [
        ("RY", 0, 1, 0.5),
        ("CNOT", 1, 0, 0.0),
        ("MultiRZ", 2, 3, 0.7),
        ("H", 3, 0, 0.0),
    ]
    x = torch.tensor(np.random.default_rng(1).uniform(0, 2 * math.pi, size=(1, n)))
    X = x.repeat(3, 1)
    F = fidelity_matrix(embed_states(seq, X, n))
    assert np.allclose(F.numpy(), 1.0, atol=1e-9)


def test_pairwise_energy_in_unit_range():
    n = 4
    X = torch.tensor(np.random.default_rng(2).uniform(0, 2 * math.pi, size=(8, n)))
    y = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1])
    seq = [("RY", q, q, 1.0) for q in range(n)]
    e = pairwise_energy(embed_states(seq, X, n), y).item()
    assert 0.0 <= e <= 1.0


def test_token_pool_size():
    # n=8: 3*8*8*5 (param 1q) + 2*8 (H,I) + 8 (CNOT) + 8*8*5 (MultiRZ) = 1304
    assert len(build_token_pool(8, 8)) == 1304
