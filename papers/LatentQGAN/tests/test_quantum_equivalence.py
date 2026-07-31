"""Verify that the Qiskit and PyTorch implementations of the LatentQGAN
sub-generator produce identical output for a given parameter setting."""

from __future__ import annotations

import numpy as np
import torch
from lib.quantum_generator import QuantumGeneratorTorch, qiskit_forward


def test_qiskit_torch_match():
    rng = np.random.default_rng(0)
    N, NA, L = 4, 1, 7
    for _ in range(3):
        alpha = rng.uniform(0, np.pi, N)
        theta = rng.uniform(0, 2 * np.pi, N * L)
        p_qk = qiskit_forward(alpha, theta, N=N, L=L, NA=NA)
        gen = QuantumGeneratorTorch(N=N, NA=NA, L=L)
        with torch.no_grad():
            gen.theta.copy_(torch.tensor(theta.reshape(L, N), dtype=torch.float32))
        p_pt = (
            gen(torch.tensor(alpha, dtype=torch.float32).unsqueeze(0))
            .detach()
            .numpy()[0]
        )
        assert np.allclose(p_qk, p_pt, atol=1e-5), (
            f"max diff {np.max(np.abs(p_qk - p_pt))}"
        )


def test_sub_param_count():
    gen = QuantumGeneratorTorch(N=4, NA=1, L=7)
    assert sum(p.numel() for p in gen.parameters()) == 4 * 7  # 28 params per sub-gen


def test_full_param_count_140():
    from lib.qgan import LatentQGenerator

    g = LatentQGenerator(T=5, N=4, NA=1, L=7)
    assert sum(p.numel() for p in g.parameters()) == 140


def test_disc_param_count_3681():
    from lib.qgan import LatentDiscriminator

    d = LatentDiscriminator(latent_dim=40, h1=64, h2=16)
    assert sum(p.numel() for p in d.parameters()) == 3681


def test_row_sums_one():
    from lib.qgan import LatentQGenerator

    g = LatentQGenerator(T=5, N=4, NA=1, L=7)
    noise = g.sample_noise(3)
    out = g(noise)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
