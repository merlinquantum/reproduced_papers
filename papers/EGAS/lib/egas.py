"""Energy-based Generative Architecture Search (EGAS) — Section III.A.

A GPT samples candidate token sequences; each is translated to an embedding circuit and
scored by the pairwise-fidelity surrogate energy (Eq. 9):

    E(s) = mean_{(i,j) in B} | delta_{y_i,y_j} - F_{Phi_s}(x_i, x_j) |.

The GPT is updated by the logit-matching loss (Eq. 10) toward a Boltzmann distribution over
the evaluated energies, with EMA energy normalisation and a top/middle/bottom selection of the
replay buffer (Appendix A.1).
"""

from __future__ import annotations

import numpy as np
import torch

from .circuits import embed_states
from .gpt import GPTQE, GPTConfig
from .statevec import fidelity_matrix


def pairwise_energy(states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """E = mean_{i!=j} |delta_{y_i,y_j} - F(x_i,x_j)| for one embedding."""
    # Ensure states and labels have matching first dimension
    if states.shape[0] == 1 and labels.shape[0] > 1:
        states = states.expand(labels.shape[0], -1)

    F = fidelity_matrix(states)  # (S, S)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).double()  # delta
    loss = (same - F).abs()
    S = states.shape[0]
    off = ~torch.eye(S, dtype=torch.bool, device=states.device)
    return loss[off].mean()


def evaluate_sequences(sequences, pool, X, y, n_qubits):
    """Energy for each token-id sequence on the (X, y) batch."""
    energies = []
    for seq_ids in sequences:
        seq = [pool[int(t)] for t in seq_ids]
        states = embed_states(seq, X, n_qubits)
        energies.append(pairwise_energy(states, y).item())
    return np.array(energies)


class EMA:
    """Exponential moving estimate of mean/std for energy normalisation."""

    def __init__(self, beta=0.9):
        self.beta = beta
        self.mean = None
        self.var = None

    def update(self, x: np.ndarray):
        m, v = float(x.mean()), float(x.var()) + 1e-8
        if self.mean is None:
            self.mean, self.var = m, v
        else:
            self.mean = self.beta * self.mean + (1 - self.beta) * m
            self.var = self.beta * self.var + (1 - self.beta) * v
        return self

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


def run_egas(
    pool,
    X,
    y,
    n_qubits,
    seq_len,
    *,
    n_iters=4000,
    n_candidates=24,
    select_k=6,
    lr=5e-5,
    weight_decay=1e-2,
    temp_max=100.0,
    temp_min=0.04,
    n_layers=8,
    n_heads=12,
    n_embd=480,
    dropout=0.2,
    grad_clip=1.0,
    seed=0,
    device="cpu",
    log_every=50,
    logger=None,
):
    """Run EGAS; return (gpt, history, buffer) where buffer is list of (seq_ids, energy).

    ``seq_ids`` in the buffer are *pool indices* (0-based, directly index ``pool``); the GPT
    reserves token id 0 as the start token so its vocabulary size is ``len(pool) + 1``
    (matching the reference ``vocab_size = |C| + 1``). Defaults for the GPT
    (``n_layers=8, n_heads=12, dropout=0.2, bias=False``) and the geometric temperature
    schedule follow the author's ``GQE.py``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    vocab = len(pool) + 1  # +1 for the reserved start token (id 0)
    gpt_config = GPTConfig(
        vocab_size=vocab,
        block_size=seq_len + 1,
        n_layer=n_layers,
        n_head=n_heads,
        n_embd=n_embd,
        dropout=dropout,
        bias=False,
    )
    gpt = GPTQE(gpt_config).to(device)
    opt = gpt.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=(0.9, 0.999),
        device_type="cpu",
    )
    ema = EMA()
    X = torch.as_tensor(X, dtype=torch.float64, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)

    buffer = []  # list of (tuple pool_indices, energy)
    seen = {}
    history = {"iter": [], "min_energy": [], "mean_energy": [], "loss": []}

    for it in range(n_iters):
        # geometric temperature schedule (author GQE.py): T_max * (T_min/T_max)^(it/n_iters)
        T = temp_max * (temp_min / temp_max) ** (it / max(1, n_iters))
        gen = gpt.generate(n_candidates, seq_len, T, device=device)[0]  # (M, D+1)
        # strip the leading start token and shift to 0-based pool indices
        seqs = (gen[:, 1:] - 1).cpu().numpy()
        energies = evaluate_sequences(seqs, pool, X, y, n_qubits)
        ema.update(energies)
        for s_ids, e in zip(seqs, energies):
            key = tuple(int(t) for t in s_ids)
            if key not in seen:
                seen[key] = e
                buffer.append((key, float(e)))

        # top/middle/bottom selection from the replay buffer (Appendix A.1).
        # Matches the author reference `utils.select_token_and_en`: top-k lowest energy,
        # bottom-k highest, and a middle group of ~k/2 sampled *evenly-spaced* across the
        # remaining middle pool (ratio 0.4 : 0.2 : 0.4 ≈ top : middle : bottom).
        buf_sorted = sorted(buffer, key=lambda z: z[1])
        nb = len(buf_sorted)
        k = min(select_k, nb)
        low = buf_sorted[:k]
        high = buf_sorted[-k:]
        mid_n = max(1, k // 2)
        mid_pool = buf_sorted[k : nb - k] if nb > 2 * k else []
        if mid_pool:
            pts = np.linspace(0, len(mid_pool) - 1, num=min(mid_n, len(mid_pool)))
            mid = [mid_pool[int(round(p))] for p in pts]
        else:
            mid = []
        sel = low + mid + high
        sel_ids = torch.tensor(
            [list(s) for s, _ in sel], dtype=torch.long, device=device
        )
        sel_e = np.array([e for _, e in sel])
        sel_e_n = ema.normalize(sel_e)
        perm = torch.randperm(len(sel))
        sel_ids = sel_ids[perm]
        target = torch.tensor(sel_e_n[perm.numpy()], dtype=torch.float64, device=device)

        # Reconstruct GPT token ids: pool index i -> token id (i + 1), prepended with the
        # reserved start token (id 0) so next-token prediction sees the full sequence.
        start = torch.zeros(sel_ids.shape[0], 1, dtype=torch.long, device=device)
        tokens = torch.cat([start, sel_ids + 1], dim=1)

        # loss
        loss = gpt.calculate_loss(tokens, target).double()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), grad_clip)
        opt.step()

        history["iter"].append(it)
        history["min_energy"].append(float(min(z[1] for z in buffer)))
        history["mean_energy"].append(float(energies.mean()))
        history["loss"].append(float(loss.item()))
        if logger and (it % log_every == 0 or it == n_iters - 1):
            logger.info(
                "EGAS it=%d T=%.3f mean_E=%.4f min_E=%.4f loss=%.4e buf=%d",
                it,
                T,
                energies.mean(),
                history["min_energy"][-1],
                loss.item(),
                len(buffer),
            )
    return gpt, history, buffer


def unique_sorted_candidates(buffer, top=10, bottom=10):
    """Return (G_sequences, B_sequences): the `top` lowest- and `bottom` highest-energy unique
    sequences as token-id tuples (Section IV.A: G and B groups)."""
    buf_sorted = sorted(buffer, key=lambda z: z[1])
    G = [s for s, _ in buf_sorted[:top]]
    Bgrp = [s for s, _ in buf_sorted[-bottom:]]
    return G, Bgrp
