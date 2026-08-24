"""Photonic Energy-based Generative Architecture Search (EGAS).

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

from .gpt import GPTQE, GPTConfig
from .photonic_bias import refine_bias
from .photonic_circuits import create_quantum_module
from .photonic_kernel_svm import (
    qksvm_accuracy as photonic_qksvm_accuracy,
)
from .statevec import fidelity_matrix


def pairwise_energy(states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """E = mean_{i!=j} |delta_{y_i,y_j} - F(x_i,x_j)| for one embedding.

    Supports binary and multiclass labels (any integer or categorical format).
    """
    F = fidelity_matrix(states)  # (S, S)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # delta (multiclass OK)
    loss = (same - F).abs()
    S = states.shape[0]
    off = ~torch.eye(S, dtype=torch.bool, device=states.device)
    return loss[off].mean()


def _create_encoder(seq, num_features, n_modes, num_photons=2, computation_space=None):
    kwargs = {
        "num_features": num_features,
        "n_modes": n_modes,
        "num_photons": num_photons,
    }
    if computation_space is not None:
        kwargs["computation_space"] = computation_space
    return create_quantum_module(seq, **kwargs)


def evaluate_sequences(
    sequences,
    pool,
    X,
    y,
    n_modes,
    *,
    num_photons=2,
    computation_space=None,
):
    """Energy for each token-id sequence on the (X, y) batch."""
    energies = []
    for seq_ids in sequences:
        seq = [pool[int(t)] for t in seq_ids]
        encoder = _create_encoder(
            seq,
            X.shape[-1],
            n_modes,
            num_photons=num_photons,
            computation_space=computation_space,
        ).to(X.device)
        encoder.eval()
        with torch.no_grad():
            states = encoder(X)
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
    n_modes,
    seq_len,
    *,
    num_photons=2,
    computation_space=None,
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
    """Run photonic EGAS; return (gpt, history, buffer).

    Mirrors the gate-based ``lib.egas.run_egas`` exactly (same GPTQE generator, same
    geometric temperature schedule, same ``vocab=|pool|+1`` start-token convention and
    logit-matching update) — only the energy evaluation differs, using the photonic
    ``QuantumModule`` from ``lib.photonic_circuits``. ``buffer`` entries are 0-based pool
    indices.
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
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)

    buffer = []  # list of (tuple pool_indices, energy)
    seen = {}
    history = {"iter": [], "min_energy": [], "mean_energy": [], "loss": []}

    for it in range(n_iters):
        # geometric temperature schedule (matches gate-based EGAS / author GQE.py)
        T = temp_max * (temp_min / temp_max) ** (it / max(1, n_iters))
        gen = gpt.generate(n_candidates, seq_len, T, device=device)[0]  # (M, D+1)
        seqs = (gen[:, 1:] - 1).cpu().numpy()  # strip start token, 0-based pool indices
        energies = evaluate_sequences(
            seqs,
            pool,
            X,
            y,
            n_modes,
            num_photons=num_photons,
            computation_space=computation_space,
        )
        ema.update(energies)
        for s_ids, e in zip(seqs, energies):
            key = tuple(int(t) for t in s_ids)
            if key not in seen:
                seen[key] = e
                buffer.append((key, float(e)))

        # top/middle/bottom selection from the replay buffer (Appendix A.1).
        # Mirrors the gate path and the author reference `utils.select_token_and_en`:
        # top-k lowest / bottom-k highest energy + ~k/2 middle sampled evenly-spaced.
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
        perm = torch.randperm(len(sel), device=device)
        sel_ids = sel_ids[perm]
        target = torch.tensor(
            sel_e_n[perm.cpu().numpy()], dtype=torch.float64, device=device
        )

        # Reconstruct GPT token ids (pool index i -> token id i + 1) with a leading
        # start token (id 0), matching the gate-based EGAS update.
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


def refine_candidates(
    candidate_ids,
    pool,
    X,
    y,
    n_modes,
    *,
    num_photons=2,
    computation_space=None,
    device="cpu",
    **refine_kwargs,
):
    """Refine selected photonic candidates and return their trained encoder models."""
    refined = []
    for sid in candidate_ids:
        seq = [pool[int(i)] for i in sid]
        kwargs = {
            "num_features": X.shape[-1],
            "num_photons": num_photons,
            "device": device,
            **refine_kwargs,
        }
        if computation_space is not None:
            kwargs["computation_space"] = computation_space
        encoder, e_before, e_after = refine_bias(seq, X, y, n_modes, **kwargs)
        refined.append(
            {
                "seq": seq,
                "encoder": encoder,
                "E_before": e_before,
                "E_after": e_after,
            }
        )
    return refined


def evaluate_candidate_accuracy(
    candidate, X_train, y_train, X_test, y_test, device="cpu"
):
    """Evaluate one raw or refined photonic candidate with the photonic QKSVM helper."""
    photonic_model = candidate["encoder"] if isinstance(candidate, dict) else candidate
    return photonic_qksvm_accuracy(
        photonic_model,
        X_train,
        y_train,
        X_test,
        y_test,
        device=device,
    )
