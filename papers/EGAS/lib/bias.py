"""Continuous parameter refinement (Section III.B).

After EGAS fixes a discrete circuit structure, a learnable MLP bias b_omega(x) is added to all
parameterized gate angles: phi_tilde(x) = r*x_i + b_omega(x).  The bias is trained with a
pairwise binary-cross-entropy fidelity loss (Eq. 12); the structure stays fixed so any change
isolates the gain from continuous refinement.  ``delta_E = E_before - E_after`` (Fig. 3/4)
reports the surrogate-energy reduction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .circuits import embed_states
from .egas import pairwise_energy
from .statevec import fidelity_matrix


class BiasMLP(nn.Module):
    """Small MLP with a zero-initialised output head; output scaled by a fixed gain (=10)."""

    def __init__(
        self, n_in: int, output_size: int = 1, hidden: int = 32, gain: float = 10.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gain = gain

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.gain * self.net(X.to(torch.float64))


def _bce_pair_loss(states, labels, eps=1e-3):
    F = fidelity_matrix(states)
    Fbar = F.clamp(eps, 1 - eps)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).double()
    S = states.shape[0]
    off = ~torch.eye(S, dtype=torch.bool, device=states.device)
    bce = -(same * torch.log(Fbar) + (1 - same) * torch.log(1 - Fbar))
    return bce[off].mean()


def refine_bias(
    seq,
    X,
    y,
    n_qubits,
    *,
    epochs=100,
    batch_samples=25,
    lr=5e-4,
    grad_clip=2.0,
    l2_bias=1e-6,
    hidden=32,
    gain=10.0,
    seed=0,
    device="cpu",
    avg_last=10,
):
    """Train a bias MLP for a fixed circuit `seq`. Returns (bias_mlp, E_before, E_after)."""
    torch.manual_seed(seed)

    output_size = 0
    for gate, _, _, _ in seq:
        if gate not in ["H", "I", "CNOT"]:
            output_size += 1

    Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    bias = (
        BiasMLP(n_qubits, output_size=output_size, hidden=hidden, gain=gain)
        .double()
        .to(device)
    )
    opt = torch.optim.RMSprop(bias.parameters(), lr=lr)

    with torch.no_grad():
        E_before = pairwise_energy(embed_states(seq, Xt, n_qubits), yt).item()

    rng = np.random.default_rng(seed)
    n = len(X)
    recent = []
    for ep in range(epochs):
        idx = rng.choice(n, size=min(batch_samples, n), replace=False)
        Xb, yb = Xt[idx], yt[idx]
        states = embed_states(seq, Xb, n_qubits, bias=bias)
        loss = _bce_pair_loss(states, yb)
        reg = l2_bias * (bias(Xb) ** 2).mean()
        opt.zero_grad()
        (loss + reg).backward()
        torch.nn.utils.clip_grad_norm_(bias.parameters(), grad_clip)
        opt.step()
        if ep >= epochs - avg_last:
            with torch.no_grad():
                recent.append(
                    pairwise_energy(
                        embed_states(seq, Xt, n_qubits, bias=bias), yt
                    ).item()
                )
    E_after = float(np.mean(recent)) if recent else E_before
    return bias, E_before, E_after
