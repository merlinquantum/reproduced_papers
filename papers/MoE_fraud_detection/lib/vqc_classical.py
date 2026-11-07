"""Classical stand-in for the VQC block (Phase 5 fair-baseline ablation, C5
in LOG.md's claim inventory).

Purpose: isolate whether the quantum block (gate or photonic) contributes
anything beyond what a similarly-sized classical block would inside the same
GQC+MoE pipeline. This is NOT part of the paper - it is an additional fair
baseline required by PAPER_REPRODUCTION_INSTRUCTIONS.md Section 4 (Baseline
Philosophy) whenever a quantum block's contribution is not otherwise
isolated.

Architecture: ``n_qubits -> hidden -> 1`` (ReLU hidden, tanh output rescaled
to roughly the gate VQC's [-1, 1] PauliZ-expectation range so the shared
``Head`` sees a comparable input distribution). ``hidden`` is chosen so the
total parameter count is close to the gate-model VQC's
``n_layers * n_qubits * 2`` trainable rotation angles (see
``matched_hidden_dim``), documented exactly (not claimed to be an exact
match - the two architectures are not directly parameter-comparable, see
LOG.md C5 caveats).
"""

from __future__ import annotations

import torch
from torch import nn


def matched_hidden_dim(n_qubits: int, target_params: int) -> int:
    """Smallest ``hidden`` such that a ``Linear(n_qubits, hidden) ->
    Linear(hidden, 1)`` block has at least ``target_params`` parameters.

    Param count = ``hidden * (n_qubits + 1) + hidden + 1``
                = ``hidden * (n_qubits + 2) + 1``.
    """
    hidden = 1
    while hidden * (n_qubits + 2) + 1 < target_params:
        hidden += 1
    return hidden


class ClassicalVQCReplacement(nn.Module):
    """Interface-compatible with :class:`lib.vqc_gate.VQCLayer` /
    :class:`lib.vqc_photonic.PhotonicVQCLayer`: ``(batch, n_qubits) ->
    (batch,)`` in ``[-1, 1]``.
    """

    def __init__(self, n_qubits: int = 6, n_layers: int = 6) -> None:
        super().__init__()
        target_params = n_layers * n_qubits * 2  # gate VQC's trainable-angle count
        hidden = matched_hidden_dim(n_qubits, target_params)
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        self.hidden_dim = hidden
        self.target_params = target_params
        self.actual_params = sum(p.numel() for p in self.net.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


__all__ = ["ClassicalVQCReplacement", "matched_hidden_dim"]
