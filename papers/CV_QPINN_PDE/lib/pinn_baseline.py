"""Classical fully-connected feed-forward PINN baseline.

Used to provide a parameter-matched comparison against the QPINN, as the
paper does in Section IV.B (Table II / Table IV). The forward network has
the same two-output interpretation as the QPINN: ``u(x)`` and a learned
``ux(x)`` linked through a consistency term during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FCNN(nn.Module):
    """Fully-connected ``tanh``-activated PINN that returns (u, ux)."""

    def __init__(
        self, in_features: int, hidden_layers: list[int], out_features: int = 2
    ) -> None:
        super().__init__()
        dims = [in_features] + hidden_layers + [out_features]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.Tanh())
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        out = self.net(x)
        u = out[..., 0]
        ux = out[..., 1]
        trace = torch.ones_like(u)
        return u, ux, trace

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def hidden_layers_for_param_count(
    target_params: int, in_features: int, out_features: int = 2, hidden_layers: int = 1
) -> list[int]:
    """Heuristic width selection for a fully-connected PINN to approximately
    match a target trainable-parameter count.

    For a single hidden layer of width ``H``: params = (in+1)*H + (H+1)*out.
    Solves for H given the target and ``hidden_layers=1``.
    """
    if hidden_layers == 1:
        h = max(
            1, round((target_params - out_features) / (in_features + 1 + out_features))
        )
        return [h]
    raise NotImplementedError("Only 1 hidden layer is supported by this heuristic")


__all__ = ["FCNN", "hidden_layers_for_param_count"]
