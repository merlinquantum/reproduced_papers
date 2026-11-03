"""Classical fair baselines for the vertical federated comparison (Phase 5).

These are independent controls, not part of the paper. Every party is a
one-hidden-layer ReLU MLP whose width is set so its trainable-parameter count matches
the quantum or photonic party it is compared with (see LOG.md, "Phase 5 Comparison
Plan"). Clients send 8 tanh-bounded values, like the gate model's Z expectations; the
server outputs 10 logits. Pixels are used as binary values (angle > 0), the same
information the photonic adaptation receives.
"""

from __future__ import annotations

import torch
from torch import nn

PIXELS = 256
PIXELS_PER_CLIENT = 32
MESSAGE_SIZE = 8
N_CLASSES = 10
SERVER_ENCODINGS = {"server_wide", "server_narrow"}


def mlp(inputs: int, hidden: int, outputs: int, dtype: torch.dtype) -> nn.Sequential:
    """One-hidden-layer ReLU network with ``(inputs + 1) * hidden + (hidden + 1) * outputs`` parameters."""
    return nn.Sequential(
        nn.Linear(inputs, hidden, dtype=dtype),
        nn.ReLU(),
        nn.Linear(hidden, outputs, dtype=dtype),
    )


def mlp_parameter_count(inputs: int, hidden: int, outputs: int) -> int:
    """Trainable parameters of :func:`mlp` (used to check the matching)."""
    return inputs * hidden + hidden + hidden * outputs + outputs


class ClassicalVerticalFederatedModel(nn.Module):
    """Classical clients on disjoint pixel slices plus a classical server.

    Parameters
    ----------
    client_slices : list of int
        For each client, the index of its 32-column slice of the permuted pixels.
    server_encoding : str
        ``"server_wide"`` (server reads all client messages) or ``"server_narrow"``
        (the single client's message, E2 layout).
    client_hidden, server_hidden : int
        Hidden widths, fixed by the parameter matching.
    dtype : torch.dtype
        Parameter dtype. Default value is torch.float32.
    """

    def __init__(
        self,
        client_slices: list[int],
        server_encoding: str,
        client_hidden: int,
        server_hidden: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if server_encoding not in SERVER_ENCODINGS:
            raise ValueError(f"Unknown server_encoding {server_encoding!r}")
        if server_encoding == "server_narrow" and len(client_slices) != 1:
            raise ValueError("server_narrow takes exactly one client")
        self.client_slices = list(client_slices)
        self.dtype = dtype
        self.clients = nn.ModuleList(
            nn.Sequential(
                mlp(PIXELS_PER_CLIENT, client_hidden, MESSAGE_SIZE, dtype), nn.Tanh()
            )
            for _ in self.client_slices
        )
        self.server = mlp(
            MESSAGE_SIZE * len(self.client_slices), server_hidden, N_CLASSES, dtype
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Class logits for ``(batch, 256)`` permuted pixels."""
        binary = (pixels > 0).to(self.dtype)
        messages = [
            client(binary[:, s * PIXELS_PER_CLIENT : (s + 1) * PIXELS_PER_CLIENT])
            for client, s in zip(self.clients, self.client_slices)
        ]
        return self.server(torch.cat(messages, dim=1))

    def trainable_parameter_counts(self) -> dict[str, int]:
        """Trainable parameters per party and in total."""
        counts = {
            f"client_{i}": sum(p.numel() for p in c.parameters())
            for i, c in enumerate(self.clients)
        }
        counts["server"] = sum(p.numel() for p in self.server.parameters())
        counts["total"] = sum(counts.values())
        return counts


class ClassicalCentralModel(nn.Module):
    """Centralised reference: one MLP on all 256 pixels (no federated constraint).

    Parameters
    ----------
    hidden : int
        Hidden width, fixed by the parameter matching.
    dtype : torch.dtype
        Parameter dtype. Default value is torch.float32.
    """

    def __init__(self, hidden: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype
        self.network = mlp(PIXELS, hidden, N_CLASSES, dtype)
        # Kept for the runner's logging, which reports the number of clients.
        self.clients = nn.ModuleList()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Class logits for ``(batch, 256)`` pixels."""
        return self.network((pixels > 0).to(self.dtype))

    def trainable_parameter_counts(self) -> dict[str, int]:
        """Trainable parameters in total."""
        return {"total": sum(p.numel() for p in self.network.parameters())}
