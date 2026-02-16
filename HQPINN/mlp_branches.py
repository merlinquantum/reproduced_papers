# mlp_branches.py
# Shared MLP branch definitions for classical-classical and classical-quantum PINNs

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple feed-forward network used as a branch in the PINN.

    Architecture:
        1 -> 16 -> 16 -> 1 with Tanh activations.

    Parameters
    ----------
    dtype : torch.dtype
        Torch dtype used for the Linear layers.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16, dtype=dtype),
            nn.Tanh(),
            nn.Linear(16, 16, dtype=dtype),
            nn.Tanh(),
            nn.Linear(16, 1, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CC_PINN(nn.Module):
    """
    Classical–Classical PINN with two parallel MLP branches:

        u(t) = u_1(t) + u_2(t)

    Parameters
    ----------
    dtype : torch.dtype
        Torch dtype used for both MLP branches.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.branch1 = MLP(dtype=dtype)
        self.branch2 = MLP(dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch1(t) + self.branch2(t)