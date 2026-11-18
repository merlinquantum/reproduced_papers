"""Classical baseline models for the fair-comparison plan (LOG.md, Fair Baseline Plan).

They take the same 16 downsampled pixels as the VQTN and return log-probabilities.
Parameter counts are reported so a baseline can be matched to the quantum model.
"""

from __future__ import annotations

import torch
from torch import nn

N_PIXELS = 16


class ClassicalClassifier(nn.Module):
    """Logistic regression (``hidden == 0``) or one-hidden-layer MLP.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    hidden : int
        Hidden width; 0 gives multinomial logistic regression.
    dtype : torch.dtype
        Parameter dtype.
    """

    def __init__(
        self, n_classes: int, hidden: int, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        if hidden == 0:
            self.network = nn.Linear(N_PIXELS, n_classes, dtype=dtype)
        else:
            self.network = nn.Sequential(
                nn.Linear(N_PIXELS, hidden, dtype=dtype),
                nn.Tanh(),
                nn.Linear(hidden, n_classes, dtype=dtype),
            )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return class log-probabilities, shape ``(batch, n_classes)``."""
        return torch.log_softmax(self.network(pixels), dim=1)
