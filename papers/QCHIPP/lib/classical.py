"""Same-information hierarchical MLP control with approximately matched budget."""

import math

from torch import nn

from .photonic import is_strict_int, residue_angles


class ClassicalQCNN(nn.Module):
    """Shared local MLP filters with the photonic model's residue-angle inputs.

    Parameters
    ----------
    hidden : int
        Hidden width of both independent stage cells. Default value is 12.
        This yields 250 trainable parameters versus the photonic model's 256;
        the budgets are approximate, not exactly iso-parameter.
    """

    def __init__(self, hidden=12):
        super().__init__()
        if not is_strict_int(hidden) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        self.stage1 = nn.Sequential(
            nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, 3), nn.Tanh()
        )
        self.stage2 = nn.Sequential(
            nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, 3), nn.Tanh()
        )
        self.readout = nn.Linear(3, 1)

    def forward(self, bits):
        """Return B binary logits from the same B x 20 bits as PhotonicQCNN.

        Parameters
        ----------
        bits : torch.Tensor
            Four ordered five-bit residue codes per sample.

        Returns
        -------
        torch.Tensor
            One logit per peptide, shape (B,).
        """
        angles = residue_angles(bits).to(self.readout.weight)
        local = self.stage1(angles.reshape(-1, 6))
        pooled_angles = local.reshape(bits.shape[0], 6) * math.pi
        return self.readout(self.stage2(pooled_angles)).squeeze(-1)
