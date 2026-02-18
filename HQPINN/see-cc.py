# see-cc.py

import torch
import torch.nn as nn

from config import SEE_CC_NUM_HIDDEN_LAYERS, SEE_CC_HIDDEN_WIDTH
from layer_classical import BranchPyTorch


class SEE_CCPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = BranchPyTorch(
            in_features=2,  # (x, t)
            out_features=3,  # (rho, u, p)
            num_hidden_layers=SEE_CC_NUM_HIDDEN_LAYERS,
            hidden_width=SEE_CC_HIDDEN_WIDTH,
        )

    def forward(self, xt):
        return self.branch(xt)
