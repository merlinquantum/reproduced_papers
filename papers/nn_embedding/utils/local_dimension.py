import numpy as np
import torch
import torch.nn as nn
from nngeometry.metrics import FIM
from nngeometry.object import FMatDense
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import gamma


def get_local_dimension(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    num_sample_per_dimension: int = 5,
) -> float:
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, shuffle=True)

    d = sum(param.numel() for param in model.parameters())
    n = y.size(0)
    gamma = 2 * np.pi * np.log(n) / n
    kappa = gamma * n / (2 * np.pi * np.log(n))
    V = (np.pi ** (d / 2)) * (epsilon**d) / gamma(d / 2 + 1)

    fisher_matrix = FIM(model, loader, FMatDense)


def create_param_ensemble(params: torch.Tensor, num_sample_per_dimension: int = 5):
    pass
