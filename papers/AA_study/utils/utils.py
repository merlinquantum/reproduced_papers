import numpy as np
from numpy.typing import NDArray
import torch
from typing import List


def trace_distance(A: NDArray, B: NDArray) -> NDArray:
    return np.linalg.norm(A - B, ord="nuc") / np.shape(A)[0]


def state_vector_to_density_matrix(x: NDArray | List | torch.Tensor) -> NDArray:
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(x, list):
        x = np.array(x)
    return np.tensordot(x, x.conjugate(), axes=0)
