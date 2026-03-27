import numpy as np
import torch
import torch.nn as nn


def create_random_pairs(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for _ in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)
    return (
        torch.stack(X1_new),
        torch.stack(X2_new),
        torch.as_tensor(Y_new, dtype=torch.float32),
    )


def pick_random_data(batch_size, X, Y):
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = torch.stack([X[i] for i in batch_index])
    Y_batch = torch.as_tensor([Y[i] for i in batch_index], dtype=torch.long)
    return X_batch, Y_batch


def calculate_distance(
    rho0: torch.Tensor, rho1: torch.Tensor, distance: str = "Trace"
) -> float:
    rho_diff = rho1 - rho0
    if distance == "Trace":
        eigvals = torch.linalg.eigvals(rho_diff)
        return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
    elif distance == "Hilbert-Schmidt":
        return torch.trace(rho_diff @ rho_diff)
    else:
        raise ValueError("No distance with that name")


class LinearLoss(nn.Module):
    def forward(self, labels, predictions):
        labels = labels.to(predictions.dtype)
        # Same as 1-(l * p) where the labels are -1 and 1
        return (
            0.5 * ((4 * labels * predictions) - (2 * labels) - (2 * predictions)).mean()
        )
