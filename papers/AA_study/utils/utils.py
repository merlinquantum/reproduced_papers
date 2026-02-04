import numpy as np
from numpy.typing import NDArray
import torch
from typing import Tuple, List
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from math import comb
import warnings


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def trace_distance(A: NDArray, B: NDArray) -> NDArray:
    return np.linalg.norm(A - B, ord="nuc") / np.shape(A)[0]


def state_vector_to_density_matrix(x: NDArray | List | torch.Tensor) -> NDArray:
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(x, list):
        x = np.array(x)
    return np.tensordot(x, x.conjugate(), axes=0)


def find_mode_photon_config(
    num_features: int,
    max_modes: int = 20,
) -> tuple[int, int]:
    """
    Find (n_modes, n_photons) with smallest n_modes such that
    C(n_modes + n_photons - 1, n_photons) >= num_features and
    n_photons <= n_modes // 2.
    """
    if num_features <= 0:
        raise ValueError("num_features must be positive.")

    best = None
    for n_modes in range(1, max_modes + 1):
        for n_photons in range(1, (n_modes // 2) + 1):
            dim = comb(n_modes + n_photons - 1, n_photons)
            if dim >= num_features:
                candidate = (n_modes, n_photons)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                break
        if best is not None and best[0] == n_modes:
            break

    if best is None:
        warnings.warn(
            "System too large for simulation: no valid (n_modes, n_photons) "
            "found with max_modes=20.",
            RuntimeWarning,
        )
        raise ValueError("System too large for simulation with max_modes=20.")
    return best


def basic_model_training(
    model: nn.Module,
    data_loader: DataLoader,
    lr: float = 0.01,
    num_epochs: int = 10,
    test_loader: DataLoader = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    accuracy_per_epoch = []
    loss_per_epoch = []

    for epoch in range(num_epochs):
        tot_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        for features, labels in data_loader:
            features = features.to(device)
            if features.dtype != next(model.parameters()).dtype:
                features = features.to(dtype=next(model.parameters()).dtype)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(features)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, labels)
            loss.backward()

            optimizer.step()
            tot_loss += loss.item()
            num_batches += 1
            preds = torch.clone(logits).detach().argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if test_loader is None:
            accuracy = correct / total
        else:
            accuracy = evaluate_model(model, test_loader)[0]
            model.train()
        accuracy_per_epoch.append(accuracy)
        avg_loss = tot_loss / max(num_batches, 1)
        loss_per_epoch.append(avg_loss)
        print(f"Epoch {epoch+1} had a loss of {avg_loss} and accuracy of {accuracy}")

    return model, accuracy_per_epoch, loss_per_epoch


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
) -> Tuple[float, float]:
    criterion = nn.NLLLoss()
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    loss_test_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            if images.dtype != next(model.parameters()).dtype:
                images = images.to(dtype=next(model.parameters()).dtype)
            labels = labels.to(device).long()
            outputs = model(images)
            log_probs = torch.log_softmax(outputs, dim=1)
            loss_test = criterion(log_probs, labels).cpu().detach().numpy()
            loss_test_list.append(loss_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
    print(f"Loss on the test set: {np.mean(loss_test_list):.2f}")

    return (
        100 * correct / total,
        np.mean(loss_test_list, dtype=float),
    )


def int_list(arg):
    """
    Parse a comma-separated string into a list of integers.

    Parameters
    ----------
    arg : str
        Comma-separated integers (e.g., "2,4,8").

    Returns
    -------
    list[int]
        Parsed list of integers.
    """
    return list(map(int, arg.split(",")))


def str_to_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    if s == "True":
        return True
    return False


def parse_args():
    """
    Parse command-line arguments for the experiment runner.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Encoding study")
    parser.add_argument(
        "--exp_to_run",
        type=str,
        default="DEFAULT",
        help="Which experiment to run between 'DEFAULT', 'BAS', 'FIG1', 'FIG2', 'FIG3' and 'FIG4'  (default: 'DEFAULT')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="The number of tests data per batch (default: 50)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs for training (default: 20)",
    )
    parser.add_argument(
        "--classical_epochs",
        type=int,
        default=20,
        help="Number of epochs for classical CNN training (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate of the optimizers (default: 0.01)",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=2000,
        help="The number of samples to create per class from the synthetic datasets of the paper (default: 2000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file to override CLI arguments",
    )
    parser.add_argument(
        "--dont_generate_graph",
        action="store_true",
        help="Disable graph generation",
    )
    return parser.parse_args()
