from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import FashionMNIST
from torchvision.datasets import KMNIST

# ==============================================================================
#
#                              fashion_kmnist
#
# ==============================================================================


def load_fashion_mnist_torch():
    """Download/load Fashion-MNIST via torchvision and return train/test tensors."""
    # Store data in the user cache to avoid committing the dataset to the repo.
    cache_root = Path.home() / ".cache" / "torchvision"

    train_dataset = FashionMNIST(root=str(cache_root), train=True, download=True)
    test_dataset = FashionMNIST(root=str(cache_root), train=False, download=True)

    X_train_tensor = train_dataset.data.unsqueeze(1).float() / 255.0
    y_train_tensor = train_dataset.targets.long()

    X_test_tensor = test_dataset.data.unsqueeze(1).float() / 255.0
    y_test_tensor = test_dataset.targets.long()

    masque_train = (y_train_tensor == 2) | (y_train_tensor == 8)
    masque_test = (y_test_tensor == 2) | (y_test_tensor == 8)

    X_train_tensor = X_train_tensor[masque_train]
    y_train_tensor = y_train_tensor[masque_train]

    X_test_tensor = X_test_tensor[masque_test]
    y_test_tensor = y_test_tensor[masque_test]

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


# ==============================================================================
#
#                                 kmnist
#
# ==============================================================================


def load_kmnist28():
    # Store data in the user cache to avoid committing the dataset to the repo.
    cache_root = Path.home() / ".cache" / "torchvision"

    train_dataset = KMNIST(root=str(cache_root), train=True, download=True)
    test_dataset = KMNIST(root=str(cache_root), train=False, download=True)

    X_train = train_dataset.data.unsqueeze(1).float() / 255.0
    y_train = train_dataset.targets.long()

    X_test = test_dataset.data.unsqueeze(1).float() / 255.0
    y_test = test_dataset.targets.long()

    masque_train = (y_train == 2) | (y_train == 8)
    masque_test = (y_test == 2) | (y_test == 8)

    X_train = X_train[masque_train]
    y_train = y_train[masque_train]

    X_test = X_test[masque_test]
    y_test = y_test[masque_test]

    return X_train, y_train, X_test, y_test


# ==============================================================================
#
#                            hidden_manifold
#
# ==============================================================================


def load_hidden_manifold():
    from lib.hidden_manifold import generate_hidden_manifold_model

    X, y = generate_hidden_manifold_model(10000, 100, 50)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return X[:5000], y[:5000], X[5000:], y[5000:]


# ==============================================================================
#
#                                plasticc
#
# ==============================================================================


def load_plasticc():
    # OpenML handles download/caching outside the repo (typically ~/.cache or ~/scikit_learn_data).
    plasticc = fetch_openml(name="PLAsTiCC", version=1, as_frame=False)

    X_np = np.asarray(plasticc.data, dtype=np.float32)
    y_raw = np.asarray(plasticc.target)
    y_np = LabelEncoder().fit_transform(y_raw).astype(np.int64)

    X_plasticc = torch.from_numpy(X_np)
    y_plasticc = torch.from_numpy(y_np)

    n_train = 2500
    n_test = 1006
    total_needed = n_train + n_test

    if X_plasticc.shape[0] < total_needed:
        raise ValueError(
            f"PLAsTiCC dataset too small: {X_plasticc.shape[0]} samples, expected at least {total_needed}."
        )

    X_train, X_test = torch.split(X_plasticc[:total_needed], [n_train, n_test], dim=0)
    y_train, y_test = torch.split(y_plasticc[:total_needed], [n_train, n_test], dim=0)

    return X_train, y_train, X_test, y_test


# ==============================================================================
#
#                                 global
#
# ==============================================================================


def data(dataset):
    if dataset == "fashion_mnist":
        return load_fashion_mnist_torch()
    if dataset == "kmnist28":
        return load_kmnist28()
    if dataset == "hidden_manifold":
        return load_hidden_manifold()
    if dataset == "plasticc":
        return load_plasticc()
    raise ValueError(f"Unsupported dataset: {dataset}")
