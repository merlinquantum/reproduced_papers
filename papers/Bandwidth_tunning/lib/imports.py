import gzip
import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import KMNIST, FashionMNIST

# ==============================================================================
#
#                              fashion_kmnist
#
# ==============================================================================


def load_fashion_mnist_torch():
    """Download/load Fashion-MNIST via torchvision and return train/test tensors."""
    # Store data in the user cache to avoid committing the dataset to the repo.
    cache_root = Path.home() / ".cache" / "torchvision"

    try:
        train_dataset = FashionMNIST(root=str(cache_root), train=True, download=True)
        test_dataset = FashionMNIST(root=str(cache_root), train=False, download=True)
    except Exception as e:
        if "SSL" in str(e) or "certificate" in str(e).lower():
            raise RuntimeError(
                f"SSL certificate error downloading Fashion-MNIST:\n{str(e)}\n\n"
                "Solutions:\n"
                "1. pip install --upgrade certifi\n"
                "2. On Windows: Run Python SSL cert installer\n"
                "3. Check internet connection and proxy settings\n"
            ) from e
        raise

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

    try:
        train_dataset = KMNIST(root=str(cache_root), train=True, download=True)
        test_dataset = KMNIST(root=str(cache_root), train=False, download=True)
    except Exception as e:
        if "SSL" in str(e) or "certificate" in str(e).lower():
            try:
                _download_kmnist_fallback(cache_root)
                train_dataset = KMNIST(root=str(cache_root), train=True, download=False)
                test_dataset = KMNIST(root=str(cache_root), train=False, download=False)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"SSL certificate error downloading KMNIST:\n{str(e)}\n\n"
                    f"Fallback download also failed:\n{str(fallback_error)}\n\n"
                    "Solutions:\n"
                    "1. pip install --upgrade certifi\n"
                    "2. On Windows: Run Python SSL cert installer\n"
                    "3. Check internet connection and proxy settings\n"
                ) from e
        else:
            raise

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


def _download_kmnist_fallback(cache_root: Path):
    """Download and extract KMNIST files from alternate mirrors.

    torchvision currently uses a single KMNIST mirror, which can fail in some SSL setups.
    This fallback pulls the exact same files from alternate hosts into the expected raw folder.
    """
    raw_dir = cache_root / "KMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    mirrors = [
        "https://codh.rois.ac.jp/kmnist/dataset/kmnist/",
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    ]

    for filename, _ in KMNIST.resources:
        gz_path = raw_dir / filename
        extracted_path = raw_dir / Path(filename).stem

        if extracted_path.exists():
            continue

        errors = []
        for mirror in mirrors:
            url = f"{mirror}{filename}"
            try:
                _download_with_ssl_fallback(url, gz_path)
                break
            except Exception as mirror_error:
                errors.append(f"{url}: {mirror_error}")
        else:
            joined = "\n".join(errors)
            raise RuntimeError(f"Unable to download {filename} from fallback mirrors:\n{joined}")

        with gzip.open(gz_path, "rb") as src, extracted_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _download_with_ssl_fallback(url: str, destination: Path):
    """Try strict TLS with certifi first, then unverified TLS as last resort."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    strict_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(url, context=strict_context) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
        return
    except ssl.SSLError:
        pass

    # Last resort for environments with broken local certificate chains.
    unsafe_context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=unsafe_context) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


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
    fetch_attempts = [
        {"name": "PLAsTiCC", "version": "active", "as_frame": False},
        {"name": "PLAsTiCC", "as_frame": False},
        {"name": "plasticc", "as_frame": False},
        {"data_id": 40900, "as_frame": False},
    ]

    plasticc = None
    errors = []
    for attempt in fetch_attempts:
        try:
            plasticc = fetch_openml(**attempt)
            break
        except Exception as e:
            errors.append(f"fetch_openml({attempt}) -> {e}")
            continue

    if plasticc is None:
        merged_errors = "\n".join(errors)
        if any(
            token in merged_errors.lower()
            for token in ["ssl", "certificate", "urlopen", "tls", "handshake"]
        ):
            raise RuntimeError(
                f"SSL certificate error downloading PLAsTiCC from OpenML:\n{merged_errors}\n\n"
                "Solutions:\n"
                "1. pip install --upgrade certifi\n"
                "2. On Windows: Run Python SSL cert installer\n"
                "3. Check internet connection and proxy settings\n"
            )

        raise RuntimeError(
            "Unable to fetch PLAsTiCC from OpenML using known identifiers.\n"
            f"Tried: {fetch_attempts}\n"
            f"Errors:\n{merged_errors}"
        )

    X_np = np.asarray(plasticc.data, dtype=np.float32)
    y_raw = np.asarray(plasticc.target)
    y_np = LabelEncoder().fit_transform(y_raw).astype(np.int64)

    n_train = 2500
    n_test = 1006
    total_needed = n_train + n_test

    if X_np.shape[0] < total_needed:
        raise ValueError(
            f"PLAsTiCC dataset too small: {X_np.shape[0]} samples, expected at least {total_needed}."
        )

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np,
        y_np,
        train_size=n_train,
        test_size=n_test,
        random_state=42,
        shuffle=True,
        stratify=y_np,
    )

    X_train = torch.from_numpy(X_train_np)
    X_test = torch.from_numpy(X_test_np)
    y_train = torch.from_numpy(y_train_np)
    y_test = torch.from_numpy(y_test_np)

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
