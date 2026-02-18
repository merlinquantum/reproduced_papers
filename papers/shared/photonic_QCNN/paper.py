"""Paper-aligned dataset loaders for photonic_QCNN (shared module)."""

from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from sklearn.datasets import load_digits

try:
    from runtime_lib.data_paths import paper_data_dir
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from runtime_lib.data_paths import paper_data_dir

REPO_ROOT = Path(__file__).resolve().parents[2]

SHARED_DATA_DIR = paper_data_dir("photonic_QCNN")
LEGACY_DATA_DIR = REPO_ROOT / "photonic_QCNN" / "data"


def _data_dir() -> Path:
    SHARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return SHARED_DATA_DIR


def _legacy_dir() -> Path:
    return LEGACY_DATA_DIR


def _resolve_data_file(filename: str) -> Path:
    primary = _data_dir() / filename
    if primary.exists():
        return primary
    fallback = _legacy_dir() / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Dataset asset '{filename}' not found in shared or legacy data directories"
    )


def get_bas():
    try:
        [ds] = qml.data.load("other", name="bars-and-stripes")
        x_train = np.array(ds.train["4"]["inputs"])
        y_train = np.array(ds.train["4"]["labels"])
        x_test = np.array(ds.test["4"]["inputs"])
        y_test = np.array(ds.test["4"]["labels"])

        x_train, x_test = (
            x_train[:400].reshape(400, 4, 4),
            x_test[:200].reshape(200, 4, 4),
        )
        y_train, y_test = y_train[:400], y_test[:200]

        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        return x_train, x_test, y_train, y_test

    except Exception as exc:
        print(f"Error loading PennyLane BAS dataset: {exc}")
        raise


def get_custom_bas():
    file_path = _resolve_data_file("FULL_DATASET_600_samples.bin")
    with open(file_path, "rb") as f:
        data_downloaded = pickle.load(f)

    random.shuffle(data_downloaded)

    x_train = np.array([data_downloaded[i][1] for i in range(400)])
    y_train = np.array([data_downloaded[i][0] for i in range(400)])
    x_test = np.array([data_downloaded[i][1] for i in range(400, 600)])
    y_test = np.array([data_downloaded[i][0] for i in range(400, 600)])

    x_train = x_train[:400].reshape(400, 4, 4)
    x_test = x_test[:200].reshape(200, 4, 4)
    y_train = y_train[:400]
    y_test = y_test[:200]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    return x_train, x_test, y_train, y_test


def get_mnist(class_list=(0, 1)):
    digits = load_digits()
    (x_train, y_train), (x_test, y_test) = (
        (digits.data[:750], digits.target[:750]),
        (digits.data[750:], digits.target[750:]),
    )
    x_train, x_test = (
        x_train.reshape(x_train.shape[0], 8, 8),
        x_test.reshape(x_test.shape[0], 8, 8),
    )

    train_list_data_array, train_list_label_array = [], []
    test_list_data_array, test_list_label_array = [], []
    for i in range(x_train.shape[0]):
        if (y_train[i] in class_list) and (len(train_list_data_array) < 500):
            train_list_data_array.append(x_train[i])
            train_list_label_array.append(int(y_train[i]))
    for i in range(x_test.shape[0]):
        if (y_test[i] in class_list) and (len(test_list_data_array) < 200):
            test_list_data_array.append(x_test[i])
            test_list_label_array.append(int(y_test[i]))

    return (
        np.array(train_list_data_array),
        np.array(test_list_data_array),
        np.array(train_list_label_array),
        np.array(test_list_label_array),
    )


__all__ = ["get_bas", "get_custom_bas", "get_mnist"]
