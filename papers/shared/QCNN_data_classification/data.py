"""Shared PCA preprocessing utilities for QCNN data classification."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

try:
    from runtime_lib.data_paths import paper_data_dir
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from runtime_lib.data_paths import paper_data_dir

_DATASET_MAP = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}


def _dataset_root() -> Path:
    return paper_data_dir("QCNN_data_classification")


def make_pca(k: int, dataset: str = "mnist"):
    dataset_key = dataset.replace("-", "").replace("_", "").lower()
    try:
        dataset_cls = _DATASET_MAP[dataset_key]
    except KeyError as exc:
        valid = ", ".join(sorted(_DATASET_MAP))
        raise ValueError(
            f"Unknown dataset '{dataset}'. Valid options: {valid}."
        ) from exc

    to_t = transforms.Compose([transforms.ToTensor()])
    root = _dataset_root()
    base_tr = dataset_cls(root, train=True, download=True, transform=to_t)
    base_te = dataset_cls(root, train=False, download=True, transform=to_t)

    def filt(base):
        xs, ys = [], []
        for img, lab in base:
            if int(lab) in (0, 1):
                xs.append(img.view(-1).float())
                ys.append(0 if int(lab) == 0 else 1)
        return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

    xtr, ytr = filt(base_tr)
    xte, yte = filt(base_te)

    pca = PCA(n_components=k, svd_solver="full", whiten=False, random_state=0)
    ztr_raw = torch.from_numpy(pca.fit_transform(xtr.numpy())).float()
    zte_raw = torch.from_numpy(pca.transform(xte.numpy())).float()

    mins = ztr_raw.min(0, keepdim=True).values
    maxs = ztr_raw.max(0, keepdim=True).values
    ztr = torch.clamp((ztr_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    zte = torch.clamp((zte_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    return (ztr, ytr), (zte, yte)


__all__ = ["make_pca"]
