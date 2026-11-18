"""Prepare the Phase 05 held-out datasets: breast cancer and MNIST 3 vs 5.

Usage (from the repository root):
    .venvs/reproduction/bin/python papers/power_of_QNN/utils/prepare_datasets.py --data-root data

Both tasks are reduced to the 4 model inputs by the transform frozen in LOG.md (Amendment
for both tasks), always fit on the training split only:

1. breast cancer only: standardize with the training mean and std;
2. PCA to 4 components (SVD of the centered training data; each component's sign is fixed
   so that its largest-magnitude loading is positive);
3. min-max scale each component to [0, 1] with the training min and max; validation and
   test values outside [0, 1] are kept.

Writes ``<data_root>/power_of_QNN/<task>/<file>.npz`` with ``x_<split>`` / ``y_<split>`` for
``split`` in train, validation, test, plus a ``MANIFEST.json`` with SHA-256 checksums and
the provenance of every step. MNIST is fetched from OpenML (``mnist_784`` version 1) into
``<data_root>/power_of_QNN/sources/openml``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from figure_io import PROJECT_DIR  # noqa: F401  (puts the repository on sys.path)
from power_of_QNN.lib.experiment_logging import sha256

SPLITS = ("train", "validation", "test")
N_COMPONENTS = 4


def stratified_indices(
    labels: np.ndarray, counts: dict[str, int], seed: int
) -> dict[str, np.ndarray]:
    """Per class, permute that class's indices with one ``RandomState(seed)`` stream
    (classes in increasing order) and cut consecutive blocks of ``counts[split]``."""
    rng = np.random.RandomState(seed)
    parts = {split: [] for split in counts}
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        start = 0
        for split, count in counts.items():
            parts[split].append(indices[start : start + count])
            start += count
    return {split: np.sort(np.concatenate(chunks)) for split, chunks in parts.items()}


def fit_reduction(x_train: np.ndarray, standardize: bool) -> dict:
    """Fit standardization (optional), 4-component PCA and [0, 1] scaling on train only."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0) if standardize else np.ones(x_train.shape[1])
    scale[scale == 0] = 1.0
    centered = (x_train - mean) / scale
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:N_COMPONENTS]
    signs = np.sign(
        components[np.arange(N_COMPONENTS), np.abs(components).argmax(axis=1)]
    )
    components = components * signs[:, None]
    projected = centered @ components.T
    return {
        "mean": mean,
        "scale": scale,
        "components": components,
        "low": projected.min(axis=0),
        "high": projected.max(axis=0),
    }


def apply_reduction(x: np.ndarray, reduction: dict) -> np.ndarray:
    projected = ((x - reduction["mean"]) / reduction["scale"]) @ reduction[
        "components"
    ].T
    return (projected - reduction["low"]) / (reduction["high"] - reduction["low"])


def _write(directory: Path, filename: str, arrays: dict, provenance: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    np.savez(path, **arrays)
    manifest = {
        **provenance,
        "files": {filename: sha256(path)},
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "class_counts": {
            split: np.bincount(arrays[f"y_{split}"]).tolist() for split in SPLITS
        },
    }
    (directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["class_counts"]), path)


def _reduced_split(x, y, indices: dict, standardize: bool) -> tuple[dict, dict]:
    reduction = fit_reduction(x[indices["train"]], standardize)
    arrays = {}
    for split in SPLITS:
        arrays[f"x_{split}"] = apply_reduction(x[indices[split]], reduction)
        arrays[f"y_{split}"] = y[indices[split]].astype(np.int64)
    centered = (x[indices["train"]] - reduction["mean"]) / reduction["scale"]
    kept = (centered @ reduction["components"].T).var(axis=0).sum()
    pca_variance = float(kept / centered.var(axis=0).sum())
    return arrays, {"pca_explained_variance_fraction_train": pca_variance}


def prepare_breast_cancer(root: Path) -> None:
    from sklearn import __version__ as sklearn_version
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    labels = data.target.astype(np.int64)
    class_sizes = np.bincount(labels)
    counts_per_class = [
        {"train": round(0.6 * n), "validation": round(0.2 * n)} for n in class_sizes
    ]
    rng = np.random.RandomState(0)
    parts = {split: [] for split in SPLITS}
    for label, counts in enumerate(counts_per_class):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n_train, n_val = counts["train"], counts["validation"]
        parts["train"].append(indices[:n_train])
        parts["validation"].append(indices[n_train : n_train + n_val])
        parts["test"].append(indices[n_train + n_val :])
    indices = {
        split: np.sort(np.concatenate(chunks)) for split, chunks in parts.items()
    }
    arrays, extra = _reduced_split(data.data, labels, indices, standardize=True)
    _write(
        root / "breast_cancer",
        "bcancer_pca4_split_rs0.npz",
        arrays,
        {
            "dataset": "Breast Cancer Wisconsin (Diagnostic), scikit-learn bundled copy",
            "source": "sklearn.datasets.load_breast_cancer",
            "sklearn_version": sklearn_version,
            "labels": "0 = malignant, 1 = benign (sklearn convention)",
            "split": "stratified per class, 60/20/20 rounded per class, RandomState(0)",
            "reduction": "standardize (train) -> PCA 4 (train SVD, sign-fixed) -> min-max [0,1] (train)",
            "license": "CC BY 4.0 (UCI ML Repository); sklearn BSD-3-Clause",
            **extra,
        },
    )


def prepare_mnist35(root: Path) -> None:
    from sklearn import __version__ as sklearn_version
    from sklearn.datasets import fetch_openml

    cache = root / "sources" / "openml"
    images, digits = fetch_openml(
        "mnist_784", version=1, as_frame=False, data_home=str(cache), return_X_y=True
    )
    digits = digits.astype(np.int64)
    pixels = images.astype(np.float64) / 255.0
    is_pair = np.isin(digits, (3, 5))
    labels = (digits == 5).astype(np.int64)
    train_pool = np.flatnonzero(is_pair[:60000])
    chosen = stratified_indices(
        labels[train_pool], {"train": 250, "validation": 250}, seed=0
    )
    indices = {
        "train": train_pool[chosen["train"]],
        "validation": train_pool[chosen["validation"]],
        "test": 60000 + np.flatnonzero(is_pair[60000:]),
    }
    arrays, extra = _reduced_split(pixels, labels, indices, standardize=False)
    _write(
        root / "mnist",
        "mnist35_pca4_train500_val500_test_rs0.npz",
        arrays,
        {
            "dataset": "MNIST digits 3 vs 5",
            "source": "OpenML mnist_784 version 1 via sklearn.datasets.fetch_openml (rows 0-59999 = train, 60000-69999 = test)",
            "local_cache": str(cache.relative_to(root.parent)),
            "sklearn_version": sklearn_version,
            "labels": "0 = digit 3, 1 = digit 5",
            "split": "train 250 + validation 250 per class drawn with RandomState(0) from the 60k training part; test = all 3s and 5s of the 10k test part",
            "reduction": "pixels/255 -> PCA 4 (train SVD, centered, sign-fixed) -> min-max [0,1] (train)",
            "license": "MNIST: CC BY-SA 3.0 (Y. LeCun, C. Cortes, C. Burges)",
            **extra,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root / "power_of_QNN"
    prepare_breast_cancer(root)
    prepare_mnist35(root)


if __name__ == "__main__":
    main()
