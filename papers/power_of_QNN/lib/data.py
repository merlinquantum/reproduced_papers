"""Datasets for the power_of_QNN reproduction.

Two real datasets sit behind one interface, ``build_dataloader(cfg, split)``:

- ``iris_binary``: Iris classes 0/1 (rows 0:100), L2 row-normalized, as in the
  authors' training scripts (Fig. 3b, Table 1). Train split only.
- ``blobs``: ``make_blobs(1000, 6 features, 2 centers, random_state=42)``,
  L2 row-normalized, 60/40 split (Fig. 6, App. B.2).
- ``iris_pair``: two Iris classes from ``iris.csv`` with a stratified
  train / validation / test split (Phase 05 fair comparison; not in the paper).
- ``prepared_split``: a binary task already reduced to 4 inputs and split by
  ``utils/prepare_datasets.py`` (Phase 05 held-out tasks: breast cancer, MNIST 3 vs 5).

The synthetic ``x ~ N(0, I)`` inputs of the Fisher estimates (Sec. 4.1) are not a
dataset here: ``lib.fisher_experiments.draw_parameters_and_inputs`` draws them from
the same seeded NumPy stream as the parameter samples, in the authors' order, so
seed 0 reproduces their cached arrays.

The real datasets are read from ``<data_root>/power_of_QNN/`` (``DATA_DIR`` or
``--data-root``), and every file is checked against the SHA-256 in its
``MANIFEST.json`` before use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .experiment_logging import sha256

logger = logging.getLogger(__name__)

PAPER_NAME = "power_of_QNN"
IRIS_FILE = ("iris", "iris_classes01_l2norm.npz")
BLOBS_FILE = ("make_blobs", "make_blobs_n1000_f6_c2_rs42_split60_40_rs0.npz")
IRIS_CSV = ("iris", "iris.csv")
DATASETS = ("iris_binary", "blobs", "iris_pair", "prepared_split")
IRIS_PAIR_SPLITS = ("train", "validation", "test")


def paper_data_dir(cfg: dict) -> Path:
    """Return ``<data_root>/power_of_QNN`` from the resolved runtime config."""
    return Path(cfg["data_root"]) / PAPER_NAME


def _verified_npz(cfg: dict, subdir: str, filename: str) -> dict[str, np.ndarray]:
    directory = paper_data_dir(cfg) / subdir
    path = directory / filename
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest["files"][filename]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _iris_pair(dataset_cfg: dict, data_dir: Path, split: str):
    """Stratified split of two Iris classes, labels remapped to 0/1 in class order.

    Each class's 50 rows are permuted by ``RandomState(split_seed)`` (one stream,
    classes in the configured order) and cut into ``per_class`` train / validation /
    test counts. Rows are L2-normalized one at a time, as in the paper, so no
    statistic is shared across splits.
    """
    if split not in IRIS_PAIR_SPLITS:
        raise ValueError(f"iris_pair has splits {IRIS_PAIR_SPLITS}; got {split!r}")
    path = data_dir / IRIS_CSV[0] / IRIS_CSV[1]
    manifest = json.loads((path.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    if sha256(path) != manifest["files"][IRIS_CSV[1]]:
        raise ValueError(f"Checksum mismatch for {path}")
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    features, classes = table[:, :4], table[:, 4].astype(np.int64)
    per_class = dataset_cfg["per_class"]
    counts = [per_class[name] for name in IRIS_PAIR_SPLITS]
    start = sum(counts[: IRIS_PAIR_SPLITS.index(split)])
    stop = start + per_class[split]
    rng = np.random.RandomState(dataset_cfg["split_seed"])
    rows, labels = [], []
    for label, iris_class in enumerate(dataset_cfg["classes"]):
        class_rows = np.flatnonzero(classes == iris_class)
        if sum(counts) > len(class_rows):
            raise ValueError(
                f"per_class {per_class} needs {sum(counts)} rows; class {iris_class} has {len(class_rows)}"
            )
        chosen = rng.permutation(class_rows)[start:stop]
        rows.append(chosen)
        labels.append(np.full(len(chosen), label, dtype=np.int64))
    rows = np.concatenate(rows)
    selected = features[rows]
    # Same as sklearn.preprocessing.normalize (L2, per row), without the dependency.
    return selected / np.linalg.norm(selected, axis=1, keepdims=True), np.concatenate(
        labels
    )


def load_arrays(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` arrays for the configured dataset and split.

    Parameters
    ----------
    cfg : dict
        Resolved config with ``dataset`` and ``data_root``.
    split : str
        ``"train"`` or ``"test"`` (``iris_binary`` has only ``"train"``;
        ``iris_pair`` also has ``"validation"``).

    Returns
    -------
    tuple
        Inputs ``(N, s_in)`` and integer labels ``(N,)``.

    Raises
    ------
    ValueError
        For an unknown dataset or split, or a checksum mismatch.
    """
    name = cfg["dataset"]["name"]
    if name == "iris_binary":
        if split != "train":
            raise ValueError(
                "iris_binary has only a train split (the paper reports training loss)"
            )
        arrays = _verified_npz(cfg, *IRIS_FILE)
        return arrays["x"], arrays["y"].astype(np.int64)
    if name == "blobs":
        if split not in ("train", "test"):
            raise ValueError(f"blobs has train and test splits; got {split!r}")
        arrays = _verified_npz(cfg, *BLOBS_FILE)
        return arrays[f"x_{split}"], arrays[f"y_{split}"].astype(np.int64)
    if name == "iris_pair":
        return _iris_pair(cfg["dataset"], paper_data_dir(cfg), split)
    if name == "prepared_split":
        if split not in IRIS_PAIR_SPLITS:
            raise ValueError(
                f"prepared_split has splits {IRIS_PAIR_SPLITS}; got {split!r}"
            )
        arrays = _verified_npz(cfg, cfg["dataset"]["subdir"], cfg["dataset"]["file"])
        return arrays[f"x_{split}"], arrays[f"y_{split}"].astype(np.int64)
    raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """Return a DataLoader over the configured dataset split.

    ``dataset.batch_size: null`` means full batch, the paper's training setting.
    Batches are ``(x, y)`` with float64 inputs and int64 labels.
    """
    x, y = load_arrays(cfg, split)
    tensors = TensorDataset(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y))
    batch_size = cfg["dataset"]["batch_size"] or len(x)
    return DataLoader(tensors, batch_size=batch_size, shuffle=False)


def dataset_record(cfg: dict) -> dict:
    """Describe the dataset for ``run_status.json``."""
    name = cfg["dataset"]["name"]
    if name == "prepared_split":
        dataset_cfg = cfg["dataset"]
        manifest = json.loads(
            (paper_data_dir(cfg) / dataset_cfg["subdir"] / "MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        return {
            "name": f"prepared_split: {manifest['dataset']} ({dataset_cfg['file']})",
            "splits": list(IRIS_PAIR_SPLITS),
            "preprocessing": f"{manifest['reduction']}; split: {manifest['split']}",
            "subset": None,
        }
    if name == "iris_pair":
        dataset_cfg = cfg["dataset"]
        return {
            "name": f"iris_pair: Iris classes {dataset_cfg['classes']} from iris.csv",
            "splits": list(IRIS_PAIR_SPLITS),
            "preprocessing": f"L2 row normalization; stratified per-class split {dataset_cfg['per_class']} with RandomState({dataset_cfg['split_seed']})",
            "subset": None,
        }
    records = {
        "iris_binary": (
            "Iris rows 0:100 (classes 0/1)",
            ["train"],
            "L2 row normalization (sklearn normalize)",
        ),
        "blobs": (
            "make_blobs n=1000 f=6 c=2 rs=42",
            ["train", "test"],
            "L2 row normalization; 60/40 split rs=0",
        ),
    }
    description, splits, preprocessing = records[name]
    return {
        "name": f"{name}: {description}",
        "splits": splits,
        "preprocessing": preprocessing,
        "subset": None,
    }
