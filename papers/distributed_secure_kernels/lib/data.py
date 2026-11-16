"""Datasets and preprocessing for the distributed secure kernel reproduction.

The four real datasets are the files the authors' code reads (README, "Data"),
copied under ``<data_root>/distributed_secure_kernels``: scikit-learn's bundled
``wine_data.csv`` and ``digits.csv.gz`` (behind ``load_wine`` / ``load_digits``),
and the authors' copies of ``parkinsons.data`` and ``framingham.csv``.

Paper-faithful preprocessing (authors' ``*_Dataset_Experiments/*.py``):

1. Parkinson's and Framingham: random minority oversampling to a 0.6
   minority / majority ratio on the *whole* dataset (``RandomOverSampler(0.6)``,
   unseeded in the authors' code; seeded here).
2. Keep only the training part of ``train_test_split(train_size=0.8 or 100,
   stratify=y, random_state=42)``; the held-out part is never used.
3. ``StandardScaler`` fitted on that subset (Digits too: the authors divide the
   pixels by 16 first, which has no effect after standardisation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from runtime_lib.data_paths import resolve_data_root

logger = logging.getLogger(__name__)

PAPER_NAME = "distributed_secure_kernels"
REAL_DATASETS = ("wine", "digits", "parkinsons", "framingham")
SPLITS = ("train", "heldout")
# Random state of the authors' stratified train_test_split.
AUTHORS_SPLIT_STATE = 42


@dataclass
class DatasetSplit:
    """One preprocessed split.

    Attributes
    ----------
    features : numpy.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    labels : numpy.ndarray
        Integer labels, shape ``(n_samples,)``.
    metadata : dict
        Name, shapes, preprocessing steps and class counts, for ``run_status.json``.
    """

    features: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def dataset_dir(cfg: dict[str, Any]) -> Path:
    """Return ``<data_root>/distributed_secure_kernels``.

    ``cfg["data_root"]`` (``--data-root``) wins, then ``DATA_DIR``. A relative root
    is resolved against this paper folder, so the default ``"../../data"`` is the
    repository-level data root both under the shared runtime and in tests.
    """

    return resolve_data_root(cfg.get("data_root"), PROJECT_DIR) / PAPER_NAME


def load_raw(name: str, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one real dataset as ``(features, labels)`` before any preprocessing.

    Parameters
    ----------
    name : str
        One of ``REAL_DATASETS``.
    root : pathlib.Path
        The paper data directory, ``<data_root>/distributed_secure_kernels``.

    Returns
    -------
    tuple of numpy.ndarray
        Float features and integer labels.

    Raises
    ------
    FileNotFoundError
        If the dataset file is missing.
    ValueError
        If ``name`` is not a known real dataset.
    """

    if name == "wine":
        # scikit-learn's wine_data.csv: header row "178,13,class_0,...", then
        # 13 features and the class index per row.
        table = np.loadtxt(root / "wine" / "wine_data.csv", delimiter=",", skiprows=1)
        return table[:, :-1], table[:, -1].astype(int)
    if name == "digits":
        table = np.loadtxt(root / "digits" / "digits.csv.gz", delimiter=",")
        return table[:, :-1], table[:, -1].astype(int)
    if name == "parkinsons":
        frame = pd.read_csv(root / "parkinsons" / "parkinsons.data")
        frame = frame.drop(columns=["name"])
        labels = frame.pop("status").to_numpy(dtype=int)
        return frame.to_numpy(dtype=float), labels
    if name == "framingham":
        # Authors' heart.py: drop 'education', drop rows with any NaN, last column is the target.
        frame = pd.read_csv(root / "framingham" / "framingham.csv")
        frame = frame.drop(columns=["education"]).dropna()
        labels = frame.iloc[:, -1].to_numpy(dtype=int)
        return frame.iloc[:, :-1].to_numpy(dtype=float), labels
    raise ValueError(f"Unknown real dataset {name!r}; expected one of {REAL_DATASETS}")


def random_oversample(
    features: np.ndarray,
    labels: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate random minority samples until minority / majority reaches ``ratio``.

    Re-implements ``imblearn.over_sampling.RandomOverSampler(sampling_strategy=ratio)``
    for binary labels: ``int(n_majority * ratio - n_minority)`` minority rows are
    drawn with replacement and appended after the original rows.

    Parameters
    ----------
    features, labels : numpy.ndarray
        Binary-labelled data.
    ratio : float
        Target minority / majority ratio (the authors use 0.6).
    rng : numpy.random.Generator
        Source of the bootstrap indices.

    Returns
    -------
    tuple of numpy.ndarray
        Oversampled features and labels.

    Raises
    ------
    ValueError
        If the labels are not binary or the minority class already reaches ``ratio``.
    """

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Oversampling expects binary labels, got classes {classes}")
    minority = classes[np.argmin(counts)]
    n_minority, n_majority = counts.min(), counts.max()
    n_added = int(n_majority * ratio - n_minority)
    if n_added < 0:
        raise ValueError(
            f"Minority ratio {n_minority / n_majority:.3f} already exceeds {ratio}"
        )
    minority_rows = np.flatnonzero(labels == minority)
    added = rng.choice(minority_rows, size=n_added, replace=True)
    rows = np.concatenate([np.arange(len(labels)), added])
    return features[rows], labels[rows]


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(labels, return_counts=True)
    return {str(int(c)): int(n) for c, n in zip(classes, counts)}


def build_dataset(
    cfg: dict[str, Any], split: str = "train", rng: np.random.Generator | None = None
) -> DatasetSplit:
    """Load and preprocess one split following ``cfg["dataset"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration. Uses ``cfg["dataset"]`` keys ``name``,
        ``oversample_ratio`` (float or null), ``subset_size`` (fraction or count of
        the authors' stratified split), ``scaling`` (``"standard"``) and ``protocol`` (``"paper"`` or ``"corrected"``).
    split : str
        ``"train"`` (the subset the paper evaluates by cross-validation) or
        ``"heldout"`` (the part the authors discard). Default value is ``"train"``.
    rng : numpy.random.Generator | None
        Generator for oversampling. If omitted, one is seeded
        from ``cfg["seed"]``. Default value is None.

    Returns
    -------
    DatasetSplit
        Features, labels and metadata.

    Raises
    ------
    KeyError
        If a required dataset key is missing.
    ValueError
        If the split, protocol or scaling is unknown.
    """

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    dataset = cfg["dataset"]
    name = dataset["name"]
    if rng is None:
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), 0]))
    protocol = dataset["protocol"]
    if protocol not in ("paper", "corrected"):
        raise ValueError(f"protocol must be 'paper' or 'corrected', got {protocol!r}")
    root = dataset_dir(cfg)
    features, labels = load_raw(name, root)
    steps = [f"read {name} from {root.name}/{name}"]
    raw_shape = list(features.shape)

    ratio = dataset["oversample_ratio"]
    if ratio is not None and protocol == "paper":
        # Paper-faithful: oversampling before the split and before CV, so
        # duplicates of one minority sample can land in train and validation folds.
        features, labels = random_oversample(features, labels, float(ratio), rng)
        steps.append(f"random minority oversampling to ratio {ratio} (whole dataset)")

    subset_size = dataset["subset_size"]
    subset_size = int(subset_size) if subset_size >= 1 else float(subset_size)
    train_x, heldout_x, train_y, heldout_y = train_test_split(
        features,
        labels,
        train_size=subset_size,
        stratify=labels,
        random_state=AUTHORS_SPLIT_STATE,
    )
    steps.append(
        f"stratified train_test_split(train_size={subset_size}, random_state={AUTHORS_SPLIT_STATE})"
    )
    chosen_x, chosen_y = (
        (train_x, train_y) if split == "train" else (heldout_x, heldout_y)
    )

    scaling = dataset["scaling"]
    if scaling == "standard":
        if protocol == "paper":
            # Authors fit the scaler on the evaluated subset itself, before CV.
            chosen_x = StandardScaler().fit_transform(chosen_x)
            steps.append("StandardScaler fitted on the subset (before CV)")
        else:
            steps.append("StandardScaler fitted inside each training fold")
    else:
        raise ValueError(f"scaling must be 'standard', got {scaling!r}")

    metadata = {
        "name": name,
        "kind": "real",
        "raw_shape": raw_shape,
        "shape": list(chosen_x.shape),
        "split": split,
        "protocol": protocol,
        "preprocessing": steps,
        "class_counts": _class_counts(chosen_y),
        "oversample_ratio": ratio,
    }
    return DatasetSplit(
        features=np.asarray(chosen_x, dtype=float), labels=chosen_y, metadata=metadata
    )


def build_dataloader(
    cfg: dict[str, Any], split: str = "train"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the split as a single full batch ``[(features, labels)]``.

    Kernel methods use every sample at once, so the "loader" is one batch. The
    interface matches the reproduction workflow's one-batch check.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    split : str
        ``"train"`` or ``"heldout"``. Default value is ``"train"``.

    Returns
    -------
    list of tuple
        One ``(features, labels)`` batch.
    """

    data = build_dataset(cfg, split)
    return [(data.features, data.labels)]
