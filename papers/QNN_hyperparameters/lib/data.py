"""Datasets, splits and preprocessing, following the author code.

The pipeline mirrors ``src/data/load_data.py``, ``experiments/*_param_opt.py``
and ``src/util/opt.py`` of https://github.com/sabrinaherbst/hyperparameters_qnn
(commit 53bc59a):

1. 60/20/20 stratified train/validation/test split, ``random_state=5``
   (test 0.20 first, then validation 0.25 of the rest), written to CSV and
   read back, as the author scripts do.
2. KDD Cup only: one-hot encode ``protocol_type``, ``service`` and ``flag``,
   align validation and test columns to train, drop one feature of every pair
   with absolute correlation in (0.9, 1).
3. PCA (7 components) on standardized features, or LDA (``min(7, n_classes-1)``
   components) on raw features, both fit on the full training split.
   ``dataset.pca_signs = "u_based"`` restores the component signs of the
   author's scikit-learn 1.2.2; the default keeps the installed version's.
4. Subsample 400 training rows and 250 validation rows with
   ``DataFrame.sample(n, random_state=12)`` (not stratified). The author
   reports metrics on these 250 validation rows; we also draw 250 test rows
   the same way, which the author code loads but never uses.
5. Optional MinMax scaling to [0, 1], fit on the 400 training rows. In the
   author code this happens only in the ``beta``, ``beta_mu`` and
   ``normal_beta`` initialization arms.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

PAPER_DATA_DIRNAME = "QNN_hyperparameters"
DATASETS = ("rice", "glass", "cover_type", "kdd_cup")
KDD_CATEGORICAL = ["protocol_type", "service", "flag"]


@dataclass
class PreparedData:
    """Model-ready arrays for one dataset/preprocessing/scaling choice.

    Attributes
    ----------
    X : dict
        ``{"train", "val", "test"}`` arrays of shape ``(n_rows, n_features)``.
    y : dict
        Matching label arrays (original label values).
    n_features : int
        Number of features after PCA/LDA, which is also the qubit count.
    identity : dict
        Dataset name, split sizes, preprocessing and subset description for
        ``run_status.json``.
    """

    X: dict
    y: dict
    n_features: int
    identity: dict


def resolve_paper_data_dir(cfg: dict) -> Path:
    """Return ``<data_root>/QNN_hyperparameters``.

    ``cfg["data_root"]`` is set by the shared runtime from ``--data-root`` or
    the ``DATA_DIR`` environment variable.
    """

    data_root = cfg.get("data_root") or os.environ.get("DATA_DIR")
    if not data_root:
        raise ValueError("data_root is not set; pass --data-root or set DATA_DIR")
    path = Path(data_root) / PAPER_DATA_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(f"paper data directory not found: {path}")
    return path


def _csv_round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """Write and re-read a frame as CSV, so dtypes match the author pipeline."""

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


def _author_split(X, y, columns: list[str]) -> dict[str, pd.DataFrame]:
    """60/20/20 stratified split with ``random_state=5`` (author ``load_data.py``)."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=5, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=5, stratify=y_train
    )
    return {
        split: _csv_round_trip(
            pd.DataFrame(np.c_[features, labels], columns=columns + ["target"])
        )
        for split, features, labels in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]
    }


def _load_raw_splits(name: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load one dataset from the acquired files and split it like the author."""

    if name == "rice":
        features = pd.read_csv(data_dir / "rice" / "rice_features.csv")
        targets = pd.read_csv(data_dir / "rice" / "rice_targets.csv")["Class"]
        labels = LabelEncoder().fit_transform(targets.to_numpy().ravel())
        return _author_split(features, labels, features.columns.tolist())
    if name == "glass":
        features = pd.read_csv(
            data_dir / "glass_identification" / "glass_identification_features.csv"
        )
        targets = pd.read_csv(
            data_dir / "glass_identification" / "glass_identification_targets.csv"
        )
        return _author_split(features, targets, features.columns.tolist())

    from sklearn.datasets import fetch_covtype, fetch_kddcup99

    sklearn_home = str(data_dir / "sklearn_data_home")
    if name == "cover_type":
        bunch = fetch_covtype(
            data_home=sklearn_home, shuffle=False, download_if_missing=False
        )
        return _author_split(bunch.data, bunch.target, list(bunch.feature_names))
    if name == "kdd_cup":
        bunch = fetch_kddcup99(
            data_home=sklearn_home, shuffle=False, download_if_missing=False
        )
        frame = pd.DataFrame(
            np.c_[bunch.data, bunch.target],
            columns=list(bunch.feature_names) + ["target"],
        )
        for column in ["target", *KDD_CATEGORICAL]:
            frame[column] = frame[column].str.decode("ascii")
        numeric = [c for c in bunch.feature_names if c not in KDD_CATEGORICAL]
        frame[numeric] = frame[numeric].apply(pd.to_numeric)
        return _author_split(
            frame.loc[:, frame.columns != "target"],
            frame["target"],
            list(bunch.feature_names),
        )
    raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")


def _kdd_feature_engineering(
    splits: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """One-hot, column alignment and correlation drop from ``kdd_param_opt.py``."""

    encoded = {
        split: pd.get_dummies(frame, columns=KDD_CATEGORICAL)
        for split, frame in splits.items()
    }
    train = encoded["train"]
    for split in ("val", "test"):
        frame = encoded[split]
        for column in set(train.columns) - set(frame.columns):
            frame[column] = 0
        encoded[split] = frame[train.columns]
    correlations = (
        train.corr(numeric_only=True)
        .abs()
        .unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
    )
    correlations = correlations[(correlations > 0.9) & (correlations < 1)]
    to_drop = [index[0] for index in correlations.index]
    return {split: frame.drop(to_drop, axis=1) for split, frame in encoded.items()}


def load_splits(name: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    """Return the author train/val/test frames, cached under ``<data_dir>/splits``.

    Parameters
    ----------
    name : str
        One of ``rice``, ``glass``, ``cover_type``, ``kdd_cup``.
    data_dir : pathlib.Path
        ``<data_root>/QNN_hyperparameters``.

    Returns
    -------
    dict
        ``{"train", "val", "test"}`` frames with a ``target`` column.
    """

    cache = data_dir / "splits"
    paths = {split: cache / f"{name}_{split}.csv" for split in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        splits = {split: pd.read_csv(path) for split, path in paths.items()}
    else:
        logger.info("DATASET_SPLIT_BUILD | dataset=%s | cache=%s", name, cache)
        splits = _load_raw_splits(name, data_dir)
        cache.mkdir(parents=True, exist_ok=True)
        for split, frame in splits.items():
            frame.to_csv(paths[split], index=False)
    if name == "kdd_cup":
        splits = _kdd_feature_engineering(splits)
    return splits


def prepare_data(cfg: dict) -> PreparedData:
    """Build the reduced, subsampled and optionally scaled arrays for a run.

    Parameters
    ----------
    cfg : dict
        Resolved run config; reads ``data_root`` and the ``dataset`` block
        (``name``, ``preprocessing``, ``n_components``, ``n_train``,
        ``n_eval``, ``subsample_seed``, ``input_scaling``, ``pca_signs``).

    Returns
    -------
    PreparedData
    """

    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]
    preprocessing = dataset_cfg["preprocessing"]
    scaling = dataset_cfg["input_scaling"]
    if preprocessing not in ("pca", "lda"):
        raise ValueError(f"preprocessing must be pca or lda, got {preprocessing!r}")
    if scaling not in ("raw", "minmax"):
        raise ValueError(f"input_scaling must be raw or minmax, got {scaling!r}")

    splits = load_splits(name, resolve_paper_data_dir(cfg))
    features = {split: frame.drop("target", axis=1) for split, frame in splits.items()}
    labels = {split: frame["target"] for split, frame in splits.items()}

    n_components = int(dataset_cfg["n_components"])
    if preprocessing == "pca":
        standardizer = StandardScaler().fit(features["train"])
        standardized = standardizer.transform(features["train"])
        reducer = PCA(n_components=n_components).fit(standardized)
        if dataset_cfg["pca_signs"] == "u_based":
            # scikit-learn < 1.5 (the author ran 1.2.2) fixed each component's
            # sign so that its largest-magnitude training score is positive;
            # 1.5 switched to the largest loading. Under MinMax scaling a flip
            # turns x into 1 - x, so the encoded inputs differ.
            scores = reducer.transform(standardized)
            largest = scores[np.argmax(np.abs(scores), axis=0), range(scores.shape[1])]
            reducer.components_ *= np.sign(largest)[:, None]
        elif dataset_cfg["pca_signs"] != "installed":
            raise ValueError("pca_signs must be installed or u_based")

        def reduce(frame):
            return reducer.transform(standardizer.transform(frame))
    else:
        n_classes = labels["train"].nunique()
        if n_classes < 8:
            n_components = n_classes - 1
        reducer = LinearDiscriminantAnalysis(n_components=n_components).fit(
            features["train"], labels["train"]
        )

        def reduce(frame):
            return reducer.transform(frame)

    subsample_seed = int(dataset_cfg["subsample_seed"])
    sizes = {
        "train": int(dataset_cfg["n_train"]),
        "val": int(dataset_cfg["n_eval"]),
        "test": int(dataset_cfg["n_eval"]),
    }
    X, y = {}, {}
    for split, size in sizes.items():
        size = min(size, len(features[split]))
        # Same call on features and labels with the same seed keeps rows aligned.
        rows = features[split].sample(size, random_state=subsample_seed)
        X[split] = np.asarray(reduce(rows), dtype=float)
        y[split] = labels[split].sample(size, random_state=subsample_seed).to_numpy()

    if scaling == "minmax":
        scaler = MinMaxScaler().fit(X["train"])
        X = {split: scaler.transform(values) for split, values in X.items()}

    identity = {
        "name": name,
        "splits": {split: int(len(values)) for split, values in X.items()},
        "preprocessing": f"{preprocessing}({X['train'].shape[1]})+{scaling}",
        "subset": (
            f"author split random_state=5; DataFrame.sample(n, random_state={subsample_seed}) "
            f"of train/val/test; full split sizes {[len(features[s]) for s in ('train', 'val', 'test')]}"
        ),
    }
    return PreparedData(X=X, y=y, n_features=X["train"].shape[1], identity=identity)


def build_dataloader(cfg: dict, split: str):
    """Return the split as a single full batch ``[(X, y)]``.

    QNN training in the paper is full-batch (the optimizers evaluate the loss
    on all 400 training rows), so a one-element list is the whole loader.

    Parameters
    ----------
    cfg : dict
        Resolved run config.
    split : str
        ``train``, ``val`` or ``test``.

    Returns
    -------
    list
        ``[(X, y)]`` with ``X`` of shape ``(n_rows, n_features)``.
    """

    data = prepare_data(cfg)
    return [(data.X[split], data.y[split])]
