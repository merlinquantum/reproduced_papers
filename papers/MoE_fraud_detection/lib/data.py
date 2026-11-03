"""Data loading, preprocessing, and CV-split generation for the MoE fraud
detection reproduction.

Implements the paper's evaluation protocol (Section 3.3-ish, as summarized in
LOG.md):

1. Repeated stratified 5-fold CV over the full ULB credit-card dataset.
2. For each fold: the sklearn "train" indices become the model-training pool;
   the sklearn "test" indices become a held-out pool that is further split
   50/25/25 (stratified) into validation / analysis / holdout.
3. A ``MinMaxScaler`` is fit on the training pool only (to avoid leakage) and
   applied to all four splits.
4. The training pool is downsampled to a 50/50 class balance (all fraud rows
   kept, non-fraud rows randomly subsampled). Validation/analysis/holdout are
   left at natural class balance so evaluation reflects real-world skew.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from runtime_lib.data_paths import paper_data_dir

PAPER_NAME = "MoE_fraud_detection"
DATASET_FILENAME = "creditcard.csv"
LABEL_COLUMN = "Class"
N_SPLITS = 5


def resolve_dataset_path(cfg: dict[str, Any] | None = None) -> Any:
    """Return the path to ``creditcard.csv`` under the shared data root."""
    cfg = cfg or {}
    data_dir = paper_data_dir(PAPER_NAME, data_root=cfg.get("data_root"))
    return data_dir / DATASET_FILENAME


def load_raw_data(
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the raw creditcard.csv into feature/label numpy arrays.

    Drops a ``Time`` column if present (defensive - the mirror used for this
    reproduction does not include one). Returns ``(X, y, feature_names)``.
    """
    csv_path = resolve_dataset_path(cfg)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"creditcard.csv not found at {csv_path}. See LOG.md 'Data "
            "Acquisition Log' for how to (re)download it."
        )
    df = pd.read_csv(csv_path)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COLUMN}' in {csv_path}")

    y = df[LABEL_COLUMN].to_numpy().astype(np.int64)
    feature_df = df.drop(columns=[LABEL_COLUMN])
    feature_names = list(feature_df.columns)
    X = feature_df.to_numpy().astype(np.float64)
    return X, y, feature_names


def _balance_downsample(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the majority (non-fraud) class to match the fraud count.

    Keeps every fraud row; randomly subsamples non-fraud rows (without
    replacement) using the supplied seeded generator. Returns a shuffled
    ``(X_bal, y_bal)`` pair.
    """
    fraud_idx = np.flatnonzero(y == 1)
    nonfraud_idx = np.flatnonzero(y == 0)
    if len(fraud_idx) > len(nonfraud_idx):
        raise ValueError(
            "Fraud rows outnumber non-fraud rows; unexpected for this dataset"
        )
    chosen_nonfraud = rng.choice(nonfraud_idx, size=len(fraud_idx), replace=False)
    combined_idx = np.concatenate([fraud_idx, chosen_nonfraud])
    rng.shuffle(combined_idx)
    return X[combined_idx], y[combined_idx]


def generate_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_repeats: int,
    n_splits: int = N_SPLITS,
    max_splits: int | None = None,
) -> list[dict[str, Any]]:
    """Generate repeated-stratified-5-fold CV splits with the paper's
    train / validation / analysis / holdout structure.

    Parameters
    ----------
    X, y : np.ndarray
        Full feature matrix and label vector.
    seed : int
        Base random seed. Derived, per-split seeds are used for the
        50/25/25 stratified splitting and the balancing downsample so that
        splits are reproducible but not correlated with each other.
    n_repeats : int
        Number of CV repeats (paper default: 3).
    n_splits : int
        Number of stratified folds per repeat (paper: 5).
    max_splits : int, optional
        If given, stop after this many (repeat, fold) splits — a debug knob
        for fast smoke iteration, not part of the paper's protocol.

    Returns
    -------
    list of dict
        One dict per (repeat, fold) split with keys: ``repeat``, ``fold``,
        ``X_train``, ``y_train`` (balanced 50/50), ``X_val``, ``y_val``,
        ``X_analysis``, ``y_analysis``, ``X_holdout``, ``y_holdout``,
        ``scaler`` (fitted MinMaxScaler).
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    splits: list[dict[str, Any]] = []
    for split_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        if max_splits is not None and split_idx >= max_splits:
            break
        repeat_idx, fold_idx = divmod(split_idx, n_splits)

        X_train_pool, y_train_pool = X[train_idx], y[train_idx]
        X_test_pool, y_test_pool = X[test_idx], y[test_idx]

        # Per-split derived seeds keep the 50/25/25 split and the balancing
        # downsample reproducible without correlating adjacent splits.
        split_seed = int(seed) * 1000 + split_idx
        X_val, X_remain, y_val, y_remain = train_test_split(
            X_test_pool,
            y_test_pool,
            test_size=0.5,
            stratify=y_test_pool,
            random_state=split_seed,
        )
        X_analysis, X_holdout, y_analysis, y_holdout = train_test_split(
            X_remain,
            y_remain,
            test_size=0.5,
            stratify=y_remain,
            random_state=split_seed + 1,
        )

        scaler = MinMaxScaler()
        scaler.fit(X_train_pool)
        X_train_pool_s = scaler.transform(X_train_pool)
        X_val_s = scaler.transform(X_val)
        X_analysis_s = scaler.transform(X_analysis)
        X_holdout_s = scaler.transform(X_holdout)

        rng = np.random.default_rng(split_seed + 2)
        X_train_bal, y_train_bal = _balance_downsample(
            X_train_pool_s, y_train_pool, rng
        )

        splits.append(
            {
                "repeat": repeat_idx,
                "fold": fold_idx,
                "X_train": X_train_bal,
                "y_train": y_train_bal,
                "X_val": X_val_s,
                "y_val": y_val,
                "X_analysis": X_analysis_s,
                "y_analysis": y_analysis,
                "X_holdout": X_holdout_s,
                "y_holdout": y_holdout,
                "scaler": scaler,
            }
        )
    return splits


def load_cv_splits(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Config-driven wrapper: load creditcard.csv and generate CV splits."""
    X, y, _feature_names = load_raw_data(cfg)
    seed = int(cfg.get("seed", 42))
    cv_cfg = cfg.get("cv", {})
    n_repeats = int(cv_cfg.get("n_repeats", 3))
    max_splits = cv_cfg.get("max_splits")
    if max_splits is not None:
        max_splits = int(max_splits)
    return generate_cv_splits(
        X, y, seed=seed, n_repeats=n_repeats, max_splits=max_splits
    )


__all__ = [
    "PAPER_NAME",
    "DATASET_FILENAME",
    "LABEL_COLUMN",
    "N_SPLITS",
    "resolve_dataset_path",
    "load_raw_data",
    "generate_cv_splits",
    "load_cv_splits",
]
