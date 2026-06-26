"""Data loading and preprocessing for QCNN-ID.

Methodological safeguards:
  raw CSVs -> concat -> shuffle -> drop class column -> object-to-numeric ->
  fill NaN -> train/test split -> PCA fitted on train only.

Two feature views are produced from the same train/test split:

* quantum branch:
    PCA(train only) -> train min-max statistics -> [0, 1] -> [0, pi].
    Test data uses the train min/max and is clipped to [0, 1] before being
    mapped to [0, pi].

* classical branch:
    PCA(train only) -> StandardScaler fitted on train PCA features only.
    No multiplication by pi is applied to the classical branch.

No learned preprocessing step, including PCA, StandardScaler, or min-max
normalisation, is fitted before the train/test split.

The paper uses the **IoT Healthcare Security Dataset** (Faisal Malik) released
at <https://github.com/imfaisalmalik/IoT-Healthcare-Security-Dataset>, also
mirrored on Kaggle as
<https://www.kaggle.com/datasets/faisalmalik/iot-healthcare-security-dataset>.
The raw archive (``ICUDatasetProcessed.zip``) contains three CSVs:

* ``Attack.csv`` -- attack traffic (label = 1),
* ``environmentMonitoring.csv`` -- environment-sensor traffic (label = 0),
* ``patientMonitoring.csv`` -- patient-sensor traffic (label = 0).

This module reads all three from ``data_dir`` (relative to the shared
``data/`` root), concatenates them, and follows the released-notebook recipe
(cell 0): shuffle with ``random_state=seed``, drop ``class``/``label`` from the
feature matrix, coerce object columns (hex strings, MAC traces) to numeric,
fill NaN with 0, then PCA.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PAPER_NAME = "QCNN_ID"
ICU_CSV_NAMES = (
    "Attack.csv",
    "environmentMonitoring.csv",
    "patientMonitoring.csv",
)
LABEL_COLUMN = "label"
TEXT_CLASS_COLUMN = "class"


@dataclass
class PreparedData:
    X_train_quantum: np.ndarray
    X_train_classical: np.ndarray
    X_test_quantum: np.ndarray
    X_test_classical: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_features_full: int
    n_components: int
    class_balance: dict[int, float]
    cumulated_var: list[float]
    n_train: int
    n_test: int


def _resolve_data_root(cfg: dict[str, Any]) -> Path:
    """Resolve the shared ``data/`` root.

    Precedence: ``cfg['data_root']`` (set by the shared runtime) > ``DATA_DIR``
    env var > the repository ``data/`` directory inferred from this file.
    """

    cfg_root = cfg.get("data_root")
    if cfg_root:
        return Path(cfg_root).expanduser().resolve()
    env_root = os.environ.get("DATA_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path(__file__).resolve().parents[3] / "data").resolve()


def _maybe_join(root: Path, item: str | os.PathLike[str]) -> Path:
    raw = Path(item).expanduser()
    if raw.is_absolute():
        return raw
    return (root / raw).resolve()


def _require_files(files: Iterable[Path]) -> None:
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Dataset file(s) not found:\n  - "
            + joined
            + "\nDownload the IoT Healthcare Security Dataset from "
            "https://github.com/imfaisalmalik/IoT-Healthcare-Security-Dataset "
            "and place Attack.csv, environmentMonitoring.csv, "
            "patientMonitoring.csv under data/QCNN_ID/."
        )


def _resolve_dataset_paths(cfg: dict[str, Any]) -> list[Path]:
    """Return the list of CSVs to concatenate.

    Resolution order:
    1. ``cfg['data_files']`` -- explicit list of file names or paths.
    2. ``cfg['data_dir']`` -- directory; load every ``ICU_CSV_NAMES`` entry.
    3. ``cfg['data_csv']`` -- single file (legacy single-CSV layout).
    """

    data_root = _resolve_data_root(cfg)

    if cfg.get("data_files"):
        files = [_maybe_join(data_root, item) for item in cfg["data_files"]]
        _require_files(files)
        return files

    if cfg.get("data_dir"):
        directory = _maybe_join(data_root, cfg["data_dir"])
        files = [directory / name for name in ICU_CSV_NAMES]
        _require_files(files)
        return files

    if cfg.get("data_csv"):
        single = _maybe_join(data_root, cfg["data_csv"])
        _require_files([single])
        return [single]

    raise KeyError(
        "Config must set one of 'data_files', 'data_dir', or 'data_csv'. "
        "For the ICU Healthcare dataset use data_dir='QCNN_ID'."
    )


def _load_and_concat(csv_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


def _preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Recipe from the released notebook (cell 0).

    Drops the text ``class`` column, casts the binary ``label`` column to int,
    coerces object columns (hex strings, MAC traces, etc.) to numeric via
    ``pd.to_numeric(errors='coerce')``, then fills the resulting NaNs with 0.
    Returns the cleaned feature frame and the integer label array.

    The function also accepts the capitalised ``Label`` spelling so tests
    that fabricate a synthetic dataset can use either convention.
    """

    label_col = LABEL_COLUMN if LABEL_COLUMN in df.columns else "Label"
    if label_col not in df.columns:
        raise KeyError(
            f"Expected a 'label' (or 'Label') column. "
            f"Got columns: {list(df.columns)[:8]}..."
        )
    y = df[label_col].astype(int).to_numpy()
    drop_cols = [c for c in (TEXT_CLASS_COLUMN, "Class", label_col) if c in df.columns]
    feats = df.drop(columns=drop_cols)
    for col in feats.select_dtypes(include=["object"]).columns:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")
    feats = feats.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return feats, y


def load_and_prepare(cfg: dict[str, Any], seed: int) -> PreparedData:
    """Run the preprocessing pipeline and return train/test arrays.

    The same ``seed`` is used for every stochastic step to ensure full
    reproducibility.

    The train/test split is performed before any learned preprocessing step.
    PCA is fitted on the training set only, then applied to the test set.

    Returns
    -------
    PreparedData
        ``X_*_q`` contains PCA-reduced quantum features angle-encoded in
        ``[0, pi]`` using train-only min-max statistics.

        ``X_*_classical`` contains PCA-reduced classical features normalised
        with a ``StandardScaler`` fitted on the training PCA features only.

        Test features are transformed using training-set preprocessing
        statistics only.
    """

    csv_paths = _resolve_dataset_paths(cfg)
    df = _load_and_concat(csv_paths)

    # Deterministic shuffle to remove concatenation-order artefacts while
    # keeping full experimental reproducibility.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    subset_size = int(cfg.get("subset_size", 0) or 0)
    if subset_size > 0 and subset_size < len(df):
        df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)

    feats, y = _preprocess(df)

    X = feats.to_numpy(dtype=np.float64)
    n_features_full = X.shape[1]

    # Split before any learned preprocessing step to avoid test-to-train
    # leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["test_size"]),
        random_state=seed,
        stratify=y,
    )

    # StandardScaler fitted on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  #

    n_components = int(cfg["n_components"])
    # PCA is fitted on the training set only. The test set is projected with
    # the PCA model learned from the training set.
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    cumulated_var = pca.explained_variance_ratio_.cumsum().tolist()

    # Quantum branch:
    # min-max statistics are learned on train PCA features only. Test values
    # are transformed with those train statistics and clipped to [0, 1] before
    # angle encoding to [0, pi].
    q_mins = X_train_pca.min(axis=0)
    q_maxs = X_train_pca.max(axis=0)
    q_spans = np.where((q_maxs - q_mins) > 1e-12, q_maxs - q_mins, 1.0)

    X_train_quantum = np.pi * (X_train_pca - q_mins) / q_spans
    X_test_quantum = np.pi * np.clip((X_test_pca - q_mins) / q_spans, 0.0, 1.0)

    # Classical branch:
    # StandardScaler is applied after PCA and is fitted on train PCA features only.
    classical_scaler = StandardScaler()
    X_train_classical = classical_scaler.fit_transform(X_train_pca)
    X_test_classical = classical_scaler.transform(X_test_pca)

    classes, counts = np.unique(y_train, return_counts=True)
    balance = {int(c): float(n / counts.sum()) for c, n in zip(classes, counts)}

    return PreparedData(
        X_train_quantum=X_train_quantum.astype(np.float32),
        X_train_classical=X_train_classical.astype(np.float32),
        X_test_quantum=X_test_quantum.astype(np.float32),
        X_test_classical=X_test_classical.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        n_features_full=n_features_full,
        n_components=n_components,
        class_balance=balance,
        cumulated_var=cumulated_var,
        n_train=len(y_train),
        n_test=len(y_test),
    )
