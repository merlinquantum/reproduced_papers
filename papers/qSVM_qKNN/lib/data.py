"""Data loading and preprocessing for the Sodar qSVM/qKNN reproduction.

The paper evaluates binary encrypted-traffic classification on CIC
ISCX-URL2016 and ISCX-IDS2012. This module keeps the preprocessing shared by
the gate-based and photonic-reservoir variants:

* clean rows and labels,
* split before learned preprocessing,
* fit categorical encoders and scalers on the training split only,
* expose a StandardScaler view for classical baselines, a train-only MinMax
  view for PennyLane encodings, and a phase-scaled view for MerLin reservoirs.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PAPER_NAME = "qSVM_qKNN"
URL_DEFAULT_FILE = Path("ISCX-URL-2016") / "Defacement_Infogain.csv"
IDS_DEFAULT_ARCHIVE = Path("ISCX-IDS-2012") / "iscxids2012-master.tar.gz"
IDS_CSV_PREFIX = "iscxids2012-master/data/CSV/"
IDS_DEFAULT_CSV_FILES = (
    "TestbedMonJun14Flows.csv",
    "TestbedSatJun12Flows.csv",
    "TestbedSunJun13Flows.csv",
    "TestbedThuJun17Flows.csv",
    "TestbedTueJun15Flows.csv",
    "TestbedWedJun16Flows.csv",
)
IDS_STRUCTURAL_COLUMNS = {
    "generated",
    "sourcePayloadAsBase64",
    "sourcePayloadAsUTF",
    "destinationPayloadAsBase64",
    "destinationPayloadAsUTF",
    "source",
    "destination",
    "startDateTime",
    "stopDateTime",
}
LABEL_CANDIDATES = ("class", "URL_Type_obf_Type", "Label", "label")
PHOTONIC_BASED_MODELS = {
    "photonic_hybrid_svm_angle",
    "photonic_hybrid_svm_amplitude",
    "photonic_fidelity_svm_angle",
    "photonic_state_svm_angle",
    "photonic_hybrid_knn_angle",
    "photonic_hybrid_knn_amplitude",
    "photonic_fidelity_knn_angle",
    "photonic_state_knn_angle",
}
SYNTHETIC_DATA_SOURCE = "synthetic"
DEFAULT_SYNTHETIC_ROWS = 96
DEFAULT_SYNTHETIC_FEATURES = 8


@dataclass
class PreparedData:
    """Prepared train/test arrays for all implemented model families.

    Parameters
    ----------
    X_train_classical : np.ndarray
        Standard-scaled training features for classical SVM/KNN baselines.
    X_test_classical : np.ndarray
        Standard-scaled test features.
    X_train_quantum : np.ndarray
        MinMax-scaled training features in the configured quantum input range.
        Angle-encoded models multiply this view by ``quantum_angle_scale`` at
        model-dispatch time; ZZFeatureMap models use ``zz_angle_scale`` instead.
        Amplitude encoding keeps this view and lets PennyLane apply the required
        L2 state normalisation.
    X_test_quantum : np.ndarray
        MinMax-scaled test features clipped to the train-fitted range.
    X_train_photonic : np.ndarray
        Train-fitted MinMax-prepared inputs for the photonic models. Angle
        reservoirs multiply this view by pi inside the MerLin encoder;
        amplitude reservoirs pad and L2-normalize this same bounded view into a
        Fock-basis state vector.
    X_test_photonic : np.ndarray
        Photonic reservoir inputs for the test split.
    y_train : np.ndarray
        Binary training labels, with benign/normal mapped to 0 and attack-like
        traffic mapped to 1.
    y_test : np.ndarray
        Binary test labels.
    train_indices : np.ndarray
        Original row indices retained in the training split.
    test_indices : np.ndarray
        Original row indices retained in the test split.
    dataset : str
        Dataset identifier from the resolved config.
    feature_names : list[str]
        Selected feature names after cleanup and optional feature limiting.
    photonic_feature_names : list[str]
        Feature names supplied to the photonic reservoir, plus padding names
        when the configured photonic basis requires extra zero columns.
    photonic_preprocessing : dict
        Metadata for the photonic train-only preprocessing.
    class_counts_clean : dict[int, int]
        Class counts after cleanup and optional pre-split subset, before
        train/test split and optional balancing.
    class_balance_clean : dict[int, float]
        Class proportions after cleanup and optional pre-split subset.
    class_balance : dict[int, float]
        Training class proportions after optional balancing.
    n_rows_raw : int
        Number of rows before cleanup.
    n_rows_clean : int
        Number of rows after cleanup and optional pre-split subset.
    n_features_raw : int
        Number of cleaned features before optional feature limiting.
    source : str
        Dataset path or archive members used for this run.
    """

    X_train_classical: np.ndarray
    X_test_classical: np.ndarray
    X_train_quantum: np.ndarray
    X_test_quantum: np.ndarray
    X_train_photonic: np.ndarray
    X_test_photonic: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    dataset: str
    feature_names: list[str]
    photonic_feature_names: list[str]
    photonic_preprocessing: dict[str, Any]
    class_counts_clean: dict[int, int]
    class_balance_clean: dict[int, float]
    class_balance: dict[int, float]
    n_rows_raw: int
    n_rows_clean: int
    n_features_raw: int
    source: str


def _resolve_data_root(cfg: dict[str, Any]) -> Path:
    """Resolve the shared repository data root.

    Parameters
    ----------
    cfg : dict
        Resolved runtime configuration.

    Returns
    -------
    pathlib.Path
        Data root. Defaults to the repository-level ``data/`` directory.
    """

    cfg_root = cfg.get("data_root")
    if cfg_root:
        return Path(cfg_root).expanduser().resolve()
    env_root = os.environ.get("DATA_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path(__file__).resolve().parents[3] / "data").resolve()


def _paper_data_dir(cfg: dict[str, Any]) -> Path:
    data_dir = Path(str(cfg.get("data_dir", PAPER_NAME))).expanduser()
    if data_dir.is_absolute():
        return data_dir
    return (_resolve_data_root(cfg) / data_dir).resolve()


def _find_label_column(df: pd.DataFrame, configured: str | None = None) -> str:
    if configured:
        if configured not in df.columns:
            raise KeyError(f"Configured label column {configured!r} is absent.")
        return configured
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Could not find a label column. Expected one of "
        f"{LABEL_CANDIDATES}; got {list(df.columns)[:12]}..."
    )


def _binary_labels(labels: pd.Series) -> np.ndarray:
    """Map CIC text labels to binary integers."""

    if pd.api.types.is_numeric_dtype(labels):
        values = labels.to_numpy()
        unique = set(pd.Series(values).dropna().astype(float).unique())
        if unique <= {0.0, 1.0}:
            return values.astype(int)
    normalized = labels.astype(str).str.strip().str.lower()
    benign = {"benign", "normal", "0", "0.0"}
    return np.where(normalized.isin(benign), 0, 1).astype(np.int64)


def _read_url2016(cfg: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    base = _paper_data_dir(cfg)
    rel = Path(str(cfg.get("dataset_file") or URL_DEFAULT_FILE))
    csv_path = rel if rel.is_absolute() else base / rel
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing ISCX-URL2016 CSV: {csv_path}. Place datasets under "
            "data/qSVM_qKNN/ISCX-URL-2016/ or override dataset_file/data_root."
        )
    return pd.read_csv(csv_path, low_memory=False), str(csv_path)


def _ids_member_name(filename: str) -> str:
    member_name = filename
    if "/" not in member_name:
        member_name = IDS_CSV_PREFIX + member_name
    return member_name


def _ids_cache_path(archive_path: Path, member_name: str) -> Path:
    target = (archive_path.parent / member_name).resolve()
    cache_root = archive_path.parent.resolve()
    if not target.is_relative_to(cache_root):
        raise ValueError(f"IDS2012 archive member escapes cache root: {member_name!r}.")
    return target


def _ensure_ids_csv_cache(
    archive_path: Path, member_names: tuple[str, ...]
) -> list[Path]:
    csv_paths = [
        _ids_cache_path(archive_path, member_name) for member_name in member_names
    ]
    missing = [
        (member_name, csv_path)
        for member_name, csv_path in zip(member_names, csv_paths)
        if not csv_path.exists()
    ]
    if not missing:
        return csv_paths
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing ISCX-IDS2012 archive: {archive_path}. Place the archive "
            "under data/qSVM_qKNN/ISCX-IDS-2012/ or keep extracted CSV files under "
            f"{archive_path.parent / IDS_CSV_PREFIX}."
        )

    with tarfile.open(archive_path) as archive:
        for member_name, csv_path in missing:
            try:
                member = archive.getmember(member_name)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Could not find {member_name!r} in {archive_path}."
                ) from exc
            if not member.isfile():
                raise FileNotFoundError(
                    f"IDS2012 member is not a file: {member_name!r}."
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Could not read {member_name!r}.")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = csv_path.with_name(f".{csv_path.name}.tmp")
            with extracted, tmp_path.open("wb") as out:
                shutil.copyfileobj(extracted, out)
            tmp_path.replace(csv_path)
    return csv_paths


def _read_ids2012(cfg: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    base = _paper_data_dir(cfg)
    archive_rel = Path(str(cfg.get("ids_archive") or IDS_DEFAULT_ARCHIVE))
    archive_path = archive_rel if archive_rel.is_absolute() else base / archive_rel
    selected = tuple(cfg.get("ids_csv_files") or IDS_DEFAULT_CSV_FILES)
    member_names = tuple(_ids_member_name(str(filename)) for filename in selected)
    csv_paths = _ensure_ids_csv_cache(archive_path, member_names)
    frames = [pd.read_csv(csv_path, low_memory=False) for csv_path in csv_paths]
    return pd.concat(frames, ignore_index=True), ",".join(
        str(path) for path in csv_paths
    )


def _uses_synthetic_data(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("data_source", "csv")).lower() == SYNTHETIC_DATA_SOURCE


def _make_synthetic_dataframe(cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Create a deterministic encrypted-traffic-like smoke dataset.

    Parameters
    ----------
    cfg : dict
        Resolved runtime configuration.
    seed : int
        Random seed used to generate the synthetic rows.

    Returns
    -------
    pandas.DataFrame
        Numeric feature table with a binary ``class`` label column.
    """

    requested_rows = int(cfg.get("subset_size", 0) or DEFAULT_SYNTHETIC_ROWS)
    n_rows = max(requested_rows, 32)
    n_features = max(
        int(cfg.get("synthetic_n_features", DEFAULT_SYNTHETIC_FEATURES)),
        int(cfg.get("feature_limit", 0) or 1),
    )

    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * ((n_rows + 1) // 2), dtype=np.int64)[:n_rows]
    rng.shuffle(y)

    X = rng.normal(loc=0.0, scale=1.0, size=(n_rows, n_features))
    signal_dims = min(4, n_features)
    signal = np.linspace(0.4, 1.1, signal_dims)
    X[:, :signal_dims] += (2 * y[:, None] - 1) * signal

    frame = pd.DataFrame(X, columns=[f"feature_{idx:02d}" for idx in range(n_features)])
    frame["protocol"] = np.where(y == 1, "tls_alert", "tls_normal")
    frame["class"] = np.where(y == 1, "attack", "benign")
    return frame


def _load_raw_dataframe(cfg: dict[str, Any], seed: int) -> tuple[pd.DataFrame, str]:
    if _uses_synthetic_data(cfg):
        return _make_synthetic_dataframe(cfg, seed), "synthetic:qSVM_qKNN"
    dataset = str(cfg["dataset"]).lower()
    if dataset in {"url2016", "iscx-url2016", "iscx_url2016"}:
        return _read_url2016(cfg)
    if dataset in {"ids2012", "iscx-ids2012", "iscx_ids2012"}:
        return _read_ids2012(cfg)
    raise ValueError(
        "dataset must be one of 'url2016' or 'ids2012', or set "
        "data_source='synthetic' for the file-free smoke dataset."
    )


def _drop_dataset_specific_columns(
    features: pd.DataFrame,
    dataset: str,
    missing_threshold: float,
) -> pd.DataFrame:
    if dataset != "ids2012":
        return features
    drop_cols = set(IDS_STRUCTURAL_COLUMNS)
    missing = features.isna().mean()
    drop_cols.update(missing[missing > missing_threshold].index.tolist())
    present = [col for col in features.columns if col in drop_cols]
    return features.drop(columns=present)


def _clean_dataframe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    dataset = str(cfg["dataset"]).lower()
    label_col = _find_label_column(df, cfg.get("label_column"))
    y = _binary_labels(df[label_col])
    features = df.drop(columns=[label_col])
    features = _drop_dataset_specific_columns(
        features,
        dataset,
        float(cfg.get("missing_column_threshold", 0.4)),
    )
    combined = features.copy()
    combined["__target__"] = y
    combined["__row_index__"] = np.arange(len(combined))
    combined = combined.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    if bool(cfg.get("drop_missing_rows", True)):
        combined = combined.dropna(axis=0)
    row_indices = combined.pop("__row_index__").to_numpy(dtype=np.int64)
    y_clean = combined.pop("__target__").to_numpy(dtype=np.int64)
    return combined.reset_index(drop=True), y_clean, row_indices, features.shape[1]


def _stratified_take(
    X: pd.DataFrame,
    y: np.ndarray,
    row_indices: np.ndarray,
    *,
    size: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if size <= 0 or size >= len(y):
        return X, y, row_indices
    selected, _ = train_test_split(
        np.arange(len(y)),
        train_size=size,
        random_state=seed,
        stratify=y,
    )
    selected = np.sort(selected)
    return (
        X.iloc[selected].reset_index(drop=True),
        y[selected],
        row_indices[selected],
    )


def _resolved_test_size(y: np.ndarray, cfg: dict[str, Any]) -> float | int:
    """Return the split test size after applying an optional absolute cap."""

    test_fraction = float(cfg["test_size"])
    max_test_size = cfg.get("max_test_size")
    if max_test_size is None or int(max_test_size) <= 0:
        return test_fraction

    n_samples = len(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    requested = int(np.ceil(test_fraction * n_samples))
    capped = min(requested, int(max_test_size))
    if bool(cfg.get("balance_classes", False)):
        expected_minority_test = int(np.floor(float(counts.min()) * test_fraction))
        balanced_cap = min(int(max_test_size), n_classes * expected_minority_test)
        per_class_target = max(1, balanced_cap // n_classes)
        class_fractions = counts / n_samples
        capped = max(
            n_classes,
            max(
                int(np.ceil(per_class_target / fraction))
                for fraction in class_fractions
            ),
        )
        capped = min(requested, capped)
    capped = max(capped, n_classes)
    if capped >= n_samples:
        return test_fraction
    return capped


def _balance_split(
    X: pd.DataFrame,
    y: np.ndarray,
    row_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels; got classes {classes.tolist()}.")
    n_each = int(counts.min())
    rng = np.random.default_rng(seed)
    selected_parts = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        selected_parts.append(rng.choice(cls_idx, size=n_each, replace=False))
    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    return (
        X.iloc[selected].reset_index(drop=True),
        y[selected],
        row_indices[selected],
    )


def _fit_transform_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=X_train.index)
    test_out = pd.DataFrame(index=X_test.index)
    for column in X_train.columns:
        train_col = X_train[column]
        test_col = X_test[column]
        numeric_train = pd.to_numeric(train_col, errors="coerce")
        numeric_test = pd.to_numeric(test_col, errors="coerce")
        if numeric_train.notna().all() and numeric_test.notna().all():
            train_out[column] = numeric_train.astype(float)
            test_out[column] = numeric_test.astype(float)
            continue
        train_text = train_col.astype(str).fillna("<MISSING>")
        categories = {
            value: idx for idx, value in enumerate(sorted(train_text.unique()))
        }
        train_out[column] = train_text.map(categories).astype(float)
        test_out[column] = (
            test_col.astype(str)
            .fillna("<MISSING>")
            .map(categories)
            .fillna(-1)
            .astype(float)
        )
    return train_out, test_out


def _select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    variances = X_train.var(axis=0)
    non_constant = variances[variances > 1e-12]
    if non_constant.empty:
        raise ValueError("All candidate features are constant after preprocessing.")
    ordered = non_constant.sort_values(ascending=False).index.tolist()
    if feature_limit is not None:
        if feature_limit <= 0:
            raise ValueError("feature_limit must be positive or null.")
        ordered = ordered[:feature_limit]
    return X_train[ordered], X_test[ordered], list(ordered)


def _class_balance(y: np.ndarray) -> dict[int, float]:
    counts = _class_counts(y)
    total = sum(counts.values())
    return {int(cls): float(count / total) for cls, count in counts.items()}


def _class_counts(y: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {int(cls): int(count) for cls, count in zip(classes, counts)}


def _fit_photonic_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Prepare train/test feature angles for the photonic reservoir.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Selected numeric training features.
    X_test : pandas.DataFrame
        Selected numeric test features.
    feature_names : list[str]
        Names of the selected source features.
    cfg : dict
        Resolved experiment configuration.

    Returns
    -------
    tuple
        Train inputs, test inputs, photonic feature names, and preprocessing
        metadata.

    Raises
    ------
    ValueError
        If the requested number of modes is incompatible with the selected
        features.
    """

    requested_modes = cfg.get("photonic_n_modes")
    n_modes = X_train.shape[1] if requested_modes is None else int(requested_modes)
    if n_modes <= 0:
        raise ValueError("photonic_n_modes must be positive.")

    train_values = X_train.to_numpy(dtype=np.float64)
    test_values = X_test.to_numpy(dtype=np.float64)

    def _pad_to_modes(
        train_matrix: np.ndarray,
        test_matrix: np.ndarray,
        names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str], int]:
        n_current = train_matrix.shape[1]
        if n_current == n_modes:
            return train_matrix, test_matrix, names, 0
        if n_current > n_modes:
            raise ValueError(
                f"Photonic preprocessing produced {n_current} columns for "
                f"{n_modes} modes."
            )
        pad_width = n_modes - n_current
        train_padded = np.pad(train_matrix, ((0, 0), (0, pad_width)))
        test_padded = np.pad(test_matrix, ((0, 0), (0, pad_width)))
        padded_names = names + [f"pad_{idx}" for idx in range(pad_width)]
        return train_padded, test_padded, padded_names, pad_width

    if train_values.shape[1] > n_modes:
        raise ValueError(
            "photonic_n_modes must be at least the selected feature count; "
            f"got {n_modes} modes for {train_values.shape[1]} features."
        )
    train_reduced, test_reduced, photonic_names, padded_modes = _pad_to_modes(
        train_values,
        test_values,
        list(feature_names),
    )

    input_range = cfg.get("photonic_input_range", [0.0, 1.0])
    if len(input_range) != 2:
        raise ValueError("photonic_input_range must contain [low, high].")
    range_low, range_high = float(input_range[0]), float(input_range[1])
    if range_low >= range_high:
        raise ValueError("photonic_input_range must be strictly increasing.")

    angle_scaler = MinMaxScaler(feature_range=(range_low, range_high))
    X_train_photonic = angle_scaler.fit_transform(train_reduced)
    X_test_photonic = angle_scaler.transform(test_reduced)
    X_train_photonic = np.clip(X_train_photonic, range_low, range_high)
    X_test_photonic = np.clip(X_test_photonic, range_low, range_high)

    metadata = {
        "n_modes": n_modes,
        "input_range": [range_low, range_high],
        "source_feature_names": list(feature_names),
        "padded_modes": padded_modes,
    }
    return (
        X_train_photonic.astype(np.float64),
        X_test_photonic.astype(np.float64),
        photonic_names,
        metadata,
    )


def _requires_photonic_inputs(cfg: dict[str, Any]) -> bool:
    models = set(cfg.get("models") or [])
    return bool(models & PHOTONIC_BASED_MODELS)


def load_and_prepare(cfg: dict[str, Any], seed: int) -> PreparedData:
    """Load the configured dataset and prepare model-specific feature views.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.
    seed : int
        Random seed used for subset selection, split, and balancing.

    Returns
    -------
    PreparedData
        Prepared arrays for classical and quantum model families.
    """

    raw_df, source = _load_raw_dataframe(cfg, seed)
    X, y, row_indices, n_features_raw = _clean_dataframe(raw_df, cfg)
    subset_size = int(cfg.get("subset_size", 0) or 0)
    X, y, row_indices = _stratified_take(
        X,
        y,
        row_indices,
        size=subset_size,
        seed=seed,
    )
    class_counts_clean = _class_counts(y)
    class_balance_clean = _class_balance(y)
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X,
        y,
        row_indices,
        test_size=_resolved_test_size(y, cfg),
        random_state=seed,
        stratify=y,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    if bool(cfg.get("balance_classes", False)):
        X_train, y_train, train_indices = _balance_split(
            X_train,
            y_train,
            train_indices,
            seed=seed,
        )
        X_test, y_test, test_indices = _balance_split(
            X_test,
            y_test,
            test_indices,
            seed=seed + 1,
        )
    X_train_numeric, X_test_numeric = _fit_transform_features(X_train, X_test)
    feature_limit = cfg.get("feature_limit")
    feature_limit = None if feature_limit is None else int(feature_limit)
    X_train_selected, X_test_selected, feature_names = _select_features(
        X_train_numeric,
        X_test_numeric,
        feature_limit,
    )
    standard_scaler = StandardScaler()
    X_train_classical = standard_scaler.fit_transform(X_train_selected)
    X_test_classical = standard_scaler.transform(X_test_selected)

    quantum_input_range = cfg.get("quantum_input_range", [0.0, 1.0])
    if len(quantum_input_range) != 2:
        raise ValueError("quantum_input_range must contain [low, high].")
    quantum_low, quantum_high = (
        float(quantum_input_range[0]),
        float(quantum_input_range[1]),
    )
    if quantum_low >= quantum_high:
        raise ValueError("quantum_input_range must be strictly increasing.")
    quantum_scaler = MinMaxScaler(feature_range=(quantum_low, quantum_high))
    X_train_quantum = quantum_scaler.fit_transform(X_train_selected)
    X_test_quantum = quantum_scaler.transform(X_test_selected)
    X_train_quantum = np.clip(X_train_quantum, quantum_low, quantum_high)
    X_test_quantum = np.clip(X_test_quantum, quantum_low, quantum_high)

    photonic_cfg = dict(cfg)
    photonic_active = _requires_photonic_inputs(cfg)
    if not photonic_active:
        photonic_cfg["photonic_n_modes"] = X_train_selected.shape[1]

    (
        X_train_photonic,
        X_test_photonic,
        photonic_feature_names,
        photonic_preprocessing,
    ) = _fit_photonic_inputs(
        X_train_selected,
        X_test_selected,
        feature_names,
        photonic_cfg,
    )
    photonic_preprocessing["active_for_requested_models"] = photonic_active

    return PreparedData(
        X_train_classical=X_train_classical.astype(np.float64),
        X_test_classical=X_test_classical.astype(np.float64),
        X_train_quantum=X_train_quantum.astype(np.float64),
        X_test_quantum=X_test_quantum.astype(np.float64),
        X_train_photonic=X_train_photonic,
        X_test_photonic=X_test_photonic,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        dataset=str(cfg["dataset"]),
        feature_names=feature_names,
        photonic_feature_names=photonic_feature_names,
        photonic_preprocessing=photonic_preprocessing,
        class_counts_clean=class_counts_clean,
        class_balance_clean=class_balance_clean,
        class_balance=_class_balance(y_train),
        n_rows_raw=len(raw_df),
        n_rows_clean=len(y),
        n_features_raw=n_features_raw,
        source=source,
    )
