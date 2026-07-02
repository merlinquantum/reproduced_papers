"""Dataset loading and preprocessing for EGAS reproduction.

Pipeline (Appendix A): pick a binary task from each dataset, reduce features to ``n`` (=8) via
PCA, rescale reduced features to [0, 2*pi], and evaluate over non-overlapping data slices of
400 train + 50 test samples (10 repeats, slice start shifted across repeats).

Datasets are public UCI sets fetched via ``ucimlrepo`` and cached as CSV under
``data/generative_quantum_embeddings/``.

ASSUMPTIONS (documented in LOG.md; paper only says "two classes with sufficient points"):
* binary task = the two most populous classes (mapped to {-1,+1}), unless the set is already
  binary.
* per-feature min-max rescaling of the PCA components to [0, 2*pi].
* PCA + scaler are fit on the full available pool (paper says "fixed PCA pipeline").
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# UCI repository ids and the rule for choosing the two classes.
DATASETS = {
    "PW": {"uci_id": 327, "classes": "binary"},  # Phishing Websites (-1/1)
    "WDGV1": {"uci_id": 107, "classes": "top2"},  # Waveform DB Generator v1 (3 classes)
    "DB": {"uci_id": 602, "classes": "top2"},  # Dry Bean (7 classes)
    "WQ": {"uci_id": 186, "classes": "top2"},  # Wine Quality (quality score)
    "WC": {"uci_id": 186, "classes": "wine_color"},  # Wine Color (red vs white)
    "MGT": {"uci_id": 159, "classes": "binary"},  # MAGIC Gamma Telescope (g/h)
    "EGSSD": {
        "uci_id": 471,
        "classes": "binary",
    },  # Electrical Grid Stability (stable/unstable)
}

TWO_PI = 2 * math.pi


def _cache_dir(data_root: str) -> Path:
    d = Path(data_root) / "generative_quantum_embeddings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_raw(name: str, data_root: str):
    """Return (features DataFrame, target Series) for a dataset, with CSV caching."""
    cache = _cache_dir(data_root)
    spec = DATASETS[name]
    fx = cache / f"{name}_X.csv"
    fy = cache / f"{name}_y.csv"
    if fx.exists() and fy.exists():
        return pd.read_csv(fx), pd.read_csv(fy).iloc[:, 0]

    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=spec["uci_id"])
    X = repo.data.features.reset_index(drop=True)
    targets = repo.data.targets.reset_index(drop=True)

    if name == "WC":
        # Wine Quality (id 186) carries the red/white label in a separate 'color' column
        # (role "Other"); pull it from the full original frame.
        y = repo.data.original.reset_index(drop=True)["color"]
    elif targets.shape[1] > 1:
        # Multiple target columns (e.g. EGSSD has continuous 'stab' + categorical 'stabf'):
        # pick the categorical label column (object dtype, else fewest unique values).
        obj_cols = [c for c in targets.columns if targets[c].dtype == object]
        col = (
            obj_cols[0]
            if obj_cols
            else min(targets.columns, key=lambda c: targets[c].nunique())
        )
        y = targets[col]
    else:
        y = targets.iloc[:, 0]
    X.to_csv(fx, index=False)
    y.to_frame(name="target").to_csv(fy, index=False)
    return X, y


def _select_binary(X: pd.DataFrame, y: pd.Series, rule: str):
    """Reduce to a binary task per `rule`; return (X_np, y_pm1) with y in {-1,+1}."""
    y = y.copy()
    if rule == "wine_color":
        pos = "red"
        mask = y.isin(["red", "white"])
        Xs, ys = X[mask], y[mask]
        labels = (ys == pos).astype(int).values
    else:
        counts = y.value_counts()
        if rule == "binary":
            chosen = counts.index[:2].tolist()
            if len(counts) != 2:
                chosen = counts.index[:2].tolist()
        elif rule == "top2":
            chosen = counts.index[:2].tolist()
        else:
            raise ValueError(rule)
        mask = y.isin(chosen)
        Xs, ys = X[mask], y[mask]
        labels = (ys == chosen[0]).astype(int).values
    X_np = Xs.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    # impute any NaNs with column means (rare; some UCI sets have a few)
    col_mean = np.nanmean(X_np, axis=0)
    inds = np.where(np.isnan(X_np))
    X_np[inds] = np.take(col_mean, inds[1])
    y_pm1 = np.where(labels == 1, 1, -1).astype(int)
    return X_np, y_pm1


def load_dataset(
    name: str,
    data_root: str = "data",
    n_components: int = 8,
    seed: int = 0,
    max_pool: int = 6000,
):
    """Load a dataset, reduce to `n_components` via PCA, rescale to [0, 2*pi].

    Returns (X (M, n_components) in [0,2pi], y (M,) in {-1,+1}).
    """
    X_raw, y_raw = _fetch_raw(name, data_root)
    X_np, y = _select_binary(X_raw, y_raw, DATASETS[name]["classes"])

    # subsample a balanced pool for tractability and class balance
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == -1)[0]
    per = min(len(idx_pos), len(idx_neg), max_pool // 2)
    sel = np.concatenate(
        [
            rng.choice(idx_pos, per, replace=False),
            rng.choice(idx_neg, per, replace=False),
        ]
    )
    rng.shuffle(sel)
    X_np, y = X_np[sel], y[sel]

    n_comp = min(n_components, X_np.shape[1])
    X_std = StandardScaler().fit_transform(X_np)
    X_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(X_std)
    X_scaled = MinMaxScaler(feature_range=(0.0, TWO_PI)).fit_transform(X_pca)
    if n_comp < n_components:  # pad if a dataset has < n features
        pad = np.zeros((X_scaled.shape[0], n_components - n_comp))
        X_scaled = np.concatenate([X_scaled, pad], axis=1)
    return X_scaled.astype(np.float64), y.astype(int)


def make_slices(
    X, y, n_train: int = 400, n_test: int = 50, n_repeats: int = 10, seed: int = 0
):
    """Non-overlapping (train, test) slices; slice start shifted across repeats.

    Yields dicts with X_train, y_train, X_test, y_test.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    Xs, ys = X[order], y[order]
    block = n_train + n_test
    slices = []
    for r in range(n_repeats):
        start = r * block
        if start + block > len(Xs):
            # wrap around if dataset is small
            start = start % max(1, len(Xs) - block)
        tr = slice(start, start + n_train)
        te = slice(start + n_train, start + block)
        slices.append(
            {
                "X_train": Xs[tr],
                "y_train": ys[tr],
                "X_test": Xs[te],
                "y_test": ys[te],
            }
        )
    return slices
