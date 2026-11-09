"""Dataset loading for the QARIMA reproduction.

Each loader returns a 1-D ``numpy.float64`` array (the univariate series) plus a
small metadata dict.  Train/OOS splitting is handled by :func:`split_series` so
the same protocol is used for every dataset.

Data files live under ``<data_root>/QARIMA/raw`` (default ``data`` at the repo
root) and are staged lazily on first use: Sunspots is generated offline from the
bundled ``statsmodels`` dataset (no network needed); CO2/AusBeer/Woolyarn/Sydney
are downloaded once from Rdatasets/NOAA and then cached on disk.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

_RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv"
_NOAA = "https://www.ncei.noaa.gov/data/global-hourly/access/2024"
_DOWNLOAD_URLS = {
    "ausbeer.csv": f"{_RDATASETS}/fpp2/ausbeer.csv",
    "woolyrnq.csv": f"{_RDATASETS}/forecast/woolyrnq.csv",
    "co2_R_datasets.csv": f"{_RDATASETS}/timeSeriesDataSets/co2_ts.csv",
    "sydney_observatory_hill_2024.csv": f"{_NOAA}/94768099999.csv",
}


def _raw_dir(data_root: str | os.PathLike | None) -> Path:
    root = Path(data_root) if data_root else Path(os.environ.get("DATA_DIR", "data"))
    raw = root / "QARIMA" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def _ensure_sunspots(raw: Path) -> Path:
    """Materialize sunspots.csv offline from the bundled statsmodels dataset."""
    dest = raw / "sunspots.csv"
    if not dest.exists():
        import statsmodels.api as sm

        sm.datasets.sunspots.load_pandas().data.to_csv(dest, index=False)
    return dest


def _ensure_downloaded(raw: Path, filename: str) -> Path:
    """Download ``filename`` into ``raw`` on first use, then reuse the cached copy.

    Downloads to a temporary sibling file and atomically replaces the cache only
    after successful completion, preventing partial files from being treated as
    valid cache entries.
    """
    dest = raw / filename
    if not dest.exists() or dest.stat().st_size == 0:
        # Download to a temporary file in the same directory for atomic replacement.
        temp_file = raw / (filename + ".tmp")
        try:
            urllib.request.urlretrieve(_DOWNLOAD_URLS[filename], temp_file)  # noqa: S310
            # Atomic replace: only move if download succeeded and temp file is nonempty.
            if temp_file.stat().st_size > 0:
                temp_file.replace(dest)
            else:
                raise RuntimeError(f"Downloaded file {temp_file} is empty")
        except Exception:
            # Clean up temp file on failure so it doesn't interfere with retries.
            if temp_file.exists():
                temp_file.unlink()
            raise
    return dest


def _parse_noaa_tmp(value: str) -> float:
    """Parse a NOAA Global-Hourly ``TMP`` field (e.g. ``+0250,1`` -> 25.0 C)."""
    try:
        raw, _quality = str(value).split(",")
        tenths = int(raw)
    except (ValueError, AttributeError):
        return np.nan
    return np.nan if tenths == 9999 else tenths / 10.0


def load_sunspots(raw: Path) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(_ensure_sunspots(raw))
    y = df["SUNACTIVITY"].to_numpy(dtype=np.float64)
    return y, {"name": "sunspots", "freq": "annual", "unit": "sunspot number"}


def load_co2(raw: Path) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(_ensure_downloaded(raw, "co2_R_datasets.csv"))
    y = df["value"].to_numpy(dtype=np.float64)
    return y, {"name": "co2", "freq": "monthly", "season": 12, "unit": "ppm"}


def load_ausbeer(raw: Path) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(_ensure_downloaded(raw, "ausbeer.csv"))
    y = df["value"].to_numpy(dtype=np.float64)
    y = y[~np.isnan(y)]
    return y, {"name": "ausbeer", "freq": "quarterly", "season": 4, "unit": "ML"}


def load_woolyarn(raw: Path) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(_ensure_downloaded(raw, "woolyrnq.csv"))
    y = df["value"].to_numpy(dtype=np.float64)
    return y, {"name": "woolyarn", "freq": "quarterly", "season": 4, "unit": "tonnes"}


def load_sydney(raw: Path) -> tuple[np.ndarray, dict]:
    """Sydney 2024 daily-mean temperature from NOAA Observatory Hill (substitute).

    The paper's station 95768099999 (North Head) has no valid temperature data, so
    we use the iconic Sydney reference station 94768099999 (Observatory Hill) and
    build a daily-mean temperature series for calendar 2024.  Labeled as a
    substitute-station reproduction.
    """
    df = pd.read_csv(
        _ensure_downloaded(raw, "sydney_observatory_hill_2024.csv"), low_memory=False
    )
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["tempC"] = df["TMP"].map(_parse_noaa_tmp)
    daily = df.set_index("DATE")["tempC"].resample("D").mean()
    daily = daily.interpolate(limit=3).dropna()
    y = daily.to_numpy(dtype=np.float64)
    return y, {
        "name": "sydney",
        "freq": "daily",
        "season": 7,
        "unit": "deg C",
        "substitute_station": "94768099999 Sydney Observatory Hill",
        "paper_station": "95768099999 North Head (no TMP data)",
    }


_LOADERS = {
    "sunspots": load_sunspots,
    "co2": load_co2,
    "ausbeer": load_ausbeer,
    "woolyarn": load_woolyarn,
    "sydney": load_sydney,
}

# Train / OOS split (number of held-out OOS points) per dataset, following the paper.
_OOS = {
    "sunspots": 128,  # paper: 181 train / 128 OOS
    "co2": 120,  # paper: 348 train / 120 OOS
    "ausbeer": 8,  # paper: last 8 quarters
    "woolyarn": 55,  # paper: 64 train / 55 OOS
    "sydney": 90,  # substitute: last ~quarter (paper split not reproducible)
}


def load_series(name: str, data_root: str | os.PathLike | None = None):
    if name not in _LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(_LOADERS)}")
    raw = _raw_dir(data_root)
    y, meta = _LOADERS[name](raw)
    meta["n_total"] = int(y.size)
    meta["oos"] = int(_OOS[name])
    return y, meta


def split_series(y: np.ndarray, n_oos: int) -> tuple[np.ndarray, np.ndarray]:
    """Split into (train, oos) with the last ``n_oos`` points held out."""
    if n_oos >= y.size:
        raise ValueError(f"n_oos={n_oos} >= series length {y.size}")
    return y[:-n_oos].copy(), y[-n_oos:].copy()
