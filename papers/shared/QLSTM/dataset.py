"""Shared dataset generators for QLSTM.

Most generator implementations live in `papers.shared.time_series.generators`.
This module keeps the QLSTM-specific CSV resolution logic (shared data root
conventions) and re-exports the same `data` factory API.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    from papers.shared.time_series import generators as ts
    from runtime_lib.data_paths import paper_data_dir
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.shared.time_series import generators as ts
    from runtime_lib.data_paths import paper_data_dir


def _csv_shared_root() -> Path:
    # Canonical shared location for generic time-series assets.
    return paper_data_dir("time_series")


def _csv_shared_root_legacy() -> Path:
    # Backwards compatibility with the previous convention.
    return paper_data_dir("QLSTM", ensure_exists=False)


def _default_csv_path() -> Path:
    shared_root = _csv_shared_root()
    shared_root.mkdir(parents=True, exist_ok=True)
    shared_path = shared_root / "airline-passengers.csv"

    legacy_root = _csv_shared_root_legacy()
    legacy_path = legacy_root / "airline-passengers.csv"
    local_path = (
        Path(__file__).resolve().parents[3]
        / "papers"
        / "QLSTM"
        / "data"
        / "airline-passengers.csv"
    )

    if not shared_path.exists() and legacy_path.exists():
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, shared_path)

    if not shared_path.exists() and local_path.exists():
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, shared_path)

    if not shared_path.exists():
        raise FileNotFoundError(
            f"CSV asset not found at {shared_path}; provide --csv-path or place the file there"
        )

    return shared_path


def _resolve_csv_path(path: str | Path | None) -> Path:
    if path:
        p = Path(path)
        if p.is_absolute():
            return p
        resolved = _csv_shared_root() / p
        return resolved
    return _default_csv_path()


class DataFactory:
    @classmethod
    def get(cls, name: str, **kw):
        if name == "csv":
            path = _resolve_csv_path(kw.get("path"))
            return ts.CSVSeriesGenerator(path)
        return ts.DataFactory.get(name, **kw)

    @classmethod
    def list(cls):
        return ts.DataFactory.list()


data = DataFactory


__all__ = [
    "BaseGenerator",
    "CSVSeriesGenerator",
    "DataFactory",
    "data",
    "SinGenerator",
    "DampedSHMGenerator",
    "CosGenerator",
    "LinearGenerator",
    "ExponentialGenerator",
    "BesselJ2Generator",
    "PopulationInversionGenerator",
    "PopulationInversionCollapseRevivalGenerator",
    "LogSineGenerator",
    "MovingAverageNoiseGenerator",
]

# Backwards-compatible re-exports
BaseGenerator = ts.BaseGenerator
CSVSeriesGenerator = ts.CSVSeriesGenerator
SinGenerator = ts.SinGenerator
DampedSHMGenerator = ts.DampedSHMGenerator
CosGenerator = ts.CosGenerator
LinearGenerator = ts.LinearGenerator
ExponentialGenerator = ts.ExponentialGenerator
BesselJ2Generator = ts.BesselJ2Generator
PopulationInversionGenerator = ts.PopulationInversionGenerator
PopulationInversionCollapseRevivalGenerator = (
    ts.PopulationInversionCollapseRevivalGenerator
)
LogSineGenerator = ts.LogSineGenerator
MovingAverageNoiseGenerator = ts.MovingAverageNoiseGenerator
