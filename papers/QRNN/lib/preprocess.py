from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

LOGGER = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in dataset '{dataset_name}'")


SZEGED_COLUMNS = [
    "date",
    "min_temperature",
    "max_temperature",
    "avg_humidity",
    "avg_wind_speed",
    "avg_pressure",
]


def preprocess_szeged_weather(raw_csv: Path, output_csv: Path) -> Path:
    """Aggregate the Szeged weather dataset into daily summary statistics."""

    if output_csv.exists():
        try:
            existing_cols = pd.read_csv(output_csv, nrows=0).columns.tolist()
            if existing_cols == SZEGED_COLUMNS:
                LOGGER.info("Using cached preprocessed dataset at %s", output_csv)
                return output_csv
            LOGGER.info(
                "Rebuilding preprocessed dataset due to mismatched columns: %s",
                output_csv,
            )
        except Exception as exc:  # pragma: no cover - best effort caching
            LOGGER.info(
                "Rebuilding preprocessed dataset due to read error (%s): %s",
                exc,
                output_csv,
            )

    df = pd.read_csv(raw_csv)
    dataset_name = raw_csv.name
    required = [
        "Formatted Date",
        "Temperature (C)",
        "Humidity",
        "Wind Speed (km/h)",
        "Pressure (millibars)",
    ]
    _require_columns(df, required, dataset_name)

    parsed_dates = pd.to_datetime(df["Formatted Date"], errors="coerce", utc=True)
    if not is_datetime64_any_dtype(parsed_dates):
        raise ValueError("Could not parse 'Formatted Date' as datetimes")
    df["Formatted Date"] = parsed_dates
    numeric_cols = [
        "Temperature (C)",
        "Humidity",
        "Wind Speed (km/h)",
        "Pressure (millibars)",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Formatted Date"] + numeric_cols)
    df["date"] = df["Formatted Date"].dt.tz_convert(None).dt.date
    df["month"] = df["Formatted Date"].dt.month

    grouped = (
        df.groupby("date")
        .agg(
            {
                "Temperature (C)": ["min", "max"],
                "Humidity": "mean",
                "Wind Speed (km/h)": "mean",
                "Pressure (millibars)": "mean",
                "month": "first",
            }
        )
        .reset_index()
        .sort_values("date")
    )

    grouped.columns = [
        "date",
        "min_temperature",
        "max_temperature",
        "avg_humidity",
        "avg_wind_speed",
        "avg_pressure",
        "month",
    ]
    grouped["date"] = grouped["date"].astype(str)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)
    LOGGER.info("Preprocessed Szeged weather dataset -> %s", output_csv)
    return output_csv


PREPROCESSORS = {
    "szeged_weather": preprocess_szeged_weather,
}

__all__ = [
    "PREPROCESSORS",
    "preprocess_szeged_weather",
]
