"""Aggregate test metrics of repeated runs (seeds) from their ``metrics.json`` files."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def collect_test_metrics(metrics_paths: dict[int, Path]) -> pd.DataFrame:
    """Long table with one row per (seed, model, metric) from full-precision JSON values.

    Parameters
    ----------
    metrics_paths : dict[int, pathlib.Path]
        ``metrics.json`` path per seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``seed``, ``model``, ``metric``, ``value``; non-numeric entries
        (candidate names) are kept in a ``label`` column instead.
    """
    rows = []
    for seed, path in sorted(metrics_paths.items()):
        metrics = json.loads(Path(path).read_text())
        for model, values in metrics["test"].items():
            for metric, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    rows.append(
                        {
                            "seed": seed,
                            "model": model,
                            "metric": metric,
                            "value": math.nan,
                            "label": str(value),
                        }
                    )
                else:
                    rows.append(
                        {
                            "seed": seed,
                            "model": model,
                            "metric": metric,
                            "value": float(value),
                            "label": None,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate(long_table: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample standard deviation, min, max and count per (model, metric) over seeds.

    Raises
    ------
    ValueError
        If any included numeric metric is non-finite, so a broken run cannot
        silently enter a mean.
    """
    numeric = long_table[long_table["label"].isna()]
    non_finite = numeric[~np.isfinite(numeric["value"])]
    if not non_finite.empty:
        raise ValueError(
            f"Non-finite metrics in sweep results:\n{non_finite.to_string()}"
        )
    grouped = numeric.groupby(["model", "metric"], sort=False)["value"]
    return grouped.agg(
        mean="mean",
        std=lambda values: values.std(ddof=1),
        min="min",
        max="max",
        n="count",
    ).reset_index()


def selected_candidates(long_table: pd.DataFrame) -> pd.DataFrame:
    """Which candidate each seed selected, per selection key (e.g. ``leak_free:qnn``)."""
    labels = long_table[long_table["metric"] == "candidate"]
    return labels.pivot(index="model", columns="seed", values="label")
