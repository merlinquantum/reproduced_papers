"""Shared helpers for aggregating sweep outputs into mean ± std summaries.

Given a sweep's ``aggregated.json`` (or any list of per-run ``metrics.json``
dicts), produce a flat table keyed by ``(point_name, temperature)`` with
``mean`` and ``std`` arrays across seeds for each tracked metric.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

METRICS_TO_TRACK = [
    "originality_L2",
    "originality_L10",
    "broken_rate_2",
    "broken_rate_3",
]


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")
    return float(value)


def extract_row(stats: dict) -> dict[str, float]:
    """Pull the four headline numbers from a single ``QRC_T=...`` metrics block."""
    originality = stats.get("originality", {})
    broken = stats.get("broken_rate_per_rule", {})
    return {
        "originality_L2": _safe_float(originality.get("2")),
        "originality_L10": _safe_float(originality.get("10")),
        "broken_rate_2": _safe_float(broken.get("2")),
        "broken_rate_3": _safe_float(broken.get("3")),
    }


def aggregate_point(
    per_seed_metrics: list[dict],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate across seeds and temperatures for one sweep point.

    Returns
    -------
    dict
        ``{temperature_str: {metric_name: {"mean": float, "std": float, "n": int}}}``.
    """
    by_T: dict[str, list[dict[str, float]]] = {}
    for metrics in per_seed_metrics:
        qrc_block = metrics.get("qrc", {})
        for label, stats in qrc_block.items():
            # label looks like "QRC_T=1.0".
            if not label.startswith("QRC_T="):
                continue
            temp = label[len("QRC_T=") :]
            row = extract_row(stats)
            by_T.setdefault(temp, []).append(row)

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for temp, rows in by_T.items():
        per_metric: dict[str, dict[str, float]] = {}
        for metric in METRICS_TO_TRACK:
            values = [r[metric] for r in rows if not math.isnan(r[metric])]
            if not values:
                per_metric[metric] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                continue
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            per_metric[metric] = {"mean": mean, "std": std, "n": len(values)}
        aggregated[temp] = per_metric
    return aggregated


def baseline_summary(per_seed_metrics: list[dict]) -> dict:
    """Extract Markov / uncorrelated baseline values from the first seed.

    Baselines are deterministic given a global seed, but they vary slightly
    between seeds because the baseline-generation RNG is seeded by the same
    global seed. We average across seeds.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("markov", "uncorrelated"):
        rows: list[dict[str, float]] = []
        for metrics in per_seed_metrics:
            stats = metrics.get("baselines", {}).get(name)
            if not stats:
                continue
            rows.append(extract_row(stats))
        if not rows:
            continue
        per_metric: dict[str, dict[str, float]] = {}
        for metric in METRICS_TO_TRACK:
            values = [r[metric] for r in rows if not math.isnan(r[metric])]
            if not values:
                per_metric[metric] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                continue
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            per_metric[metric] = {"mean": mean, "std": std, "n": len(values)}
        out[name] = per_metric
    return out


def load_aggregated(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
