"""Flatten study records into per-seed rows and aggregate them over seeds.

Row identity is ``(component, quantity, embedding, n, train_size, protocol)``,
matching :func:`lib.paper_values.load_paper_values` keys (without protocol) so
paper values attach programmatically. Dataset (C) values are converted from
accuracy to classification error, as the paper plots error.

Paired quantum-minus-classical differences are computed per seed before
aggregation (``*_minus_best_classical``), so their intervals reflect the
paired design.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import stats

QUANTUM_FAMILY_PREFIXES = ("quantum_kernel", "projected_quantum_kernel")


def _task_rows(
    base: dict, component: str, task_record: dict, as_error: bool
) -> list[dict]:
    rows = []
    for protocol, summary in task_record["protocols"].items():

        def value(score: float) -> float:
            return 1.0 - score if as_error else score

        family_values = {
            family: value(entry["test_score"])
            for family, entry in summary["families"].items()
        }
        best = value(summary["best_classical"]["test_score"])
        entries = dict(family_values)
        entries["best_classical"] = best
        if "best_classical_with_extension" in summary:
            entries["best_classical_with_extension"] = value(
                summary["best_classical_with_extension"]["test_score"]
            )
        for family, family_value in family_values.items():
            if family.startswith(QUANTUM_FAMILY_PREFIXES):
                entries[f"{family}_minus_best_classical"] = family_value - best
        for quantity, quantity_value in entries.items():
            rows.append(
                {
                    **base,
                    "component": component,
                    "quantity": quantity,
                    "train_size": task_record["train_size"],
                    "protocol": protocol,
                    "value": quantity_value,
                }
            )
    return rows


def record_rows(record: dict) -> list[dict]:
    """Per-seed rows of one study or training-size record."""
    base = {
        "seed": record["seed"],
        "n": record["n"],
        "embedding": record.get("embedding"),
    }
    rows = []
    geometry_record = record.get("dimension_geometry")
    if geometry_record:
        for kernel_name, d_value in geometry_record["d"].items():
            rows.append(
                {
                    **base,
                    "component": "dimension_geometry",
                    "quantity": f"d_{kernel_name}",
                    "train_size": None,
                    "protocol": None,
                    "value": d_value,
                }
            )
        for kernel_name, g_record in geometry_record["g"].items():
            rows.append(
                {
                    **base,
                    "component": "dimension_geometry",
                    "quantity": f"g_{kernel_name}",
                    "train_size": None,
                    "protocol": None,
                    "value": g_record["g"],
                }
            )
        for kernel_name, g_record in geometry_record.get(
            "g_with_extension", {}
        ).items():
            rows.append(
                {
                    **base,
                    "component": "dimension_geometry",
                    "quantity": f"g_{kernel_name}_with_extension",
                    "train_size": None,
                    "protocol": None,
                    "value": g_record["g"],
                }
            )
    if "quantum_dataset" in record:
        rows += _task_rows(
            base, "quantum_dataset", record["quantum_dataset"], as_error=False
        )
    if "bound" in record:
        rows.append(
            {
                **base,
                "component": "bound",
                "quantity": "prediction_error_bound",
                "train_size": record["bound"]["train_size"],
                "protocol": None,
                "value": record["bound"]["bound"],
            }
        )
    for component in ("engineered_pqk", "engineered_qk"):
        component_record = record.get(component)
        if not component_record or "skipped" in component_record:
            continue
        for task_record in component_record["by_train_size"].values():
            rows += _task_rows(base, component, task_record, as_error=False)
    if "classical_dataset" in record:
        rows += _task_rows(
            base, "classical_dataset", record["classical_dataset"], as_error=True
        )
    if record.get("component") == "training_size":
        for task_record in record["by_train_size"].values():
            rows += _task_rows(base, "training_size", task_record, as_error=False)
    return rows


def aggregate_rows(rows: list[dict], paper_values: dict | None = None) -> list[dict]:
    """Mean, sample std, count and 95% t-interval per row identity.

    Parameters
    ----------
    rows : list of dict
        Per-seed rows from :func:`record_rows`.
    paper_values : dict or None
        Output of :func:`lib.paper_values.load_paper_values`; attached as
        ``paper_value`` when the identity matches. Default value is None.
    """
    groups: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (
            row["component"],
            row["quantity"],
            row["embedding"],
            row["n"],
            row["train_size"],
            row["protocol"],
        )
        groups[key].append((row["seed"], row["value"]))
    summary = []
    for key in sorted(
        groups, key=lambda k: tuple("" if v is None else str(v).zfill(6) for v in k)
    ):
        seeds, values = zip(*groups[key])
        values = np.asarray(values, dtype=float)
        count = values.size
        std = float(values.std(ddof=1)) if count > 1 else math.nan
        half_width = (
            float(stats.t.ppf(0.975, count - 1) * std / math.sqrt(count))
            if count > 1
            else math.nan
        )
        component, quantity, embedding, n, train_size, protocol = key
        entry = {
            "component": component,
            "quantity": quantity,
            "embedding": embedding,
            "n": n,
            "train_size": train_size,
            "protocol": protocol,
            "mean": float(values.mean()),
            "std": std,
            "ci95_half_width": half_width,
            "count": int(count),
            "seeds": sorted(seeds),
        }
        if paper_values is not None:
            entry["paper_value"] = paper_values.get(
                (component, quantity, embedding, n, train_size)
            )
        summary.append(entry)
    return summary


__all__ = ["aggregate_rows", "record_rows"]
