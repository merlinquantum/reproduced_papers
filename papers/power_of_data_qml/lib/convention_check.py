"""EXP-AMB: test inferred circuit conventions against the journal Source Data.

Each variant overrides part of the ``conventions``, ``kernels`` or
``engineered`` config blocks (e.g. the E2 ZZ coefficient, the E3 register, the
PQK geometry bandwidth). The study runs once per variant at small system sizes,
and every summary row with a paper value is scored by its relative error. The
variant with the smallest mean relative error per (component, quantity,
embedding) is the one most consistent with the paper; the selection rule is
fixed here, before any result is seen (LOG.md, Reproducibility Assessment).
"""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict
from pathlib import Path

from runtime_lib.config import deep_update

from . import embeddings, study, summary

LOGGER = logging.getLogger(__name__)

OVERRIDABLE_BLOCKS = ("conventions", "kernels", "engineered")


def variant_config(cfg: dict, variant: dict) -> dict:
    """Return ``cfg`` with one variant's overrides applied.

    Raises
    ------
    KeyError
        If the variant has no ``name`` or overrides an unknown block.
    """
    name = variant["name"]
    variant_cfg = copy.deepcopy(cfg)
    for block, overrides in variant.items():
        if block == "name":
            continue
        if block not in OVERRIDABLE_BLOCKS:
            raise KeyError(f"Variant {name}: cannot override block '{block}'")
        unknown = set(overrides) - set(variant_cfg[block])
        if unknown:
            raise KeyError(
                f"Variant {name}: unknown keys in '{block}': {sorted(unknown)}"
            )
        variant_cfg[block] = deep_update(variant_cfg[block], overrides)
    return variant_cfg


def relative_error_table(summary_rows_by_variant: dict[str, list[dict]]) -> list[dict]:
    """Mean relative error to the paper per variant and (component, quantity, embedding).

    Only rows with a paper value and protocol ``None`` or ``"paper"`` are scored.
    """
    table = []
    for variant_name, rows in summary_rows_by_variant.items():
        errors = defaultdict(list)
        for row in rows:
            paper = row.get("paper_value")
            if paper is None or row["protocol"] not in (None, "paper") or paper == 0.0:
                continue
            key = (row["component"], row["quantity"], row["embedding"])
            errors[key].append(abs(row["mean"] - paper) / abs(paper))
        for (component, quantity, embedding), values in sorted(errors.items(), key=str):
            table.append(
                {
                    "variant": variant_name,
                    "component": component,
                    "quantity": quantity,
                    "embedding": embedding,
                    "mean_relative_error": sum(values) / len(values),
                    "points": len(values),
                }
            )
    return table


def best_variants(table: list[dict]) -> list[dict]:
    """Variant with the smallest mean relative error per (component, quantity, embedding)."""
    best: dict[tuple, dict] = {}
    for row in table:
        key = (row["component"], row["quantity"], row["embedding"])
        incumbent = best.get(key)
        if (
            incumbent is None
            or row["mean_relative_error"] < incumbent["mean_relative_error"]
        ):
            best[key] = row
    return [
        {**row, "selected_variant": row["variant"]}
        for row in sorted(
            best.values(),
            key=lambda r: (r["component"], r["quantity"], str(r["embedding"])),
        )
    ]


def run_convention_check(
    cfg, pca_features, complex_dtype, run_dir: Path, paper_values, on_unit_completed
) -> dict:
    """Run the study for every variant and score each against the Source Data.

    Returns
    -------
    dict
        ``{"variants": {name: summary_rows}, "relative_errors": [...],
        "best_variants": [...], "records_files": {name: path}}``.
    """
    variants = cfg["convention_check"]["variants"]
    if not variants:
        raise ValueError("convention_check.variants is empty")
    summaries, records_files = {}, {}
    total = len(variants)
    for index, variant in enumerate(variants, start=1):
        name = variant["name"]
        variant_cfg = variant_config(cfg, variant)
        settings = study.StudySettings.from_config(variant_cfg)
        conventions = embeddings.EmbeddingConventions.from_config(
            variant_cfg["conventions"]
        )
        records_path = run_dir / f"records_{name}.jsonl"
        LOGGER.info(
            "Convention variant %d/%d started | name=%s | overrides=%s",
            index,
            total,
            name,
            {k: v for k, v in variant.items() if k != "name"},
        )

        def unit_callback(done, units, record, _name=name):
            on_unit_completed(done, units, record, variant=_name)

        records = study.run_study(
            pca_features,
            settings,
            conventions,
            complex_dtype,
            records_path,
            unit_callback,
        )
        rows = [row for record in records for row in summary.record_rows(record)]
        summaries[name] = summary.aggregate_rows(rows, paper_values)
        records_files[name] = records_path.name
    table = relative_error_table(summaries)
    for row in table:
        if not math.isfinite(row["mean_relative_error"]):
            raise ValueError(f"Non-finite relative error for {row}")
    return {
        "variants": summaries,
        "relative_errors": table,
        "best_variants": best_variants(table),
        "records_files": records_files,
    }


__all__ = [
    "best_variants",
    "relative_error_table",
    "run_convention_check",
    "variant_config",
]
