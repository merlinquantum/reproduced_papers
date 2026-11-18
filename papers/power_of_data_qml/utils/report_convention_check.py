"""Write the complete candidate table of a convention-check (EXP-AMB) sweep.

Usage (from the paper folder)::

    python utils/report_convention_check.py \
        --run outdir/run_20261017-033107 --stage "A (attempt 2)" \
        --run outdir/run_20261016-163902 --stage "A2" \
        --output results/convention_check_candidates

``--run`` and ``--stage`` pair by position. Each row is a candidate variant, and
each cell is its mean relative error to the journal Source Data over the
checked system sizes (lower is better). The three right-most columns are the
predeclared selection sums, so the selected variant can be re-derived from the
table (LOG.md, "Sweep plan: EXP-AMB stage A2").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUANTITIES = (
    ("d_quantum_kernel", "d_QK"),
    ("g_quantum_kernel", "g_QK"),
    ("d_projected_quantum_kernel", "d_PQK"),
    ("g_projected_quantum_kernel", "g_PQK"),
)
EMBEDDINGS = ("E1", "E2", "E3")


def relative_error_sum(errors, variant, quantities, embeddings) -> float:
    """Sum the mean relative errors of one variant over quantities and embeddings."""
    return sum(
        errors[(variant, quantity, embedding)]
        for quantity in quantities
        for embedding in embeddings
    )


def variant_table(metrics: dict) -> tuple[list[str], list[dict]]:
    """Return the column headers and one row per variant."""
    errors = {
        (r["variant"], r["quantity"], r["embedding"]): r["mean_relative_error"]
        for r in metrics["relative_errors"]
    }
    columns = [
        (quantity, embedding, f"{label} {embedding}")
        for quantity, label in QUANTITIES
        for embedding in EMBEDDINGS
    ]
    rows = []
    for variant in metrics["variants"]:
        cells = {
            header: errors.get((variant, quantity, embedding))
            for quantity, embedding, header in columns
        }

        rows.append(
            {
                "variant": variant,
                "cells": cells,
                "scale_sum": relative_error_sum(
                    errors,
                    variant,
                    ("d_quantum_kernel", "g_quantum_kernel"),
                    ("E1", "E2"),
                ),
                "e3_sum": relative_error_sum(
                    errors, variant, ("d_quantum_kernel", "g_quantum_kernel"), ("E3",)
                ),
                "gamma_sum": relative_error_sum(
                    errors,
                    variant,
                    ("d_projected_quantum_kernel", "g_projected_quantum_kernel"),
                    EMBEDDINGS,
                ),
            }
        )
    return [header for _, _, header in columns], rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--stage", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.run) != len(args.stage):
        parser.error("--run and --stage must be given the same number of times")
    lines = [
        "# Convention-check candidates (EXP-AMB)",
        "",
        "Mean relative error to the journal Source Data, n = 3-8, seeds 0-2. Lower is better.",
        "Selection sums: scale = d_QK+g_QK over E1,E2; E3 = d_QK+g_QK over E3;",
        "gamma = d_PQK+g_PQK over E1-E3.",
        "",
    ]
    for run, stage in zip(args.run, args.stage):
        metrics = json.loads((run / "metrics.json").read_text())
        headers, rows = variant_table(metrics)
        lines += [
            f"## Stage {stage} (`{run.name}`, {len(rows)} variants)",
            "",
            "| variant | "
            + " | ".join(headers)
            + " | scale sum | E3 sum | gamma sum |",
            "|---|" + "---|" * (len(headers) + 3),
        ]
        for row in rows:
            cells = " | ".join(f"{row['cells'][h]:.3f}" for h in headers)
            lines.append(
                f"| {row['variant']} | {cells} | {row['scale_sum']:.3f} | "
                f"{row['e3_sum']:.3f} | {row['gamma_sum']:.3f} |"
            )
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
