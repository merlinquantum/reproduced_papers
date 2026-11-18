#!/usr/bin/env python3
"""Export compact per-seed metrics of every evidence sweep to a curated file.

The raw sweeps live in the disposable ``outdir/sweeps/``. The notebook needs the
individual seed values (not only aggregates), so this script copies the few
scalar metrics it uses into ``results/per_seed_metrics.json``. Only COMPLETED
runs recorded by a sweep coordinator are included, and each metrics file must
match the SHA-256 recorded in its ``run_status.json``.

Example (from the paper folder)::

    python utils/export_per_seed_metrics.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parents[1]
SWEEPS_DIR = PAPER_DIR / "outdir" / "sweeps"
OUTPUT = PAPER_DIR / "results" / "per_seed_metrics.json"

SWEEPS = (
    "e1_fig2a", "e2_fig2b", "e3_ff_depth", "e4_scaled", "e4b_scale_init", "e5_fig3",
    "e5b_oracle_scale", "lr_robustness", "e6_table1", "e6_controls", "e6_minmax",
    "e7_appc", "e7_appc_retry", "e7_scaling_controls", "g1_e7_ff_diag",
    "m1_fig2a", "m2_fig2b", "m4_scaled", "m5_fig3",
    "m6_table1_tf", "m6_table1_ff", "m6_table1_ffs",
)  # fmt: skip
FIELDS = ("u", "v", "p")


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the scalar metrics the notebook displays."""
    row: dict[str, Any] = {
        "seconds_per_iteration": metrics["seconds_per_iteration"],
        "training_seconds": metrics["training_seconds"],
    }
    if metrics["task"] == "toy_regression":
        row["train_mse"] = metrics["train_mse"]
        row["dense_mse"] = metrics["dense_mse"]
        row["encoding_multipliers"] = metrics.get("encoding_multipliers")
    else:
        for field in FIELDS:
            row[f"{field}_mean_maerm"] = metrics["maerm_table"][
                f"{field}_maerm_abs_median"
            ]["mean"]
        row["final_loss"] = metrics["final_loss"]
        row["cost_factor"] = metrics["cost_factor"]
    return row


def export() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sweep in SWEEPS:
        status = json.loads(
            (SWEEPS_DIR / sweep / "sweep_status.json").read_text(encoding="utf-8")
        )
        by_candidate: dict[str, dict[str, Any]] = {}
        for entry in status["runs"]:
            if entry["status"] != "COMPLETED":
                continue
            metrics_path = SWEEPS_DIR / sweep / entry["metrics_path"]
            run_status = json.loads(
                (metrics_path.parent / "run_status.json").read_text(encoding="utf-8")
            )
            if (
                hashlib.sha256(metrics_path.read_bytes()).hexdigest()
                != run_status["metrics_sha256"]
            ):
                raise ValueError(f"Stale metrics: {metrics_path}")
            seeds = by_candidate.setdefault(entry["candidate"], {})
            if str(entry["seed"]) in seeds:
                raise ValueError(
                    f"Duplicate run: {sweep} {entry['candidate']} {entry['seed']}"
                )
            seeds[str(entry["seed"])] = compact(
                json.loads(metrics_path.read_text(encoding="utf-8"))
            )
        result[sweep] = {
            "coverage": f"{sum(len(s) for s in by_candidate.values())} / {status['expected_runs']}",
            "candidates": by_candidate,
        }
    return result


def main() -> int:
    OUTPUT.write_text(json.dumps(export(), indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(PAPER_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
