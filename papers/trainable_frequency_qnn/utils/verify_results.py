#!/usr/bin/env python3
"""Read-only recalculation of the headline result tables (no files written).

Recomputes, from each completed run's ``metrics.json`` only, the per-candidate
means used in LOG.md / README and compares them with ``results/*_results.json``.
Exits non-zero if any value differs by more than 1e-9 relative.

Example (from the paper folder)::

    python utils/verify_results.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
SWEEPS = PAPER_DIR / "outdir" / "sweeps"
RESULTS = PAPER_DIR / "results"

TOY = {
    "e1_fig2a": ["e1_fig2a"],
    "e2_fig2b": ["e2_fig2b"],
    "e3_ff_depth": ["e3_ff_depth"],
    "e4_scaled": ["e4_scaled"],
    "e4b_scale_init": ["e4b_scale_init"],
}
NS = {
    "e6_table1": ["e6_table1"],
    "e6_table1_with_controls": ["e6_table1", "e6_controls"],
    "e6_minmax": ["e6_minmax"],
    "e7_appc_combined": ["e7_appc", "e7_appc_retry"],
    "e7_with_scaling_controls": ["e7_appc", "e7_appc_retry", "e7_scaling_controls"],
}


def completed_metrics(sweeps: list[str]) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for sweep in sweeps:
        status = json.loads(
            (SWEEPS / sweep / "sweep_status.json").read_text(encoding="utf-8")
        )
        for entry in status["runs"]:
            if entry["status"] == "COMPLETED":
                metrics = json.loads(
                    (SWEEPS / sweep / entry["metrics_path"]).read_text(encoding="utf-8")
                )
                by_seed = runs.setdefault(entry["candidate"], {})
                if entry["seed"] in by_seed:
                    raise ValueError(
                        f"duplicate {entry['candidate']} seed {entry['seed']}"
                    )
                by_seed[entry["seed"]] = metrics
    return runs


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


def main() -> int:
    checked = mismatches = 0
    for name, sweeps in TOY.items():
        stored = json.loads(
            (RESULTS / f"{name}_results.json").read_text(encoding="utf-8")
        )
        for candidate, by_seed in completed_metrics(sweeps).items():
            value = statistics.fmean(m["train_mse"] for m in by_seed.values())
            checked += 1
            if not close(value, stored["candidates"][candidate]["train_mse"]["mean"]):
                mismatches += 1
                print(
                    f"MISMATCH {name} {candidate} train_mse {value} vs {stored['candidates'][candidate]['train_mse']['mean']}"
                )
    for name, sweeps in NS.items():
        stored = json.loads(
            (RESULTS / f"{name}_results.json").read_text(encoding="utf-8")
        )
        for candidate, by_seed in completed_metrics(sweeps).items():
            for field in ("u", "v", "p"):
                value = statistics.fmean(
                    m["maerm_table"][f"{field}_maerm_abs_median"]["mean"]
                    for m in by_seed.values()
                )
                checked += 1
                if not close(
                    value, stored["candidates"][candidate][f"{field}_mean"]["mean"]
                ):
                    mismatches += 1
                    print(f"MISMATCH {name} {candidate} {field} {value}")
    e5 = json.loads((RESULTS / "e5_fig3_results.json").read_text(encoding="utf-8"))
    for candidate, by_seed in completed_metrics(["e5_fig3"]).items():
        model, k = candidate.rsplit("_k", 1)
        value = statistics.fmean(m["train_mse"] for m in by_seed.values())
        checked += 1
        if not close(value, e5["models"][model]["per_dataset_mean_train_mse"][k]):
            mismatches += 1
            print(f"MISMATCH e5_fig3 {candidate} {value}")
    print(
        f"verify_results: {checked} values recomputed from metrics.json, {mismatches} mismatches"
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
