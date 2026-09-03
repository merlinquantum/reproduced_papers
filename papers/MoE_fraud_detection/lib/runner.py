"""Runtime entry point for the MoE fraud-detection (gate-model) reproduction.

Dispatches on ``cfg.get("task")``:

- unset / any value other than ``"latency_benchmark"`` (default): runs the
  full MoE CV experiment (XGBoost primary expert + PennyLane GQC hybrid
  secondary expert + XGBoost MoE router, threshold-swept), writing
  ``run_dir/metrics.json`` and ``run_dir/metrics_table.csv`` -- unchanged
  from before ``task`` existed, so every existing config (none of which set
  ``task``) keeps working exactly as before.
- ``"latency_benchmark"``: runs the Table 1 reproduction (QMKL vs GFM vs GQC
  per-sample inference latency, :mod:`.latency_benchmark`), writing
  ``run_dir/latency_benchmark.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .latency_benchmark import run_latency_benchmark
from .pipeline import BASELINE_KEY, run_cv_experiment


def _run_latency_benchmark_task(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    results = run_latency_benchmark(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latency_benchmark.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    for method, stats in results.items():
        logger.info(
            "Table 1 latency: %-4s %.4f +/- %.4f ms/sample (CV %.2f%%)",
            method,
            stats["ms_per_sample"],
            stats["std_ms_per_sample"],
            stats["cv_pct"],
        )
    logger.info(
        "Wrote latency benchmark results to %s", run_dir / "latency_benchmark.json"
    )
    return results


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    task = cfg.get("task")
    if task == "latency_benchmark":
        return _run_latency_benchmark_task(cfg, run_dir)

    logger = logging.getLogger(__name__)
    payload = run_cv_experiment(cfg, run_dir)

    table = payload["table"]
    logger.info(
        "MoE fraud detection CV experiment complete (%d splits).", payload["n_splits"]
    )
    baseline_row = table[BASELINE_KEY]
    logger.info(
        "XGBoost baseline: AUCPR=%.4f+/-%.4f AP=%.4f+/-%.4f",
        baseline_row["aucpr_mean"],
        baseline_row["aucpr_std"],
        baseline_row["ap_mean"],
        baseline_row["ap_std"],
    )
    for gamma in payload["router_thresholds"]:
        key = f"{gamma:g}"
        row = table[key]
        logger.info(
            "MoE gamma=%s: AUCPR=%.4f+/-%.4f AP=%.4f+/-%.4f routed_fraction=%.4f",
            key,
            row["aucpr_mean"],
            row["aucpr_std"],
            row["ap_mean"],
            row["ap_std"],
            row["routed_fraction_mean"],
        )

    logger.info(
        "Wrote metrics to %s and %s",
        run_dir / "metrics.json",
        run_dir / "metrics_table.csv",
    )
    return payload


__all__ = ["train_and_evaluate"]
