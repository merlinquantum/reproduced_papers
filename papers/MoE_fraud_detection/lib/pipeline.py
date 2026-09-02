"""Full CV experiment: XGBoost primary expert + GQC secondary expert + MoE
router, evaluated per router threshold, aggregated across CV splits.

Reproduces the paper's Tables 2-5 shape: rows = XGBoost-only baseline (E1) +
one row per router threshold (E2), columns = AUCPR / AP / precision / recall
(mean, std, median) plus routed_fraction (mean, std).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from xgboost import XGBClassifier

from .calibration import apply_temperature, fit_temperature, probs_to_logits
from .data import load_cv_splits
from .gqc_model import GQCModel
from .moe import (
    build_router_targets,
    combine_predictions,
    evaluate_binary,
    youden_j_threshold,
)
from .train_gqc import train_gqc_model

logger = logging.getLogger(__name__)

BASELINE_KEY = "xgboost_baseline"


def _parse_router_thresholds(cfg: dict[str, Any]) -> list[float]:
    thresholds = cfg.get("evaluation", {}).get(
        "router_thresholds", [0.5, 0.6, 0.7, 0.8, 0.9]
    )
    if isinstance(thresholds, str):
        thresholds = [
            float(chunk.strip()) for chunk in thresholds.split(",") if chunk.strip()
        ]
    return [float(t) for t in thresholds]


def _make_xgb_classifier(
    cfg: dict[str, Any], seed: int, *, early_stopping: bool = False
) -> XGBClassifier:
    """Build an XGBoost classifier from ``cfg["model"]["xgboost"]``.

    ``n_jobs`` defaults to 2 (not -1/all-cores) so multiple experiment
    configs can be run as separate OS processes in parallel on this
    container's cores without oversubscription -- see LOG.md "Parallel
    Statistically-Powered Runs" for the reasoning. Override via
    ``cfg["model"]["xgboost"]["n_jobs"]``.

    ``early_stopping`` is only honored for the PRIMARY expert (never the
    router, which has no natural held-out eval set of its own beyond the
    analysis/validation split it's already trained on) -- see
    ``_run_single_split``. When true, the caller must pass ``eval_set`` to
    ``.fit()``.
    """
    xgb_cfg = cfg.get("model", {}).get("xgboost", {})
    kwargs: dict[str, Any] = {
        "n_estimators": int(xgb_cfg.get("n_estimators", 200)),
        "max_depth": int(xgb_cfg.get("max_depth", 4)),
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": int(xgb_cfg.get("n_jobs", 2)),
    }
    if early_stopping and xgb_cfg.get("early_stopping_rounds"):
        kwargs["early_stopping_rounds"] = int(xgb_cfg["early_stopping_rounds"])
    return XGBClassifier(**kwargs)


def _predict_gqc(model: GQCModel, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)
        p_hat = model.predict_proba(x_t)
    return p_hat.numpy()


def _run_single_split(
    cfg: dict[str, Any], split: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Train both experts + router on one CV split and evaluate on holdout."""
    router_thresholds = _parse_router_thresholds(cfg)
    input_dim = split["X_train"].shape[1]
    xgb_cfg = cfg.get("model", {}).get("xgboost", {})
    early_stopping = bool(xgb_cfg.get("early_stopping_rounds"))
    router_split = str(cfg.get("model", {}).get("router_split", "analysis"))
    if router_split not in {"analysis", "validation"}:
        raise ValueError(
            f"model.router_split must be 'analysis' or 'validation', got {router_split!r}"
        )

    # --- Primary expert: XGBoost, trained on the balanced train pool.
    # Optionally early-stopped on the validation split -- the paper's own
    # text explicitly allows this ("optionally using the validation set for
    # early stopping", Section 2.2). ---
    xgb1 = _make_xgb_classifier(cfg, seed, early_stopping=early_stopping)
    if early_stopping:
        xgb1.fit(
            split["X_train"],
            split["y_train"],
            eval_set=[(split["X_val"], split["y_val"])],
            verbose=False,
        )
    else:
        xgb1.fit(split["X_train"], split["y_train"])

    # --- Secondary expert: GQC hybrid, trained on the same balanced pool. ---
    gqc = GQCModel(input_dim, cfg)
    train_gqc_model(gqc, split["X_train"], split["y_train"], cfg)

    # --- Calibrate both experts on the validation split. ---
    p1_val_raw = xgb1.predict_proba(split["X_val"])[:, 1]
    p2_val_raw = _predict_gqc(gqc, split["X_val"])
    t1 = fit_temperature(probs_to_logits(p1_val_raw), split["y_val"])
    t2 = fit_temperature(probs_to_logits(p2_val_raw), split["y_val"])
    p1_val_cal = apply_temperature(probs_to_logits(p1_val_raw), t1)
    p2_val_cal = apply_temperature(probs_to_logits(p2_val_raw), t2)
    tau1 = youden_j_threshold(p1_val_cal, split["y_val"])
    tau2 = youden_j_threshold(p2_val_cal, split["y_val"])

    # --- Build router targets + train the XGBoost router.
    #
    # ``router_split="analysis"`` (default): matches the paper's own
    # described train/validation/analysis/holdout procedure (Section 3.3,
    # Fig. 2) -- validation is reserved for calibration/threshold-selection
    # only, and the router trains on the analysis split.
    #
    # ``router_split="validation"``: matches the paper's PROSE literally
    # ("train an XGBoost router on the validation features and these router
    # targets") -- the router reuses the same validation split (and the
    # tau1/tau2 computed from it) for its own targets and training features,
    # and the analysis split goes unused. See lib/moe.py module docstring
    # for the full paper-internal-inconsistency discussion. ---
    if router_split == "analysis":
        router_X, router_y = split["X_analysis"], split["y_analysis"]
        p1_router_raw = xgb1.predict_proba(router_X)[:, 1]
        p2_router_raw = _predict_gqc(gqc, router_X)
    else:
        router_X, router_y = split["X_val"], split["y_val"]
        p1_router_raw, p2_router_raw = p1_val_raw, p2_val_raw
    p1_router_cal = apply_temperature(probs_to_logits(p1_router_raw), t1)
    p2_router_cal = apply_temperature(probs_to_logits(p2_router_raw), t2)
    z_router = build_router_targets(
        p1_router_cal, router_y, p2_router_cal, tau1, tau2, router_y
    )
    router = _make_xgb_classifier(cfg, seed)
    router.fit(router_X, z_router)

    # --- Holdout evaluation. ---
    p1_holdout_raw = xgb1.predict_proba(split["X_holdout"])[:, 1]
    p2_holdout_raw = _predict_gqc(gqc, split["X_holdout"])
    p1_holdout_cal = apply_temperature(probs_to_logits(p1_holdout_raw), t1)
    p2_holdout_cal = apply_temperature(probs_to_logits(p2_holdout_raw), t2)
    router_probs_holdout = router.predict_proba(split["X_holdout"])[:, 1]
    y_holdout = split["y_holdout"]

    baseline_metrics = evaluate_binary(p1_holdout_cal, y_holdout)
    baseline_metrics["routed_fraction"] = 0.0

    threshold_metrics: dict[str, dict[str, float]] = {}
    for gamma in router_thresholds:
        p_comb = combine_predictions(
            p1_holdout_cal, p2_holdout_cal, router_probs_holdout, gamma
        )
        m = evaluate_binary(p_comb, y_holdout)
        r = (router_probs_holdout > gamma).astype(int)
        m["routed_fraction"] = float(r.mean())
        threshold_metrics[f"{gamma:g}"] = m

    return {
        "repeat": split["repeat"],
        "fold": split["fold"],
        "router_split": router_split,
        "tau1": tau1,
        "tau2": tau2,
        "temperature_primary": t1,
        "temperature_secondary": t2,
        BASELINE_KEY: baseline_metrics,
        "thresholds": threshold_metrics,
    }


def _aggregate_metric(values: list[float], stat: str) -> float:
    arr = np.asarray(values, dtype=float)
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "std":
        return float(np.std(arr))
    if stat == "median":
        return float(np.median(arr))
    raise ValueError(f"Unknown stat {stat}")


def _aggregate_rows(
    per_split_results: list[dict[str, Any]], router_thresholds: list[float]
) -> dict[str, Any]:
    row_keys = [BASELINE_KEY] + [f"{gamma:g}" for gamma in router_thresholds]
    metric_names = ["aucpr", "ap", "precision", "recall", "routed_fraction"]
    table: dict[str, Any] = {}
    for row_key in row_keys:
        values_by_metric: dict[str, list[float]] = {m: [] for m in metric_names}
        for split_result in per_split_results:
            row = (
                split_result[BASELINE_KEY]
                if row_key == BASELINE_KEY
                else split_result["thresholds"][row_key]
            )
            for m in metric_names:
                values_by_metric[m].append(row[m])
        row_summary: dict[str, float] = {}
        for m in metric_names:
            row_summary[f"{m}_mean"] = _aggregate_metric(values_by_metric[m], "mean")
            row_summary[f"{m}_std"] = _aggregate_metric(values_by_metric[m], "std")
            if m != "routed_fraction":
                row_summary[f"{m}_median"] = _aggregate_metric(
                    values_by_metric[m], "median"
                )
        table[row_key] = row_summary
    return table


def _write_metrics_json(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )


def _write_metrics_csv(run_dir: Path, table: dict[str, Any]) -> None:
    columns = [
        "row",
        "aucpr_mean",
        "aucpr_std",
        "aucpr_median",
        "ap_mean",
        "ap_std",
        "ap_median",
        "precision_mean",
        "precision_std",
        "precision_median",
        "recall_mean",
        "recall_std",
        "recall_median",
        "routed_fraction_mean",
        "routed_fraction_std",
    ]
    with (run_dir / "metrics_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row_key, row in table.items():
            writer.writerow([row_key] + [row.get(col) for col in columns[1:]])


def run_cv_experiment(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run the full repeated-CV MoE experiment and write results to run_dir.

    Returns the aggregated results dict (also written to
    ``run_dir/metrics.json``; a flattened version is written to
    ``run_dir/metrics_table.csv``).
    """
    seed = int(cfg.get("seed", 42))
    router_thresholds = _parse_router_thresholds(cfg)
    splits = load_cv_splits(cfg)
    logger.info(
        "Loaded %d CV splits (n_repeats=%s)",
        len(splits),
        cfg.get("cv", {}).get("n_repeats"),
    )

    per_split_results = []
    for i, split in enumerate(splits):
        logger.info(
            "Running split %d/%d (repeat=%d fold=%d): train=%d val=%d analysis=%d holdout=%d",
            i + 1,
            len(splits),
            split["repeat"],
            split["fold"],
            len(split["y_train"]),
            len(split["y_val"]),
            len(split["y_analysis"]),
            len(split["y_holdout"]),
        )
        result = _run_single_split(cfg, split, seed=seed + i)
        per_split_results.append(result)
        logger.info(
            "  split %d: baseline AUCPR=%.4f AP=%.4f | best-threshold AUCPR=%.4f",
            i + 1,
            result[BASELINE_KEY]["aucpr"],
            result[BASELINE_KEY]["ap"],
            max(result["thresholds"][k]["aucpr"] for k in result["thresholds"]),
        )

    table = _aggregate_rows(per_split_results, router_thresholds)

    payload = {
        "n_splits": len(splits),
        "n_repeats": int(cfg.get("cv", {}).get("n_repeats", 3)),
        "router_thresholds": router_thresholds,
        "per_split": per_split_results,
        "table": table,
    }

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_json(run_dir, payload)
    _write_metrics_csv(run_dir, table)

    return payload


__all__ = ["run_cv_experiment", "BASELINE_KEY"]
