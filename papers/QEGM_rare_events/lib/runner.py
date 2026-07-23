"""Shared-runtime entry point for the QEGM rare-event reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .data import build_gmm_dataset
from .metrics import (
    coverage_calibration,
    rare_event_recall,
    summarize,
    tail_kl_divergence,
)
from .models import build_model
from .training import generate, train_one

logger = logging.getLogger(__name__)


def _parse_list(value, *, cast):
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v) for v in str(value).split(",") if v.strip()]


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Train each requested variant for each seed and write metrics/figures."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = str(cfg.get("device", "cpu"))

    training_cfg = cfg["training"]
    seeds = _parse_list(training_cfg["seeds"], cast=int)
    variants = _parse_list(training_cfg["models"], cast=str)

    eval_cfg = cfg["evaluation"]
    n_generated = int(eval_cfg["n_generated"])
    n_tail_bins = int(eval_cfg["n_tail_bins"])
    tail_kl_eps = float(eval_cfg["tail_kl_eps"])

    results: dict = {variant: [] for variant in variants}
    histories: dict = {variant: [] for variant in variants}
    hw_settings: dict = {}

    dataset_seed = int(cfg.get("seed", 42))
    dataset = build_gmm_dataset(cfg, seed=dataset_seed)
    real_test = dataset.test.numpy().flatten()
    np.save(run_dir / "real_samples_test.npy", real_test)

    for variant in variants:
        for seed in seeds:
            logger.info("Training variant=%s seed=%d", variant, seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = build_model(variant, cfg, in_dim=1).to(device)
            if variant == "qegm_merlin":
                hw_settings.setdefault(variant, model.hardware_settings())

            train_result = train_one(model, dataset, cfg, seed=seed)

            generated = generate(model, n=n_generated, device=device).flatten()
            metrics = {
                "tail_kl": tail_kl_divergence(
                    real_test,
                    generated,
                    tail_threshold=dataset.tail_threshold,
                    n_bins=n_tail_bins,
                    eps=tail_kl_eps,
                ),
                "rare_recall": rare_event_recall(
                    real_test,
                    generated,
                    tail_threshold=dataset.tail_threshold,
                ),
                "coverage": coverage_calibration(real_test, generated),
                "train_time_s": train_result.train_time_s,
                "n_params": int(sum(p.numel() for p in model.parameters())),
            }
            results[variant].append(metrics)
            histories[variant].append(
                {
                    "seed": seed,
                    "losses": train_result.losses,
                    "val_losses": train_result.val_losses,
                }
            )
            np.save(run_dir / f"samples_{variant}_seed{seed}.npy", generated)

    summary = {variant: summarize(results[variant]) for variant in variants}
    metrics_payload = {
        "per_seed": results,
        "summary": summary,
        "config_dataset": {
            "means": list(dataset.means),
            "stds": list(dataset.stds),
            "weights": list(dataset.weights),
            "tail_threshold": dataset.tail_threshold,
        },
        "hardware_settings": hw_settings,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (run_dir / "histories.json").write_text(json.dumps(histories, indent=2))

    for variant in variants:
        s = summary[variant]
        if "tail_kl" in s and "rare_recall" in s:
            logger.info(
                "summary %-12s tail_kl=%.3f±%.3f  rare_recall=%.3f±%.3f",
                variant,
                s["tail_kl"]["mean"],
                s["tail_kl"]["std"],
                s["rare_recall"]["mean"],
                s["rare_recall"]["std"],
            )

    try:
        from utils.plot_results import generate_figures

        generate_figures(run_dir)
    except Exception as exc:  # plotting is non-fatal
        logger.warning("Figure generation skipped: %s", exc)
