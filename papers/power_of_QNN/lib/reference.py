"""V-00 validation anchor: compare this implementation with the authors' cached arrays.

Upstream cache: ``<data_root>/power_of_QNN/sources/effective_dimension`` (git tag
v1.0.0 of amyami187/effective_dimension). Three checks:

1. ``effective_dimension`` applied to the cached Fig. 3a F_hat arrays reproduces
   ``Effective_dimension_plots/plot_fig/source_data/effective_dimension.txt``.
2. The torch qubit models, fed the authors' draws (``RandomState(0)``: theta ~ U[0,1],
   then x ~ N(0,1)), reproduce the cached Fig. 2 quantum F_hat element-wise.
3. The classical [4,4,4,2] F_hat cannot be matched element-wise: the authors' networks
   came from ``torch.manual_seed`` streams of torch 1.3.1. The check compares
   distribution statistics (mean rank, normalized effective dimension) instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .data import paper_data_dir
from .fisher import effective_dimension, spectrum_summary
from .fisher_experiments import fisher_for_draw
from .models import build_model

logger = logging.getLogger(__name__)


def upstream_dir(cfg: dict) -> Path:
    """Return the cached upstream repository path and fail if it is absent."""
    path = paper_data_dir(cfg) / "sources" / "effective_dimension"
    if not (path / "effective_dimension" / "effective_dimension.py").exists():
        raise FileNotFoundError(
            f"Upstream cache missing at {path}; recreate it with the command in LOG.md (Available Resources)"
        )
    return path


def run_reference_check(cfg: dict, run_dir: Path, device: torch.device) -> dict:
    """Run the three V-00 checks and report pass/fail against configured tolerances."""
    reference_cfg = cfg["reference"]
    fisher_cfg = cfg["fisher"]
    upstream = upstream_dir(cfg)
    n_values = [float(n) for n in fisher_cfg["n_values"]]
    logger.info("DATASET_READY | name=upstream cache | path=%s", upstream)
    results: dict = {"upstream": str(upstream), "checks": {}}

    logger.info("EVALUATION_STARTED | split=reference | check=fig3a_formula")
    published = np.loadtxt(
        upstream
        / "Effective_dimension_plots/plot_fig/source_data/effective_dimension.txt"
    )
    formula_rows = {}
    for row, filename in enumerate(reference_cfg["fig3a_arrays"]):
        f_hat = np.load(upstream / "Effective_dimension_plots/data" / filename)
        ours = (
            effective_dimension(f_hat, n_values, fisher_cfg["gamma"]) / f_hat.shape[1]
        )
        formula_rows[filename] = {
            "d": int(f_hat.shape[1]),
            "max_abs_diff": float(np.abs(ours - published[row]).max()),
            "ours": ours.tolist(),
            "published": published[row].tolist(),
        }
    formula_pass = all(
        entry["max_abs_diff"] <= reference_cfg["formula_tolerance"]
        for entry in formula_rows.values()
    )
    results["checks"]["fig3a_formula"] = {"passed": formula_pass, "rows": formula_rows}
    logger.info(
        "EVALUATION_COMPLETED | split=reference | check=fig3a_formula | passed=%s",
        formula_pass,
    )

    elementwise = {}
    for model_cfg in reference_cfg["quantum_models"]:
        logger.info(
            "EVALUATION_STARTED | split=reference | check=quantum_fhat | model=%s",
            model_cfg["name"],
        )
        model = build_model(model_cfg)
        ours = fisher_for_draw(model, fisher_cfg, 0, device)["f_hat"]
        cached = np.load(
            upstream
            / "Eigenvalue_distribution_plots/main_figure/data"
            / model_cfg["reference_file"]
        )
        difference = float(np.abs(ours - cached).max())
        elementwise[model_cfg["name"]] = {
            "max_abs_diff": difference,
            "cached_max_abs": float(np.abs(cached).max()),
            "passed": difference <= reference_cfg["fhat_tolerance"],
        }
        logger.info(
            "EVALUATION_COMPLETED | split=reference | check=quantum_fhat | model=%s | max_abs_diff=%.3e | passed=%s",
            model_cfg["name"],
            difference,
            elementwise[model_cfg["name"]]["passed"],
        )
    results["checks"]["quantum_fhat_elementwise"] = elementwise

    classical_cfg = reference_cfg["classical_model"]
    logger.info("EVALUATION_STARTED | split=reference | check=classical_distribution")
    model = build_model(classical_cfg)
    cached = np.load(
        upstream
        / "Eigenvalue_distribution_plots/main_figure/data"
        / classical_cfg["reference_file"]
    )
    cached_rank = spectrum_summary(cached, fisher_cfg["zero_tolerance"])["mean_rank"]
    cached_ed = effective_dimension(cached, n_values, fisher_cfg["gamma"]) / model.d
    draws = [
        fisher_for_draw(model, fisher_cfg, seed, device)
        for seed in fisher_cfg["draw_seeds"]
    ]
    draw_ranks = np.asarray([draw["spectrum"]["mean_rank"] for draw in draws])
    draw_eds = np.asarray([draw["effective_dimension"] / model.d for draw in draws])
    rank_band = (
        draw_ranks.mean() - reference_cfg["band_sigmas"] * draw_ranks.std(),
        draw_ranks.mean() + reference_cfg["band_sigmas"] * draw_ranks.std(),
    )
    ed_low = draw_eds.mean(axis=0) - reference_cfg["band_sigmas"] * draw_eds.std(axis=0)
    ed_high = draw_eds.mean(axis=0) + reference_cfg["band_sigmas"] * draw_eds.std(
        axis=0
    )
    classical_pass = bool(
        rank_band[0] <= cached_rank <= rank_band[1]
        and np.all((cached_ed >= ed_low) & (cached_ed <= ed_high))
    )
    results["checks"]["classical_distribution"] = {
        "passed": classical_pass,
        "cached_mean_rank": cached_rank,
        "draw_mean_ranks": draw_ranks.tolist(),
        "cached_normalized_effective_dimension": cached_ed.tolist(),
        "draws_normalized_effective_dimension_mean": draw_eds.mean(axis=0).tolist(),
        "draws_normalized_effective_dimension_std": draw_eds.std(axis=0).tolist(),
        "band_sigmas": reference_cfg["band_sigmas"],
    }
    logger.info(
        "EVALUATION_COMPLETED | split=reference | check=classical_distribution | cached_rank=%.2f | draw_rank_mean=%.2f | passed=%s",
        cached_rank,
        draw_ranks.mean(),
        classical_pass,
    )
    results["all_passed"] = bool(
        formula_pass
        and all(entry["passed"] for entry in elementwise.values())
        and classical_pass
    )
    return results
