"""Fisher spectrum and effective-dimension experiments (Figs. 2, 3a, 8, 9; ablations A-01..A-03).

Sampling follows the authors' ``EffectiveDimension``: for draw seed ``s``, a NumPy
``RandomState(s)`` first draws ``theta ~ U[low, high]^{N_theta x d}``, then
``x ~ N(0, 1)^{N_x x s_in}``. Seed 0 with ``[low, high] = [0, 1]`` reproduces the
draws behind the cached upstream arrays. Each model gets a fresh stream, as in the
authors' scripts (one ``EffectiveDimension`` per model).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from .classical import FeedForwardNetwork, enumerate_topologies
from .fisher import (
    effective_dimension,
    empirical_fisher,
    normalize_fisher,
    spectrum_summary,
)
from .models import build_model, describe_model

logger = logging.getLogger(__name__)


def draw_parameters_and_inputs(
    seed: int,
    n_thetas: int,
    d: int,
    n_inputs: int,
    input_size: int,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``theta`` then ``x`` from ``RandomState(seed)`` in the authors' order."""
    stream = np.random.RandomState(seed)
    thetas = stream.uniform(low, high, size=(n_thetas, d))
    inputs = stream.normal(0.0, 1.0, size=(n_inputs, input_size))
    return thetas, inputs


def fisher_for_draw(model, fisher_cfg: dict, seed: int, device: torch.device) -> dict:
    """Compute the normalized Fisher and its summaries for one draw of (theta, x)."""
    thetas, inputs = draw_parameters_and_inputs(
        seed,
        fisher_cfg["num_thetas"],
        model.d,
        fisher_cfg["num_inputs"],
        model.input_size,
        fisher_cfg["theta_low"],
        fisher_cfg["theta_high"],
    )
    fishers = (
        empirical_fisher(
            model,
            torch.as_tensor(thetas, dtype=torch.float64, device=device),
            torch.as_tensor(inputs, dtype=torch.float64, device=device),
            fisher_cfg["chunk_size"],
        )
        .cpu()
        .numpy()
    )
    f_hat, mean_trace = normalize_fisher(fishers)
    effdim = effective_dimension(f_hat, fisher_cfg["n_values"], fisher_cfg["gamma"])
    return {
        "f_hat": f_hat,
        "eigenvalues": np.clip(np.linalg.eigvalsh(f_hat), 0.0, None),
        "mean_trace": mean_trace,
        "effective_dimension": effdim,
        "spectrum": spectrum_summary(f_hat, fisher_cfg["zero_tolerance"]),
    }


def postselection_success(
    model, fisher_cfg: dict, seed: int, device: torch.device
) -> float:
    """Mean probability of a collision-free outcome over one draw's (theta, x) pairs.

    This is the success probability of MerLin's UNBUNCHED postselection for the same
    draws the Fisher uses (theta row i paired with every input).
    """
    thetas, inputs = draw_parameters_and_inputs(
        seed,
        fisher_cfg["num_thetas"],
        model.d,
        fisher_cfg["num_inputs"],
        model.input_size,
        fisher_cfg["theta_low"],
        fisher_cfg["theta_high"],
    )
    theta = torch.as_tensor(
        thetas, dtype=torch.float64, device=device
    ).repeat_interleave(len(inputs), dim=0)
    x = torch.as_tensor(inputs, dtype=torch.float64, device=device).repeat(
        len(thetas), 1
    )
    with torch.no_grad():
        return float(model.collision_free_probability(x, theta).mean())


def run_fisher_effective_dimension(
    cfg: dict, run_dir: Path, device: torch.device
) -> dict:
    """Evaluate every configured model on every draw seed and write arrays.

    Writes ``fisher_<model>.npz`` per model with ``eigenvalues`` ``(draws, N_theta, d)``,
    ``effective_dimension`` ``(draws, len(n))``, ``mean_trace`` ``(draws,)`` and, for
    the seeds in ``fisher.save_fhat_draws``, ``f_hat_seed<s>`` ``(N_theta, d, d)``.

    Returns
    -------
    dict
        Metrics: per model, per-draw and aggregated (mean, std) normalized effective
        dimension, unnormalized mean trace, and spectrum summaries.
    """
    fisher_cfg = cfg["fisher"]
    n_values = [float(n) for n in fisher_cfg["n_values"]]
    metrics = {
        "n_values": n_values,
        "gamma": fisher_cfg["gamma"],
        "fisher": fisher_cfg,
        "models": {},
    }
    for model_cfg in cfg["models"]:
        model = build_model(model_cfg)
        name = model_cfg["name"]
        draws, arrays = [], {}
        for seed in fisher_cfg["draw_seeds"]:
            logger.info(
                "EVALUATION_STARTED | split=fisher_draw | model=%s | d=%d | draw_seed=%d",
                name,
                model.d,
                seed,
            )
            started = time.time()
            result = fisher_for_draw(model, fisher_cfg, seed, device)
            normalized = result["effective_dimension"] / model.d
            logger.info(
                "EVALUATION_COMPLETED | split=fisher_draw | model=%s | draw_seed=%d | ed_norm_first=%.6f | "
                "ed_norm_last=%.6f | mean_trace=%.6g | mean_rank=%.2f | seconds=%.1f",
                name,
                seed,
                normalized[0],
                normalized[-1],
                result["mean_trace"],
                result["spectrum"]["mean_rank"],
                time.time() - started,
            )
            draws.append(
                {
                    "draw_seed": seed,
                    "effective_dimension": result["effective_dimension"].tolist(),
                    "normalized_effective_dimension": normalized.tolist(),
                    "mean_trace": result["mean_trace"],
                    "spectrum": result["spectrum"],
                }
            )
            if getattr(model, "detection", None) == "unbunched":
                draws[-1]["postselection_success_mean"] = postselection_success(
                    model, fisher_cfg, seed, device
                )
                logger.info(
                    "EVALUATION_COMPLETED | split=postselection | model=%s | draw_seed=%d | success_mean=%.6f",
                    name,
                    seed,
                    draws[-1]["postselection_success_mean"],
                )
            arrays.setdefault("eigenvalues", []).append(result["eigenvalues"])
            arrays.setdefault("effective_dimension", []).append(
                result["effective_dimension"]
            )
            arrays.setdefault("mean_trace", []).append(result["mean_trace"])
            if seed in fisher_cfg["save_fhat_draws"]:
                arrays[f"f_hat_seed{seed}"] = result["f_hat"]
        array_path = run_dir / f"fisher_{name}.npz"
        np.savez_compressed(
            array_path,
            draw_seeds=np.asarray(fisher_cfg["draw_seeds"]),
            **{k: np.asarray(v) for k, v in arrays.items()},
        )
        logger.info("ARTIFACT_WRITTEN | path=%s", array_path.name)
        normalized_all = np.asarray(
            [draw["normalized_effective_dimension"] for draw in draws]
        )
        metrics["models"][name] = {
            **describe_model(model_cfg, model),
            "config": model_cfg,
            "draws": draws,
            "normalized_effective_dimension_mean": normalized_all.mean(axis=0).tolist(),
            "normalized_effective_dimension_std": normalized_all.std(axis=0).tolist(),
            "mean_trace_mean": float(np.mean([draw["mean_trace"] for draw in draws])),
            "mean_rank_mean": float(
                np.mean([draw["spectrum"]["mean_rank"] for draw in draws])
            ),
            "fraction_near_zero_mean": float(
                np.mean([draw["spectrum"]["fraction_near_zero"] for draw in draws])
            ),
            "arrays": array_path.name,
        }
    return metrics


def run_topology_search(cfg: dict, run_dir: Path, device: torch.device) -> dict:
    """Fair classical baseline search B-1 / ablation A-02 (paper footnotes 5 and 18).

    Every bias-free (or biased) fully connected topology with exactly ``target_d``
    parameters and at most ``max_hidden_layers`` hidden layers is evaluated. The
    predeclared selection rule is the highest mean F_hat rank over the draws, then the
    highest normalized effective dimension at ``selection_n``.
    """
    search_cfg = cfg["topology_search"]
    fisher_cfg = cfg["fisher"]
    topologies = enumerate_topologies(
        search_cfg["input_size"],
        search_cfg["output_size"],
        search_cfg["target_d"],
        search_cfg["max_hidden_layers"],
        search_cfg["bias"],
    )
    if not topologies:
        raise ValueError(
            f"No topology has exactly d={search_cfg['target_d']} parameters"
        )
    selection_index = [float(n) for n in fisher_cfg["n_values"]].index(
        float(search_cfg["selection_n"])
    )
    logger.info(
        "EVALUATION_STARTED | split=topology_search | candidates=%d | target_d=%d",
        len(topologies),
        search_cfg["target_d"],
    )
    candidates = []
    for position, sizes in enumerate(topologies, start=1):
        model = FeedForwardNetwork(
            sizes, search_cfg["activation"], search_cfg["head"], search_cfg["bias"]
        )
        ranks, normalized = [], []
        for seed in fisher_cfg["draw_seeds"]:
            result = fisher_for_draw(model, fisher_cfg, seed, device)
            ranks.append(result["spectrum"]["mean_rank"])
            normalized.append(result["effective_dimension"] / model.d)
        normalized = np.asarray(normalized)
        candidate = {
            "sizes": sizes,
            "mean_rank": float(np.mean(ranks)),
            "normalized_effective_dimension_mean": normalized.mean(axis=0).tolist(),
            "selection_value": float(normalized.mean(axis=0)[selection_index]),
        }
        candidates.append(candidate)
        logger.info(
            "EVALUATION_COMPLETED | split=topology_candidate | candidate=%d/%d | sizes=%s | mean_rank=%.2f | ed_norm_at_selection_n=%.6f",
            position,
            len(topologies),
            sizes,
            candidate["mean_rank"],
            candidate["selection_value"],
        )
    best_rank = max(candidate["mean_rank"] for candidate in candidates)
    top = [candidate for candidate in candidates if candidate["mean_rank"] == best_rank]
    best_value = max(candidate["selection_value"] for candidate in top)
    tied = [
        candidate["sizes"]
        for candidate in top
        if abs(candidate["selection_value"] - best_value) <= search_cfg["tie_tolerance"]
    ]
    logger.info(
        "EVALUATION_COMPLETED | split=topology_search | selected=%s | mean_rank=%.2f",
        tied,
        best_rank,
    )
    return {
        "selection_rule": "max mean F_hat rank, then max normalized effective dimension at selection_n (predeclared, LOG.md Fair Baseline Plan)",
        "search": search_cfg,
        "fisher": fisher_cfg,
        "candidates": candidates,
        "selected": tied,
    }


def run_sensitivity(cfg: dict, run_dir: Path, device: torch.device) -> dict:
    """Fig. 8 / App. C.1: effective dimension vs the number of Monte Carlo samples.

    For every model and every sample count ``m`` in ``fisher.sample_sizes``, the
    effective dimension is estimated with ``N_theta = N_x = m`` for each draw seed.
    Writes ``sensitivity_<model>.npz`` with ``normalized_effective_dimension``
    ``(len(sample_sizes), draws, len(n))``.
    """
    fisher_cfg = cfg["fisher"]
    metrics = {
        "n_values": [float(n) for n in fisher_cfg["n_values"]],
        "fisher": fisher_cfg,
        "models": {},
    }
    for model_cfg in cfg["models"]:
        model = build_model(model_cfg)
        name = model_cfg["name"]
        table = []
        for sample_size in fisher_cfg["sample_sizes"]:
            sized_cfg = {
                **fisher_cfg,
                "num_thetas": sample_size,
                "num_inputs": sample_size,
            }
            logger.info(
                "EVALUATION_STARTED | split=sensitivity | model=%s | samples=%d",
                name,
                sample_size,
            )
            normalized = np.asarray(
                [
                    fisher_for_draw(model, sized_cfg, seed, device)[
                        "effective_dimension"
                    ]
                    / model.d
                    for seed in fisher_cfg["draw_seeds"]
                ]
            )
            table.append(normalized)
            logger.info(
                "EVALUATION_COMPLETED | split=sensitivity | model=%s | samples=%d | ed_norm_last_mean=%.6f | ed_norm_last_std=%.6f",
                name,
                sample_size,
                normalized[:, -1].mean(),
                normalized[:, -1].std(),
            )
        table = np.asarray(table)
        array_path = run_dir / f"sensitivity_{name}.npz"
        np.savez_compressed(
            array_path,
            normalized_effective_dimension=table,
            sample_sizes=np.asarray(fisher_cfg["sample_sizes"]),
        )
        logger.info("ARTIFACT_WRITTEN | path=%s", array_path.name)
        metrics["models"][name] = {
            **describe_model(model_cfg, model),
            "config": model_cfg,
            "sample_sizes": fisher_cfg["sample_sizes"],
            "normalized_effective_dimension_mean": table.mean(axis=1).tolist(),
            "normalized_effective_dimension_std": table.std(axis=1).tolist(),
            "arrays": array_path.name,
        }
    return metrics
