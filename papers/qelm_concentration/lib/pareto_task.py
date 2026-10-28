"""Multi-objective hyperparameter search (E1, paper Fig. 2 and App. "Hyperparameter search").

Objectives per hyperparameter setting, each averaged over ``n_realisations`` realisations
(new block parameters and a new NARMA sequence, paper: 10):

- cumulative NARMA R^2 over orders ``1..floor(N/2)``, noiseless;
- the same with additive Gaussian noise on the expectations (shot-noise proxy);
- mean minimal MPS bond dimension of output states of NARMA windows;
- variability: mean over observables of ``Var_{u ~ Unif(-1,1)}[<R_i>]``.

All four are maximised (paper: "target high bond dimension and output variability, as
well as good NARMA performances"). Realisation ``r`` reuses the same random draws for
every trial (common random numbers), so differences between trials are not sampling noise.
Table I is enqueued as the first trial so its position relative to the fronts is exact.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import optuna
import torch

from .data import narma_sequence_from_config
from .mps import minimal_bond_dimension
from .narma_task import cumulative_r2, orders_r2
from .qelm import (
    apply_feature_noise,
    architecture_from_config,
    real_dtype_from_config,
    ring_features,
)
from .readout import alpha_grid
from .reservoir import (
    KickedIsingReservoir,
    ReservoirHyperparameters,
    sample_block_parameters,
)

logger = logging.getLogger(__name__)

OBJECTIVES = ("narma_noiseless", "narma_noisy", "bond_dimension", "variability")
SEARCHED_KEYS = (
    "input_strength",
    "b_mean",
    "b_std",
    "h_mean",
    "h_std",
    "J_mean",
    "J_std",
)


def evaluate_hyperparameters(
    cfg: dict, hyper: ReservoirHyperparameters, search_seed: int
) -> dict[str, float]:
    """Average the four objectives over the configured realisations.

    The NARMA score sums R^2 over orders ``n <= score_max_order``. The paper's text says
    ``n = 1..floor(N/2)``, but its Fig. 2a scores exceed that maximum; the configs use
    ``n = 1..N`` (see LOG.md). Per-order mean R^2 values are also returned
    (``noiseless_r2_n{k}``, ``noisy_r2_n{k}``) so any order range can be recomputed.
    """
    exp = cfg["pareto_search"]
    score_max_order = int(exp["score_max_order"])
    arch = architecture_from_config(cfg["reservoir"])
    alphas = alpha_grid(cfg["readout"])
    real_dtype = real_dtype_from_config(cfg)
    chunk = int(cfg["reservoir"]["chunk_size"])
    window = arch.n_blocks
    per_realisation = {key: [] for key in OBJECTIVES}
    per_order = {"noiseless": [], "noisy": []}

    for realisation in range(int(exp["n_realisations"])):
        params = sample_block_parameters(
            hyper, arch, np.random.default_rng([search_seed, realisation, 1])
        )
        reservoir = KickedIsingReservoir(arch, hyper, params, cfg["device"], real_dtype)
        sequence = narma_sequence_from_config(
            cfg, window, np.random.default_rng([search_seed, realisation, 2])
        )
        windows = np.concatenate([sequence.train_windows, sequence.test_windows])
        features, weights = ring_features(reservoir, windows, exp["feature_set"], chunk)
        n_train = len(sequence.train_windows)
        noiseless = orders_r2(features[:n_train], features[n_train:], sequence, alphas)
        noise_rng = np.random.default_rng([search_seed, realisation, 3])
        noisy_features = apply_feature_noise(features, weights, exp["noise"], noise_rng)
        noisy = orders_r2(
            noisy_features[:n_train], noisy_features[n_train:], sequence, alphas
        )
        per_realisation["narma_noiseless"].append(
            cumulative_r2(noiseless, sequence.orders, score_max_order)
        )
        per_realisation["narma_noisy"].append(
            cumulative_r2(noisy, sequence.orders, score_max_order)
        )
        per_order["noiseless"].append(noiseless)
        per_order["noisy"].append(noisy)

        bond_windows = sequence.test_windows[: int(exp["bond_windows"])]
        with torch.no_grad():
            states = (
                reservoir.evolve(torch.as_tensor(bond_windows))
                .to(torch.complex128)
                .cpu()
                .numpy()
            )
        bond = [
            minimal_bond_dimension(
                state, arch.n_qubits, float(exp["fidelity_threshold"])
            )
            for state in states
        ]
        per_realisation["bond_dimension"].append(float(np.mean(bond)))

        uniform_inputs = np.random.default_rng([search_seed, realisation, 4]).uniform(
            -1.0, 1.0, size=(int(exp["variability_samples"]), window)
        )
        uniform_features, _ = ring_features(
            reservoir, uniform_inputs, exp["feature_set"], chunk
        )
        per_realisation["variability"].append(
            float(uniform_features.var(axis=0).mean())
        )

    result = {key: float(np.mean(values)) for key, values in per_realisation.items()}
    for condition, values in per_order.items():
        means = np.mean(values, axis=0)
        for order, value in zip(sequence.orders, means, strict=True):
            result[f"{condition}_r2_n{order}"] = float(value)
    return result


def _hyper_from_values(values: dict, h_min: float) -> ReservoirHyperparameters:
    return ReservoirHyperparameters(
        **{key: float(values[key]) for key in SEARCHED_KEYS}, h_min=h_min
    )


def pareto_ranks(points: np.ndarray) -> np.ndarray:
    """Non-dominated sorting rank (0 = Pareto front) for maximisation of every column."""
    n = len(points)
    ranks = np.full(n, -1)
    remaining = np.arange(n)
    rank = 0
    while remaining.size:
        subset = points[remaining]
        dominated = np.zeros(remaining.size, dtype=bool)
        for i in range(remaining.size):
            better_eq = np.all(subset >= subset[i], axis=1)
            strictly = np.any(subset > subset[i], axis=1)
            dominated[i] = np.any(better_eq & strictly)
        ranks[remaining[~dominated]] = rank
        remaining = remaining[dominated]
        rank += 1
    return ranks


def front_distance(points: np.ndarray, index: int) -> float:
    """Distance from ``points[index]`` to the nearest front point in min-max units."""
    span = points.max(axis=0) - points.min(axis=0)
    span = np.where(span > 0, span, 1.0)
    normalised = (points - points.min(axis=0)) / span
    front = normalised[pareto_ranks(points) == 0]
    return float(np.min(np.linalg.norm(front - normalised[index], axis=1)))


def analyse_trials(rows: list[dict], reference_index: int, text_max_order: int) -> dict:
    """Position of the reference (Table I) trial in the full space and the Fig. 2 projections.

    ``fig2a_text_definition`` repeats the Fig. 2a projection with the paper's stated score
    (noisy R^2 summed over ``n <= text_max_order = floor(N/2)``), recomputed from the
    per-order columns of the same trials.
    """
    text_score = np.array(
        [
            sum(row[f"noisy_r2_n{order}"] for order in range(1, text_max_order + 1))
            for row in rows
        ]
    )
    values = np.array([[row[key] for key in OBJECTIVES] for row in rows])
    values = np.column_stack([values, text_score])
    projections = {
        "all_objectives": list(range(len(OBJECTIVES))),
        "fig2a_narma_noisy_vs_bond": [1, 2],
        "fig2b_variability_vs_bond": [3, 2],
        "fig2a_text_definition": [4, 2],
    }
    analysis = {}
    for name, columns in projections.items():
        points = values[:, columns]
        ranks = pareto_ranks(points)
        analysis[name] = {
            "reference_rank": int(ranks[reference_index]),
            "front_size": int(np.sum(ranks == 0)),
            "reference_front_distance": front_distance(points, reference_index),
            "reference_rank_percentile": float(np.mean(ranks < ranks[reference_index])),
        }
    best_noisy = int(np.argmax(values[:, 1]))
    analysis["single_objective_narma_noisy"] = {
        "best_trial": rows[best_noisy]["trial"],
        "best_value": float(values[best_noisy, 1]),
        "reference_value": float(values[reference_index, 1]),
    }
    return analysis


def run_pareto_search(cfg: dict, run_dir: Path) -> dict:
    """Optuna multi-objective search at one qubit count; writes ``trials.csv``."""
    exp = cfg["pareto_search"]
    search_seed = int(cfg["seed"])
    space = exp["search_space"]
    reference = cfg["hyperparameters"]
    h_min = float(reference["h_min"])

    if exp["sampler"] == "nsga2":
        sampler = optuna.samplers.NSGAIISampler(
            seed=search_seed, population_size=int(exp["population_size"])
        )
    elif exp["sampler"] == "random":
        sampler = optuna.samplers.RandomSampler(seed=search_seed)
    else:
        raise ValueError(f"Unknown sampler {exp['sampler']!r}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=["maximize"] * len(OBJECTIVES), sampler=sampler
    )
    study.enqueue_trial({key: float(reference[key]) for key in SEARCHED_KEYS})
    n_trials = int(exp["n_trials"])
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> tuple[float, ...]:
        values = {
            key: trial.suggest_float(key, float(space[key][0]), float(space[key][1]))
            for key in SEARCHED_KEYS
        }
        result = evaluate_hyperparameters(
            cfg, _hyper_from_values(values, h_min), search_seed
        )
        rows.append(
            {"trial": trial.number, "is_table_I": trial.number == 0, **values, **result}
        )
        logger.info(
            "TRAIN_TRIAL_COMPLETED | trial=%d/%d | narma_noiseless=%.4f | narma_noisy=%.4f | bond_dimension=%.2f | variability=%.3e",
            trial.number + 1,
            n_trials,
            result["narma_noiseless"],
            result["narma_noisy"],
            result["bond_dimension"],
            result["variability"],
        )
        return tuple(result[key] for key in OBJECTIVES)

    arch = architecture_from_config(cfg["reservoir"])
    narma_cfg = cfg["dataset"]["narma"]
    logger.info(
        "DATASET_READY | dataset=narma | sequences=one_per_realisation | realisations=%s | window=%d | orders=%s | t_train=%s | t_test=%s",
        exp["n_realisations"],
        arch.n_blocks,
        narma_cfg["orders"],
        narma_cfg["t_train"],
        narma_cfg["t_test"],
    )
    logger.info(
        "TRAINING_STARTED | optimizer=optuna_%s | n_qubits=%d | layers=%d | trials=%d | realisations=%s",
        exp["sampler"],
        arch.n_qubits,
        arch.n_layers,
        n_trials,
        exp["n_realisations"],
    )
    study.optimize(objective, n_trials=n_trials)

    trials_path = run_dir / "trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", trials_path.name, len(rows))

    logger.info("EVALUATION_STARTED | analysis=pareto_position_of_table_I")
    analysis = analyse_trials(rows, reference_index=0, text_max_order=arch.n_blocks)
    logger.info(
        "EVALUATION_COMPLETED | analysis=pareto_position_of_table_I | fig2a_rank=%d | fig2b_rank=%d | all_rank=%d",
        analysis["fig2a_narma_noisy_vs_bond"]["reference_rank"],
        analysis["fig2b_variability_vs_bond"]["reference_rank"],
        analysis["all_objectives"]["reference_rank"],
    )
    return {
        "n_qubits": arch.n_qubits,
        "n_layers": arch.n_layers,
        "n_trials": n_trials,
        "table_I_objectives": {key: rows[0][key] for key in OBJECTIVES},
        "analysis": analysis,
        "trials_path": trials_path.name,
    }
