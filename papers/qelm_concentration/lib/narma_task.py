"""NARMA-n forecasting with the QELM and matched classical baselines (E2, paper Fig. 3a).

For each (sequence seed, reservoir seed) the reservoir is run once on the windows; every
task order and noise condition reuses those features. Baselines see the same sequences.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baselines import EchoStateNetwork, polynomial_features, random_tanh_features
from .data import narma_sequence_from_config
from .qelm import (
    apply_feature_noise,
    architecture_from_config,
    build_reservoir,
    real_dtype_from_config,
    ring_features,
)
from .readout import alpha_grid, fit_ridge_regression
from .reservoir import ReservoirHyperparameters

logger = logging.getLogger(__name__)

# ``noisy(condition, rng)`` returns the reservoir features under one configured noise condition.
NoisyFeatures = Callable[[dict, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class ReservoirFeatureSource:
    """Reservoir features of NARMA windows, independent of the simulated platform.

    Attributes
    ----------
    window : int
        Window length ``L`` the reservoir encodes.
    metadata : dict
        Size fields written to ``metrics.json`` (qubits or modes/photons, layers).
    compute : callable
        ``compute(reservoir_seed, windows) -> (exact_features, noisy)`` for windows of
        shape ``(n_windows, window)``.
    """

    window: int
    metadata: dict
    compute: Callable[[int, np.ndarray], tuple[np.ndarray, NoisyFeatures]]


def qubit_feature_source(cfg: dict) -> ReservoirFeatureSource:
    """Kicked-Ising QELM features (``cfg["reservoir"]``, Table I hyperparameters)."""
    exp = cfg["narma_scaling"]
    arch = architecture_from_config(cfg["reservoir"])
    hyper = ReservoirHyperparameters.from_config(cfg["hyperparameters"])
    real_dtype = real_dtype_from_config(cfg)
    chunk = int(cfg["reservoir"]["chunk_size"])

    def compute(reservoir_seed: int, windows: np.ndarray):
        reservoir = build_reservoir(
            arch, hyper, int(reservoir_seed), cfg["device"], real_dtype
        )
        features, weights = ring_features(reservoir, windows, exp["feature_set"], chunk)
        return features, lambda condition, rng: apply_feature_noise(
            features, weights, condition, rng
        )

    return ReservoirFeatureSource(
        window=arch.n_blocks,
        metadata={"n_qubits": arch.n_qubits, "n_layers": arch.n_layers},
        compute=compute,
    )


def orders_r2(train_x, test_x, sequence, alphas) -> list[float]:
    """Test R^2 of one feature set for every task order of ``sequence``."""
    return [
        fit_ridge_regression(
            train_x,
            sequence.train_targets[:, j],
            test_x,
            sequence.test_targets[:, j],
            alphas,
        )["r2"]
        for j in range(len(sequence.orders))
    ]


def esn_r2(
    sequence, esn_cfg: dict, alphas, seed: int
) -> tuple[list[float], list[dict]]:
    """ESN test R^2 per order with (spectral radius, input scale) chosen on a train tail."""
    steps = sequence.first_window_index + np.arange(
        len(sequence.train_targets) + len(sequence.test_targets)
    )
    n_train = len(sequence.train_targets)
    n_fit = int(round(n_train * (1 - float(esn_cfg["validation_fraction"]))))
    candidates = []
    for radius in esn_cfg["spectral_radius"]:
        for scale in esn_cfg["input_scale"]:
            esn = EchoStateNetwork(
                int(esn_cfg["n_units"]), float(radius), float(scale), seed
            )
            states = esn.states(sequence.raw_inputs)[steps]
            candidates.append(((float(radius), float(scale)), states))

    r2_values, chosen = [], []
    for j in range(len(sequence.orders)):
        train_y = sequence.train_targets[:, j]
        best = None
        for params, states in candidates:
            score = fit_ridge_regression(
                states[:n_fit],
                train_y[:n_fit],
                states[n_fit:n_train],
                train_y[n_fit:],
                alphas,
            )["r2"]
            if best is None or score > best[0]:
                best = (score, params, states)
        _, params, states = best
        r2_values.append(
            fit_ridge_regression(
                states[:n_train],
                train_y,
                states[n_train:],
                sequence.test_targets[:, j],
                alphas,
            )["r2"]
        )
        chosen.append({"spectral_radius": params[0], "input_scale": params[1]})
    return r2_values, chosen


def cumulative_r2(
    r2_values: list[float], orders: tuple[int, ...], max_order: int
) -> float:
    """Sum of R^2 over orders ``1..max_order`` (the paper's NARMA score)."""
    return float(
        sum(
            r2
            for r2, order in zip(r2_values, orders, strict=True)
            if order <= max_order
        )
    )


def run_narma_scaling(
    cfg: dict, run_dir: Path, source: ReservoirFeatureSource | None = None
) -> dict:
    """QELM and baselines on NARMA-n for one reservoir size (E2).

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    run_dir : pathlib.Path
        Run directory (unused; kept for the task signature).
    source : ReservoirFeatureSource | None
        Reservoir features. If omitted, the kicked-Ising qubit QELM of
        `qubit_feature_source` is used. Default value is None.
    """
    exp = cfg["narma_scaling"]
    if source is None:
        source = qubit_feature_source(cfg)
    alphas = alpha_grid(cfg["readout"])
    window = source.window
    records = []

    n_realisations = len(exp["sequence_seeds"]) * len(exp["reservoir_seeds"])
    completed_realisations = 0
    logger.info(
        "TRAINING_STARTED | readout=closed_form_ridge | %s | realisations=%d | orders=%d",
        " | ".join(f"{key}={value}" for key, value in source.metadata.items()),
        n_realisations,
        len(cfg["dataset"]["narma"]["orders"]),
    )
    logger.info(
        "DATASET_READY | dataset=narma | sequence_seeds=%s | window=%d | orders=%s",
        list(exp["sequence_seeds"]),
        window,
        list(cfg["dataset"]["narma"]["orders"]),
    )
    for sequence_seed in exp["sequence_seeds"]:
        sequence = narma_sequence_from_config(
            cfg, window, np.random.default_rng(int(sequence_seed))
        )
        logger.info(
            "DATASET_SPLIT_READY | dataset=narma | sequence_seed=%s | window=%d | train=%d | test=%d | orders=%s",
            sequence_seed,
            window,
            len(sequence.train_targets),
            len(sequence.test_targets),
            list(sequence.orders),
        )
        for reservoir_seed in exp["reservoir_seeds"]:
            logger.info(
                "SIMULATION_STARTED | sequence_seed=%s | reservoir_seed=%s",
                sequence_seed,
                reservoir_seed,
            )
            windows = np.concatenate([sequence.train_windows, sequence.test_windows])
            features, noisy_features = source.compute(int(reservoir_seed), windows)
            n_train = len(sequence.train_windows)
            for condition in exp["noise_conditions"]:
                noise_rng = np.random.default_rng(
                    [int(sequence_seed), int(reservoir_seed), 11]
                )
                noisy = noisy_features(condition, noise_rng)
                r2_values = orders_r2(
                    noisy[:n_train], noisy[n_train:], sequence, alphas
                )
                records.append(
                    {
                        "model": "qelm",
                        "condition": condition["label"],
                        "sequence_seed": int(sequence_seed),
                        "reservoir_seed": int(reservoir_seed),
                        "n_features": int(features.shape[1]),
                        "r2": r2_values,
                        "cumulative_r2": cumulative_r2(
                            r2_values, sequence.orders, window
                        ),
                    }
                )
                logger.info(
                    "EVALUATION_COMPLETED | model=qelm | condition=%s | sequence_seed=%s | reservoir_seed=%s | cumulative_r2=%.4f",
                    condition["label"],
                    sequence_seed,
                    reservoir_seed,
                    records[-1]["cumulative_r2"],
                )
            completed_realisations += 1
            logger.info(
                "TRAIN_REALISATION_COMPLETED | realisation=%d/%d | sequence_seed=%s | reservoir_seed=%s",
                completed_realisations,
                n_realisations,
                sequence_seed,
                reservoir_seed,
            )

        baselines = exp["baselines"]
        n_features_qelm = int(features.shape[1])
        baseline_features = {}
        if "raw_window" in baselines:
            baseline_features["raw_window"] = (
                sequence.train_windows,
                sequence.test_windows,
            )
        if "degree2_window" in baselines:
            baseline_features["degree2_window"] = polynomial_features(
                sequence.train_windows, sequence.test_windows
            )
        if "matched_elm" in baselines:
            baseline_features["matched_elm"] = random_tanh_features(
                sequence.train_windows,
                sequence.test_windows,
                n_features_qelm,
                float(exp["elm_input_scale"]),
                int(sequence_seed),
            )
        for name, (train_x, test_x) in baseline_features.items():
            r2_values = orders_r2(train_x, test_x, sequence, alphas)
            records.append(
                {
                    "model": name,
                    "condition": "none",
                    "sequence_seed": int(sequence_seed),
                    "n_features": int(train_x.shape[1]),
                    "r2": r2_values,
                    "cumulative_r2": cumulative_r2(r2_values, sequence.orders, window),
                }
            )
            logger.info(
                "EVALUATION_COMPLETED | model=%s | sequence_seed=%s | cumulative_r2=%.4f",
                name,
                sequence_seed,
                records[-1]["cumulative_r2"],
            )
        if "esn" in baselines:
            r2_values, chosen = esn_r2(sequence, exp["esn"], alphas, int(sequence_seed))
            records.append(
                {
                    "model": "esn",
                    "condition": "none",
                    "sequence_seed": int(sequence_seed),
                    "n_features": int(exp["esn"]["n_units"]),
                    "r2": r2_values,
                    "cumulative_r2": cumulative_r2(r2_values, sequence.orders, window),
                    "chosen_params": chosen,
                }
            )
            logger.info(
                "EVALUATION_COMPLETED | model=esn | sequence_seed=%s | cumulative_r2=%.4f",
                sequence_seed,
                records[-1]["cumulative_r2"],
            )

    orders = [int(order) for order in cfg["dataset"]["narma"]["orders"]]
    return {
        **source.metadata,
        "window": window,
        "orders": orders,
        "records": records,
        "summary": summarise_narma(records, orders),
    }


def summarise_narma(records: list[dict], orders: list[int]) -> dict:
    """Mean and std of R^2 per order and of the cumulative score for each model/condition."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(f"{record['model']}|{record['condition']}", []).append(record)
    summary = {}
    for label, items in groups.items():
        r2 = np.array([item["r2"] for item in items])
        cumulative = np.array([item["cumulative_r2"] for item in items])
        summary[label] = {
            "n": len(items),
            "orders": orders,
            "r2_mean": r2.mean(axis=0).tolist(),
            "r2_std": r2.std(axis=0).tolist(),
            "cumulative_r2_mean": float(cumulative.mean()),
            "cumulative_r2_std": float(cumulative.std()),
        }
    return summary
