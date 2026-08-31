from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.lib_datasets import get_qorc_dataset
from lib.lib_qorc_encoding_and_linear_training import (
    create_perceval_noise_model,
    create_qorc_reservoir_classifier,
)
from lib.lib_remote_qorc import create_remote_qorc_processor

QPU_COLOR = "#ff8c00"
SIMULATION_COLOR = "#2f80c9"
ERROR_COLOR = "#000080"


def _standardize_distribution(distribution: np.ndarray) -> np.ndarray:
    mean = float(np.mean(distribution))
    standard_deviation = float(np.std(distribution))
    if standard_deviation == 0:
        raise ValueError("Cannot standardize a distribution with zero variance.")
    return (distribution - mean) / standard_deviation


def _prepare_reservoir(cfg: dict, training_data: np.ndarray, noise) -> object:
    reservoir = create_qorc_reservoir_classifier(
        n_photons=cfg["n_photons"],
        n_components=cfg["n_modes"] - 1,
        seed=cfg["seed"],
        device_name=cfg["device"],
        b_no_bunching=True,
        input_features=training_data.shape[1],
        n_classes=10,
        cache=False,
        noise=noise,
    )
    reservoir.fit_reservoir(training_data)
    return reservoir


def _encode_distribution(reservoir, image: np.ndarray, shots: int) -> np.ndarray:
    reduced_image = reservoir._transform_and_normalize_input(image.reshape(1, -1))
    inputs = np.asarray(reduced_image, dtype=np.float32)
    distribution = reservoir.layer(
        torch.as_tensor(inputs),
        shots=shots,
    )
    return distribution.detach().cpu().numpy()[0]


def _validate_config(cfg: dict) -> None:
    if cfg["n_photons"] != 3 or cfg["n_modes"] != 12:
        raise ValueError("Fig. 5 requires n_photons=3 and n_modes=12.")
    if cfg["distribution_start"] < 0 or cfg["distribution_end"] > 220:
        raise ValueError("Fig. 5 distribution indices must be within [0, 220].")
    if cfg["distribution_end"] <= cfg["distribution_start"]:
        raise ValueError("distribution_end must be greater than distribution_start.")
    if cfg["n_simulation_runs"] <= 0 or cfg["shots"] <= 0:
        raise ValueError("n_simulation_runs and shots must be positive.")
    if not cfg["use_qpu"]:
        raise ValueError("Fig. 5 requires use_qpu=true to produce experimental bars.")
    if not cfg["qpu_device"].startswith("qpu:"):
        raise ValueError("Fig. 5 requires a qpu:* qpu_device.")


def run_fig5_distribution(cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    """Reproduce Fig. 5 by comparing noisy simulation and QPU distributions.

    Parameters
    ----------
    cfg : dict
        Resolved Fig. 5 experiment configuration.
    run_dir : pathlib.Path
        Timestamped output directory for experiment artifacts.
    logger : logging.Logger
        Logger receiving experiment progress messages.

    Returns
    -------
    None
        Writes the distribution data and figure to ``run_dir``.

    Raises
    ------
    ValueError
        If the Fig. 5 configuration is invalid.
    """
    _validate_config(cfg)
    training_data, _, _, _ = get_qorc_dataset(
        "mnist", sampling="full", seed=cfg["seed"]
    )
    training_data = (
        training_data.reshape(training_data.shape[0], -1).astype(np.float32) / 255.0
    )
    image_index = int(np.random.default_rng(cfg["seed"]).integers(len(training_data)))
    image = training_data[image_index]

    simulation_reservoir = _prepare_reservoir(
        cfg,
        training_data,
        create_perceval_noise_model(
            enabled=True,
            indistinguishability=cfg["noise_indistinguishability"],
            g2=cfg["noise_g2"],
            g2_distinguishable=cfg["noise_g2_distinguishable"],
        ),
    )
    qpu_reservoir = _prepare_reservoir(cfg, training_data, noise=None)
    qpu_processor = create_remote_qorc_processor(
        cfg["qpu_device"],
        qpu_reservoir.layer,
        cfg["shots"],
        logger,
    )

    qpu_input = np.asarray(
        qpu_reservoir._transform_and_normalize_input(image.reshape(1, -1)),
        dtype=np.float32,
    )
    qpu_distribution = (
        qpu_processor.forward(
            qpu_reservoir.layer,
            torch.as_tensor(qpu_input),
        )
        .detach()
        .cpu()
        .numpy()[0]
    )
    standardized_qpu = _standardize_distribution(qpu_distribution)

    simulation_values = np.empty((cfg["n_simulation_runs"], 220), dtype=np.float64)
    for run_index in range(cfg["n_simulation_runs"]):
        logger.info("Fig. 5 simulation %s/%s", run_index + 1, cfg["n_simulation_runs"])
        simulation_distribution = _encode_distribution(
            simulation_reservoir, image, cfg["shots"]
        )
        simulation_values[run_index] = _standardize_distribution(
            simulation_distribution
        )

    start = cfg["distribution_start"]
    end = cfg["distribution_end"]
    selected_simulation_values = simulation_values[:, start:end]
    summary = {
        "image_index": image_index,
        "distribution_indices": list(range(start, end)),
        "qpu_standardized": standardized_qpu[start:end].tolist(),
        "simulation_mean": selected_simulation_values.mean(axis=0).tolist(),
        "simulation_min": selected_simulation_values.min(axis=0).tolist(),
        "simulation_max": selected_simulation_values.max(axis=0).tolist(),
        "simulation_histogram_peak": [
            int(np.histogram(values, bins=20)[0].max())
            for values in selected_simulation_values.T
        ],
        "configuration": cfg,
    }
    (run_dir / "fig5_distribution.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (run_dir / "fig5_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distribution_index",
                "qpu",
                "simulation_mean",
                "simulation_min",
                "simulation_max",
            ]
        )
        writer.writerows(
            zip(
                summary["distribution_indices"],
                summary["qpu_standardized"],
                summary["simulation_mean"],
                summary["simulation_min"],
                summary["simulation_max"],
            )
        )
    _plot_fig5(summary, run_dir / "fig5_distribution.png")


def _plot_fig5(summary: dict, output_path: Path) -> None:
    distribution_indices = np.asarray(summary["distribution_indices"])
    qpu_values = np.asarray(summary["qpu_standardized"])
    simulation_mean = np.asarray(summary["simulation_mean"])
    simulation_min = np.asarray(summary["simulation_min"])
    simulation_max = np.asarray(summary["simulation_max"])
    histogram_peak = np.asarray(summary["simulation_histogram_peak"], dtype=float)
    line_widths = 0.8 + 2.4 * histogram_peak / histogram_peak.max()

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.bar(distribution_indices, qpu_values, color=QPU_COLOR, alpha=0.55, label="QPU")
    axis.vlines(
        distribution_indices,
        simulation_min,
        simulation_max,
        color=ERROR_COLOR,
        linewidth=line_widths,
        zorder=3,
    )
    axis.scatter(
        distribution_indices,
        simulation_mean,
        color=SIMULATION_COLOR,
        s=24,
        label="Simulation",
        zorder=4,
    )
    axis.set_xlabel("Distribution entry index")
    axis.set_ylabel("Value (a.u.)")
    axis.set_xlim(distribution_indices[0] - 0.5, distribution_indices[-1] + 0.5)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
