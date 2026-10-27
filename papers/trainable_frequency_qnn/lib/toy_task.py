"""Cosine-series regression experiments E1-E5 (paper Sec. V, Figs. 2-3)."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import build_dataloader, nyquist_point_count, resolve_torch_dtype
from .models import build_network, parameter_summary
from .qnn import TrainableFrequencyQNN
from .spectrum import accessible_frequencies, dominant_frequencies, prediction_spectrum
from .training import build_optimizer, endless_batches

logger = logging.getLogger(__name__)


def _mean_squared_error(
    model: torch.nn.Module, loader, device
) -> tuple[float, np.ndarray]:
    predictions = []
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            squared_error += float(((outputs - targets) ** 2).sum())
            count += targets.numel()
            predictions.append(outputs.cpu().numpy())
    return squared_error / count, np.concatenate(predictions)


def run_toy_regression(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Train one model on one cosine series and write its artifacts.

    Artifacts written to ``run_dir``: ``losses.csv`` (per-iteration batch loss and
    periodic full-train MSE), ``predictions.npz`` (train and dense grids) and,
    for QNNs, ``spectrum.npz`` (DFT of the trained prediction).

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``task = "toy_regression"``).
    run_dir : pathlib.Path
        Run directory.

    Returns
    -------
    dict
        Metrics for ``metrics.json``.
    """
    dtype = resolve_torch_dtype(cfg)
    device = torch.device(cfg["device"])
    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]
    frequencies = [float(w) for w in cfg["dataset"]["frequencies"]]

    train_loader = build_dataloader(cfg, "train")
    train_eval_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=evaluation_cfg["batch_size"], shuffle=False
    )
    dense_loader = build_dataloader(cfg, "dense")
    logger.info(
        "DATASET_READY | name=cosine_series | frequencies=%s | train_points=%d | dense_points=%d",
        frequencies,
        len(train_loader.dataset),
        len(dense_loader.dataset),
    )

    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    model = build_network(cfg["model"], 1, dtype, generator).to(device)
    parameters = parameter_summary(model)
    logger.info(
        "MODEL_READY | family=%s | parameters=%s", cfg["model"]["family"], parameters
    )
    optimizer = build_optimizer(model.parameters(), training_cfg)

    iterations = training_cfg["iterations"]
    evaluate_every = evaluation_cfg["every"]
    batches = endless_batches(train_loader, device)
    history: list[dict[str, float]] = []
    logger.info(
        "TRAINING_STARTED | iterations=%d | batch_size=%d | optimizer=adam | lr=%g",
        iterations,
        training_cfg["batch_size"],
        training_cfg["lr"],
    )
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        inputs, targets = next(batches)
        optimizer.zero_grad()
        loss = torch.mean((model(inputs) - targets) ** 2)
        loss.backward()
        optimizer.step()
        record = {"iteration": iteration, "batch_loss": loss.item()}
        if iteration % evaluate_every == 0 or iteration == iterations:
            record["train_mse"], _ = _mean_squared_error(
                model, train_eval_loader, device
            )
            logger.info(
                "TRAIN_ITERATION_COMPLETED | iteration=%d/%d | batch_loss=%.6e | train_mse=%.6e",
                iteration,
                iterations,
                record["batch_loss"],
                record["train_mse"],
            )
        else:
            logger.info(
                "TRAIN_ITERATION_COMPLETED | iteration=%d/%d | batch_loss=%.6e",
                iteration,
                iterations,
                record["batch_loss"],
            )
        history.append(record)
    training_seconds = time.perf_counter() - started
    logger.info("TRAINING_COMPLETED | seconds=%.2f", training_seconds)

    losses_path = run_dir / "losses.csv"
    with losses_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["iteration", "batch_loss", "train_mse"]
        )
        writer.writeheader()
        writer.writerows(history)
    logger.info("ARTIFACT_WRITTEN | path=%s", losses_path.name)

    model.eval()
    logger.info("EVALUATION_STARTED | split=train")
    train_mse, train_predictions = _mean_squared_error(model, train_eval_loader, device)
    logger.info("EVALUATION_COMPLETED | split=train | mse=%.6e", train_mse)
    logger.info("EVALUATION_STARTED | split=dense")
    dense_mse, dense_predictions = _mean_squared_error(model, dense_loader, device)
    logger.info("EVALUATION_COMPLETED | split=dense | mse=%.6e", dense_mse)

    predictions_path = run_dir / "predictions.npz"
    np.savez(
        predictions_path,
        train_x=train_loader.dataset.tensors[0].numpy()[:, 0],
        train_y=train_loader.dataset.tensors[1].numpy(),
        train_prediction=train_predictions,
        dense_x=dense_loader.dataset.tensors[0].numpy()[:, 0],
        dense_y=dense_loader.dataset.tensors[1].numpy(),
        dense_prediction=dense_predictions,
    )
    logger.info("ARTIFACT_WRITTEN | path=%s", predictions_path.name)

    metrics: dict[str, Any] = {
        "task": "toy_regression",
        "data_frequencies": frequencies,
        "n_train_points": nyquist_point_count(frequencies, cfg["dataset"]["domain_pi"]),
        "train_mse": train_mse,
        "dense_mse": dense_mse,
        "final_batch_loss": history[-1]["batch_loss"],
        "parameters": parameters,
        "training_seconds": training_seconds,
        "seconds_per_iteration": training_seconds / iterations,
    }

    if isinstance(model, TrainableFrequencyQNN):
        multipliers = [float(w) for w in model.encoding_frequencies().detach().cpu()]
        metrics["encoding_multipliers"] = multipliers
        metrics["accessible_frequencies"] = accessible_frequencies(multipliers)
        spectrum_cfg = evaluation_cfg["spectrum"]
        angular, magnitudes = prediction_spectrum(
            model,
            spectrum_cfg["half_width_pi"],
            spectrum_cfg["n_points"],
            dtype,
            device,
        )
        metrics["dft_peaks"] = dominant_frequencies(
            angular, magnitudes, spectrum_cfg["n_peaks"]
        )
        spectrum_path = run_dir / "spectrum.npz"
        np.savez(spectrum_path, angular_frequency=angular, magnitude=magnitudes)
        logger.info(
            "EVALUATION_COMPLETED | split=spectrum | multipliers=%s | dft_peaks=%s",
            [round(w, 4) for w in multipliers],
            [round(w, 4) for w in metrics["dft_peaks"]],
        )
        logger.info("ARTIFACT_WRITTEN | path=%s", spectrum_path.name)
    elif cfg["model"]["family"] == "fourier":
        metrics["learned_frequencies"] = [
            float(w) for w in model.frequencies.detach().cpu()
        ]

    return metrics
