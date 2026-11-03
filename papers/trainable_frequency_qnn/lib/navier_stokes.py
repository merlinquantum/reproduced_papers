"""Physics-informed (DQC) Navier-Stokes experiments E6-E7 (paper Sec. VI, App. C).

Two networks are trained jointly (paper Sec. VI): one predicts the pressure
``p``, the other the stream function ``psi`` with ``u = d psi / dy`` and
``v = -d psi / dx``, so continuity holds by construction. The loss is
``L = L_PDE + L_data`` with the squared x/y momentum residuals (paper Eqs. 5-7)
on collocation points and squared u, v, p errors on the sparse data sub-grid.
Derivatives with respect to the inputs come from autograd; the paper uses
derivative quantum circuits with the parameter-shift rule, which give the same
values in noiseless simulation (LOG.md, A13).
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import (
    build_dataloader,
    coordinate_bounds,
    input_scale_factors,
    load_cylinder_wake,
    resolve_torch_dtype,
)
from .models import build_network, circuit_evaluation_cost_factor, parameter_summary
from .training import build_optimizer, endless_batches

logger = logging.getLogger(__name__)

FIELDS = ("u", "v", "p")


class StreamPressureModel(nn.Module):
    """Pair of networks predicting (psi, p) from model inputs (x, y, t)."""

    def __init__(self, stream_network: nn.Module, pressure_network: nn.Module) -> None:
        super().__init__()
        self.stream_network = stream_network
        self.pressure_network = pressure_network


def _input_gradient(output: torch.Tensor, inputs: torch.Tensor, create_graph: bool):
    # Each output depends only on its own input row, so the gradient of the sum
    # gives every per-sample derivative at once.
    return torch.autograd.grad(output.sum(), inputs, create_graph=create_graph)[0]


def velocity_pressure(
    model: StreamPressureModel,
    inputs: torch.Tensor,
    scales: torch.Tensor,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    """Return predicted u, v, p at model inputs.

    ``scales`` holds d(model input)/d(physical coordinate) for (x, y, t), so
    derivatives are taken in physical units for any coordinate transform.
    """
    inputs = inputs.requires_grad_(True)
    psi = model.stream_network(inputs)
    psi_gradient = _input_gradient(psi, inputs, create_graph) * scales
    return {
        "u": psi_gradient[:, 1],
        "v": -psi_gradient[:, 0],
        "p": model.pressure_network(inputs),
    }


def momentum_residuals(
    model: StreamPressureModel,
    inputs: torch.Tensor,
    scales: torch.Tensor,
    reynolds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the x and y momentum residuals zeta_x, zeta_y (paper Eqs. 5-6)."""
    fields = velocity_pressure(model, inputs, scales, create_graph=True)
    u, v, p = fields["u"], fields["v"], fields["p"]
    u_gradient = _input_gradient(u, inputs, True) * scales
    v_gradient = _input_gradient(v, inputs, True) * scales
    p_gradient = _input_gradient(p, inputs, True) * scales
    u_x, u_y, u_t = u_gradient[:, 0], u_gradient[:, 1], u_gradient[:, 2]
    v_x, v_y, v_t = v_gradient[:, 0], v_gradient[:, 1], v_gradient[:, 2]
    u_xx = _input_gradient(u_x, inputs, True)[:, 0] * scales[0]
    u_yy = _input_gradient(u_y, inputs, True)[:, 1] * scales[1]
    v_xx = _input_gradient(v_x, inputs, True)[:, 0] * scales[0]
    v_yy = _input_gradient(v_y, inputs, True)[:, 1] * scales[1]
    zeta_x = u_t + u * u_x + v * u_y - (u_xx + u_yy) / reynolds + p_gradient[:, 0]
    zeta_y = v_t + u * v_x + v * v_y - (v_xx + v_yy) / reynolds + p_gradient[:, 1]
    return zeta_x, zeta_y


def maerm(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Mean absolute error relative to the median (paper Sec. VI), in percent.

    ``maerm`` divides by ``|median(reference)|`` as written in the paper;
    ``maerm_abs_median`` divides by ``median(|reference|)`` (LOG.md, A10).
    """
    absolute_error = np.abs(prediction - reference)
    return {
        "maerm": 100 * float(np.mean(absolute_error) / abs(np.median(reference))),
        "maerm_abs_median": 100
        * float(np.mean(absolute_error) / np.median(np.abs(reference))),
    }


def _evaluate(
    model: StreamPressureModel, cfg: dict[str, Any], scales: torch.Tensor, device
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    loader = build_dataloader(cfg, "evaluation")
    predictions, references = [], []
    for inputs, targets in loader:
        fields = velocity_pressure(model, inputs.to(device), scales, create_graph=False)
        predictions.append(
            torch.stack([fields[f] for f in FIELDS], 1).detach().cpu().numpy()
        )
        references.append(targets.numpy())
    evaluation_times = list(cfg["dataset"]["evaluation_times"])
    n_times = len(evaluation_times)
    predictions = np.concatenate(predictions).reshape(n_times, -1, len(FIELDS))
    references = np.concatenate(references).reshape(n_times, -1, len(FIELDS))

    per_time = []
    for time_index, time_value in enumerate(evaluation_times):
        row: dict[str, Any] = {"t": time_value}
        for field_index, field in enumerate(FIELDS):
            errors = maerm(
                predictions[time_index, :, field_index],
                references[time_index, :, field_index],
            )
            row[f"{field}_maerm"] = errors["maerm"]
            row[f"{field}_maerm_abs_median"] = errors["maerm_abs_median"]
        per_time.append(row)

    table_times = set(cfg["evaluation"]["table_times"])
    table_rows = [row for row in per_time if row["t"] in table_times]
    if len(table_rows) != len(table_times):
        raise ValueError(
            "Every evaluation.table_times entry must be in dataset.evaluation_times."
        )
    table = {}
    for field in FIELDS:
        for metric in ("maerm", "maerm_abs_median"):
            values = np.array([row[f"{field}_{metric}"] for row in table_rows])
            table[f"{field}_{metric}"] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
    return {"per_time": per_time, "table": table}, predictions, references


def run_navier_stokes(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Train the (psi, p) model with the DQC loss and write its artifacts.

    Artifacts written to ``run_dir``: ``losses.csv`` (per-iteration PDE, data and
    total loss), ``predictions.npz`` (u, v, p predictions and references at every
    evaluation time) and ``model_state.pt``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``task = "navier_stokes"``).
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
    dataset_cfg = cfg["dataset"]

    wake = load_cylinder_wake(cfg)
    bounds = coordinate_bounds(wake)
    scales = torch.as_tensor(
        input_scale_factors(dataset_cfg["coordinate_transform"], bounds),
        dtype=dtype,
        device=device,
    )
    collocation_loader = build_dataloader(cfg, "collocation")
    data_loader = build_dataloader(cfg, "data")
    logger.info(
        "DATASET_READY | name=cylinder_wake | time_window=%s | transform=%s | "
        "collocation_points=%d | data_points=%d | evaluation_times=%s",
        dataset_cfg["time_window"],
        dataset_cfg["coordinate_transform"],
        len(collocation_loader.dataset),
        len(data_loader.dataset),
        dataset_cfg["evaluation_times"],
    )

    # One seeded generator, stream network first, so FF and TF runs with the
    # same seed share their ansatz initialisation (paired seeds).
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    stream_network = build_network(cfg["model"], 3, dtype, generator)
    pressure_network = build_network(cfg["model"], 3, dtype, generator)
    model = StreamPressureModel(stream_network, pressure_network).to(device)
    summaries = [parameter_summary(stream_network), parameter_summary(pressure_network)]
    cost_factor = circuit_evaluation_cost_factor(summaries)
    logger.info(
        "MODEL_READY | family=%s | stream=%s | pressure=%s | cost_factor=%.4f",
        cfg["model"]["family"],
        summaries[0],
        summaries[1],
        cost_factor,
    )
    optimizer = build_optimizer(model.parameters(), training_cfg)

    reynolds = float(dataset_cfg["reynolds"])
    iterations = training_cfg["iterations"]
    collocation_batches = endless_batches(collocation_loader, device)
    data_batches = endless_batches(data_loader, device)
    history: list[dict[str, float]] = []
    logger.info(
        "TRAINING_STARTED | iterations=%d | batch_size=%d | optimizer=adam | lr=%g",
        iterations,
        training_cfg["batch_size"],
        training_cfg["lr"],
    )
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        collocation_inputs, _ = next(collocation_batches)
        data_inputs, data_targets = next(data_batches)
        optimizer.zero_grad()
        zeta_x, zeta_y = momentum_residuals(model, collocation_inputs, scales, reynolds)
        pde_loss = torch.mean(zeta_x**2 + zeta_y**2)
        fields = velocity_pressure(model, data_inputs, scales, create_graph=True)
        predicted = torch.stack([fields[f] for f in FIELDS], 1)
        data_loss = torch.mean(((predicted - data_targets) ** 2).sum(dim=1))
        loss = pde_loss + data_loss
        loss.backward()
        optimizer.step()
        record = {
            "iteration": iteration,
            "loss": loss.item(),
            "pde_loss": pde_loss.item(),
            "data_loss": data_loss.item(),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        logger.info(
            "TRAIN_ITERATION_COMPLETED | iteration=%d/%d | loss=%.6e | pde_loss=%.6e | "
            "data_loss=%.6e | elapsed_s=%.1f",
            iteration,
            iterations,
            record["loss"],
            record["pde_loss"],
            record["data_loss"],
            record["seconds"],
        )
    training_seconds = time.perf_counter() - started
    logger.info("TRAINING_COMPLETED | seconds=%.2f", training_seconds)

    losses_path = run_dir / "losses.csv"
    with losses_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["iteration", "loss", "pde_loss", "data_loss", "seconds"]
        )
        writer.writeheader()
        writer.writerows(history)
    logger.info("ARTIFACT_WRITTEN | path=%s", losses_path.name)

    state_path = run_dir / "model_state.pt"
    torch.save(model.state_dict(), state_path)
    logger.info("ARTIFACT_WRITTEN | path=%s", state_path.name)

    logger.info(
        "EVALUATION_STARTED | split=evaluation | times=%s",
        dataset_cfg["evaluation_times"],
    )
    evaluation, predictions, references = _evaluate(model, cfg, scales, device)
    table = evaluation["table"]
    logger.info(
        "EVALUATION_COMPLETED | split=evaluation | mean_maerm_u=%.2f | mean_maerm_v=%.2f | "
        "mean_maerm_p=%.2f",
        table["u_maerm"]["mean"],
        table["v_maerm"]["mean"],
        table["p_maerm"]["mean"],
    )
    predictions_path = run_dir / "predictions.npz"
    np.savez(
        predictions_path,
        times=np.asarray(dataset_cfg["evaluation_times"]),
        fields=np.asarray(FIELDS),
        prediction=predictions,
        reference=references,
    )
    logger.info("ARTIFACT_WRITTEN | path=%s", predictions_path.name)

    frequency_parameters = {}
    for name, network in (("stream", stream_network), ("pressure", pressure_network)):
        if hasattr(network, "encoding_frequencies"):
            frequency_parameters[name] = [
                float(w) for w in network.encoding_frequencies().detach().cpu()
            ]

    return {
        "task": "navier_stokes",
        "final_loss": history[-1]["loss"],
        "final_pde_loss": history[-1]["pde_loss"],
        "final_data_loss": history[-1]["data_loss"],
        "maerm_table": table,
        "maerm_per_time": evaluation["per_time"],
        "parameters": {"stream": summaries[0], "pressure": summaries[1]},
        "cost_factor": cost_factor,
        "encoding_multipliers": frequency_parameters,
        "training_seconds": training_seconds,
        "seconds_per_iteration": training_seconds / iterations,
    }
