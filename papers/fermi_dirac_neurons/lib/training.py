"""Full-batch gradient descent shared by all models (Sec. VI C, Eq. (119)).

All models of one run are updated in lock step, one epoch per gradient step,
as in the authors' notebooks: at each epoch the loss is evaluated at the
current parameters, recorded, and then the parameters move by
``-learning_rate * gradient``. ``history.csv`` therefore stores, for epoch e,
the loss *before* update e, matching ``outputs/*.csv`` in the authors' repo.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np

from .models import PREDICTION_RULES, Dataset

logger = logging.getLogger(__name__)


def evaluate_model(
    model, theta: np.ndarray, data: Dataset, task_kind: str, loss: str
) -> dict:
    """Return evaluation metrics of one model on one split.

    Parameters
    ----------
    model : HamiltonianNeuron or PauliFeatureNeuron
        Trained model.
    theta : numpy.ndarray
        Parameters.
    data : Dataset
        Evaluation split.
    task_kind : str
        ``"regression"`` (squared loss, Figs. 7 and 9) or ``"classification"``
        (logistic loss, Fig. 8 and Table II).
    loss : str
        Training loss name, evaluated on the split as well.

    Returns
    -------
    dict
        ``loss`` and, for regression, ``mse``; for classification,
        ``accuracy_<rule>`` for every prediction rule.
    """
    value, _ = model.loss_and_gradient(theta, data, loss)
    metrics = {"loss": value}
    if task_kind == "regression":
        residuals = model.outputs(theta, data) - data.targets
        metrics["mse"] = float(np.mean(residuals**2))
    else:
        for rule in PREDICTION_RULES:
            predictions = model.classify(theta, data, rule)
            metrics[f"accuracy_{rule}"] = float(np.mean(predictions == data.targets))
    return metrics


def train_models(
    models: dict,
    initial_parameters: dict[str, np.ndarray],
    train_data: Dataset,
    evaluation_data: dict[str, Dataset],
    training_cfg: dict,
    task_kind: str,
    history_path: Path,
    shot_streams: dict | None = None,
    learning_rates: dict | None = None,
) -> dict:
    """Train several models with full-batch gradient descent.

    Parameters
    ----------
    models : dict
        Model name -> model instance.
    initial_parameters : dict
        Model name -> initial parameter vector.
    train_data : Dataset
        Training split.
    evaluation_data : dict
        Split name -> dataset evaluated every ``eval_every`` epochs and at the
        end.
    training_cfg : dict
        Keys ``loss``, ``learning_rate``, ``epochs``, ``eval_every``.
        Optional (Phase 5): ``optimizer`` (``"gd"``, the paper's full-batch
        gradient descent, or ``"adam"``; default ``"gd"``) and
        ``on_nonfinite`` (``"raise"``, or ``"freeze"`` to stop updating a
        model whose loss or gradient becomes non-finite and mark it
        diverged; default ``"raise"``).
    task_kind : str
        ``"regression"`` or ``"classification"``.
    history_path : pathlib.Path
        CSV file receiving the per-epoch training loss of every model.
    shot_streams : dict or None
        Model name -> random generator for finite-shot gradient estimates
        (``training_cfg["finite_shots"]`` shots per state and parameter,
        squared loss only). None trains with exact gradients, as in the
        paper. Default value is None.
    learning_rates : dict or None
        Model name -> learning rate overriding ``training_cfg["learning_rate"]``
        (per-model tuned rates, Phase 5). Default value is None.

    Returns
    -------
    dict
        Final parameters, final training loss, evaluation history and final
        evaluation metrics per model.

    Raises
    ------
    FloatingPointError
        If a loss or gradient becomes non-finite.
    """
    loss_name = training_cfg["loss"]
    optimizer = training_cfg.get("optimizer", "gd")
    on_nonfinite = training_cfg.get("on_nonfinite", "raise")
    if optimizer not in ("gd", "adam"):
        raise ValueError(f"optimizer must be gd or adam, got {optimizer!r}")
    if on_nonfinite not in ("raise", "freeze"):
        raise ValueError(f"on_nonfinite must be raise or freeze, got {on_nonfinite!r}")
    rates = {
        name: float((learning_rates or {}).get(name, training_cfg["learning_rate"]))
        for name in models
    }
    epochs = int(training_cfg["epochs"])
    eval_every = int(training_cfg["eval_every"])
    parameters = {
        name: np.array(theta, dtype=float) for name, theta in initial_parameters.items()
    }
    names = list(models)
    evaluations: list[dict] = []
    diverged: dict[str, int | None] = dict.fromkeys(names)
    seconds = dict.fromkeys(names, 0.0)
    # Adam moments (Kingma & Ba, beta = 0.9, 0.999, eps = 1e-8), per model.
    first_moment = {name: np.zeros_like(parameters[name]) for name in names}
    second_moment = {name: np.zeros_like(parameters[name]) for name in names}

    def run_evaluation(epoch: int) -> None:
        for split, data in evaluation_data.items():
            logger.info("EVALUATION_STARTED | epoch=%d | split=%s", epoch, split)
            for name in names:
                metrics = evaluate_model(
                    models[name], parameters[name], data, task_kind, loss_name
                )
                evaluations.append(
                    {"epoch": epoch, "split": split, "model": name, **metrics}
                )
                logger.info(
                    "EVALUATION_COMPLETED | epoch=%d | split=%s | model=%s | %s",
                    epoch,
                    split,
                    name,
                    " | ".join(f"{key}={value:.10g}" for key, value in metrics.items()),
                )

    logger.info(
        "TRAINING_STARTED | models=%s | loss=%s | optimizer=full-batch %s | lr=%s | epochs=%d | gradients=%s",
        ",".join(names),
        loss_name,
        "gradient descent" if optimizer == "gd" else "Adam",
        ",".join(f"{name}:{rates[name]:g}" for name in names),
        epochs,
        "exact"
        if shot_streams is None
        else f"finite shots ({training_cfg['finite_shots']} per state and parameter)",
    )
    # A diverged model keeps NaN: its last loss is not a final-epoch value.
    final_loss = dict.fromkeys(names, float("nan"))
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *[f"{name}_train_loss" for name in names]])
        for epoch in range(epochs):
            if eval_every > 0 and epoch % eval_every == 0:
                run_evaluation(epoch)
            losses = []
            for name in names:
                if diverged[name] is not None:
                    losses.append(float("nan"))
                    continue
                started = time.perf_counter()
                if shot_streams is None:
                    value, gradient = models[name].loss_and_gradient(
                        parameters[name], train_data, loss_name
                    )
                else:
                    value, gradient = models[name].finite_shot_loss_and_gradient(
                        parameters[name],
                        train_data,
                        int(training_cfg["finite_shots"]),
                        shot_streams[name],
                    )
                if not (np.isfinite(value) and np.all(np.isfinite(gradient))):
                    if on_nonfinite == "raise":
                        raise FloatingPointError(
                            f"Non-finite loss or gradient for model {name!r} at epoch {epoch}"
                        )
                    # The parameters stay at their last finite value.
                    diverged[name] = epoch
                    final_loss[name] = float("nan")
                    logger.warning("MODEL_DIVERGED | model=%s | epoch=%d", name, epoch)
                    losses.append(float("nan"))
                    continue
                if optimizer == "gd":
                    step = rates[name] * gradient
                else:
                    first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                    second_moment[name] = (
                        0.999 * second_moment[name] + 0.001 * gradient**2
                    )
                    corrected_first = first_moment[name] / (1 - 0.9 ** (epoch + 1))
                    corrected_second = second_moment[name] / (1 - 0.999 ** (epoch + 1))
                    step = (
                        rates[name]
                        * corrected_first
                        / (np.sqrt(corrected_second) + 1e-8)
                    )
                candidate = parameters[name] - step
                if not np.all(np.isfinite(candidate)):
                    if on_nonfinite == "raise":
                        raise FloatingPointError(
                            f"Non-finite parameters for model {name!r} at epoch {epoch}"
                        )
                    diverged[name] = epoch
                    logger.warning("MODEL_DIVERGED | model=%s | epoch=%d", name, epoch)
                else:
                    parameters[name] = candidate
                seconds[name] += time.perf_counter() - started
                losses.append(value)
                final_loss[name] = value if diverged[name] is None else float("nan")
            writer.writerow([epoch, *[f"{value:.17g}" for value in losses]])
            logger.info(
                "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | %s",
                epoch + 1,
                epochs,
                " | ".join(
                    f"{name}_loss={value:.10g}" for name, value in zip(names, losses)
                ),
            )
    logger.info("ARTIFACT_WRITTEN | path=%s", history_path.name)
    final_metrics = {}
    for split, data in evaluation_data.items():
        logger.info("EVALUATION_STARTED | epoch=%d | split=%s | final", epochs, split)
        for name in names:
            metrics = evaluate_model(
                models[name], parameters[name], data, task_kind, loss_name
            )
            final_metrics.setdefault(name, {})[split] = metrics
            logger.info(
                "EVALUATION_COMPLETED | epoch=%d | split=%s | model=%s | final | %s",
                epochs,
                split,
                name,
                " | ".join(f"{key}={value:.10g}" for key, value in metrics.items()),
            )
    return {
        "final_parameters": {
            name: theta.tolist() for name, theta in parameters.items()
        },
        "final_train_loss_before_last_update": final_loss,
        "evaluations": evaluations,
        "final_metrics": final_metrics,
        "learning_rates": rates,
        "optimizer": optimizer,
        "diverged_at_epoch": diverged,
        "training_seconds": seconds,
    }
