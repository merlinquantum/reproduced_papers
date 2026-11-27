"""Phase 05 fair training comparison: per-model learning-rate selection, then fresh seeds.

Each model is trained at every learning rate of ``comparison.lr_grid`` on the
selection seeds. The predeclared rule picks the lr with the lowest mean final
cross-entropy on ``comparison.selection_split`` (the training split for the paper's
C-04 question, the validation split for held-out tasks). The selected lr is then
trained on fresh seeds and evaluated on every available split. Grid, selection and
final training run inside one run, so the selection is computed from full-precision
values in code, never by hand.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_arrays
from .models import build_model, describe_model
from .training import initial_parameters, per_seed_accuracy, per_seed_loss, train

logger = logging.getLogger(__name__)


def _seed_range(seeds_cfg: dict) -> list[int]:
    return list(range(seeds_cfg["start"], seeds_cfg["start"] + seeds_cfg["count"]))


def select_learning_rate(
    mean_losses: dict[float, float], tie_tolerance: float
) -> tuple[float, list[float]]:
    """Return the lr with the lowest mean loss and every lr tied with it.

    Parameters
    ----------
    mean_losses : dict
        Mean final selection-split loss per candidate lr.
    tie_tolerance : float
        Candidates within this distance of the minimum are reported as tied.

    Returns
    -------
    tuple
        ``(selected_lr, tied_lrs)``; ``tied_lrs`` includes the selected lr.

    Raises
    ------
    ValueError
        If a candidate loss is not finite.
    """
    if not all(np.isfinite(value) for value in mean_losses.values()):
        raise ValueError(f"Non-finite selection loss in {mean_losses}")
    selected = min(mean_losses, key=mean_losses.get)
    best = mean_losses[selected]
    tied = sorted(
        lr for lr, value in mean_losses.items() if value - best <= tie_tolerance
    )
    return selected, tied


def _evaluate(model, splits: dict, theta: torch.Tensor) -> dict:
    """Per-seed loss and accuracy on every split at the final parameters."""
    evaluation = {}
    with torch.no_grad():
        for split, (x, y) in splits.items():
            evaluation[split] = {
                "loss": per_seed_loss(model, x, y, theta).cpu().numpy(),
                "accuracy": per_seed_accuracy(model, x, y, theta).cpu().numpy(),
            }
    return evaluation


def _train_seeds(
    model, splits, seeds, init, training_cfg, device
) -> tuple[dict, dict, float]:
    theta0 = initial_parameters(len(seeds), model.d, seeds, init["low"], init["high"])
    x_train, y_train = splits["train"]
    started = time.perf_counter()
    result = train(
        model,
        x_train,
        y_train,
        torch.as_tensor(theta0, dtype=torch.float64, device=device),
        training_cfg,
    )
    elapsed = time.perf_counter() - started
    final_theta = torch.as_tensor(result["theta"], device=device)
    return result, _evaluate(model, splits, final_theta), elapsed


def run_fair_comparison(cfg: dict, run_dir: Path, device: torch.device) -> dict:
    """Run the predeclared lr selection and the fresh-seed final training per model.

    Writes ``comparison_<model>.npz`` with the final-seed training ``losses``
    ``(T + 1, n_seeds)``, per-split final ``<split>_loss`` / ``<split>_accuracy``
    ``(n_seeds,)``, the selection-grid losses ``grid_losses`` ``(n_lr, n_selection_seeds)``,
    and the initial and final parameters.

    Returns
    -------
    dict
        Per model: the grid means, the selected and tied lr, and the final-seed
        mean and std of every split's loss and accuracy, plus wall-clock times.
    """
    comparison = cfg["comparison"]
    training_cfg = cfg["training"]
    selection_split = comparison["selection_split"]
    lr_grid = [float(lr) for lr in comparison["lr_grid"]]
    selection_seeds = _seed_range(comparison["selection_seeds"])
    final_seeds = _seed_range(comparison["final_seeds"])
    if set(selection_seeds) & set(final_seeds):
        raise ValueError("Selection and final seeds must be disjoint")

    splits = {}
    class_counts = {}
    for split in comparison["splits"]:
        x_np, y_np = load_arrays(cfg, split)
        splits[split] = (
            torch.as_tensor(x_np, dtype=torch.float64, device=device),
            torch.as_tensor(y_np, device=device),
        )
        class_counts[split] = np.bincount(y_np).tolist()
    logger.info(
        "DATASET_READY | name=%s | class_counts=%s",
        cfg["dataset"]["name"],
        class_counts,
    )
    if selection_split not in splits:
        raise ValueError(
            f"selection_split {selection_split!r} is not in {list(splits)}"
        )

    metrics = {
        "comparison": comparison,
        "training": training_cfg,
        "selection_seeds": selection_seeds,
        "final_seeds": final_seeds,
        "models": {},
    }
    logger.info(
        "TRAINING_STARTED | models=%s | lr_grid=%s | selection_seeds=%d | final_seeds=%d | iterations=%d",
        [model_cfg["name"] for model_cfg in cfg["models"]],
        lr_grid,
        len(selection_seeds),
        len(final_seeds),
        training_cfg["iterations"],
    )
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        model = build_model(model_cfg)
        model_training = {**training_cfg, **model_cfg["training"]}
        init = model_training["init"]
        logger.info(
            "MODEL_STARTED | model=%s | d=%d | init=U[%g, %g] | optimizer=%s",
            name,
            model.d,
            init["low"],
            init["high"],
            model_training["optimizer"],
        )
        grid_losses, grid_seconds = [], []
        for lr in lr_grid:
            logger.info(
                "EVALUATION_STARTED | split=%s | phase=selection | model=%s | lr=%g",
                selection_split,
                name,
                lr,
            )
            _, evaluation, elapsed = _train_seeds(
                model,
                splits,
                selection_seeds,
                init,
                {**model_training, "lr": lr},
                device,
            )
            grid_losses.append(evaluation[selection_split]["loss"])
            grid_seconds.append(elapsed)
            logger.info(
                "EVALUATION_COMPLETED | split=%s | phase=selection | model=%s | lr=%g | loss_mean=%.6f | loss_std=%.6f | seconds=%.1f",
                selection_split,
                name,
                lr,
                grid_losses[-1].mean(),
                grid_losses[-1].std(),
                elapsed,
            )
        grid_means = {
            lr: float(losses.mean()) for lr, losses in zip(lr_grid, grid_losses)
        }
        selected_lr, tied = select_learning_rate(
            grid_means, comparison["tie_tolerance"]
        )
        logger.info(
            "SELECTION_COMPLETED | model=%s | selected_lr=%g | tied_lrs=%s",
            name,
            selected_lr,
            tied,
        )

        logger.info(
            "EVALUATION_STARTED | split=%s | phase=final | model=%s", list(splits), name
        )
        result, evaluation, final_seconds = _train_seeds(
            model,
            splits,
            final_seeds,
            init,
            {**model_training, "lr": selected_lr},
            device,
        )
        losses = result["losses"]
        threshold = training_cfg["loss_threshold"]
        reached = (losses <= threshold).any(axis=0)
        first_hit = np.where(reached, (losses <= threshold).argmax(axis=0), -1)
        summary = {
            split: {
                "loss_mean": float(values["loss"].mean()),
                "loss_std": float(values["loss"].std()),
                "accuracy_mean": float(values["accuracy"].mean()),
                "accuracy_std": float(values["accuracy"].std()),
            }
            for split, values in evaluation.items()
        }
        logger.info(
            "EVALUATION_COMPLETED | split=%s | phase=final | model=%s | lr=%g | %s | seconds=%.1f",
            list(splits),
            name,
            selected_lr,
            " | ".join(
                f"{split}_loss={values['loss_mean']:.6f}+-{values['loss_std']:.6f} {split}_acc={values['accuracy_mean']:.4f}"
                for split, values in summary.items()
            ),
            final_seconds,
        )
        array_path = run_dir / f"comparison_{name}.npz"
        arrays = {
            "losses": losses,
            "accuracies": result["accuracies"],
            "final_theta": result["theta"],
            "final_seeds": np.asarray(final_seeds),
            "grid_lrs": np.asarray(lr_grid),
            "grid_losses": np.stack(grid_losses),
        }
        for split, values in evaluation.items():
            arrays[f"{split}_loss"] = values["loss"]
            arrays[f"{split}_accuracy"] = values["accuracy"]
        np.savez_compressed(array_path, **arrays)
        logger.info("ARTIFACT_WRITTEN | path=%s", array_path.name)
        metrics["models"][name] = {
            **describe_model(model_cfg, model),
            "config": model_cfg,
            "effective_training": model_training,
            "grid_mean_loss": {str(lr): value for lr, value in grid_means.items()},
            "grid_std_loss": {
                str(lr): float(losses.std()) for lr, losses in zip(lr_grid, grid_losses)
            },
            "grid_seconds": {str(lr): s for lr, s in zip(lr_grid, grid_seconds)},
            "selected_lr": selected_lr,
            "tied_lrs": tied,
            "final": summary,
            "final_seconds": final_seconds,
            "loss_curve_mean": losses.mean(axis=1).tolist(),
            "loss_curve_std": losses.std(axis=1).tolist(),
            "loss_threshold": threshold,
            "fraction_reaching_threshold": float(reached.mean()),
            "median_iterations_to_threshold": float(np.median(first_hit[reached]))
            if reached.any()
            else None,
            "arrays": array_path.name,
        }
    return metrics
