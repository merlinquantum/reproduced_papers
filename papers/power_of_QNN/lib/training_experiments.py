"""Training experiments: Iris loss curves and Fisher-Rao norms (Fig. 3b, Table 1) and the
effective dimension under label randomization (Fig. 6, App. B.2)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .data import load_arrays
from .fisher import (
    effective_dimension,
    empirical_fisher,
    fisher_rao_norm,
    normalize_fisher,
)
from .models import build_model, describe_model
from .training import initial_parameters, per_seed_accuracy, per_seed_loss, train

logger = logging.getLogger(__name__)


def _seeds(training_cfg: dict) -> list[int]:
    return list(
        range(
            training_cfg["seeds"]["start"],
            training_cfg["seeds"]["start"] + training_cfg["seeds"]["count"],
        )
    )


def _model_training_cfg(training_cfg: dict, model_cfg: dict) -> dict:
    """Merge per-model overrides (optimizer, gradient, lr, init) into the shared settings."""
    merged = {key: value for key, value in training_cfg.items() if key != "seeds"}
    merged.update(model_cfg["training"])
    return merged


def local_normalized_fisher(
    model, theta: np.ndarray, inputs: torch.Tensor, chunk_size: int
) -> np.ndarray:
    """Return ``F_hat`` at a single trained ``theta`` (trace normalized to ``d``), as the
    authors compute it for Table 1 and Fig. 6 (one parameter set, training inputs)."""
    fisher = (
        empirical_fisher(
            model,
            torch.as_tensor(theta, dtype=torch.float64, device=inputs.device).unsqueeze(
                0
            ),
            inputs,
            chunk_size,
        )
        .cpu()
        .numpy()
    )
    return normalize_fisher(fisher)[0][0]


def run_training(cfg: dict, run_dir: Path, device: torch.device) -> dict:
    """Train every configured model for every seed on the full training batch.

    Writes ``training_<model>.npz`` with ``losses`` and ``accuracies``
    ``(T + 1, n_seeds)``, ``initial_theta`` and ``final_theta`` ``(n_seeds, d)``, and
    ``fisher_rao_norm`` ``(n_seeds,)`` when enabled.

    Returns
    -------
    dict
        Per model: final-loss mean and std (the Table 1 metric), mean loss curve,
        iterations to reach ``loss_threshold``, accuracy, and Fisher-Rao norm.
    """
    training_cfg = cfg["training"]
    x_np, y_np = load_arrays(cfg, "train")
    x = torch.as_tensor(x_np, dtype=torch.float64, device=device)
    y = torch.as_tensor(y_np, device=device)
    logger.info(
        "DATASET_READY | name=%s | x=%s | class_counts=%s",
        cfg["dataset"]["name"],
        tuple(x.shape),
        np.bincount(y_np).tolist(),
    )
    seeds = _seeds(training_cfg)
    metrics = {"seeds": seeds, "training": training_cfg, "models": {}}
    logger.info(
        "TRAINING_STARTED | models=%s | seeds=%d | iterations=%d",
        [model_cfg["name"] for model_cfg in cfg["models"]],
        len(seeds),
        training_cfg["iterations"],
    )
    for model_cfg in cfg["models"]:
        model = build_model(model_cfg)
        name = model_cfg["name"]
        model_training = _model_training_cfg(training_cfg, model_cfg)
        init = model_training["init"]
        theta0 = initial_parameters(
            len(seeds), model.d, seeds, init["low"], init["high"]
        )
        logger.info(
            "MODEL_STARTED | model=%s | d=%d | init=U[%g, %g]",
            name,
            model.d,
            init["low"],
            init["high"],
        )
        result = train(
            model,
            x,
            y,
            torch.as_tensor(theta0, dtype=torch.float64, device=device),
            model_training,
        )
        losses = result["losses"]
        final = losses[-1]
        threshold = training_cfg["loss_threshold"]
        reached = (losses <= threshold).any(axis=0)
        first_hit = np.where(reached, (losses <= threshold).argmax(axis=0), -1)
        logger.info("EVALUATION_STARTED | split=train | model=%s", name)
        fr_norms = None
        if training_cfg["fisher_rao"]:
            fr_norms = np.asarray(
                [
                    fisher_rao_norm(
                        theta,
                        local_normalized_fisher(
                            model, theta, x, training_cfg["fisher_chunk_size"]
                        ),
                    )
                    for theta in result["theta"]
                ]
            )
        logger.info(
            "EVALUATION_COMPLETED | split=train | model=%s | final_loss_mean=%.6f | final_loss_std=%.6f | "
            "final_accuracy_mean=%.4f | fisher_rao_mean=%s",
            name,
            final.mean(),
            final.std(),
            result["accuracies"][-1].mean(),
            "n/a" if fr_norms is None else f"{fr_norms.mean():.4f}",
        )
        array_path = run_dir / f"training_{name}.npz"
        arrays = {
            "losses": losses,
            "accuracies": result["accuracies"],
            "initial_theta": theta0,
            "final_theta": result["theta"],
            "seeds": np.asarray(seeds),
        }
        if fr_norms is not None:
            arrays["fisher_rao_norm"] = fr_norms
        np.savez_compressed(array_path, **arrays)
        logger.info("ARTIFACT_WRITTEN | path=%s", array_path.name)
        metrics["models"][name] = {
            **describe_model(model_cfg, model),
            "config": model_cfg,
            "effective_training": model_training,
            "final_loss_mean": float(final.mean()),
            "final_loss_std": float(final.std()),
            "loss_curve_mean": losses.mean(axis=1).tolist(),
            "loss_curve_std": losses.std(axis=1).tolist(),
            "final_accuracy_mean": float(result["accuracies"][-1].mean()),
            "loss_threshold": threshold,
            "fraction_reaching_threshold": float(reached.mean()),
            "median_iterations_to_threshold": float(np.median(first_hit[reached]))
            if reached.any()
            else None,
            "fisher_rao_norm_mean": None
            if fr_norms is None
            else float(fr_norms.mean()),
            "fisher_rao_norm_std": None if fr_norms is None else float(fr_norms.std()),
            "final_theta_norm_squared_mean": float(
                (result["theta"] ** 2).sum(axis=1).mean()
            ),
            "arrays": array_path.name,
        }
    return metrics


def randomize_labels(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Overwrite the first ``int(fraction * N)`` labels with uniform random labels.

    This is the authors' procedure (``generalisation_random_0*.py``): about half of
    the overwritten labels keep their value, so the realized flip fraction is close
    to ``fraction / 2``. The stream is seeded per run here (the authors' was not).
    """
    randomized = labels.copy()
    count = int(fraction * len(labels))
    randomized[:count] = np.random.RandomState(seed).randint(0, 2, size=count)
    return randomized


def run_generalisation(cfg: dict, run_dir: Path, device: torch.device) -> dict:
    """Fig. 6: effective dimension at the trained parameters vs label randomization.

    For each fraction, ``runs_per_fraction`` networks are trained in one batch, each
    with its own randomized labels. The effective dimension is the single-point
    quantity of the authors' code: F_hat normalized at the trained theta, training
    inputs, ``n = n_value``.
    """
    generalisation_cfg = cfg["generalisation"]
    training_cfg = cfg["training"]
    (model_cfg,) = cfg["models"]
    model = build_model(model_cfg)
    x_train_np, y_train_np = load_arrays(cfg, "train")
    x_test_np, y_test_np = load_arrays(cfg, "test")
    x_train = torch.as_tensor(x_train_np, dtype=torch.float64, device=device)
    x_test = torch.as_tensor(x_test_np, dtype=torch.float64, device=device)
    y_test = torch.as_tensor(y_test_np, device=device)
    logger.info(
        "DATASET_READY | name=%s | train=%s | test=%s",
        cfg["dataset"]["name"],
        tuple(x_train.shape),
        tuple(x_test.shape),
    )
    seeds = _seeds(training_cfg)
    model_training = _model_training_cfg(training_cfg, model_cfg)
    metrics = {
        "generalisation": generalisation_cfg,
        "training": training_cfg,
        **describe_model(model_cfg, model),
        "fractions": [],
    }
    arrays = {}
    logger.info(
        "TRAINING_STARTED | fractions=%s | runs_per_fraction=%d | iterations=%d",
        generalisation_cfg["randomised_fractions"],
        len(seeds),
        training_cfg["iterations"],
    )
    for fraction in generalisation_cfg["randomised_fractions"]:
        label_sets = np.stack(
            [
                randomize_labels(
                    y_train_np, fraction, generalisation_cfg["label_seed_base"] + seed
                )
                for seed in seeds
            ]
        )
        realized = (label_sets != y_train_np).mean(axis=1)
        theta0 = initial_parameters(
            len(seeds),
            model.d,
            seeds,
            model_training["init"]["low"],
            model_training["init"]["high"],
        )
        logger.info(
            "MODEL_STARTED | fraction=%.2f | runs=%d | realized_flip_mean=%.4f",
            fraction,
            len(seeds),
            realized.mean(),
        )
        result = train(
            model,
            x_train,
            torch.as_tensor(label_sets, device=device),
            torch.as_tensor(theta0, dtype=torch.float64, device=device),
            model_training,
        )
        theta = torch.as_tensor(result["theta"], device=device)
        with torch.no_grad():
            test_loss = per_seed_loss(model, x_test, y_test, theta).cpu().numpy()
        test_accuracy = per_seed_accuracy(model, x_test, y_test, theta).cpu().numpy()
        logger.info(
            "EVALUATION_STARTED | split=train_fisher_and_test | fraction=%.2f", fraction
        )
        effdims = np.asarray(
            [
                effective_dimension(
                    local_normalized_fisher(
                        model, theta_row, x_train, training_cfg["fisher_chunk_size"]
                    )[None],
                    [generalisation_cfg["n_value"]],
                )[0]
                for theta_row in result["theta"]
            ]
        )
        normalized = effdims / model.d
        logger.info(
            "EVALUATION_COMPLETED | split=train_fisher_and_test | fraction=%.2f | ed_norm_mean=%.6f | ed_norm_std=%.6f | "
            "train_loss_mean=%.6f | test_loss_mean=%.6f | test_accuracy_mean=%.4f",
            fraction,
            normalized.mean(),
            normalized.std(),
            result["losses"][-1].mean(),
            test_loss.mean(),
            test_accuracy.mean(),
        )
        key = f"{fraction:.2f}"
        arrays[f"normalized_effective_dimension_{key}"] = normalized
        arrays[f"test_loss_{key}"] = test_loss
        arrays[f"train_losses_{key}"] = result["losses"][
            :: generalisation_cfg["loss_stride"]
        ]
        metrics["fractions"].append(
            {
                "fraction": fraction,
                "realized_flip_fraction_mean": float(realized.mean()),
                "normalized_effective_dimension_mean": float(normalized.mean()),
                "normalized_effective_dimension_std": float(normalized.std()),
                "final_train_loss_mean": float(result["losses"][-1].mean()),
                "final_train_accuracy_mean": float(result["accuracies"][-1].mean()),
                "test_loss_mean": float(test_loss.mean()),
                "test_loss_std": float(test_loss.std()),
                "test_accuracy_mean": float(test_accuracy.mean()),
            }
        )
    array_path = run_dir / "generalisation.npz"
    np.savez_compressed(array_path, **arrays)
    logger.info("ARTIFACT_WRITTEN | path=%s", array_path.name)
    metrics["arrays"] = array_path.name
    return metrics
