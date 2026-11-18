"""Full-batch training loop shared by every model class (App. F).

Step semantics follow the author code (``optimise_model_adam``): at step t the
objective and its gradient are computed at the current parameters, the
optimiser updates them, and the test KL divergence is then evaluated at the
updated parameters. ``history.csv`` row t therefore holds the pre-update
training objective and the post-update test KL, exactly like the curves of
Figs. 8, 9 and 11. The objective includes the L2 penalty, as the author code's
plotted "negative log likelihood" does; the pure NLL is recorded separately.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import torch
from torch import nn

from .data import SplitData
from .metrics import average_kl, negative_log_likelihood, zero_sum_violation

logger = logging.getLogger(__name__)

HISTORY_FIELDS = ("step", "train_objective", "train_nll", "test_kl", "validation_nll")


def build_optimizer(model: nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    """Adam with optax-equivalent defaults, or plain gradient descent.

    ``torch.optim.Adam`` defaults (betas 0.9/0.999, eps 1e-8 added after the
    square root, bias correction) match ``optax.adam``;
    ``tests/test_training.py`` checks the trajectories against optax.
    """
    name = training_cfg["optimizer"]
    lr = float(training_cfg["lr"])
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"training.optimizer must be 'adam' or 'sgd', got {name!r}")


def l2_penalty(model: nn.Module) -> torch.Tensor:
    """Sum of squared parameters (the author code penalises every weight)."""
    return sum((parameter**2).sum() for parameter in model.parameters())


@torch.no_grad()
def evaluate(model: nn.Module, split: SplitData, kl_floor: float) -> dict:
    """KL to the true marginals, NLL (labelled splits) and bias violation."""
    expectations = model(split.features)
    result = {
        "kl": average_kl(split.win_probabilities, expectations, kl_floor),
        "zero_sum_violation": zero_sum_violation(expectations),
    }
    if split.labels is not None:
        result["nll"] = float(negative_log_likelihood(expectations, split.labels))
    return result


def train(
    model: nn.Module,
    train_split: SplitData,
    test_split: SplitData,
    validation_split: SplitData | None,
    training_cfg: dict,
    evaluation_cfg: dict,
    history_path: Path,
) -> dict:
    """Train ``model`` and write the per-step history.

    Parameters
    ----------
    model : torch.nn.Module
        Maps ``(N, 6)`` features to ``(N, 3)`` expectations E[Y_k | x].
    train_split, test_split : SplitData
        Labelled training data and the test strategies.
    validation_split : SplitData or None
        Optional labelled validation data (fair-comparison tuning).
    training_cfg : dict
        ``steps``, ``optimizer``, ``lr`` and ``l2`` keys.
    evaluation_cfg : dict
        ``every`` (evaluation period in steps) and ``kl_floor`` keys.
    history_path : pathlib.Path
        Destination CSV with one row per evaluated step.

    Returns
    -------
    dict
        Final and best-observed metrics, plus wall-clock time.

    Raises
    ------
    FloatingPointError
        If the training objective becomes non-finite (for example a HardTanh
        surrogate assigning probability 0 to an observed label).
    """
    steps = int(training_cfg["steps"])
    l2 = float(training_cfg["l2"])
    every = int(evaluation_cfg["every"])
    kl_floor = float(evaluation_cfg["kl_floor"])
    if steps < 1 or every < 1:
        raise ValueError(
            f"training.steps and evaluation.every must be >= 1, got {steps}, {every}"
        )
    optimizer = build_optimizer(model, training_cfg)
    logger.info(
        "TRAINING_STARTED | steps=%d | optimizer=%s | lr=%g | l2=%g | eval_every=%d | n_train=%d | n_test=%d",
        steps,
        training_cfg["optimizer"],
        float(training_cfg["lr"]),
        l2,
        every,
        train_split.size,
        test_split.size,
    )
    started = time.perf_counter()
    best = {"test_kl": math.inf, "step": None}
    best_validation = {"validation_nll": math.inf, "step": None, "test_kl": None}
    last_row: dict = {}
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for step in range(steps):
            optimizer.zero_grad()
            expectations = model(train_split.features)
            nll = negative_log_likelihood(expectations, train_split.labels)
            objective = nll + l2 * l2_penalty(model) if l2 else nll
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"Non-finite training objective at step {step}: {objective.item()}"
                )
            objective.backward()
            optimizer.step()
            row = {
                "step": step,
                "train_objective": objective.item(),
                "train_nll": nll.item(),
            }
            if step % every == 0 or step == steps - 1:
                test_metrics = evaluate(model, test_split, kl_floor)
                row["test_kl"] = test_metrics["kl"]
                if test_metrics["kl"] < best["test_kl"]:
                    best = {"test_kl": test_metrics["kl"], "step": step}
                if validation_split is not None:
                    row["validation_nll"] = evaluate(model, validation_split, kl_floor)[
                        "nll"
                    ]
                    if row["validation_nll"] < best_validation["validation_nll"]:
                        best_validation = {
                            "validation_nll": row["validation_nll"],
                            "step": step,
                            "test_kl": row["test_kl"],
                        }
                writer.writerow(row)
                logger.info(
                    "TRAIN_STEP_COMPLETED | step=%d/%d | train_objective=%.8f | train_nll=%.8f | test_kl=%.8e%s",
                    step + 1,
                    steps,
                    row["train_objective"],
                    row["train_nll"],
                    row["test_kl"],
                    f" | validation_nll={row['validation_nll']:.8f}"
                    if "validation_nll" in row
                    else "",
                )
            else:
                logger.debug(
                    "TRAIN_STEP | step=%d/%d | train_objective=%.8f",
                    step + 1,
                    steps,
                    row["train_objective"],
                )
            last_row = row
    elapsed = time.perf_counter() - started
    logger.info("ARTIFACT_WRITTEN | path=%s", history_path.name)

    logger.info(
        "EVALUATION_STARTED | split=train,test%s | checkpoint=final",
        ",validation" if validation_split else "",
    )
    final_train = evaluate(model, train_split, kl_floor)
    final_test = evaluate(model, test_split, kl_floor)
    final = {
        "train_nll": final_train["nll"],
        "train_kl": final_train["kl"],
        "test_kl": final_test["kl"],
        "test_zero_sum_violation": final_test["zero_sum_violation"],
    }
    if validation_split is not None:
        final["validation_nll"] = evaluate(model, validation_split, kl_floor)["nll"]
    logger.info(
        "EVALUATION_COMPLETED | checkpoint=final | test_kl=%.8e | train_nll=%.8f | test_zero_sum_violation=%.3e",
        final["test_kl"],
        final["train_nll"],
        final["test_zero_sum_violation"],
    )
    return {
        "final": final,
        "last_step_objective": last_row["train_objective"],
        "min_test_kl": best,
        "min_validation_nll": best_validation if validation_split is not None else None,
        "train_seconds": elapsed,
    }
