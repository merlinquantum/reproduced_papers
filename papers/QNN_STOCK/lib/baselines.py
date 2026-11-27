"""Classical baselines B0-B4 (LOG.md, Fair Baseline Plan).

Every baseline returns a conditional table ``q(next | context)`` of shape
``(levels**T, levels)``. Trainable models fit the context counts with
full-batch Adam on the next-step negative log-likelihood (the counts are a
sufficient statistic, so this equals training on every window).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import joint_distribution, split_validation
from .metrics import normalize_rows, sequence_metrics

logger = logging.getLogger(__name__)


def counts_table(windows: np.ndarray, levels: int, outcome_levels: int) -> np.ndarray:
    return joint_distribution(windows, levels, outcome_levels) * len(windows)


def uniform_table(n_contexts: int, levels: int) -> np.ndarray:
    return np.full((n_contexts, levels), 1.0 / levels)


def marginal_table(
    windows: np.ndarray, outcome_levels: int, n_contexts: int
) -> np.ndarray:
    marginal = np.bincount(windows[:, -1], minlength=outcome_levels) / len(windows)
    return np.tile(marginal, (n_contexts, 1))


def frequency_table(
    windows: np.ndarray, levels: int, outcome_levels: int, alpha: float
) -> np.ndarray:
    """B0: empirical ``P(outcome | context)`` with additive (Laplace) smoothing."""

    return normalize_rows(counts_table(windows, levels, outcome_levels) + alpha)


def pooled_shrinkage_table(
    windows: np.ndarray,
    pooled_conditional: np.ndarray,
    levels: int,
    outcome_levels: int,
    strength: float,
) -> np.ndarray:
    """B4: per-asset counts shrunk toward the pooled multi-asset conditional."""

    counts = counts_table(windows, levels, outcome_levels)
    return normalize_rows(counts + strength * pooled_conditional)


def context_features(n_contexts: int, levels: int) -> torch.Tensor:
    """Binary digits of every context index (``T * log2(levels)`` features)."""

    n_bits = int(np.log2(n_contexts))
    rows = [
        [(c >> (n_bits - 1 - b)) & 1 for b in range(n_bits)] for c in range(n_contexts)
    ]
    return torch.tensor(rows, dtype=torch.float64)


class ConditionalModel(nn.Module):
    """Logistic (``hidden == 0``) or one-hidden-layer tanh MLP over context bits."""

    def __init__(self, n_features: int, levels: int, hidden: int) -> None:
        super().__init__()
        out_dim = 1 if levels == 2 else levels
        if hidden == 0:
            self.net = nn.Linear(n_features, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, out_dim)
            )
        self.levels = levels
        self.double()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        if self.levels == 2:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        return torch.log_softmax(logits, dim=1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def fit_conditional_model(
    counts: np.ndarray,
    hidden: int,
    weight_decay: float,
    lr: float,
    steps: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, int]:
    """Fit B2 (``hidden=0``) or B3 on context counts; return the table and #params."""

    torch.manual_seed(seed)
    n_contexts, levels = counts.shape
    features = context_features(n_contexts, levels).to(device)
    weights = torch.as_tensor(counts / counts.sum(), dtype=torch.float64, device=device)
    model = ConditionalModel(features.shape[1], levels, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = -(weights * model(features)).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        table = model(features).exp().cpu().numpy()
    return table, model.n_params()


def _select_and_refit(
    name: str,
    grid: list[dict[str, Any]],
    build_table,
    train_windows: np.ndarray,
    validation_fraction: float,
    levels: int,
) -> dict[str, Any]:
    """Pick the grid point with the lowest validation log-loss, refit on all training windows."""

    fit_windows, validation_windows = split_validation(
        train_windows, validation_fraction
    )
    scored = []
    for point in grid:
        table, _ = build_table(fit_windows, **point)
        validation = sequence_metrics(validation_windows, levels, table)["log_loss"]
        scored.append((validation, point))
        logger.debug(
            "Baseline %s candidate %s validation log-loss %.6f", name, point, validation
        )
    best_validation, best_point = min(scored, key=lambda item: item[0])
    table, n_params = build_table(train_windows, **best_point)
    logger.info(
        "TRAIN_FIT_COMPLETED | baseline=%s | selected=%s | validation_log_loss=%.6f | params=%d",
        name,
        best_point,
        best_validation,
        n_params,
    )
    return {
        "conditional": table,
        "n_params": n_params,
        "selected": best_point,
        "validation_log_loss": best_validation,
    }


def fit_baselines(
    train_windows: np.ndarray,
    levels: int,
    outcome_levels: int,
    baseline_cfg: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    """Fit B0-B3 for one asset with validation-based selection.

    ``levels`` is the number of symbols per context step and ``outcome_levels``
    the number of prediction outcomes (``levels ** horizon``).
    """

    n_contexts = levels ** (train_windows.shape[1] - 1)
    validation_fraction = float(baseline_cfg["validation_fraction"])
    results: dict[str, dict[str, Any]] = {
        "B1_uniform": {
            "conditional": uniform_table(n_contexts, outcome_levels),
            "n_params": 0,
        },
        "B1_marginal": {
            "conditional": marginal_table(train_windows, outcome_levels, n_contexts),
            "n_params": outcome_levels - 1,
        },
    }
    logger.info("TRAIN_FIT_COMPLETED | baseline=B1_uniform | params=0")
    logger.info(
        "TRAIN_FIT_COMPLETED | baseline=B1_marginal | params=%d", outcome_levels - 1
    )

    results["B0_frequency"] = _select_and_refit(
        "B0_frequency",
        [{"alpha": alpha} for alpha in baseline_cfg["laplace_alpha"]],
        lambda windows, alpha: (
            frequency_table(windows, levels, outcome_levels, alpha),
            n_contexts * (outcome_levels - 1),
        ),
        train_windows,
        validation_fraction,
        levels,
    )

    torch_cfg = baseline_cfg["torch"]

    def torch_table(windows, hidden, weight_decay, lr):
        return fit_conditional_model(
            counts_table(windows, levels, outcome_levels),
            hidden,
            weight_decay,
            lr,
            int(torch_cfg["steps"]),
            seed,
            device,
        )

    hidden_options = {
        "B2_logistic": [0],
        **{f"B3_mlp_h{h}": [h] for h in torch_cfg["mlp_hidden"]},
    }
    for name, hiddens in hidden_options.items():
        grid = [
            {"hidden": h, "weight_decay": wd, "lr": lr}
            for h, wd, lr in itertools.product(
                hiddens, torch_cfg["weight_decay"], torch_cfg["lr"]
            )
        ]
        results[name] = _select_and_refit(
            name, grid, torch_table, train_windows, validation_fraction, levels
        )
    return results


def fit_pooled_baseline(
    task_windows: list[np.ndarray],
    levels: int,
    outcome_levels: int,
    baseline_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """B4: shrink each asset's frequency table toward the pooled table."""

    validation_fraction = float(baseline_cfg["validation_fraction"])
    # Selection pools only the fit parts, so no asset's validation slice leaks
    # into the pooled table; the refit pools every asset's full training split.
    pooled_fit = normalize_rows(
        sum(
            counts_table(
                split_validation(w, validation_fraction)[0], levels, outcome_levels
            )
            for w in task_windows
        )
    )
    pooled_full = normalize_rows(
        sum(counts_table(w, levels, outcome_levels) for w in task_windows)
    )
    n_params = pooled_full.shape[0] * (outcome_levels - 1)
    results = []
    for windows in task_windows:
        fit_windows, validation_windows = split_validation(windows, validation_fraction)
        scored = []
        for strength in baseline_cfg["pooled_strength"]:
            table = pooled_shrinkage_table(
                fit_windows, pooled_fit, levels, outcome_levels, strength
            )
            scored.append(
                (
                    sequence_metrics(validation_windows, levels, table)["log_loss"],
                    strength,
                )
            )
        best_validation, best_strength = min(scored)
        logger.info(
            "TRAIN_FIT_COMPLETED | baseline=B4_pooled_shrinkage | selected=%s | "
            "validation_log_loss=%.6f | params=%d",
            {"strength": best_strength},
            best_validation,
            n_params,
        )
        results.append(
            {
                "conditional": pooled_shrinkage_table(
                    windows, pooled_full, levels, outcome_levels, best_strength
                ),
                "n_params": n_params,
                "selected": {"strength": best_strength},
                "validation_log_loss": best_validation,
            }
        )
    return results
