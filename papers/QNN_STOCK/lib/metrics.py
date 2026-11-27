"""Distribution and sequence metrics.

The paper's Table 1 reports a "KL divergence" without its direction, support
or split (LOG.md, F5). Every candidate definition is computed so E9 can decide
which one reproduces the paper's magnitudes.
"""

from __future__ import annotations

import numpy as np

from .data import context_indices

# Model probabilities are clipped at this value inside logarithms so that a
# zero-probability prediction gives a large finite divergence instead of inf.
PROBABILITY_FLOOR = 1e-12


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """``KL(p || q)`` over the entries where ``p > 0``."""

    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.clip(np.asarray(q, dtype=float).reshape(-1), PROBABILITY_FLOOR, None)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def normalize_rows(joint: np.ndarray) -> np.ndarray:
    """Row-normalize ``(contexts, levels)``; empty rows become uniform."""

    totals = joint.sum(axis=1, keepdims=True)
    uniform = np.full_like(joint, 1.0 / joint.shape[1])
    return np.where(totals > 0, joint / np.where(totals > 0, totals, 1.0), uniform)


def distribution_metrics(
    target_joint: np.ndarray, model_joint: np.ndarray, model_conditional: np.ndarray
) -> dict[str, float]:
    """Compare a model with an empirical ``P(context, next)``.

    ``model_joint`` is compared on the full ``(context, next)`` support;
    ``model_conditional`` is compared per context. Context rows absent from
    the target are skipped in the conditional sums.
    """

    target_conditional = normalize_rows(target_joint)
    context_weight = target_joint.sum(axis=1)
    present = context_weight > 0
    forward = np.array(
        [
            kl_divergence(target_conditional[c], model_conditional[c])
            for c in np.flatnonzero(present)
        ]
    )
    reverse = np.array(
        [
            kl_divergence(model_conditional[c], target_conditional[c])
            for c in np.flatnonzero(present)
        ]
    )
    tvd = np.array(
        [
            total_variation(target_conditional[c], model_conditional[c])
            for c in np.flatnonzero(present)
        ]
    )
    return {
        "kl_joint": kl_divergence(target_joint, model_joint),
        "kl_joint_reverse": kl_divergence(model_joint, target_joint),
        "tvd_joint": total_variation(target_joint, model_joint),
        "kl_conditional_sum": float(forward.sum()),
        "kl_conditional_mean": float(forward.mean()),
        "kl_conditional_weighted": float(np.dot(context_weight[present], forward)),
        "kl_conditional_sum_reverse": float(reverse.sum()),
        "tvd_conditional_mean": float(tvd.mean()),
    }


def sequence_metrics(
    windows: np.ndarray, levels: int, model_conditional: np.ndarray
) -> dict[str, float]:
    """Next-step log-loss and argmax accuracy over individual windows."""

    contexts = context_indices(windows, levels)
    outcomes = windows[:, -1]
    probs = np.clip(model_conditional[contexts, outcomes], PROBABILITY_FLOOR, None)
    predicted = np.argmax(model_conditional[contexts], axis=1)
    return {
        "log_loss": float(-np.mean(np.log(probs))),
        "accuracy": float(np.mean(predicted == outcomes)),
        "n_windows": int(len(windows)),
    }


def path_step_metrics(
    windows: np.ndarray, horizon: int, model_conditional: np.ndarray
) -> dict[str, float]:
    """Per-step log-loss and accuracy of a binary path model ``q(path | c)``.

    Step ``k`` uses the model's marginal ``q(step k up | c)``, summed over the
    other steps of the path. The outcome index holds the earliest step as its
    most significant bit (``lib.data.outcome_windows``).
    """

    contexts = context_indices(windows, 2)
    paths = np.arange(2**horizon)
    metrics = {}
    for step in range(horizon):
        shift = horizon - 1 - step
        up_paths = ((paths >> shift) & 1).astype(bool)
        prob_up = model_conditional[:, up_paths].sum(axis=1)[contexts]
        moved_up = ((windows[:, -1] >> shift) & 1).astype(bool)
        prob_true = np.clip(
            np.where(moved_up, prob_up, 1.0 - prob_up), PROBABILITY_FLOOR, None
        )
        metrics[f"step{step + 1}_log_loss"] = float(-np.mean(np.log(prob_true)))
        metrics[f"step{step + 1}_accuracy"] = float(
            np.mean((prob_up > 0.5) == moved_up)
        )
    return metrics
