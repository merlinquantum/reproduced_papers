"""Mixture-of-Experts router: thresholding, router-target construction, and
hard-mixture prediction combination (paper Section 3.3, Eq. 15-16).

NOTE on router training features (documented per task spec / to be folded
into LOG.md's "Paper vs Reference Code Disagreements" table): the paper's
prose says the router is "trained on the validation features and these
router targets", but its own procedure (Section 3.3) computes the router
targets from predictions made on the ANALYSIS split (using tau1/tau2 that
were themselves calibrated on the validation split) and describes analysis
as the router's training data. We follow the analysis-split reading here:
``pipeline.py`` trains the XGBoost router on the analysis split's raw
features against targets built from the analysis split's calibrated
predictions/labels. The validation split is reserved for calibration
(temperature fitting + tau1/tau2 threshold selection) only. This is a
terminology inconsistency in the paper text vs. its own described procedure,
not an ambiguity we introduced.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def youden_j_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    """Return the probability threshold maximizing Youden's J = TPR - FPR.

    ``roc_curve`` prepends a synthetic ``np.inf`` threshold (the
    "predict nothing positive" operating point, J = 0). On folds where no
    real threshold achieves positive J, a bare argmax selects it, and the
    resulting ``probs >= inf`` mask silently rejects every sample —
    corrupting the MoE router's training targets. Restrict the argmax to
    finite thresholds so the best *achievable* operating point is returned
    instead.
    """
    fpr, tpr, thresholds = roc_curve(y, probs)
    j_scores = np.where(np.isfinite(thresholds), tpr - fpr, -np.inf)
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def build_router_targets(
    p1_hat: np.ndarray,
    y1_true_labels: np.ndarray,
    p2_hat: np.ndarray,
    tau1: float,
    tau2: float,
    y_true: np.ndarray,
) -> np.ndarray:
    """Build router training targets per Eq. 15.

    ``z_i = 1`` iff ``(p2_hat_i > tau2) == y_true_i`` (secondary expert
    correct at its threshold) AND ``(p1_hat_i > tau1) != y_true_i`` (primary
    expert incorrect at its threshold); else 0.

    ``y1_true_labels`` and ``y_true`` are both accepted (matching the task's
    specified function signature) but must refer to the same underlying
    label array for the same sample set — the paper's Eq. 15 uses a single
    ground-truth vector. We validate that they agree rather than silently
    picking one, since a mismatch would indicate a caller bug.
    """
    y1_true_labels = np.asarray(y1_true_labels)
    y_true = np.asarray(y_true)
    if not np.array_equal(y1_true_labels, y_true):
        raise ValueError(
            "build_router_targets: y1_true_labels and y_true must be the same "
            "label array (both refer to the same sample set's ground truth)."
        )

    pred1_correct = (np.asarray(p1_hat) > tau1).astype(int) == y_true
    pred2_correct = (np.asarray(p2_hat) > tau2).astype(int) == y_true
    z = (pred2_correct & ~pred1_correct).astype(int)
    return z


def combine_predictions(
    p1_hat: np.ndarray,
    p2_hat: np.ndarray,
    router_probs: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Hard mixture per Eq. 16: route to expert 2 where router_probs > gamma."""
    r = (np.asarray(router_probs) > gamma).astype(int)
    p_comb = (1 - r) * np.asarray(p1_hat) + r * np.asarray(p2_hat)
    return p_comb


def routed_fraction(router_probs: np.ndarray, gamma: float) -> float:
    """Fraction of samples routed to the secondary (quantum-hybrid) expert."""
    r = (np.asarray(router_probs) > gamma).astype(int)
    return float(r.mean())


def evaluate_binary(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute AUCPR (trapezoidal), AP, and Youden's-J-thresholded
    precision/recall for a (probs, y) pair.

    AUCPR here is the trapezoidal-rule area under the (recall, precision)
    curve from ``precision_recall_curve`` -- distinct from ``average_precision
    _score`` (AP), which the paper's tables report separately.
    """
    probs = np.asarray(probs)
    y = np.asarray(y)

    precision, recall, _ = precision_recall_curve(y, probs)
    aucpr = float(sk_auc(recall, precision))
    ap = float(average_precision_score(y, probs))

    tau = youden_j_threshold(probs, y)
    # NOTE: sklearn's roc_curve (which youden_j_threshold uses internally to
    # pick tau) treats a sample as predicted-positive when score >= threshold.
    # We use the same ">=" convention here so the reported precision/recall
    # are consistent with the TPR/FPR that produced tau. This is a distinct
    # design choice from combine_predictions()/Eq. 16's strict "> gamma",
    # which governs the MoE router's routing decision, not this general
    # single-expert reporting helper.
    y_pred = (probs >= tau).astype(int)
    prec_at_tau = float(precision_score(y, y_pred, zero_division=0))
    rec_at_tau = float(recall_score(y, y_pred, zero_division=0))

    return {
        "aucpr": aucpr,
        "ap": ap,
        "precision": prec_at_tau,
        "recall": rec_at_tau,
        "routed_fraction": None,
    }


__all__ = [
    "youden_j_threshold",
    "build_router_targets",
    "combine_predictions",
    "routed_fraction",
    "evaluate_binary",
]
