"""Binary classification metrics with an explicit fixed decision threshold."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def binary_metrics(targets, probabilities) -> dict:
    """Compute held-out metrics; require finite probabilities and both classes."""
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid prediction probabilities")
    prediction = p >= 0.5
    return {
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "bce": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "threshold": 0.5,
    }
