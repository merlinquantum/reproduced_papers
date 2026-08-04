"""Linear classifiers used as the QKS post-processing head, per the LB rule.

Wraps scikit-learn's `LogisticRegression` and `LinearSVC` so the runner can
swap them out from JSON configs.  Also exposes an SVM-RBF baseline used as a
non-linear classical reference in Fig. 5 of arXiv:1806.08321.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def _filter_keys(d: Dict[str, Any], allowed: set[str]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in allowed}


@dataclass
class ClassifierResult:
    train_accuracy: float
    test_accuracy: float
    train_error: float
    test_error: float
    name: str
    fit_seconds: float


def train_classifier(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Dict[str, Any] | None = None,
    seed: int = 0,
) -> ClassifierResult:
    """Train a small classifier and report accuracies. `name` is one of:

    - `logistic_regression`
    - `svm_rbf`
    """
    import time

    cfg = dict(cfg or {})
    C = float(cfg.get("C", 1.0))
    max_iter = int(cfg.get("max_iter", 2000))

    t0 = time.perf_counter()
    if name == "logistic_regression":
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    elif name == "svm_rbf":
        clf = SVC(C=C, kernel="rbf", gamma=cfg.get("gamma", "scale"), random_state=seed)
    elif name == "svm_linear":
        from sklearn.svm import LinearSVC

        clf = LinearSVC(C=C, max_iter=max_iter, random_state=seed, dual="auto")
    else:
        raise ValueError(f"Unknown classifier: {name}")
    clf.fit(X_train, y_train)
    dt = time.perf_counter() - t0

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    return ClassifierResult(
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        train_error=1.0 - train_acc,
        test_error=1.0 - test_acc,
        name=name,
        fit_seconds=dt,
    )


__all__ = ["ClassifierResult", "train_classifier"]
