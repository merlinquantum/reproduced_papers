"""Ridge readouts shared by the QELM and the classical baselines.

The paper trains ``y = w . r + w0`` with an l2 penalty but does not report ``lambda``.
Every ridge model here selects ``alpha`` on the training split only, over
``logspace(log_min, log_max, num)`` (config ``readout.alphas``): leave-one-out for
classification, contiguous 5-fold CV for time-series regression.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, r2_score
from sklearn.model_selection import KFold

CLASSIFICATION_METRICS = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
)


def alpha_grid(readout_cfg: dict) -> np.ndarray:
    """Return the ridge penalty grid from ``readout_cfg["alphas"]``."""
    grid = readout_cfg["alphas"]
    return np.logspace(float(grid["log_min"]), float(grid["log_max"]), int(grid["num"]))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, macro/weighted F1 and macro/weighted precision (paper Fig. 4, Fig. A2)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def fit_ridge_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, float]:
    """Ridge on one-hot (+-1) labels with argmax prediction; returns test metrics and alpha.

    ``RidgeClassifierCV`` encodes classes as +-1 targets; with a fitted intercept this is
    an affine reparametrisation of ridge on {0, 1} one-hot labels and gives the same
    argmax. ``alpha`` is chosen by efficient leave-one-out, which also works for the
    10-sample training sets of the learning curve (5-fold stratified CV needs at least
    5 samples per class).
    """
    model = RidgeClassifierCV(alphas=alphas, cv=None, fit_intercept=True)
    model.fit(train_x, train_y)
    metrics = classification_metrics(test_y, model.predict(test_x))
    train_metrics = classification_metrics(train_y, model.predict(train_x))
    metrics["train_f1_macro"] = train_metrics["f1_macro"]
    metrics["train_accuracy"] = train_metrics["accuracy"]
    metrics["alpha"] = float(model.alpha_)
    return metrics


def fit_ridge_regression(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, float]:
    """Ridge regression on standardised features; returns test R^2 and alpha.

    Folds are contiguous (no shuffling) because NARMA targets are time-ordered.
    Standardisation uses training statistics only.
    """
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    model = RidgeCV(
        alphas=alphas, cv=KFold(n_splits=5, shuffle=False), fit_intercept=True
    )
    model.fit((train_x - mean) / std, train_y)
    prediction = model.predict((test_x - mean) / std)
    return {"r2": float(r2_score(test_y, prediction)), "alpha": float(model.alpha_)}
