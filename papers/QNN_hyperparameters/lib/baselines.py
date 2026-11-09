"""E-F: classical baselines on the exact QNN inputs.

Logistic regression, RBF-SVM and a one-hidden-layer MLP are trained on the
same 400 training rows and the same reduced (and optionally MinMax-scaled)
features the QNN sees, and evaluated on the same 250 validation rows and 250
test rows. Hyperparameters are chosen by 5-fold cross-validation on the 400
training rows only. The MLP hidden width is chosen so its trainable
parameter count is close to ``mlp_parameter_target`` (28 = RealAmplitudes /
TwoLocal at 7 qubits and 3 reps; 56 = EfficientSU2).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

MODELS = ("logistic_regression", "rbf_svm", "mlp")


def mlp_hidden_width(n_features: int, n_classes: int, parameter_target: int) -> int:
    """Smallest hidden width whose parameter count reaches ``parameter_target``.

    One hidden layer: ``(n_features + 1) * h + (h + 1) * n_outputs`` with
    ``n_outputs = 1`` for binary problems (sklearn's logistic output).
    """

    outputs = 1 if n_classes == 2 else n_classes
    width = 1
    while (n_features + 1) * width + (width + 1) * outputs < parameter_target:
        width += 1
    return width


def build_search(
    model: str,
    n_features: int,
    n_classes: int,
    parameter_target: int,
    seed: int,
    cv_folds: int,
):
    """Return a ``GridSearchCV`` with the predeclared grid for ``model``."""

    min_class_count_folds = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=seed
    )
    if model == "logistic_regression":
        estimator, grid = LogisticRegression(max_iter=5000), {"C": [0.01, 0.1, 1, 10]}
    elif model == "rbf_svm":
        estimator, grid = (
            SVC(kernel="rbf"),
            {"C": [0.1, 1, 10], "gamma": ["scale", 0.1, 1]},
        )
    elif model == "mlp":
        width = mlp_hidden_width(n_features, n_classes, parameter_target)
        estimator = MLPClassifier(
            hidden_layer_sizes=(width,), max_iter=500, random_state=seed
        )
        grid = {"learning_rate_init": [1e-3, 1e-2]}
    else:
        raise ValueError(f"unknown baseline {model!r}; expected one of {MODELS}")
    return GridSearchCV(estimator, grid, cv=min_class_count_folds, scoring="accuracy")


def count_parameters(fitted) -> int:
    """Trainable parameter count of a fitted baseline (support vectors excluded)."""

    if isinstance(fitted, MLPClassifier):
        return int(
            sum(w.size for w in fitted.coefs_) + sum(b.size for b in fitted.intercepts_)
        )
    if isinstance(fitted, LogisticRegression):
        return int(fitted.coef_.size + fitted.intercept_.size)
    return int(fitted.dual_coef_.size + fitted.intercept_.size)


def fit_and_evaluate(
    model: str, X: dict, y: dict, parameter_target: int, seed: int, cv_folds: int
) -> dict:
    """Tune on the training rows by CV, refit, and score validation and test.

    Returns
    -------
    dict
        Best hyperparameters, CV accuracy, parameter count, and accuracy /
        weighted F1 on ``train``, ``val`` and ``test``.
    """

    n_classes = len(np.unique(y["train"]))
    smallest_class = int(np.min(np.unique(y["train"], return_counts=True)[1]))
    folds = max(2, min(cv_folds, smallest_class))
    search = build_search(
        model, X["train"].shape[1], n_classes, parameter_target, seed, folds
    )
    search.fit(X["train"], y["train"])
    logger.info(
        "TRAIN_CV_COMPLETED | model=%s | folds=%d | best=%s | cv_accuracy=%.6f",
        model,
        folds,
        search.best_params_,
        search.best_score_,
    )
    scores = {
        "best_params": search.best_params_,
        "cv_accuracy": float(search.best_score_),
        "cv_folds": folds,
    }
    scores["parameters"] = count_parameters(search.best_estimator_)
    for split in ("train", "val", "test"):
        logger.info("EVALUATION_STARTED | model=%s | split=%s", model, split)
        predictions = search.best_estimator_.predict(X[split])
        scores[split] = {
            "accuracy": float(accuracy_score(y[split], predictions)),
            "f1_weighted": float(f1_score(y[split], predictions, average="weighted")),
        }
        logger.info(
            "EVALUATION_COMPLETED | model=%s | split=%s | accuracy=%.6f",
            model,
            split,
            scores[split]["accuracy"],
        )
    return scores
