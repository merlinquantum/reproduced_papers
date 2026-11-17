"""SVM evaluation over random partitions for quantum and classical kernels.

Protocol (paper, "Classification of data via ABS"; SI Note 5): soft-margin SVM
with C = lambda = 50, 80/20 random train/test split, test accuracy averaged over
100 partitions. Two arms are evaluated on identical partitions:

- ``paper``: fixed C (50) for every model; RBF gamma from the median heuristic on
  the training split; no tuning.
- ``tuned``: inner 3-fold stratified CV on the training split only, same grid for
  every model (C for precomputed kernels; C x gamma or C x degree for feature
  SVMs; k for kNN). The test split is never used for selection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .kernels import median_heuristic_gamma

logger = logging.getLogger(__name__)

ARMS = ("paper", "tuned")


def _inner_cv(partition_seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=partition_seed)


def fit_precomputed(
    kernel: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    arm: str,
    evaluation_cfg: dict[str, Any],
    partition_seed: int,
) -> tuple[float, float, dict[str, Any]]:
    """Fit an SVM on a precomputed point kernel; return train/test accuracy.

    Parameters
    ----------
    kernel : numpy.ndarray
        ``(N, N)`` kernel over all points of the dataset.
    labels : numpy.ndarray
        ``(N,)`` labels in {-1, +1}.
    train, test : numpy.ndarray
        Index arrays of one partition.
    arm : str
        ``"paper"`` or ``"tuned"``.
    evaluation_cfg : dict
        Needs ``svm_C`` and ``tuning.C_grid``.
    partition_seed : int
        Seed of the inner CV shuffle (tuned arm).

    Returns
    -------
    tuple
        (train accuracy, test accuracy, selected hyperparameters).
    """
    train_kernel = kernel[np.ix_(train, train)]
    test_kernel = kernel[np.ix_(test, train)]
    if arm == "paper":
        model = SVC(kernel="precomputed", C=evaluation_cfg["svm_C"])
        model.fit(train_kernel, labels[train])
        chosen = {"C": evaluation_cfg["svm_C"]}
    elif arm == "tuned":
        search = GridSearchCV(
            SVC(kernel="precomputed"),
            {"C": evaluation_cfg["tuning"]["C_grid"]},
            cv=_inner_cv(partition_seed),
            scoring="accuracy",
        )
        search.fit(train_kernel, labels[train])
        model = search.best_estimator_
        chosen = dict(search.best_params_)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return (
        float(model.score(train_kernel, labels[train])),
        float(model.score(test_kernel, labels[test])),
        chosen,
    )


def fit_feature_model(
    model_cfg: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    arm: str,
    evaluation_cfg: dict[str, Any],
    partition_seed: int,
) -> tuple[float, float, dict[str, Any]]:
    """Fit a classical baseline on explicit features; return train/test accuracy.

    ``model_cfg["estimator"]`` is ``linear``, ``rbf``, ``poly`` or ``knn``.
    """
    estimator = model_cfg["estimator"]
    tuning = evaluation_cfg["tuning"]
    x_train, x_test = features[train], features[test]
    if estimator == "knn":
        if arm == "paper":
            model = KNeighborsClassifier(n_neighbors=model_cfg["k"]).fit(
                x_train, labels[train]
            )
            chosen = {"n_neighbors": model_cfg["k"]}
        else:
            search = GridSearchCV(
                KNeighborsClassifier(),
                {"n_neighbors": tuning["knn_grid"]},
                cv=_inner_cv(partition_seed),
            ).fit(x_train, labels[train])
            model, chosen = search.best_estimator_, dict(search.best_params_)
    else:
        if arm == "paper":
            params: dict[str, Any] = {"C": evaluation_cfg["svm_C"]}
            if estimator == "rbf":
                params["gamma"] = median_heuristic_gamma(x_train)
            if estimator == "poly":
                params.update(degree=model_cfg["degree"], coef0=1.0, gamma="scale")
            model = SVC(kernel=estimator, **params).fit(x_train, labels[train])
            chosen = {key: value for key, value in params.items() if key != "coef0"}
        else:
            grid: dict[str, Any] = {"C": tuning["C_grid"]}
            fixed: dict[str, Any] = {}
            if estimator == "rbf":
                grid["gamma"] = tuning["gamma_grid"]
            if estimator == "poly":
                grid["degree"] = tuning["degree_grid"]
                fixed = {"coef0": 1.0, "gamma": "scale"}
            search = GridSearchCV(
                SVC(kernel=estimator, **fixed), grid, cv=_inner_cv(partition_seed)
            ).fit(x_train, labels[train])
            model, chosen = search.best_estimator_, dict(search.best_params_)
    return (
        float(model.score(x_train, labels[train])),
        float(model.score(x_test, labels[test])),
        chosen,
    )


def evaluate_over_partitions(
    fit_one: Callable[[np.ndarray, np.ndarray, str, int], tuple[float, float, dict]],
    partitions: list[tuple[int, np.ndarray, np.ndarray]],
    arm: str,
) -> dict[str, Any]:
    """Run one model on every partition and summarise accuracies.

    Parameters
    ----------
    fit_one : Callable
        ``fit_one(train, test, arm, partition_seed) -> (train_acc, test_acc, params)``.
    partitions : list of (seed, train, test)
        Identical for every model in a run.
    arm : str
        Evaluation arm.

    Returns
    -------
    dict
        Mean/std of train and test accuracy, per-partition test accuracies and the
        chosen hyperparameters of each partition.
    """
    train_accuracy, test_accuracy, chosen = [], [], []
    for seed, train, test in partitions:
        train_acc, test_acc, params = fit_one(train, test, arm, seed)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        chosen.append(params)
        logger.debug(
            "partition=%d | arm=%s | train_acc=%.4f | test_acc=%.4f",
            seed,
            arm,
            train_acc,
            test_acc,
        )
    return {
        "arm": arm,
        "n_partitions": len(partitions),
        "mean_test_accuracy": float(np.mean(test_accuracy)),
        "std_test_accuracy": float(np.std(test_accuracy)),
        "mean_train_accuracy": float(np.mean(train_accuracy)),
        "std_train_accuracy": float(np.std(train_accuracy)),
        "test_accuracy_per_partition": [float(value) for value in test_accuracy],
        "chosen_hyperparameters": chosen,
    }
