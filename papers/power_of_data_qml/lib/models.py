"""Model families, hyperparameter grids, and the two selection protocols.

Grids are those of App. J.5 (classical models) and App. J.6 (quantum kernel
models). Every hyperparameter setting is a ``Candidate``; a model family is
the set of candidates sharing a family name (for regression the SVR and
KernelRidge variants of one kernel form one family, since the paper takes "the
best between SVR and KernelRidge").

Selection protocols (LOG.md, Fair Baseline Plan):

* ``paper``: fit every candidate on the training split and report each family's
  best test score. This is test-set selection, as the paper reports
  "the prediction performance under the best hyper-parameter"; it is optimistic
  and applied symmetrically to every model.
* ``cv``: choose each family's candidate by K-fold cross-validation on the
  training split, refit it on the whole training split, and report its test
  score. Ties keep the earliest candidate in grid order.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

LOGGER = logging.getLogger(__name__)

PAPER_GRIDS = {
    # App. J.5 / J.6.
    "C": (
        0.006,
        0.015,
        0.03,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        256.0,
        512.0,
        1024.0,
    ),
    "svm_gamma_scales": (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0),
    "mlp_hidden": (10, 25, 50, 75, 100, 125, 150, 200),
    "tree_depth": (2, 3, 4, 5),
    "tree_estimators": (25, 50, 100, 200, 500),
    "adaboost_estimators": (25, 50, 100, 200, 500),
    "polynomial_degrees": (2, 3),
}
SMOKE_GRIDS = {
    # Tiny grids for smoke tests only; never used for reported results.
    "C": (1.0, 64.0),
    "svm_gamma_scales": (1.0,),
    "mlp_hidden": (25,),
    "tree_depth": (3,),
    "tree_estimators": (25,),
    "adaboost_estimators": (25,),
    "polynomial_degrees": (2,),
}
CLASSICAL_FAMILIES = (
    "linear_kernel",
    "gaussian_kernel",
    "neural_network",
    "random_forest",
    "gradient_boosting",
    "adaboost",
)
EXTENSION_FAMILIES = ("polynomial_kernel",)


@dataclass(frozen=True)
class Candidate:
    """One hyperparameter setting of one model family.

    Attributes
    ----------
    family : str
        Model family name (e.g. ``"gaussian_kernel"``).
    params : dict
        Hyperparameters, recorded in the metrics.
    input_key : str
        Key of the input matrix in the task's ``inputs`` mapping.
    input_kind : str
        ``"kernel"`` (precomputed square kernel) or ``"features"``.
    make_estimator : callable
        Zero-argument factory of an unfitted scikit-learn estimator.
    """

    family: str
    params: dict
    input_key: str
    input_kind: str
    make_estimator: Callable = field(compare=False)


@dataclass
class Task:
    """A supervised task over one subsample.

    Attributes
    ----------
    kind : str
        ``"classification"`` (accuracy, maximized) or ``"regression"`` (MAE,
        minimized).
    labels : numpy.ndarray
        Targets for all subsample points.
    train_indices, test_indices : numpy.ndarray
        Disjoint positions into the subsample.
    inputs : dict
        Input name to matrix (kernel ``(N, N)`` or features ``(N, d)``).
    """

    kind: str
    labels: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    inputs: dict


def _grids(grid_name: str) -> dict:
    if grid_name == "paper":
        return PAPER_GRIDS
    if grid_name == "smoke":
        return SMOKE_GRIDS
    raise ValueError(f"Unknown grid: {grid_name}")


def _kernel_estimators(
    kind: str, C: float, svr_epsilon: float
) -> list[tuple[str, Callable]]:
    if kind == "classification":
        return [("svc", lambda C=C: SVC(kernel="precomputed", C=C))]
    # KernelRidge alpha = 1 / (2C) is the standard correspondence to the SVM C (AMB-9).
    return [
        ("svr", lambda C=C: SVR(kernel="precomputed", C=C, epsilon=svr_epsilon)),
        (
            "kernel_ridge",
            lambda C=C: KernelRidge(kernel="precomputed", alpha=1.0 / (2.0 * C)),
        ),
    ]


def kernel_family_candidates(
    family: str,
    kind: str,
    kernel_keys: list[tuple[str, dict]],
    *,
    grid_name: str,
    svr_epsilon: float,
) -> list[Candidate]:
    """Candidates of a precomputed-kernel family over the C grid.

    Parameters
    ----------
    family : str
        Family name.
    kind : str
        Task kind.
    kernel_keys : list of (str, dict)
        Input keys of the family's kernel matrices with their own parameters
        (e.g. one key per bandwidth).
    grid_name : str
        ``"paper"`` or ``"smoke"``.
    svr_epsilon : float
        SVR epsilon (regression only).
    """
    candidates = []
    for input_key, kernel_params in kernel_keys:
        for C in _grids(grid_name)["C"]:
            for estimator_name, factory in _kernel_estimators(kind, C, svr_epsilon):
                candidates.append(
                    Candidate(
                        family,
                        {**kernel_params, "C": C, "estimator": estimator_name},
                        input_key,
                        "kernel",
                        factory,
                    )
                )
    return candidates


def feature_family_candidates(
    kind: str, *, grid_name: str, random_state: int, mlp_max_iter: int
) -> list[Candidate]:
    """Candidates of the non-kernel classical families on the PCA features."""
    grids = _grids(grid_name)
    classify = kind == "classification"
    candidates = []
    for hidden in grids["mlp_hidden"]:
        estimator_cls = MLPClassifier if classify else MLPRegressor
        candidates.append(
            Candidate(
                "neural_network",
                {"hidden_layer_size": hidden},
                "features",
                "features",
                lambda hidden=hidden, cls=estimator_cls: cls(
                    hidden_layer_sizes=(hidden,),
                    max_iter=mlp_max_iter,
                    random_state=random_state,
                ),
            )
        )
    for family, classifier, regressor in (
        ("random_forest", RandomForestClassifier, RandomForestRegressor),
        ("gradient_boosting", GradientBoostingClassifier, GradientBoostingRegressor),
    ):
        estimator_cls = classifier if classify else regressor
        for depth in grids["tree_depth"]:
            for n_estimators in grids["tree_estimators"]:
                candidates.append(
                    Candidate(
                        family,
                        {"max_depth": depth, "n_estimators": n_estimators},
                        "features",
                        "features",
                        lambda d=depth, e=n_estimators, cls=estimator_cls: cls(
                            max_depth=d, n_estimators=e, random_state=random_state
                        ),
                    )
                )
    for n_estimators in grids["adaboost_estimators"]:
        estimator_cls = AdaBoostClassifier if classify else AdaBoostRegressor
        candidates.append(
            Candidate(
                "adaboost",
                {"n_estimators": n_estimators},
                "features",
                "features",
                lambda e=n_estimators, cls=estimator_cls: cls(
                    n_estimators=e, random_state=random_state
                ),
            )
        )
    return candidates


def score(kind: str, truth: np.ndarray, predictions: np.ndarray) -> float:
    """Accuracy (classification) or mean absolute error (regression)."""
    if kind == "classification":
        return float(np.mean(predictions == truth))
    return float(np.mean(np.abs(predictions - truth)))


def is_better(kind: str, candidate_score: float, incumbent: float | None) -> bool:
    """Strict improvement under the task's metric direction."""
    if incumbent is None:
        return True
    return (
        candidate_score > incumbent
        if kind == "classification"
        else candidate_score < incumbent
    )


def _fit_predict(
    candidate: Candidate,
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    evaluate: np.ndarray,
) -> np.ndarray:
    estimator = candidate.make_estimator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if candidate.input_kind == "kernel":
            estimator.fit(matrix[np.ix_(train, train)], labels[train])
            return estimator.predict(matrix[np.ix_(evaluate, train)])
        estimator.fit(matrix[train], labels[train])
        return estimator.predict(matrix[evaluate])


def _evaluate_candidate(kind, candidate, matrix, labels, train, evaluate):
    predictions = _fit_predict(candidate, matrix, labels, train, evaluate)
    return score(kind, labels[evaluate], predictions)


def run_protocol(
    task: Task,
    candidates: list[Candidate],
    *,
    protocol: str,
    n_jobs: int,
    cv_folds: int,
    random_state: int,
) -> dict:
    """Select and score every family's candidates under one protocol.

    Parameters
    ----------
    task : Task
        The supervised task.
    candidates : list of Candidate
        All candidates of all families to compare.
    protocol : str
        ``"paper"`` or ``"cv"``.
    n_jobs : int
        Parallel joblib workers.
    cv_folds : int
        Folds for the ``cv`` protocol.
    random_state : int
        Seed of the fold split.

    Returns
    -------
    dict
        Family name to ``{"test_score", "params", "selection_score"}``.
    """
    train, test = task.train_indices, task.test_indices
    parallel = Parallel(n_jobs=n_jobs)
    families: dict[str, dict] = {}
    if protocol == "paper":
        scores = parallel(
            delayed(_evaluate_candidate)(
                task.kind, c, task.inputs[c.input_key], task.labels, train, test
            )
            for c in candidates
        )
        for candidate, test_score in zip(candidates, scores):
            entry = families.get(candidate.family)
            if entry is None or is_better(task.kind, test_score, entry["test_score"]):
                families[candidate.family] = {
                    "test_score": test_score,
                    "params": candidate.params,
                    "selection_score": test_score,
                }
        return families

    if protocol == "cv":
        splitter_cls = StratifiedKFold if task.kind == "classification" else KFold
        splitter = splitter_cls(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        folds = [
            (train[fold_train], train[fold_valid])
            for fold_train, fold_valid in splitter.split(train, task.labels[train])
        ]
        fold_scores = parallel(
            delayed(_evaluate_candidate)(
                task.kind,
                c,
                task.inputs[c.input_key],
                task.labels,
                fold_train,
                fold_valid,
            )
            for c in candidates
            for fold_train, fold_valid in folds
        )
        mean_scores = (
            np.asarray(fold_scores).reshape(len(candidates), len(folds)).mean(axis=1)
        )
        selected: dict[str, tuple[Candidate, float]] = {}
        for candidate, cv_score in zip(candidates, mean_scores):
            incumbent = selected.get(candidate.family)
            if incumbent is None or is_better(task.kind, float(cv_score), incumbent[1]):
                selected[candidate.family] = (candidate, float(cv_score))
        refit_scores = parallel(
            delayed(_evaluate_candidate)(
                task.kind, c, task.inputs[c.input_key], task.labels, train, test
            )
            for c, _ in selected.values()
        )
        for (family, (candidate, cv_score)), test_score in zip(
            selected.items(), refit_scores
        ):
            families[family] = {
                "test_score": test_score,
                "params": candidate.params,
                "selection_score": cv_score,
            }
        return families

    raise ValueError(f"Unknown protocol: {protocol}")


def best_of(kind: str, family_results: dict, families: tuple[str, ...]) -> dict:
    """Best test score among ``families`` (the paper's "best classical ML")."""
    best = None
    for family in families:
        if family not in family_results:
            continue
        entry = family_results[family]
        if best is None or is_better(kind, entry["test_score"], best["test_score"]):
            best = {"family": family, "test_score": entry["test_score"]}
    if best is None:
        raise ValueError(f"None of {families} were evaluated")
    return best


__all__ = [
    "CLASSICAL_FAMILIES",
    "EXTENSION_FAMILIES",
    "PAPER_GRIDS",
    "SMOKE_GRIDS",
    "Candidate",
    "Task",
    "best_of",
    "feature_family_candidates",
    "is_better",
    "kernel_family_candidates",
    "run_protocol",
    "score",
]
