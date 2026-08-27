"""Training and evaluation helpers for qSVM gate-based and photonic variants."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import PreparedData
from .models import MODEL_REGISTRY, PHOTONIC_BASED_MODELS, make_model


@dataclass
class ModelResult:
    """Structured output for one model variant and one seed."""

    name: str
    metrics: dict[str, Any]
    train_time_s: float
    predict_time_s: float
    n_train: int
    n_test: int
    n_features: int
    n_qubits: int | None
    y_train_true: list[int]
    y_train_pred: list[int]
    y_test_true: list[int]
    y_test_pred: list[int]
    y_train_score: list[float] | None
    y_test_score: list[float] | None
    train_indices: list[int]
    test_indices: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return {
            "name": self.name,
            "metrics": self.metrics,
            "train_time_s": self.train_time_s,
            "predict_time_s": self.predict_time_s,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "n_qubits": self.n_qubits,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
            "metadata": self.metadata,
            "train_predictions": {
                "y_true": self.y_train_true,
                "y_pred": self.y_train_pred,
                "y_score": self.y_train_score,
            },
            "test_predictions": {
                "y_true": self.y_test_true,
                "y_pred": self.y_test_pred,
                "y_score": self.y_test_score,
            },
        }


def _metric_average(cfg: dict[str, Any]) -> str:
    return str(cfg.get("metric_average", "binary"))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray | None = None,
    average: str = "binary",
) -> dict[str, Any]:
    """Compute the paper's classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    average : str, optional
        sklearn averaging mode for precision/recall/F1. Default value is
        "binary".
    y_score : np.ndarray | None, optional
        Continuous score for the positive class, used to compute ROC-AUC when
        available. Default value is None.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1-score, confusion matrix, and the full
        sklearn classification report.
    """

    labels = [0, 1]
    kwargs: dict[str, Any] = {"zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = 1
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, average=average, **kwargs)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, **kwargs)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    if y_score is not None:
        score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) == 2 and score.shape[0] == len(y_true):
            metrics["roc_auc"] = float(roc_auc_score(y_true, score))
    return metrics


def data_for_model(
    name: str,
    prepared: PreparedData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the train/test arrays consumed by a configured model.

    Parameters
    ----------
    name : str
        Model name from ``MODEL_REGISTRY``.
    prepared : PreparedData
        Prepared dataset with classical, gate-based, and photonic feature views.

    Returns
    -------
    tuple of np.ndarray
        ``X_train``, ``X_test``, ``y_train``, ``y_test``, train indices, and
        test indices for the model family.
    """

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_REGISTRY}.")
    if name in {"svm_classical", "knn_classical"}:
        return (
            prepared.X_train_classical,
            prepared.X_test_classical,
            prepared.y_train,
            prepared.y_test,
            prepared.train_indices,
            prepared.test_indices,
        )
    if name in PHOTONIC_BASED_MODELS:
        X_train = prepared.X_train_photonic
        X_test = prepared.X_test_photonic
        y_train = prepared.y_train
        y_test = prepared.y_test
        train_indices = prepared.train_indices
        test_indices = prepared.test_indices
        return X_train, X_test, y_train, y_test, train_indices, test_indices
    X_train = prepared.X_train_quantum
    X_test = prepared.X_test_quantum
    y_train = prepared.y_train
    y_test = prepared.y_test
    train_indices = prepared.train_indices
    test_indices = prepared.test_indices
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def _score_to_list(score: np.ndarray | None) -> list[float] | None:
    if score is None:
        return None
    return np.asarray(score, dtype=float).tolist()


def run_model_variant(
    name: str,
    prepared: PreparedData,
    cfg: dict[str, Any],
    *,
    seed: int,
    logger: logging.Logger | None = None,
    feature_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> ModelResult:
    """Train and evaluate a configured model variant.

    Parameters
    ----------
    name : str
        Model name from ``MODEL_REGISTRY``.
    prepared : PreparedData
        Prepared train/test arrays.
    cfg : dict
        Resolved experiment configuration.
    seed : int
        Current seed.
    logger : logging.Logger | None, optional
        Logger for progress messages. Default value is None.
    feature_cache : dict | None, optional
        Per-seed cache for photonic reservoir features. Default value is None.

    Returns
    -------
    ModelResult
        Structured metrics and predictions.
    """

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_REGISTRY}.")
    X_train, X_test, y_train, y_test, train_indices, test_indices = data_for_model(
        name,
        prepared,
    )
    model = make_model(name, cfg, feature_cache=feature_cache)
    n_qubits_for_log = model.n_qubits_for(X_train.shape[1])
    if logger:
        logger.info(
            "Running %s: train=%d test=%d features=%d qubits=%s",
            name,
            len(y_train),
            len(y_test),
            X_train.shape[1],
            n_qubits_for_log,
        )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    y_test_pred, y_test_score = model.predict_with_scores(X_test)
    predict_time = time.perf_counter() - start
    y_train_pred, y_train_score = model.predict_train_with_scores()
    metadata = model.metadata()
    result_n_features = int(
        model.n_features_ if model.n_features_ is not None else X_train.shape[1]
    )
    n_qubits = model.n_qubits_ if model.n_qubits_ is not None else n_qubits_for_log

    metrics = evaluate_predictions(
        y_test,
        y_test_pred,
        y_score=y_test_score,
        average=_metric_average(cfg),
    )
    return ModelResult(
        name=name,
        metrics=metrics,
        train_time_s=float(train_time),
        predict_time_s=float(predict_time),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        n_features=result_n_features,
        n_qubits=n_qubits,
        y_train_true=y_train.astype(int).tolist(),
        y_train_pred=np.asarray(y_train_pred, dtype=int).tolist(),
        y_test_true=y_test.astype(int).tolist(),
        y_test_pred=np.asarray(y_test_pred, dtype=int).tolist(),
        y_train_score=_score_to_list(y_train_score),
        y_test_score=_score_to_list(y_test_score),
        train_indices=np.asarray(train_indices, dtype=int).tolist(),
        test_indices=np.asarray(test_indices, dtype=int).tolist(),
        metadata=metadata,
    )
