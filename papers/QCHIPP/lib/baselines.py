"""Fitted scikit-learn controls on the same residue inputs as the QCNNs."""

from __future__ import annotations

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .data import one_hot_features

BASELINE_MODELS = ("logistic", "rf", "logistic_onehot", "rf_onehot")


def baseline_features(name: str, bits: torch.Tensor) -> torch.Tensor:
    """Return the input representation a baseline is trained on.

    ``*_onehot`` controls see the same four residues as 80 one-hot columns;
    the others see the 20 category-rank bits used by the quantum models.
    """
    return one_hot_features(bits) if name.endswith("_onehot") else bits


def build_baseline(cfg: dict):
    """Construct the configured logistic-regression or random-forest control.

    Parameters
    ----------
    cfg : dict
        Needs ``model.name``, ``seed`` and the matching ``baselines.logistic``
        (``C``, ``max_iter``) or ``baselines.random_forest`` (``n_estimators``)
        hyperparameters.

    Returns
    -------
    sklearn estimator
        Unfitted classifier.
    """
    name = cfg["model"]["name"]
    seed = int(cfg["seed"])
    if name in ("logistic", "logistic_onehot"):
        settings = cfg["baselines"]["logistic"]
        return LogisticRegression(
            C=float(settings["C"]),
            max_iter=int(settings["max_iter"]),
            random_state=seed,
        )
    if name in ("rf", "rf_onehot"):
        settings = cfg["baselines"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=int(settings["n_estimators"]), random_state=seed, n_jobs=1
        )
    raise ValueError(f"Unsupported baseline model: {name}")


def fit_baseline(cfg: dict, bits: torch.Tensor, targets: torch.Tensor):
    """Fit a baseline on the full released training split.

    Returns
    -------
    tuple
        Fitted estimator and a training record compatible with
        :func:`lib.training.train_model`.
    """
    name = cfg["model"]["name"]
    model = build_baseline(cfg)
    model.fit(baseline_features(name, bits).numpy(), targets.numpy())
    # Random forests have no fixed parameter vector, so the count is None.
    parameter_count = (
        int(model.coef_.size + model.intercept_.size)
        if name.startswith("logistic")
        else None
    )
    training = {
        "updates": 1,
        "optimizer": "sklearn native solver",
        "losses": [],
        "parameters": {
            "quantum": 0,
            "classical": parameter_count,
            "total": parameter_count,
        },
    }
    return model, training


def predict_baseline(model, name: str, bits: torch.Tensor):
    """Return positive-class probabilities as a NumPy array."""
    return model.predict_proba(baseline_features(name, bits).numpy())[:, 1]
