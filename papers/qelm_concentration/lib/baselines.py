"""Classical baselines of the Fair Baseline Plan (LOG.md).

- Paper Table A2: nine untuned scikit-learn classifiers on standardised raw Landsat
  features. ``AdaBoostClassifier(algorithm="SAMME.R")`` no longer exists in
  scikit-learn >= 1.6, so the current default (SAMME) is used.
- Matched classical random-feature ELMs (tanh projection and random Fourier features)
  whose feature count equals the QELM readout dimension, with the same ridge readout.
- Ridge on degree-2 polynomial features.
- Tuned RF / kNN / RBF-SVM on a validation split carved from the training set.
- NARMA: ridge on the raw window, on degree-2 window features, a matched random-feature
  ELM, and a 400-unit echo state network (true memory; an upper reference).
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .readout import classification_metrics

logger = logging.getLogger(__name__)

# Models that are deterministic given the data; one initialisation is enough.
DETERMINISTIC_PAPER_BASELINES = {"ridge", "knn", "svm_rbf", "naive_bayes", "qda"}


def paper_baseline(name: str, seed: int):
    """Return one Table A2 model inside a StandardScaler pipeline."""
    models = {
        "ridge": lambda: RidgeClassifierCV(fit_intercept=True),
        "knn": lambda: KNeighborsClassifier(n_neighbors=7),
        "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=seed),
        "random_forest": lambda: RandomForestClassifier(random_state=seed),
        "mlp": lambda: MLPClassifier(max_iter=1000, random_state=seed),
        "adaboost": lambda: AdaBoostClassifier(random_state=seed),
        "naive_bayes": lambda: GaussianNB(),
        "qda": lambda: QuadraticDiscriminantAnalysis(),
    }
    if name not in models:
        raise ValueError(
            f"Unknown paper baseline {name!r}; expected one of {sorted(models)}"
        )
    return make_pipeline(StandardScaler(), models[name]())


def run_paper_baselines(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    names: list[str],
    n_inits: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Mean and std of test metrics over ``n_inits`` model seeds for each baseline."""
    results = {}
    for name in names:
        repeats = 1 if name in DETERMINISTIC_PAPER_BASELINES else n_inits
        runs = []
        for init in range(repeats):
            model = paper_baseline(name, seed + init)
            model.fit(train_x, train_y)
            runs.append(classification_metrics(test_y, model.predict(test_x)))
        results[name] = summarise_runs(runs)
        results[name]["n_inits"] = repeats
        logger.info(
            "EVALUATION_COMPLETED | baseline=%s | inits=%d | f1_macro=%.4f",
            name,
            repeats,
            results[name]["f1_macro_mean"],
        )
    return results


def summarise_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation of every metric over a list of runs."""
    summary = {}
    for key in runs[0]:
        values = np.array([run[key] for run in runs], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def random_tanh_features(
    train: np.ndarray, test: np.ndarray, n_features: int, input_scale: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Classical ELM features ``tanh(W x + c)`` on inputs already scaled to ``[-1, 1]``.

    ``W ~ N(0, input_scale**2 / d)``, ``c ~ U(-1, 1)``; ``d`` is the input dimension.
    """
    rng = np.random.default_rng(seed)
    dim = train.shape[1]
    weights = rng.normal(0.0, input_scale / np.sqrt(dim), size=(dim, n_features))
    offset = rng.uniform(-1.0, 1.0, size=n_features)
    return np.tanh(train @ weights + offset), np.tanh(test @ weights + offset)


def random_fourier_features(
    train: np.ndarray, test: np.ndarray, n_features: int, bandwidth: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random Fourier features ``cos(W x + c)`` of an RBF kernel with the given bandwidth."""
    rng = np.random.default_rng(seed)
    dim = train.shape[1]
    weights = rng.normal(0.0, 1.0 / bandwidth, size=(dim, n_features))
    offset = rng.uniform(0.0, 2 * np.pi, size=n_features)
    return np.cos(train @ weights + offset), np.cos(test @ weights + offset)


def polynomial_features(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Degree-2 polynomial features without the constant column."""
    transform = PolynomialFeatures(degree=2, include_bias=False)
    return transform.fit_transform(train), transform.transform(test)


def tune_and_evaluate(
    name: str,
    grid: dict[str, list],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> dict:
    """Select hyperparameters on a stratified validation split, refit on train, test once."""
    fit_x, val_x, fit_y, val_y = train_test_split(
        train_x,
        train_y,
        test_size=validation_fraction,
        stratify=train_y,
        random_state=seed,
    )
    constructors = {
        "random_forest": lambda **kw: RandomForestClassifier(random_state=seed, **kw),
        "knn": lambda **kw: KNeighborsClassifier(**kw),
        "svm_rbf": lambda **kw: SVC(kernel="rbf", random_state=seed, **kw),
    }
    if name not in constructors:
        raise ValueError(f"No tuning recipe for {name!r}")
    best_score, best_params = -np.inf, None
    keys = sorted(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values, strict=True))
        model = make_pipeline(StandardScaler(), constructors[name](**params))
        model.fit(fit_x, fit_y)
        score = classification_metrics(val_y, model.predict(val_x))["f1_macro"]
        logger.debug("Tuning %s params=%s val_f1_macro=%.4f", name, params, score)
        if score > best_score:
            best_score, best_params = score, params
    model = make_pipeline(StandardScaler(), constructors[name](**best_params))
    model.fit(train_x, train_y)
    metrics = classification_metrics(test_y, model.predict(test_x))
    return {"params": best_params, "validation_f1_macro": float(best_score), **metrics}


class EchoStateNetwork:
    """Leaky-free echo state network driven by the scalar NARMA input.

    Parameters
    ----------
    n_units : int
        Reservoir size (400 in paper App. C).
    spectral_radius : float
        Spectral radius of the recurrent matrix.
    input_scale : float
        Scale of the input weights, applied to ``4u - 1`` (``u ~ Unif(0, 0.5)`` mapped to
        ``[-1, 1]``).
    seed : int
        Seed of the random matrices.
    """

    def __init__(
        self, n_units: int, spectral_radius: float, input_scale: float, seed: int
    ):
        rng = np.random.default_rng(seed)
        recurrent = rng.normal(size=(n_units, n_units))
        recurrent *= spectral_radius / np.max(np.abs(np.linalg.eigvals(recurrent)))
        self.recurrent = recurrent
        self.input_weights = rng.uniform(-input_scale, input_scale, size=n_units)
        self.bias = rng.uniform(-0.1, 0.1, size=n_units)

    def states(self, raw_inputs: np.ndarray) -> np.ndarray:
        """Reservoir state after each input of the whole sequence; shape ``(T, n_units)``."""
        state = np.zeros(self.recurrent.shape[0])
        collected = np.empty((len(raw_inputs), state.size))
        for t, value in enumerate(4 * raw_inputs - 1):
            state = np.tanh(
                self.recurrent @ state + self.input_weights * value + self.bias
            )
            collected[t] = state
        return collected
