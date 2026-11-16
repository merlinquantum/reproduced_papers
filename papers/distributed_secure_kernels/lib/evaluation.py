"""Cross-validated kernel SVM and kernel PCA, under two protocols.

Paper-faithful protocol (authors' ``tester.py`` and ``library.py``):

- one kernel matrix on the whole evaluated subset;
- kernel SVM: ``SVC(kernel='precomputed')`` with default C = 1, scored by
  ``StratifiedKFold(5, shuffle=True, random_state=0)``;
- kernel PCA: double-centred kernel, ``KernelPCA(kernel='precomputed')``
  fitted on *all* samples, then a linear ``SVC`` scored by the same CV, for
  2..7 components; the paper reports the best component count, i.e. a choice
  made on the evaluation folds (reported here as ``best_on_test_folds``).

Corrected protocol (fixed before any run): for each outer fold,
oversampling only duplicates training rows, the scaler and kernel PCA are
fitted on the training fold, and C (and the component count) are chosen by an
inner stratified 5-fold CV on the training fold.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# KernelFn(train_features, other_features, symmetric) -> kernel block.
KernelFn = Callable[[np.ndarray, np.ndarray, bool], np.ndarray]


def stratified_folds(
    labels: np.ndarray, n_splits: int, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold indices of ``StratifiedKFold(n_splits, shuffle=True, random_state)``."""

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def _fold_scores(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
    }


def summarize(folds: list[dict[str, float]]) -> dict[str, Any]:
    """Mean and population std over folds (the paper's ``scores.std()``)."""

    summary: dict[str, Any] = {"folds": folds}
    for metric in ("accuracy", "balanced_accuracy"):
        values = np.array([fold[metric] for fold in folds])
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def double_center(kernel: np.ndarray) -> np.ndarray:
    """``library.py::center_kernel_matrix``: ``K - 1K - K1 + 1K1`` with ``1 = ones / N``."""

    n = kernel.shape[0]
    ones = np.full((n, n), 1.0 / n)
    return kernel - ones @ kernel - kernel @ ones + ones @ kernel @ ones


# --------------------------------------------------------------------------- #
# Paper-faithful protocol
# --------------------------------------------------------------------------- #


def paper_kernel_svm(
    kernel: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    tag: str,
) -> dict[str, Any]:
    """Precomputed-kernel SVM (C = 1) scored on the given folds."""

    results = []
    for index, (train, test) in enumerate(folds, start=1):
        model = SVC(kernel="precomputed").fit(
            kernel[np.ix_(train, train)], labels[train]
        )
        scores = _fold_scores(labels[test], model.predict(kernel[np.ix_(test, train)]))
        results.append(scores)
        logger.info(
            "TRAIN_FOLD_COMPLETED | method=%s | model=kernel_svm | fold=%d/%d | accuracy=%.4f",
            tag,
            index,
            len(folds),
            scores["accuracy"],
        )
    return summarize(results)


def paper_kernel_pca(
    kernel: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    components: list[int],
    tag: str,
) -> dict[str, Any]:
    """``library.py::compute_pca_metrics_and_svm``: kernel PCA on all samples, then linear SVM CV."""

    centred = double_center(kernel)
    per_count: dict[str, Any] = {}
    for count in components:
        projected = KernelPCA(n_components=count, kernel="precomputed").fit_transform(
            centred
        )
        results = []
        for train, test in folds:
            model = SVC(kernel="linear").fit(projected[train], labels[train])
            results.append(_fold_scores(labels[test], model.predict(projected[test])))
        per_count[str(count)] = summarize(results)
        logger.info(
            "TRAIN_KPCA_COMPLETED | method=%s | components=%d | accuracy=%.4f",
            tag,
            count,
            per_count[str(count)]["accuracy"]["mean"],
        )
    best = max(components, key=lambda c: per_count[str(c)]["accuracy"]["mean"])
    return {
        "per_component_count": per_count,
        # Paper rule: component count with the highest CV accuracy, i.e. selected
        # on the evaluation folds themselves (optimistic).
        "best_on_test_folds": {"components": best, **per_count[str(best)]},
    }


# --------------------------------------------------------------------------- #
# Corrected protocol
# --------------------------------------------------------------------------- #


def oversample_indices(
    labels: np.ndarray, ratio: float | None, rng: np.random.Generator
) -> np.ndarray:
    """Training row indices after random minority duplication (identity if ``ratio`` is None)."""

    rows = np.arange(len(labels))
    if ratio is None:
        return rows
    classes, counts = np.unique(labels, return_counts=True)
    minority = classes[np.argmin(counts)]
    n_added = int(counts.max() * ratio - counts.min())
    if n_added <= 0:
        return rows
    added = rng.choice(np.flatnonzero(labels == minority), size=n_added, replace=True)
    return np.concatenate([rows, added])


def _best_point(
    grid: list[dict[str, Any]], mean_score: Callable[[dict[str, Any]], float]
) -> dict[str, Any]:
    """Grid point with the highest mean inner-CV accuracy (first one on ties)."""

    best_point, best_score = grid[0], -np.inf
    for point in grid:
        score = mean_score(point)
        if score > best_score + 1e-12:
            best_point, best_score = point, score
    return {**best_point, "inner_cv_accuracy": float(best_score)}


def corrected_cv(
    features: np.ndarray,
    labels: np.ndarray,
    kernel_grid: list[dict[str, Any]],
    kernel_factory: Callable[[dict[str, Any], np.random.Generator], KernelFn],
    eval_cfg: dict[str, Any],
    oversample_ratio: float | None,
    standardize: bool,
    rng: np.random.Generator,
    tag: str,
) -> dict[str, Any]:
    """Nested CV of a kernel SVM (and optionally kernel PCA + linear SVM).

    Oversampling duplicates rows of the fitting set only (outer training fold,
    and inner training folds during selection); validation and test rows are
    never duplicated. A duplicated row reuses its sample's kernel row.

    Parameters
    ----------
    features, labels : numpy.ndarray
        The evaluated subset, *not* oversampled and *not* scaled.
    kernel_grid : list of dict
        Kernel hyperparameter points (``[{}]`` for kernels without any).
    kernel_factory : callable
        ``kernel_factory(point, rng)`` returns a :data:`KernelFn`.
    eval_cfg : dict
        ``n_splits``, ``cv_random_state``, ``inner_splits``, ``C_grid``,
        ``kernel_pca`` (bool) and ``kpca_components``.
    oversample_ratio : float | None
        Minority / majority ratio for duplicating fitting rows (None: no oversampling).
    standardize : bool
        Fit a ``StandardScaler`` on each training fold.
    rng : numpy.random.Generator
        Oversampling and shot-noise generator.
    tag : str
        Method name for the log.

    Returns
    -------
    dict
        ``kernel_svm`` (and ``kernel_pca``) summaries with the selected
        hyperparameters per fold.
    """

    outer = stratified_folds(
        labels, int(eval_cfg["n_splits"]), int(eval_cfg["cv_random_state"])
    )
    c_grid = [{"C": float(c)} for c in eval_cfg["C_grid"]]
    svm_folds, pca_folds = [], []
    for index, (train, test) in enumerate(outer, start=1):
        train_x, test_x = features[train], features[test]
        if standardize:
            scaler = StandardScaler().fit(train_x)
            train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)
        train_y, test_y = labels[train], labels[test]
        inner = stratified_folds(
            train_y, int(eval_cfg["inner_splits"]), int(eval_cfg["cv_random_state"])
        )
        # One oversampling draw per fitting set, shared by every grid point, so
        # that hyperparameters are compared on identical training rows.
        inner_sets = [
            (rows[oversample_indices(train_y[rows], oversample_ratio, rng)], held)
            for rows, held in inner
        ]
        outer_rows = oversample_indices(train_y, oversample_ratio, rng)

        best: dict[str, Any] | None = None
        for point in kernel_grid:
            kernel_fn = kernel_factory(point, rng)
            kernel_train = kernel_fn(train_x, train_x, True)
            cross = kernel_fn(test_x, train_x, False)

            def svm_score(svm_point, _k=kernel_train, _y=train_y, _sets=inner_sets):
                scores = []
                for rows, held in _sets:
                    model = SVC(kernel="precomputed", C=svm_point["C"])
                    model.fit(_k[np.ix_(rows, rows)], _y[rows])
                    predicted = model.predict(_k[np.ix_(held, rows)])
                    scores.append(accuracy_score(_y[held], predicted))
                return float(np.mean(scores))

            selected = _best_point(c_grid, svm_score)
            if (
                best is None
                or selected["inner_cv_accuracy"]
                > best["selected"]["inner_cv_accuracy"] + 1e-12
            ):
                best = {
                    "selected": {**point, **selected},
                    "kernel_train": kernel_train,
                    "cross": cross,
                }
        kernel_train, cross, selected = (
            best["kernel_train"],
            best["cross"],
            best["selected"],
        )
        model = SVC(kernel="precomputed", C=selected["C"]).fit(
            kernel_train[np.ix_(outer_rows, outer_rows)], train_y[outer_rows]
        )
        scores = _fold_scores(test_y, model.predict(cross[:, outer_rows]))
        svm_folds.append({**scores, "selected": selected})
        logger.info(
            "TRAIN_FOLD_COMPLETED | method=%s | model=kernel_svm | protocol=corrected | fold=%d/%d | accuracy=%.4f | selected=%s",
            tag,
            index,
            len(outer),
            scores["accuracy"],
            selected,
        )
        if eval_cfg["kernel_pca"]:
            pca_folds.append(
                _corrected_kernel_pca_fold(
                    kernel_train,
                    cross,
                    train_y,
                    test_y,
                    inner_sets,
                    outer_rows,
                    eval_cfg,
                )
            )
            logger.info(
                "TRAIN_FOLD_COMPLETED | method=%s | model=kernel_pca | protocol=corrected | fold=%d/%d | accuracy=%.4f | selected=%s",
                tag,
                index,
                len(outer),
                pca_folds[-1]["accuracy"],
                pca_folds[-1]["selected"],
            )
    result: dict[str, Any] = {"kernel_svm": summarize(svm_folds)}
    if eval_cfg["kernel_pca"]:
        result["kernel_pca"] = summarize(pca_folds)
    return result


def _project(kernel, rows, targets, n_components):
    """Kernel PCA fitted on ``rows``; projections of ``rows`` and of ``targets``.

    ``targets`` is either row indices into ``kernel`` or a ready cross-kernel block.
    Components are sorted by decreasing eigenvalue, so the first ``c`` columns are
    the ``c``-component projection (one fit serves every component count).
    """

    pca = KernelPCA(n_components=n_components, kernel="precomputed")
    fitted = pca.fit_transform(kernel[np.ix_(rows, rows)])
    block = kernel[np.ix_(targets, rows)] if targets.ndim == 1 else targets
    return fitted, pca.transform(block)


def _corrected_kernel_pca_fold(
    kernel_train, cross, train_y, test_y, inner_sets, outer_rows, eval_cfg
):
    """Kernel PCA fitted on the (oversampled) training fold; components and C by inner CV."""

    counts = sorted(int(count) for count in eval_cfg["kpca_components"])
    projections = [
        (*_project(kernel_train, rows, held, counts[-1]), rows, held)
        for rows, held in inner_sets
    ]

    def score(point):
        values = []
        for fitted, held_proj, rows, held in projections:
            c = point["components"]
            model = SVC(kernel="linear", C=point["C"]).fit(fitted[:, :c], train_y[rows])
            values.append(
                accuracy_score(train_y[held], model.predict(held_proj[:, :c]))
            )
        return float(np.mean(values))

    grid = [
        {"components": count, "C": float(c)}
        for count in counts
        for c in eval_cfg["C_grid"]
    ]
    selected = _best_point(grid, score)
    fitted, test_proj = _project(
        kernel_train, outer_rows, cross[:, outer_rows], counts[-1]
    )
    c = selected["components"]
    model = SVC(kernel="linear", C=selected["C"]).fit(
        fitted[:, :c], train_y[outer_rows]
    )
    scores = _fold_scores(test_y, model.predict(test_proj[:, :c]))
    return {**scores, "selected": selected}
