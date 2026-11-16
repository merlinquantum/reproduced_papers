from __future__ import annotations

import numpy as np
from common import load_runtime_ready_config
from distributed_secure_kernels.lib.data import build_dataset
from distributed_secure_kernels.lib.evaluation import (
    corrected_cv,
    double_center,
    paper_kernel_svm,
    stratified_folds,
)


def test_wine_linear_svm_reproduces_table_1():
    # Table 1, Wine, centralised classical: 0.9860 +- 0.0172. (The released log
    # wine_dataset_1005332.txt prints 0.98621 from an older script version.)
    cfg = load_runtime_ready_config()
    data = build_dataset(cfg)
    folds = stratified_folds(data.labels, 5, 0)
    result = paper_kernel_svm(
        data.features @ data.features.T, data.labels, folds, "test"
    )
    assert round(result["accuracy"]["mean"], 4) == 0.9860
    assert round(result["accuracy"]["std"], 4) == 0.0172


def test_double_centering_is_idempotent():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 3))
    centred = double_center(a @ a.T)
    np.testing.assert_allclose(double_center(centred), centred, atol=1e-12)


def test_corrected_cv_never_puts_duplicates_in_test_folds():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 3))
    labels = np.array([0] * 15 + [1] * 45)
    seen = []

    def factory(point, generator):
        def kernel(left, right, symmetric):
            seen.append((left.shape[0], right.shape[0]))
            return left @ right.T

        return kernel

    evaluation = {
        "n_splits": 5,
        "cv_random_state": 0,
        "inner_splits": 3,
        "C_grid": [1.0],
        "kernel_pca": False,
    }
    result = corrected_cv(
        features, labels, [{}], factory, evaluation, 0.6, True, rng, "test"
    )
    assert len(result["kernel_svm"]["folds"]) == 5
    # Kernels are computed on unique rows only: 48 train rows, 12 test rows per fold.
    assert set(seen) == {(48, 48), (12, 48)}


def test_sliced_kernel_pca_matches_separate_fits_up_to_sign():
    from distributed_secure_kernels.lib.evaluation import _project
    from sklearn.decomposition import KernelPCA

    rng = np.random.default_rng(3)
    points = rng.normal(size=(40, 5))
    kernel = (points @ points.T) ** 2
    rows, held = np.arange(30), np.arange(30, 40)
    fitted, projected = _project(kernel, rows, held, 7)
    for count in (2, 5):
        separate = KernelPCA(n_components=count, kernel="precomputed").fit(
            kernel[np.ix_(rows, rows)]
        )
        reference = separate.transform(kernel[np.ix_(held, rows)])
        np.testing.assert_allclose(
            np.abs(projected[:, :count]), np.abs(reference), rtol=1e-6, atol=1e-8
        )
