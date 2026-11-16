from __future__ import annotations

import json

import common  # noqa: F401
import numpy as np
import pytest
from common import PROJECT_DIR
from distributed_secure_kernels.lib.evaluation import corrected_cv
from distributed_secure_kernels.lib.runner import kernel_fn_for, kernel_grid
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

PLANNED_GAMMA = [0.001, 0.01, 0.1, 1.0, 10.0]
PLANNED_C = [0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0]


@pytest.mark.parametrize("dataset", ["wine", "parkinsons", "framingham"])
def test_fair_classical_configs_carry_the_phase1_grids(dataset):
    cfg = json.loads(
        (
            PROJECT_DIR / "configs" / f"accuracy_{dataset}_fair_classical.json"
        ).read_text()
    )
    rbf, poly = cfg["methods"]
    assert rbf["kernel"] == "rbf" and rbf["gamma_multipliers"] == PLANNED_GAMMA
    assert (
        poly["kernel"] == "poly"
        and poly["degrees"] == [2, 3]
        and poly["coef0s"] == [0.0, 1.0]
    )
    expected_c = (
        [c for c in PLANNED_C if c <= 10.0] if dataset == "framingham" else PLANNED_C
    )
    assert cfg["evaluation"]["C_grid"] == expected_c
    assert cfg["dataset"]["protocol"] == "corrected"


def test_rbf_and_poly_kernels_use_the_training_scale():
    rng = np.random.default_rng(0)
    left, right = rng.normal(size=(4, 3)), rng.normal(size=(6, 3))
    gamma = 0.1 / (3 * right.var())
    rbf = kernel_fn_for({"kernel": "rbf"}, {"gamma_multiplier": 0.1}, rng)(
        left, right, False
    )
    np.testing.assert_allclose(rbf, rbf_kernel(left, right, gamma=gamma))
    poly = kernel_fn_for({"kernel": "poly"}, {"degree": 3, "coef0": 1.0}, rng)(
        left, right, False
    )
    np.testing.assert_allclose(
        poly,
        polynomial_kernel(
            left, right, degree=3, gamma=1 / (3 * right.var()), coef0=1.0
        ),
    )


def test_kernel_grid_expands_the_configured_points():
    assert len(kernel_grid({"kernel": "rbf", "gamma_multipliers": PLANNED_GAMMA})) == 5
    assert (
        len(kernel_grid({"kernel": "poly", "degrees": [2, 3], "coef0s": [0.0, 1.0]}))
        == 4
    )
    assert kernel_grid({"kernel": "squared_overlap"}) == [{}]


def test_tuned_rbf_selects_inside_the_grid_and_is_deterministic():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(80, 4))
    labels = (features[:, 0] + 0.5 * rng.normal(size=80) > 0).astype(int)
    method = {"kernel": "rbf", "gamma_multipliers": PLANNED_GAMMA}
    evaluation = {
        "n_splits": 5,
        "cv_random_state": 0,
        "inner_splits": 3,
        "C_grid": PLANNED_C,
        "kernel_pca": False,
    }

    def run(seed):
        return corrected_cv(
            features,
            labels,
            kernel_grid(method),
            lambda point, generator: kernel_fn_for(method, point, generator),
            evaluation,
            None,
            True,
            np.random.default_rng(seed),
            "rbf",
        )

    first, second = run(0), run(0)
    assert first == second
    for fold in first["kernel_svm"]["folds"]:
        assert fold["selected"]["gamma_multiplier"] in PLANNED_GAMMA
        assert fold["selected"]["C"] in PLANNED_C
    assert first["kernel_svm"]["accuracy"]["mean"] > 0.7
