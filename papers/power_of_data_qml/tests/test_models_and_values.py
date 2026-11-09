"""Selection protocols, summary aggregation, and Source Data loading."""

from __future__ import annotations

import math
from pathlib import Path

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
from lib import models, paper_values, summary
from lib.kernels import linear_kernel

DATA_ROOT = Path(common.REPO_ROOT) / "data"


def _linear_task(kind: str):
    rng = np.random.default_rng(0)
    features = rng.standard_normal((80, 3))
    target = features @ np.array([1.0, -2.0, 0.5])
    labels = np.where(target > 0, 1, -1) if kind == "classification" else target
    inputs = {"features": features, "linear": linear_kernel(features)}
    positions = np.arange(80)
    return models.Task(kind, labels, positions[:60], positions[60:], inputs)


@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("protocol", ["paper", "cv"])
def test_protocols_return_every_family(kind, protocol):
    task = _linear_task(kind)
    candidates = models.kernel_family_candidates(
        "linear_kernel", kind, [("linear", {})], grid_name="smoke", svr_epsilon=0.1
    ) + models.feature_family_candidates(
        kind, grid_name="smoke", random_state=0, mlp_max_iter=500
    )
    results = models.run_protocol(
        task, candidates, protocol=protocol, n_jobs=1, cv_folds=3, random_state=0
    )
    assert set(results) == {
        "linear_kernel",
        "neural_network",
        "random_forest",
        "gradient_boosting",
        "adaboost",
    }
    best = models.best_of(kind, results, models.CLASSICAL_FAMILIES)
    if kind == "classification":
        assert results["linear_kernel"]["test_score"] > 0.9
        assert best["test_score"] == max(r["test_score"] for r in results.values())
    else:
        assert results["linear_kernel"]["test_score"] < 0.2
        assert best["test_score"] == min(r["test_score"] for r in results.values())


def test_paper_protocol_reports_best_test_score():
    task = _linear_task("classification")
    candidates = models.kernel_family_candidates(
        "linear_kernel",
        "classification",
        [("linear", {})],
        grid_name="paper",
        svr_epsilon=0.1,
    )
    results = models.run_protocol(
        task, candidates, protocol="paper", n_jobs=1, cv_folds=3, random_state=0
    )
    individual = [
        models.score(
            "classification",
            task.labels[task.test_indices],
            models._fit_predict(
                c,
                task.inputs[c.input_key],
                task.labels,
                task.train_indices,
                task.test_indices,
            ),
        )
        for c in candidates
    ]
    assert results["linear_kernel"]["test_score"] == max(individual)


def test_paper_grid_sizes_match_appendix_j():
    grid = models.PAPER_GRIDS
    assert len(grid["C"]) == 18 and grid["C"][0] == 0.006 and grid["C"][-1] == 1024.0
    assert grid["svm_gamma_scales"] == (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0)
    assert grid["mlp_hidden"] == (10, 25, 50, 75, 100, 125, 150, 200)
    regression = models.kernel_family_candidates(
        "gaussian_kernel",
        "regression",
        [(f"rbf_{s}", {}) for s in grid["svm_gamma_scales"]],
        grid_name="paper",
        svr_epsilon=0.1,
    )
    assert len(regression) == 8 * 18 * 2  # SVR and KernelRidge per (gamma, C)


def test_summary_aggregates_paired_differences():
    records = []
    for seed, (classical_acc, pqk_acc) in enumerate(
        [(0.6, 0.8), (0.7, 0.85), (0.65, 0.9)]
    ):
        task_record = {
            "train_size": 600,
            "test_size": 200,
            "metric": "accuracy",
            "protocols": {
                "paper": {
                    "families": {
                        "gaussian_kernel": {"test_score": classical_acc, "params": {}},
                        "projected_quantum_kernel": {
                            "test_score": pqk_acc,
                            "params": {},
                        },
                    },
                    "best_classical": {
                        "family": "gaussian_kernel",
                        "test_score": classical_acc,
                    },
                }
            },
        }
        records.append(
            {
                "seed": seed,
                "n": 5,
                "embedding": "E3",
                "engineered_pqk": {
                    "construction": {},
                    "by_train_size": {"600": task_record},
                },
            }
        )
    rows = [row for record in records for row in summary.record_rows(record)]
    aggregated = {r["quantity"]: r for r in summary.aggregate_rows(rows, {})}
    difference = aggregated["projected_quantum_kernel_minus_best_classical"]
    assert difference["mean"] == pytest.approx((0.2 + 0.15 + 0.25) / 3)
    assert difference["count"] == 3 and math.isfinite(difference["ci95_half_width"])


@pytest.mark.skipif(
    not paper_values.source_data_dir(DATA_ROOT).is_dir(),
    reason="Source Data CSVs not cached",
)
def test_source_data_values_load_with_correct_columns():
    values = paper_values.load_paper_values(DATA_ROOT)
    # Spot checks against the CSVs recorded in LOG.md (Experiment Inventory).
    assert values[
        ("engineered_pqk", "projected_quantum_kernel", "E3", 10, 600)
    ] == pytest.approx(0.8549, abs=1e-4)
    assert values[("engineered_pqk", "best_classical", "E3", 10, 600)] == pytest.approx(
        0.6660, abs=1e-4
    )
    # Third geometric column is Q (E3) despite its "Q (E2)" header (AMB-12).
    assert values[
        ("dimension_geometry", "g_quantum_kernel", "E3", 28, None)
    ] == pytest.approx(0.9449, abs=1e-4)
    assert values[
        ("dimension_geometry", "d_quantum_kernel", "E2", 3, None)
    ] == pytest.approx(1.9235, abs=1e-4)
    assert values[
        ("classical_dataset", "best_classical", None, 28, 600)
    ] == pytest.approx(0.0547, abs=1e-4)
    assert (
        "engineered_qk",
        "best_classical",
        "E1",
        29,
        600,
    ) not in values  # spurious rows excluded
    assert (
        "engineered_pqk",
        "quantum_kernel",
        "E3",
        30,
        600,
    ) not in values  # QK stops at n=28
    training = paper_values.load_training_size_values(DATA_ROOT)
    # n = 10-12 have seeds 0-9; the unplotted n = 13 has only seeds 0-7.
    assert len(training) == 3 * 10 * 5 + 8 * 5
    assert {row["seed"] for row in training if row["n"] == 13} == set(range(8))
