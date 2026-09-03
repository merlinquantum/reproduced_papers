from __future__ import annotations

import math

from common import REPO_ROOT
from lib.latency_benchmark import (
    gfm_kernel_row,
    prepare_benchmark_pools,
    qmkl_kernel_row,
    run_latency_benchmark,
)


def _smoke_cfg() -> dict:
    return {
        "seed": 42,
        "data_root": str(REPO_ROOT / "data"),
        "dataset": {
            "n_samples": 2000,
            "test_fraction": 0.1,
            "train_balance_per_class": 20,
        },
        "pca": {"n_components": 2},
        "gqc_model": {
            "n_qubits": 6,
            "backend": "gate",
            "autoencoder": {"hidden_dims": [32, 16]},
            "vqc": {"n_layers": 2},
            "head": {"hidden_dim": 4},
        },
        "benchmark": {"n_runs": 3, "n_reps": 2, "batch_size": 5},
    }


def test_prepare_benchmark_pools_shapes_and_balance():
    pools = prepare_benchmark_pools(_smoke_cfg())

    n_per_class = 20
    y_train = pools["y_train"]
    assert int((y_train == 1).sum()) == n_per_class
    assert int((y_train == 0).sum()) == n_per_class
    assert pools["X_train_full"].shape == (2 * n_per_class, 29)
    assert pools["X_train_pca"].shape == (2 * n_per_class, 2)
    assert pools["X_test_pca"].shape[1] == 2
    assert pools["X_test_full"].shape[0] == pools["y_test"].shape[0]
    # Fraud must actually be present in the test pool too (not just train).
    assert pools["y_test"].sum() > 0


def test_kernel_rows_are_valid_probabilities():
    pools = prepare_benchmark_pools(_smoke_cfg())
    x_test = pools["X_test_pca"][0]
    train = pools["X_train_pca"]

    qmkl_row = qmkl_kernel_row(x_test, train)
    gfm_row = gfm_kernel_row(x_test, train)

    assert qmkl_row.shape == (train.shape[0],)
    assert gfm_row.shape == (train.shape[0],)
    for row in (qmkl_row, gfm_row):
        assert bool(((row >= -1e-9) & (row <= 1 + 1e-9)).all())
        assert all(math.isfinite(v) for v in row)

    # A point kernel-evaluated against itself should be close to a perfect
    # overlap (k(x, x) ~= 1) for every feature map used.
    self_row_qmkl = qmkl_kernel_row(train[0], train[:1])
    self_row_gfm = gfm_kernel_row(train[0], train[:1])
    assert self_row_qmkl[0] > 0.99
    assert self_row_gfm[0] > 0.99


def test_run_latency_benchmark_end_to_end_smoke():
    cfg = _smoke_cfg()
    results = run_latency_benchmark(cfg)

    assert set(results.keys()) == {"QMKL", "GFM", "GQC"}
    for stats in results.values():
        assert stats["ms_per_sample"] > 0
        assert math.isfinite(stats["ms_per_sample"])
        assert stats["cv_pct"] >= 0
        assert math.isfinite(stats["cv_pct"])


def test_run_latency_benchmark_via_runner_dispatch(tmp_path):
    from lib.runner import train_and_evaluate

    cfg = _smoke_cfg()
    cfg["task"] = "latency_benchmark"
    payload = train_and_evaluate(cfg, tmp_path)

    assert set(payload.keys()) == {"QMKL", "GFM", "GQC"}
    assert (tmp_path / "latency_benchmark.json").exists()
