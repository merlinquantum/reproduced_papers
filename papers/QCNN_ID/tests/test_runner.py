"""Smoke tests for the QCNN-ID shared-runtime runner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib import runner  # noqa: E402
from lib.data import PreparedData  # noqa: E402
from lib.runner import (  # noqa: E402
    _aggregate_histories,
    _aggregate_seeds,
    _learning_rate_for,
)
from lib.training import EpochMetrics, TrainHistory  # noqa: E402


def test_runner_creates_run_dir_and_uses_feature_view_dim(monkeypatch, tmp_path):
    """The runner builds CNNClassifier from the feature view it actually trains on."""

    rng = np.random.default_rng(0)
    prepared = PreparedData(
        X_train_quantum=rng.random((8, 3), dtype=np.float32),
        X_train_classical=rng.normal(size=(8, 3)).astype(np.float32),
        X_test_quantum=rng.random((4, 3), dtype=np.float32),
        X_test_classical=rng.normal(size=(4, 3)).astype(np.float32),
        y_train=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        y_test=np.array([0, 1, 0, 1], dtype=np.int64),
        n_features_full=50,
        n_components=3,
        class_balance={0: 0.5, 1: 0.5},
        cumulated_var=[0.5, 0.75, 0.9],
        n_train=8,
        n_test=4,
    )
    monkeypatch.setattr(runner, "load_and_prepare", lambda cfg, seed: prepared)

    cfg = {
        "models": ["cnn_classifier"],
        "seeds": [42],
        "epochs": 1,
        "batch_size": 4,
        "lr_cnn": 0.001,
        "lr_qcnn": 0.005,
        "n_qubits": 3,
        "n_components": 3,
        "ansatz_reps": 1,
        "test_size": 0.25,
        "subset_size": 12,
        "device": "cpu",
    }

    run_dir = tmp_path / "fresh_run_dir"
    payload = runner.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "run.log").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "train_predictions.csv").exists()
    assert (run_dir / "test_predictions.csv").exists()
    expected_params = 3 * 128 + 128 + 128 * 64 + 64 + 64 * 1 + 1
    assert payload["summary"]["cnn_classifier"]["param_count"] == expected_params


def _history(name: str, accuracy: float, cm: list[list[int]]) -> TrainHistory:
    return TrainHistory(
        name=name,
        epochs=[
            EpochMetrics(
                loss=1.0 - accuracy,
                accuracy=accuracy,
                precision=accuracy - 0.1,
                recall=accuracy - 0.2,
                time_s=2.0,
            )
        ],
        final_accuracy=accuracy,
        final_precision=accuracy - 0.1,
        final_recall=accuracy - 0.2,
        confusion_matrix=cm,
        param_count=3,
        train_time_s=5.0,
    )


def test_multiseed_summary_sums_confusion_matrices():
    per_seed = {
        "toy": [
            _history("toy", 0.8, [[8, 1], [2, 7]]),
            _history("toy", 0.6, [[6, 3], [4, 5]]),
        ]
    }

    summary = _aggregate_seeds(per_seed)["toy"]

    assert summary["seeds"] == 2
    assert summary["accuracy_mean"] == 0.7
    assert summary["confusion_matrix"] == [[14, 4], [6, 12]]


def test_multiseed_plot_history_uses_mean_curve_and_summed_confusion():
    per_seed = {
        "toy": [
            _history("toy", 0.8, [[8, 1], [2, 7]]),
            _history("toy", 0.6, [[6, 3], [4, 5]]),
        ]
    }

    history = _aggregate_histories(per_seed)[0]

    assert history.name == "toy"
    assert history.epochs[0].accuracy == 0.7
    assert history.confusion_matrix == [[14, 4], [6, 12]]


def test_photonic_learning_rate_can_be_configured_separately():
    cfg = {"lr_cnn": 0.001, "lr_qcnn": 0.01, "lr_photonic": 0.05}

    assert _learning_rate_for("cnn_classifier", cfg) == 0.001
    assert _learning_rate_for("qcnn_classifier", cfg) == 0.01
    assert _learning_rate_for("photonic_classifier", cfg) == 0.05


def test_photonic_learning_rate_falls_back_to_qcnn_rate():
    cfg = {"lr_cnn": 0.001, "lr_qcnn": 0.01}

    assert _learning_rate_for("photonic_classifier", cfg) == 0.01
