"""Smoke tests for the qSVM shared-runtime runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib import runner  # noqa: E402
from lib.data import PreparedData  # noqa: E402
from lib.models import MODEL_REGISTRY  # noqa: E402

CURATED_ALL_MODELS = {
    "svm_classical",
    "knn_classical",
    "state_svm_angle",
    "state_svm_amplitude",
    "state_svm_zz",
    "state_knn_angle",
    "state_knn_amplitude",
    "state_knn_zz",
    "hybrid_svm_angle",
    "hybrid_svm_amplitude",
    "hybrid_svm_zz",
    "hybrid_knn_angle",
    "hybrid_knn_amplitude",
    "hybrid_knn_zz",
    "photonic_hybrid_svm_angle",
    "photonic_hybrid_svm_amplitude",
    "photonic_state_svm_angle",
    "photonic_hybrid_knn_angle",
    "photonic_hybrid_knn_amplitude",
    "photonic_state_knn_angle",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_default_config_is_file_free_smoke_covering_registered_models():
    config_dir = PAPER_ROOT / "configs"
    cfg = _read_json(config_dir / "defaults.json")

    assert cfg["data_source"] == "synthetic"
    assert "dataset_file" not in cfg
    assert "ids_archive" not in cfg
    assert set(cfg["models"]) == set(MODEL_REGISTRY)


def test_all_models_url2016_config_covers_registered_models_on_real_data():
    config_dir = PAPER_ROOT / "configs"
    cfg = _read_json(config_dir / "all_models_url2016.json")

    assert cfg["data_source"] == "csv"
    assert cfg["dataset"] == "url2016"
    assert set(cfg["models"]) == CURATED_ALL_MODELS


def test_state_fidelity_configs_include_experimental_models_only_where_requested():
    config_dir = PAPER_ROOT / "configs"
    url_cfg = _read_json(config_dir / "kernel_vs_state_fidelity_url2016.json")
    ids_cfg = _read_json(config_dir / "kernel_vs_state_fidelity_ids2012.json")

    expected = {
        "qsvm_angle",
        "qsvm_amplitude",
        "qsvm_zz",
        "state_svm_angle",
        "state_svm_amplitude",
        "state_svm_zz",
        "qknn_angle",
        "qknn_amplitude",
        "qknn_zz",
        "state_knn_angle",
        "state_knn_amplitude",
        "state_knn_zz",
        "photonic_fidelity_svm_angle",
        "photonic_state_svm_angle",
        "photonic_fidelity_knn_angle",
        "photonic_state_knn_angle",
    }
    assert set(url_cfg["models"]) == expected
    assert set(ids_cfg["models"]) == expected


def test_fast_models_url2016_declares_all_pca_prediction_region_plots():
    cfg = _read_json(PAPER_ROOT / "configs" / "fast_models_url2016.json")

    specs = runner._decision_boundary_plot_specs(cfg)

    assert [spec["filename"] for spec in specs] == [
        "pca_prediction_regions.png",
        "pca_photonic_hybrid_svm_angle_vs_amplitude.png",
    ]
    assert specs[0]["models"] == [
        "svm_classical",
        "state_svm_angle",
        "photonic_hybrid_svm_angle",
    ]
    assert specs[1]["models"] == [
        "photonic_hybrid_svm_angle",
        "photonic_hybrid_svm_amplitude",
    ]


def _prepared() -> PreparedData:
    X_train = np.array(
        [
            [-1.0, -0.9],
            [-0.8, -0.7],
            [0.8, 0.7],
            [1.0, 0.9],
            [-0.9, -1.0],
            [0.9, 1.0],
        ],
        dtype=float,
    )
    X_test = np.array([[-0.85, -0.95], [0.85, 0.95]], dtype=float)
    return PreparedData(
        X_train_classical=X_train,
        X_test_classical=X_test,
        X_train_quantum=np.clip((X_train + 1.0) / 2.0, 0.0, 1.0),
        X_test_quantum=np.clip((X_test + 1.0) / 2.0, 0.0, 1.0),
        X_train_photonic=np.clip((X_train + 1.0) / 2.0, 0.0, 1.0),
        X_test_photonic=np.clip((X_test + 1.0) / 2.0, 0.0, 1.0),
        y_train=np.array([0, 0, 1, 1, 0, 1], dtype=np.int64),
        y_test=np.array([0, 1], dtype=np.int64),
        train_indices=np.arange(6),
        test_indices=np.arange(6, 8),
        dataset="toy",
        feature_names=["f0", "f1"],
        photonic_feature_names=["f0", "f1"],
        photonic_preprocessing={
            "n_modes": 2,
            "input_range": [0.0, 1.0],
            "source_feature_names": ["f0", "f1"],
            "padded_modes": 0,
        },
        class_counts_clean={0: 4, 1: 4},
        class_balance_clean={0: 0.5, 1: 0.5},
        class_balance={0: 0.5, 1: 0.5},
        n_rows_raw=8,
        n_rows_clean=8,
        n_features_raw=2,
        source="synthetic",
    )


def test_runner_writes_metrics_summary_and_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_and_prepare", lambda cfg, seed: _prepared())
    cfg = {
        "models": ["svm_classical", "knn_classical"],
        "seeds": [42],
        "dataset": "toy",
        "subset_size": 0,
        "feature_limit": 2,
        "svm_kernel": "linear",
        "svm_c": 1.0,
        "svm_gamma": "scale",
        "knn_neighbors": 1,
        "knn_metric": "euclidean",
        "knn_weights": "distance",
        "metric_average": "binary",
    }

    payload = runner.train_and_evaluate(cfg, tmp_path)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "train_predictions.csv").exists()
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "svm_knn_comparison.png").exists()
    assert (tmp_path / "roc_curves.csv").exists()
    assert (tmp_path / "svm_roc_curves.png").exists()
    assert (tmp_path / "knn_roc_curves.png").exists()
    assert (tmp_path / "svm_radar_chart.png").exists()
    assert (tmp_path / "knn_radar_chart.png").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["summary"]["svm_classical"]["accuracy_mean"] == 1.0
    assert saved["summary"]["svm_classical"]["roc_auc_mean"] == 1.0
    assert saved["data"][0]["n_classes"] == 2
    assert saved["data"][0]["class_counts_clean"] == {"0": 4, "1": 4}
    assert saved["data"][0]["class_counts_train"] == {"0": 3, "1": 3}
    assert saved["data"][0]["class_counts_test"] == {"0": 1, "1": 1}
    assert saved["data"][0]["class_balance_clean"] == {"0": 0.5, "1": 0.5}
    assert saved["data"][0]["class_balance_train"] == {"0": 0.5, "1": 0.5}
    assert saved["data"][0]["class_balance_test"] == {"0": 0.5, "1": 0.5}
    assert payload["summary"]["knn_classical"]["accuracy_mean"] == 1.0
