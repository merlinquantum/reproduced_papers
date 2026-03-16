from __future__ import annotations

import json

import numpy as np
from lib import architecture_grid_run as arch_module
from lib import tau_alpha_grid_run as tau_module
from lib.benchmark_artifacts import (
    compute_decision_grid,
    compute_grid_spec,
    compute_probability_features,
)


class DummyModel:
    def __init__(self, dimension, num_layers, design, alpha):
        self.dimension = dimension
        self.num_layers = num_layers
        self.design = design
        self.alpha = alpha

    def fit(self, X, y, **kwargs):
        return self

    def score(self, X, y):
        return 1.0

    def predict_proba(self, X):
        score = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.column_stack([1 - score, score])

    def get_quantum_features(self, X):
        return np.column_stack([X[:, 0]])


class DummyDataset:
    def __init__(self, n_train=10, n_test=4):
        rng = np.random.default_rng(0)
        self.X_train = rng.normal(size=(n_train, 2))
        self.y_train = (self.X_train[:, 0] > 0).astype(int)
        self.X_test = rng.normal(size=(n_test, 2))
        self.y_test = (self.X_test[:, 0] > 0).astype(int)

    @property
    def train(self):
        return self.X_train, self.y_train

    @property
    def test(self):
        return self.X_test, self.y_test


def test_benchmark_artifact_helpers():
    dataset = DummyDataset(n_train=6, n_test=3)
    X_tr, _ = dataset.train
    X_te, _ = dataset.test
    grid_spec = compute_grid_spec(X_tr, X_te, resolution=7)
    assert grid_spec.resolution == 7

    model = DummyModel(dimension=2, num_layers=1, design="AA", alpha=0.1)
    diff = compute_decision_grid(model, grid_spec)
    assert diff.shape == (grid_spec.resolution, grid_spec.resolution)

    feat_tr, feat_te = compute_probability_features(model, X_tr, X_te)
    assert feat_tr.shape[0] == X_tr.shape[0]
    assert feat_te is not None
    assert feat_te.shape[0] == X_te.shape[0]


def test_run_architecture_grid_saves_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(arch_module, "MerlinReuploadingClassifier", DummyModel)
    monkeypatch.setattr(arch_module, "CirclesDataset", DummyDataset)
    monkeypatch.setattr(arch_module, "MoonsDataset", DummyDataset)

    cfg = {
        "dataset": {"train_size": 6, "test_size": 3, "batch_size": 4},
        "experiment": {
            "type": "design_benchmark",
            "dataset": "circles",
            "alpha": 0.1,
            "tau": 1.0,
            "depths": [1],
            "designs": ["AA"],
            "grid_resolution": 5,
        },
        "training": {"epochs": 1, "lr": 0.01, "patience": 1},
    }

    arch_module.run_architecture_grid(cfg, tmp_path)
    results_path = tmp_path / "design_benchmark_results.json"
    figure_path = tmp_path / "design_benchmark_figure_data.npz"
    assert results_path.exists()
    assert figure_path.exists()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(results["metrics"]) == 1
    with np.load(figure_path) as data:
        assert "diff_L1_AA" in data
        assert data["diff_L1_AA"].shape == (5, 5)


def test_run_tau_alpha_grid_saves_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_module, "MerlinReuploadingClassifier", DummyModel)
    monkeypatch.setattr(tau_module, "CirclesDataset", DummyDataset)
    monkeypatch.setattr(tau_module, "MoonsDataset", DummyDataset)

    cfg = {
        "dataset": {"train_size": 6, "test_size": 3, "batch_size": 4},
        "experiment": {
            "type": "tau_alpha_benchmark",
            "dataset": "moons",
            "depths": [1],
            "tau_values": [0.1],
            "alpha_values": [0.2],
            "grid_resolution": 5,
        },
        "training": {"epochs": 1, "lr": 0.01, "patience": 1},
    }

    tau_module.run_tau_alpha_grid(cfg, tmp_path)
    results_path = tmp_path / "tau_alpha_benchmark_results.json"
    figure_path = tmp_path / "tau_alpha_benchmark_figure_data.npz"
    assert results_path.exists()
    assert figure_path.exists()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(results["metrics"]) == 1
    key = "diff_L1_tau0.100000_alpha0.200000"
    with np.load(figure_path) as data:
        assert key in data
        assert data[key].shape == (5, 5)
