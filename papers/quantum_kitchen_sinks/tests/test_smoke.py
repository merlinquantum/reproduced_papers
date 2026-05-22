from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import load_runtime_ready_config


def test_picture_frames_dataset_shapes():
    from lib.data import load_picture_frames

    X_tr, y_tr, X_te, y_te = load_picture_frames(n_train=100, n_test=40, seed=0)
    assert X_tr.shape == (100, 2)
    assert y_tr.shape == (100,)
    assert X_te.shape == (40, 2)
    assert y_te.shape == (40,)
    assert set(np.unique(y_tr)) == {0, 1}


def test_qks_featurizer_shape():
    from lib.qks_model import QKSFeaturizer

    feat = QKSFeaturizer(
        circuit="cnot2",
        n_qubits=2,
        n_episodes=8,
        sigma=1.0,
        encoding="split",
    )
    feat.fit_episodes(input_dim=2, seed=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))
    features = feat.transform(X, seed=0)
    assert features.shape == (5, 8 * 2)
    assert features.dtype == np.float32
    assert set(np.unique(features.astype(int))) <= {0, 1}


def test_train_and_evaluate_writes_artifact(tmp_path):
    from lib.runner import train_and_evaluate

    cfg = load_runtime_ready_config()
    cfg["dataset"]["n_train"] = 60
    cfg["dataset"]["n_test"] = 20
    cfg["qks"]["n_episodes"] = 16
    cfg["data_root"] = str(tmp_path)
    run_dir = tmp_path / "run"
    train_and_evaluate(cfg, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "results" in metrics
    assert "summary" in metrics
    assert metrics["summary"]["best_test_accuracy"] >= 0.0


def test_encoding_split_and_tile():
    from lib.encoding import make_episodes

    eps_split = make_episodes(
        n_episodes=3,
        input_dim=2,
        n_gate_params=2,
        sigma=1.0,
        encoding="split",
        seed=0,
    )
    for ep in eps_split:
        nonzeros_per_row = (ep.omega != 0.0).sum(axis=1)
        assert (nonzeros_per_row == 1).all()
        assert ep.omega.shape == (2, 2)
        assert ep.beta.shape == (2,)

    eps_tile = make_episodes(
        n_episodes=3,
        input_dim=8,
        n_gate_params=4,
        sigma=1.0,
        encoding="tile",
        seed=0,
    )
    for ep in eps_tile:
        nonzeros_per_row = (ep.omega != 0.0).sum(axis=1)
        assert (nonzeros_per_row == 2).all()
        assert ep.omega.shape == (4, 8)
        assert ep.beta.shape == (4,)
