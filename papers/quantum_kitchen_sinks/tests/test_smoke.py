from __future__ import annotations

import json

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


def test_photonic_dual_rail_input_state_and_shape():
    from lib.photonic_qks import PhotonicQKSFeaturizer

    feat = PhotonicQKSFeaturizer(
        n_modes=6,
        n_photons=3,
        n_episodes=4,
        sigma=0.05,
        encoding="tile",
        input_modes=[0, 2, 4],
        computation_space="DUAL_RAIL",
    )
    assert feat.input_state == [1, 0, 1, 0, 1, 0]
    feat.fit_episodes(input_dim=6, seed=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 6))
    features = feat.transform(X, seed=0)
    assert features.shape == (3, 4 * 6)
    assert features.dtype == np.float32


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


def test_dual_rail_outcome_table_matches_the_measured_basis():
    """The outcome->click-pattern table must match the computation space.

    DUAL_RAIL reports 2**(m/2) outcomes, not C(m, k). Indexing a C(m, k) table
    with dual-rail outcome indices does not raise -- it silently maps outcomes
    onto unrelated click patterns and leaves mode 0 permanently occupied, i.e.
    a constant (dead) feature that a linear classifier cannot recover from.
    """
    import merlin as ml
    from lib.photonic_qks import PhotonicQKSFeaturizer

    dual = PhotonicQKSFeaturizer(
        n_modes=6,
        n_photons=3,
        n_episodes=1,
        sigma=0.05,
        encoding="tile",
        computation_space=ml.ComputationSpace.DUAL_RAIL,
    )
    table = dual._build_outcome_table()
    assert table.shape == (8, 6), "dual rail has 2**(m/2) outcomes"
    assert (table.sum(axis=1) == 3).all(), "one photon per logical qubit"
    for pair in range(3):
        assert (table[:, 2 * pair] + table[:, 2 * pair + 1] == 1).all(), (
            "exactly one rail of each pair is occupied"
        )
    assert len({row.tobytes() for row in table}) == 8, "outcomes must be distinct"
    assert table.var(axis=0).min() > 0, "no constant (dead) feature column"

    unbunched = PhotonicQKSFeaturizer(
        n_modes=4,
        n_photons=2,
        n_episodes=1,
        sigma=0.05,
        encoding="tile",
        computation_space=ml.ComputationSpace.UNBUNCHED,
    )
    assert unbunched._build_outcome_table().shape == (6, 4), "C(4, 2) outcomes"
