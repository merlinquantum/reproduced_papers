"""Phase 5 B6 tuned-RBM baseline: equivalence, matching, protocol and runner wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from photonic_MMD_generative.lib import rbm_classical

TUNING = {
    "sigmas": [1.0, 2.0],
    "hidden_units": [4, 8],
    "learning_rates": [0.01],
    "gibbs_steps": [2, 5],
    "batch_size": 32,
    "n_iter": 3,
    "validation_size": 50,
    "split_seed": 20261015,
    "num_samples": 100,
}


def _binary_data(num=200, m=8, n=2, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((num, m), dtype=np.int64)
    X[np.arange(num)[:, None], np.argsort(rng.random((num, m)), axis=1)[:, :n]] = 1
    return X


def test_iso_parameter_point_matches_qcbm():
    # Haar QCBM at m = 256 has 2 m^2 real parameters; 512 hidden units is the
    # closest RBM (declared in the Phase 5 plan).
    assert rbm_classical.num_parameters(256, 512) == 131_840
    assert abs(rbm_classical.num_parameters(256, 512) - 2 * 256**2) < 0.01 * 2 * 256**2
    assert rbm_classical.num_parameters(256, 64) == 16_704


def test_epoch_loop_equals_sklearn_fit():
    from qml_benchmarks.models.energy_based_model import RestrictedBoltzmannMachine

    X = _binary_data().astype(float)
    cfg = {"batch_size": 32, "n_iter": 4}
    logged, _ = rbm_classical.train_rbm_logged(X, 6, 0.05, cfg, seed=3, tag="t")
    reference = RestrictedBoltzmannMachine(
        n_components=6, learning_rate=0.05, batch_size=32, n_iter=4, random_state=3
    )
    reference.fit(X)
    np.testing.assert_array_equal(logged.components_, reference.components_)
    np.testing.assert_array_equal(
        logged.intercept_visible_, reference.intercept_visible_
    )
    np.testing.assert_array_equal(logged.intercept_hidden_, reference.intercept_hidden_)


def test_rejects_count_data():
    X = _binary_data()
    X[0, 0] = 2
    with pytest.raises(ValueError, match="binarized"):
        rbm_classical.train_rbm_logged(X, 4, 0.01, TUNING, seed=0, tag="t")


def test_validation_split_is_fixed_and_disjoint():
    X = np.arange(100)[:, None]
    fit_a, val_a = rbm_classical.validation_split(X, 20, 7)
    fit_b, val_b = rbm_classical.validation_split(X, 20, 7)
    np.testing.assert_array_equal(val_a, val_b)
    assert len(fit_a) == 80 and len(val_a) == 20
    assert not set(fit_a.ravel()) & set(val_a.ravel())
    with pytest.raises(ValueError):
        rbm_classical.validation_split(X, 100, 7)


def test_gibbs_chain_reads_are_prefixes_of_one_chain():
    X = _binary_data().astype(float)
    rbm, _ = rbm_classical.train_rbm_logged(X, 4, 0.01, TUNING, seed=1, tag="t")
    chains = rbm_classical.gibbs_chain_samples(rbm, 50, [2, 5])
    assert set(chains) == {2, 5}
    assert chains[2].shape == (50, 8) and set(np.unique(chains[5])) <= {0, 1}


def test_tune_and_refit_protocol_and_determinism():
    X = _binary_data()
    X[:10, 0] = 2  # counts are binarized before training
    record_a, samples_a = rbm_classical.tune_and_refit(X, TUNING, [1.0, 2.0], seed=4)
    record_b, samples_b = rbm_classical.tune_and_refit(X, TUNING, [1.0, 2.0], seed=4)
    assert record_a["fit_size"] == 150 and record_a["validation_size"] == 50
    assert len(record_a["grid"]) == 2  # hidden {4, 8} x lr {0.01}
    for sigma, selected in record_a["selected"].items():
        # The selected candidate is the validation minimum over the declared grid.
        scores = [
            entry["validation_gaussian_mmd_binarized"][str(steps)][sigma]
            for entry in record_a["grid"]
            for steps in TUNING["gibbs_steps"]
        ]
        assert selected["validation_mmd"] == min(scores)
        assert selected["gibbs_steps"] in TUNING["gibbs_steps"]
    assert set(samples_a) == {"sigma1.0", "sigma2.0"}
    for key in samples_a:
        np.testing.assert_array_equal(samples_a[key], samples_b[key])
        assert samples_a[key].shape == (TUNING["num_samples"], 8)


def test_runner_rbm_tuned_writes_evidence(tmp_path):
    from photonic_MMD_generative.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(tmp_path / "data")
    cfg["model"]["backend"] = "rbm_tuned"
    cfg["tuning"] = dict(TUNING, validation_size=100)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["backend"] == "rbm_tuned"
    assert set(metrics["sample_metrics_by_sigma"]) == {"1.0", "2.0"}
    for entry in metrics["sample_metrics_by_sigma"].values():
        assert {"walsh_mmd", "gaussian_mmd_binarized", "gaussian_mmd_raw"} <= set(
            entry["rbm_tuned"]
        )
    assert (run_dir / "samples.npz").exists()


def test_fig3a_b6_config_matches_frozen_protocol():
    cfg = json.loads(
        (
            Path(PROJECT_DIR) / "configs" / "fig3a_bs_rbm_tuned_classical.json"
        ).read_text()
    )
    fig3a = json.loads(
        (Path(PROJECT_DIR) / "configs" / "fig3a_bs_original.json").read_text()
    )
    assert cfg["dataset"] == fig3a["dataset"]
    assert cfg["evaluation"]["n_operators"] == fig3a["evaluation"]["n_operators"]
    assert cfg["tuning"]["num_samples"] == fig3a["evaluation"]["num_model_samples"]
    assert cfg["tuning"]["hidden_units"] == [64, 256, 512]
    assert cfg["tuning"]["learning_rates"] == [0.01, 0.001]
    assert cfg["tuning"]["gibbs_steps"] == [10, 100, 1000]
    assert cfg["tuning"]["sigmas"] == [2.0, 3.0, 4.5]
    assert cfg["tuning"]["n_iter"] == 2000 and cfg["tuning"]["batch_size"] == 64
