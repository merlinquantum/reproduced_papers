from __future__ import annotations

import copy
import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config
from qelm_concentration.lib import runner as paper_runner
from qelm_concentration.lib.data import landsat_directory

from runtime_lib.config import deep_update, load_config
from runtime_lib.dtypes import resolve_config_dtypes

CONFIG_DIR = PROJECT_DIR / "configs"


def _config(*overlays: str | dict) -> dict:
    cfg = copy.deepcopy(load_runtime_ready_config())
    for overlay in overlays:
        data = (
            load_config(CONFIG_DIR / overlay) if isinstance(overlay, str) else overlay
        )
        cfg = deep_update(cfg, copy.deepcopy(data))
    return resolve_config_dtypes(cfg)


def _assert_completed(run_dir):
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED", status["error"]
    log = (run_dir / "run.log").read_text()
    # Logging contract: exactly one of each lifecycle event, whatever the seed count.
    for event in ("RUN_STARTED", "DATASET_READY", "METRICS_WRITTEN", "RUN_COMPLETED"):
        assert log.count(f"{event} |") == 1, event
    assert log.count("TRAINING_STARTED |") <= 1
    for name in ("run.log", "config_snapshot.json", "metrics.json"):
        assert (run_dir / name).exists(), name
    return json.loads((run_dir / "metrics.json").read_text())


def _skip_without_landsat(cfg):
    if not (landsat_directory(cfg) / "sat.trn").exists():
        pytest.skip("Landsat data not downloaded (see README Data section)")


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "name",
    sorted(
        path.name for path in CONFIG_DIR.glob("*.json") if path.name != "defaults.json"
    ),
)
def test_named_configs_resolve(name):
    cfg = _config(name)
    assert cfg["task"] in paper_runner.TASKS
    assert cfg[cfg["task"]], f"{name} has no section for task {cfg['task']}"


def test_landsat_readouts_smoke(tmp_path):
    cfg = _config(
        "smoke.json",
        {
            "dataset": {"landsat": {"n_subsample": 120}},
            "landsat_readouts": {"split_seeds": [0, 1]},
        },
    )
    _skip_without_landsat(cfg)
    metrics = _assert_completed_run(cfg, tmp_path)
    summary = metrics["summary"]["readouts"]
    assert "readout=w2_etcut_scaled" in summary
    assert (
        summary["readout=w1_et_scaled"]["n_features_mean"]
        == 3 * cfg["reservoir"]["n_qubits"]
    )


def test_landsat_classical_smoke(tmp_path):
    cfg = _config(
        "landsat_classical.json",
        {
            "dataset": {"landsat": {"n_subsample": 150}},
            "landsat_classical": {
                "split_seeds": [0, 1],
                "paper_baselines": ["ridge", "knn", "random_forest"],
                "n_inits": 2,
                "matched_elm": {
                    "n_features": [24],
                    "families": ["tanh", "fourier"],
                    "feature_noise_std": [0.0],
                    "n_inits": 1,
                },
                "tuned_baselines": {"grids": {"knn": {"n_neighbors": [3, 7]}}},
            },
        },
    )
    _skip_without_landsat(cfg)
    metrics = _assert_completed_run(cfg, tmp_path)
    assert set(metrics["splits"]["0"]["paper_baselines"]) == {
        "ridge",
        "knn",
        "random_forest",
    }


def test_narma_scaling_smoke(tmp_path):
    cfg = _config(
        "narma_scaling_reduced_n8.json",
        {
            "dtype": "float32",
            "dataset": {
                "narma": {
                    "orders": [1, 2, 5],
                    "t_init": 20,
                    "t_train": 60,
                    "t_test": 40,
                }
            },
            "narma_scaling": {
                "sequence_seeds": [0, 1],
                "reservoir_seeds": [0],
                "esn": {"n_units": 20, "spectral_radius": [0.9], "input_scale": [0.5]},
            },
        },
    )
    metrics = _assert_completed_run(cfg, tmp_path)
    assert "qelm|shadow_1e5_shots" in metrics["summary"]
    assert "esn|none" in metrics["summary"]


def test_pareto_search_smoke(tmp_path):
    cfg = _config("pareto_search_smoke.json", {"pareto_search": {"n_trials": 3}})
    metrics = _assert_completed_run(cfg, tmp_path)
    assert (tmp_path / "run" / "trials.csv").exists()
    assert metrics["analysis"]["all_objectives"]["reference_rank"] >= 0


def test_concentration_smoke(tmp_path):
    cfg = _config(
        "concentration.json",
        {
            "dtype": "float32",
            "concentration": {
                "n_qubits": [4, 6],
                "n_layers": [4],
                "reservoir_seeds": [0],
                "n_samples": 16,
            },
        },
    )
    metrics = _assert_completed_run(cfg, tmp_path)
    assert len(metrics["records"]) == 2 * 2


def _assert_completed_run(cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)
    return _assert_completed(run_dir)


def test_landsat_readouts_reuse_saved_probabilities(tmp_path):
    first = _config(
        "smoke.json",
        {
            "dataset": {"landsat": {"n_subsample": 120}},
            "landsat_readouts": {"save_probabilities": True},
        },
    )
    _skip_without_landsat(first)
    source = tmp_path / "source"
    source.mkdir()
    simulated = paper_runner.train_and_evaluate(first, source)

    reused = _config(
        "smoke.json",
        {
            "dataset": {"landsat": {"n_subsample": 120}},
            "landsat_readouts": {
                "realisations": [[0, 0]],
                "probability_sources": [str(source)],
            },
        },
    )
    target = tmp_path / "run"
    target.mkdir()
    metrics = paper_runner.train_and_evaluate(reused, target)
    _assert_completed(target)
    assert "ARTIFACT_LOADED" in (target / "run.log").read_text()
    assert metrics["summary"] == simulated["summary"]

    mismatched = copy.deepcopy(reused)
    mismatched["reservoir"]["n_layers"] += 4
    other = tmp_path / "mismatch"
    other.mkdir()
    with pytest.raises(ValueError, match="different configuration"):
        paper_runner.train_and_evaluate(mismatched, other)
