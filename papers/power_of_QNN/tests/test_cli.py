from __future__ import annotations

import json
from copy import deepcopy

import pytest
from common import (
    PROJECT_DIR,
    REPO_ROOT,
    build_project_cli_parser,
    load_runtime_ready_config,
)

from runtime_lib.config import deep_update, load_config
from runtime_lib.dtypes import resolve_config_dtypes

DATA_ROOT = REPO_ROOT / "data"


def _overlay(name: str, **changes) -> dict:
    cfg = deep_update(
        deepcopy(load_config(PROJECT_DIR / "configs" / "defaults.json")),
        load_config(PROJECT_DIR / "configs" / name),
    )
    cfg = deep_update(cfg, changes)
    cfg["data_root"] = str(DATA_ROOT)
    return resolve_config_dtypes(cfg)


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_every_config_overlays_defaults():
    for path in sorted((PROJECT_DIR / "configs").glob("*.json")):
        cfg = deep_update(
            deepcopy(load_config(PROJECT_DIR / "configs" / "defaults.json")),
            load_config(path),
        )
        assert cfg["experiment"]["kind"] in {
            "reference_check",
            "fisher_effective_dimension",
            "sensitivity",
            "topology_search",
            "training",
            "generalisation",
            "fair_comparison",
        }, path.name


def test_defaults_smoke_run_writes_contract_artifacts(tmp_path):
    from power_of_QNN.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert set(metrics["models"]) == {"qnn", "easy_qnn", "classical_4442"}
    assert all(entry["d"] == 40 for entry in metrics["models"].values())
    for name in ("fisher_qnn.npz", "fig_spectra.png", "fig_effective_dimension.png"):
        assert (run_dir / name).exists(), name


@pytest.mark.skipif(
    not (DATA_ROOT / "power_of_QNN" / "iris").exists(),
    reason="Iris snapshot absent (see LOG.md Data Acquisition Log)",
)
def test_tiny_training_run(tmp_path):
    from power_of_QNN.lib import runner as paper_runner

    cfg = _overlay(
        "training_iris_original.json",
        training={"iterations": 2, "seeds": {"start": 0, "count": 2}},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for entry in metrics["models"].values():
        assert len(entry["loss_curve_mean"]) == 3
        assert entry["fisher_rao_norm_mean"] > 0
