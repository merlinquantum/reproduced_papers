from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config

from runtime_lib.config import deep_update, load_config

EXPECTED_FLAGS = (
    "--task",
    "--iterations",
    "--lr",
    "--noise-model",
    "--noise-level",
    "--penalty",
    "--vqem-budget",
    "--vqc-budget",
    "--seed-data",
    "--classical-kind",
    "--vvqe-budget",
    "--half-distance",
    "--vqc-merlin-budget",
)


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_exposes_experiment_flags():
    parser, _ = build_project_cli_parser()
    option_strings = {
        flag for action in parser._actions for flag in action.option_strings
    }
    assert set(EXPECTED_FLAGS) <= option_strings


def _tiny_config(overlay_name: str | None, **overrides) -> dict:
    cfg = load_runtime_ready_config()
    if overlay_name is not None:
        cfg = deep_update(cfg, load_config(PROJECT_DIR / "configs" / overlay_name))
    cfg["data_root"] = str(PROJECT_DIR.parents[1] / "data")
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        cfg[section][key] = value
    return cfg


TINY_RUNS = {
    "vqem_defaults": (None, {}),
    "sdp": ("e0_sdp_original.json", {}),
    "vqem_budget_noisy": (
        "e3_vqem_budget_noisy_original.json",
        {"vqem__depth": 1, "training__iterations": 3},
    ),
    "vqc": ("e4_vqc_original.json", {"vqc__budget": 2.0, "training__iterations": 3}),
    "classical": ("e4_vqc_classical.json", {"training__iterations": 3}),
    "vvqe": ("e5_vvqe_reduced.json", {"vvqe__depth": 1, "training__iterations": 2}),
}


@pytest.mark.parametrize("name", list(TINY_RUNS))
def test_tiny_run_writes_contract_artifacts(name, tmp_path):
    from virtual_qnn.lib import runner

    overlay, overrides = TINY_RUNS[name]
    if name == "sdp":
        pytest.importorskip("cvxpy")
    if name == "vvqe":
        pytest.importorskip("pennylane")
    cfg = _tiny_config(overlay, **overrides)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["metrics_path"] == "metrics.json"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "config_snapshot.json").is_file()
    log = (run_dir / "run.log").read_text()
    for event in ("RUN_STARTED", "DATASET_READY", "METRICS_WRITTEN", "RUN_COMPLETED"):
        assert log.count(event) == 1, event
    assert "EVALUATION_COMPLETED" in log
    if name != "sdp":
        assert log.count("TRAINING_STARTED") == 1
        assert "TRAIN_ITERATION_COMPLETED" in log


def test_unknown_task_fails_clearly(tmp_path):
    from virtual_qnn.lib import runner

    cfg = load_runtime_ready_config()
    cfg["experiment"]["task"] = "not_a_task"
    with pytest.raises(ValueError, match="Unsupported experiment.task"):
        runner.train_and_evaluate(cfg, tmp_path)
