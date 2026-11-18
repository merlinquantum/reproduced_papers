from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config

VALIDATOR = (
    Path.home()
    / ".claude/plugins/cache/alps-paper-reproduction/alps-paper-reproduction/0.1.0"
    / "skills/experiment-logging/scripts/validate_logging.py"
)


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_every_config_overlays_defaults_with_known_task():
    for path in sorted((PROJECT_DIR / "configs").glob("*.json")):
        cfg = load_runtime_ready_config(
            None if path.name == "defaults.json" else path.name
        )
        assert cfg["task"] in {"toy_regression", "navier_stokes"}, path.name
        assert "<<" not in json.dumps({k: v for k, v in cfg.items() if k != "dtype"}), (
            path.name
        )


def test_toy_run_writes_contract_artifacts(tmp_path):
    from trainable_frequency_qnn.lib import runner as paper_runner

    cfg = load_runtime_ready_config("fig2b_tf_original.json")
    cfg["training"]["iterations"] = 4
    cfg["evaluation"]["every"] = 2
    cfg["evaluation"]["spectrum"] = {"half_width_pi": 8, "n_points": 512, "n_peaks": 3}
    run_dir = tmp_path / "run_test"

    metrics = paper_runner.train_and_evaluate(cfg, run_dir)

    for name in ("run.log", "run_status.json", "config_snapshot.json", "metrics.json"):
        assert (run_dir / name).exists(), name
    for name in ("losses.csv", "predictions.npz", "spectrum.npz"):
        assert (run_dir / name).exists(), name
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert len(metrics["encoding_multipliers"]) == 3
    if VALIDATOR.exists():
        result = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--run-dir",
                str(run_dir),
                "--require-training",
                "--require-evaluation",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
