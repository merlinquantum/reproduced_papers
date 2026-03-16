from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_20250101-000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "VQC_[1, 0, 0]": {
            "loss_curves": [
                {"losses": [1.0, 0.8, 0.6], "train_mses": [1.0, 0.7, 0.5]},
                {"losses": [1.1, 0.9, 0.7], "train_mses": [1.1, 0.8, 0.6]},
            ],
            "initial_state": [1, 0, 0],
        }
    }
    _write_json(run_dir / "metrics.json", metrics)

    config_snapshot = {
        "plotting": {"colors": ["#1f77b4"]},
        "training": {"initial_states": [[1, 0, 0]]},
    }
    _write_json(run_dir / "config_snapshot.json", config_snapshot)

    learned_dir = run_dir / "learned_function"
    learned_dir.mkdir(parents=True, exist_ok=True)
    predictions = {
        "x": [0.0, 0.1, 0.2],
        "target": [0.2, 0.25, 0.3],
        "entries": [
            {
                "label": "VQC_[1, 0, 0]",
                "initial_state": [1, 0, 0],
                "color": "#1f77b4",
                "predictions": [0.2, 0.24, 0.28],
            }
        ],
    }
    _write_json(learned_dir / "predictions.json", predictions)
    return run_dir


def test_visu_scripts_with_previous_run(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)

    training_script = (
        REPO_ROOT
        / "papers/fock_state_expressivity/VQC_fourier_series/utils/visu_training_curves.py"
    )
    learned_script = (
        REPO_ROOT
        / "papers/fock_state_expressivity/VQC_fourier_series/utils/visu_learned_functions.py"
    )

    training_result = subprocess.run(
        [sys.executable, str(training_script), "--previous-run", str(run_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Saved training curves to" in training_result.stdout
    assert (run_dir / "figures" / "training_curves.png").exists()

    learned_result = subprocess.run(
        [sys.executable, str(learned_script), "--previous-run", str(run_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Saved learned functions to" in learned_result.stdout
    assert (run_dir / "figures" / "learned_vs_target.png").exists()
