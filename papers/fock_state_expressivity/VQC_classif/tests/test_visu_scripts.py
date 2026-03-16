from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_20250101-000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "linear": {
            "runs": [
                {
                    "losses": [1.0, 0.9],
                    "train_accuracies": [0.6, 0.7],
                    "test_accuracies": [0.55, 0.65],
                    "final_test_acc": 0.65,
                }
            ],
            "final_test_accs": [0.65],
            "avg_final_test_acc": 0.65,
        }
    }
    _write_json(run_dir / "metrics.json", metrics)

    boundary_dir = run_dir / "decision_boundaries"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    boundary_payload = [
        {
            "dataset": "linear",
            "model_type": "vqc",
            "requested_model_type": "vqc",
            "activation": "none",
            "initial_state": [1, 1, 1],
            "best_acc": 0.9,
            "x_train": [[0.0, 0.0], [1.0, 1.0]],
            "y_train": [0, 1],
            "x_test": [[0.5, 0.5]],
            "y_test": [1],
            "grid_x": [[0.0, 1.0], [0.0, 1.0]],
            "grid_y": [[0.0, 0.0], [1.0, 1.0]],
            "class_map": [[0.0, 1.0], [0.0, 1.0]],
        }
    ]
    _write_json(boundary_dir / "boundary_data.json", boundary_payload)
    return run_dir


def test_visu_scripts_with_previous_run(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)

    training_script = (
        REPO_ROOT
        / "papers/fock_state_expressivity/VQC_classif/utils/visu_training_metrics.py"
    )
    boundary_script = (
        REPO_ROOT
        / "papers/fock_state_expressivity/VQC_classif/utils/visu_decision_boundaries.py"
    )

    training_result = subprocess.run(
        [sys.executable, str(training_script), "--previous-run", str(run_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Saved training metrics to" in training_result.stdout
    assert (run_dir / "figures" / "training_metrics.png").exists()

    boundary_result = subprocess.run(
        [sys.executable, str(boundary_script), "--previous-run", str(run_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Saved decision boundaries to" in boundary_result.stdout
    assert (run_dir / "figures" / "decision_boundaries" / "linear_vqc.png").exists()
