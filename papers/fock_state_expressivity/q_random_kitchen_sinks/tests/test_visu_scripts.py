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

    config_snapshot = {"sweep": {"r_values": [1], "gamma_values": [1]}}
    _write_json(run_dir / "config_snapshot.json", config_snapshot)

    metrics = [
        {"method": "quantum", "r": 1, "gamma": 1, "repeat": 0, "accuracy": 0.8},
        {"method": "classical", "r": 1, "gamma": 1, "repeat": 0, "accuracy": 0.7},
    ]
    _write_json(run_dir / "metrics.json", metrics)

    visualization_dir = run_dir / "visualization_data"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    dataset_payload = {
        "x_train": [[0.0, 0.0], [1.0, 1.0]],
        "x_test": [[0.5, 0.5]],
        "y_train": [0, 1],
        "y_test": [1],
    }
    _write_json(visualization_dir / "dataset.json", dataset_payload)

    decision_payload = {
        "method": "quantum",
        "r_values": [1],
        "gamma_values": [1],
        "grid_x": [[0.0, 1.0], [0.0, 1.0]],
        "grid_y": [[0.0, 0.0], [1.0, 1.0]],
        "entries": [
            {"r": 1, "gamma": 1, "accuracy": 0.8, "class_map": [[0, 1], [0, 1]]}
        ],
        "dataset": dataset_payload,
    }
    _write_json(visualization_dir / "combined_decisions_quantum.json", decision_payload)
    decision_payload["method"] = "classical"
    _write_json(
        visualization_dir / "combined_decisions_classical.json", decision_payload
    )

    kernel_payload = {
        "kernel": [[1.0, 0.2], [0.2, 1.0]],
        "y_train": [0, 1],
        "title": "Kernel heatmap",
    }
    _write_json(visualization_dir / "kernel_quantum.json", kernel_payload)
    _write_json(visualization_dir / "kernel_classical.json", kernel_payload)

    return run_dir


def test_visu_scripts_with_previous_run(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)

    scripts = [
        "visu_dataset.py",
        "visu_accuracies.py",
        "visu_decisions.py",
        "visu_kernel_heatmap.py",
    ]
    for script in scripts:
        script_path = (
            REPO_ROOT
            / "papers/fock_state_expressivity/q_random_kitchen_sinks/utils"
            / script
        )
        result = subprocess.run(
            [sys.executable, str(script_path), "--previous-run", str(run_dir)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Saved" in result.stdout

    figures_dir = run_dir / "figures"
    assert (figures_dir / "moons_dataset.png").exists()
    assert (figures_dir / "accuracy_bar.png").exists()
    assert (figures_dir / "decision_boundary_quantum.png").exists()
    assert (figures_dir / "decision_boundary_classical.png").exists()
    assert (figures_dir / "kernel_quantum.png").exists()
    assert (figures_dir / "kernel_classical.png").exists()
