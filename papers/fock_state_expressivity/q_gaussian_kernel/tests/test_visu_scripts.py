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

    visualization_dir = run_dir / "sampler" / "visualization_data"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    learned_payload = {
        "x_on_pi": [0.0, 0.5, 1.0],
        "sigma_labels": ["sigma=1.00"],
        "sigma_values": [1.0],
        "targets": [[0.9, 0.5, 0.1]],
        "predictions": [
            {
                "sigma_label": "sigma=1.00",
                "sigma_value": 1.0,
                "photons": 2,
                "prediction": [0.85, 0.45, 0.15],
            }
        ],
    }
    _write_json(visualization_dir / "learned_functions.json", learned_payload)

    datasets_payload = {
        "circular": {
            "x_train": [[0.0, 0.0], [1.0, 1.0]],
            "y_train": [0, 1],
            "x_test": [[0.5, 0.5]],
            "y_test": [1],
        },
        "moon": {
            "x_train": [[0.0, 1.0], [1.0, 0.0]],
            "y_train": [0, 1],
            "x_test": [[0.5, 0.5]],
            "y_test": [0],
        },
        "blob": {
            "x_train": [[-1.0, -1.0], [1.0, 1.0]],
            "y_train": [0, 1],
            "x_test": [[0.0, 0.0]],
            "y_test": [1],
        },
    }
    classify_visualization = run_dir / "classify" / "visualization_data"
    classify_visualization.mkdir(parents=True, exist_ok=True)
    _write_json(
        classify_visualization / "classification_datasets.json", datasets_payload
    )

    quantum_metrics = [
        {
            "sigma_label": "sigma=1.00",
            "n_photons": 2,
            "circular_acc": 0.7,
            "moon_acc": 0.6,
            "blob_acc": 0.8,
        }
    ]
    classical_metrics = [
        {
            "sigma_label": "sigma=1.00",
            "circular_acc": 0.75,
            "moon_acc": 0.65,
            "blob_acc": 0.85,
        }
    ]
    classify_dir = run_dir / "classify"
    classify_dir.mkdir(parents=True, exist_ok=True)
    _write_json(classify_dir / "quantum_metrics.json", quantum_metrics)
    _write_json(classify_dir / "classical_metrics.json", classical_metrics)
    return run_dir


def test_visu_scripts_with_previous_run(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)

    scripts = [
        "visu_learned_functions.py",
        "visu_dataset_examples.py",
        "visu_accuracy_bars.py",
    ]
    for script in scripts:
        script_path = (
            REPO_ROOT
            / "papers/fock_state_expressivity/q_gaussian_kernel/utils"
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

    sampler_figures = run_dir / "sampler" / "figures"
    classify_figures = run_dir / "classify" / "figures"
    assert (sampler_figures / "learned_vs_target.png").exists()
    assert (classify_figures / "datasets" / "classification_datasets.png").exists()
    assert (classify_figures / "svm_accuracy.png").exists()
