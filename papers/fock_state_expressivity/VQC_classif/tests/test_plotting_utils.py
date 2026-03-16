from __future__ import annotations

from pathlib import Path

from VQC_classif.utils.plotting import (  # noqa: E402
    plot_decision_boundary_from_payload,
    plot_training_metrics,
)


def test_plot_training_metrics_saves(tmp_path: Path) -> None:
    results = {
        "linear": {
            "runs": [
                {
                    "losses": [1.0, 0.9],
                    "train_accuracies": [0.6, 0.7],
                    "test_accuracies": [0.55, 0.65],
                }
            ]
        }
    }
    output_path = tmp_path / "training_metrics.png"
    plot_training_metrics(results, output_path)
    assert output_path.exists()


def test_plot_decision_boundary_from_payload_saves(tmp_path: Path) -> None:
    payload = {
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
    output_path = tmp_path / "decision_boundary.png"
    plot_decision_boundary_from_payload(payload, output_path)
    assert output_path.exists()
