from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = None
for parent in Path(__file__).resolve().parents:
    if parent.name == "q_random_kitchen_sinks":
        PROJECT_ROOT = parent
        break
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate q_random_kitchen_sinks project root.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from utils.visualization import (  # noqa: E402
    plot_combined_decisions_from_payload,
    plot_dataset_from_payload,
    save_kernel_heatmap_from_payload,
)


def test_plot_dataset_from_payload_saves(tmp_path: Path) -> None:
    payload = {
        "x_train": [[0.0, 0.0], [1.0, 1.0]],
        "x_test": [[0.5, 0.5]],
        "y_train": [0, 1],
        "y_test": [1],
    }
    output_path = tmp_path / "dataset.png"
    plot_dataset_from_payload(payload, output_path)
    assert output_path.exists()


def test_save_kernel_heatmap_from_payload_saves(tmp_path: Path) -> None:
    payload = {
        "kernel": [[1.0, 0.2], [0.2, 1.0]],
        "y_train": [0, 1],
        "title": "Kernel heatmap",
    }
    output_path = tmp_path / "kernel.png"
    save_kernel_heatmap_from_payload(payload, output_path)
    assert output_path.exists()


def test_plot_combined_decisions_from_payload_saves(tmp_path: Path) -> None:
    payload = {
        "method": "quantum",
        "r_values": [1],
        "gamma_values": [1],
        "grid_x": [[0.0, 1.0], [0.0, 1.0]],
        "grid_y": [[0.0, 0.0], [1.0, 1.0]],
        "entries": [
            {"r": 1, "gamma": 1, "accuracy": 0.9, "class_map": [[0, 1], [0, 1]]}
        ],
        "dataset": {
            "x_train": [[0.0, 0.0], [1.0, 1.0]],
            "x_test": [[0.5, 0.5]],
            "y_train": [0, 1],
            "y_test": [1],
        },
    }
    output_path = tmp_path / "combined.png"
    plot_combined_decisions_from_payload(payload, output_path)
    assert output_path.exists()
