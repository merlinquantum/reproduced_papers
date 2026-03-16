from __future__ import annotations

from pathlib import Path

from q_gaussian_kernel.utils.plotting import (  # noqa: E402
    plot_accuracy_bars,
    plot_dataset_examples_from_payload,
    plot_gaussian_fits_from_payload,
)


def test_plot_gaussian_fits_from_payload_saves(tmp_path: Path) -> None:
    payload = {
        "x_on_pi": [0.0, 0.5, 1.0],
        "sigma_labels": ["sigma=1.00", "sigma=0.50"],
        "sigma_values": [1.0, 0.5],
        "targets": [[0.9, 0.5, 0.1], [0.8, 0.4, 0.05]],
        "predictions": [
            {
                "sigma_label": "sigma=1.00",
                "sigma_value": 1.0,
                "photons": 2,
                "prediction": [0.85, 0.45, 0.15],
            },
            {
                "sigma_label": "sigma=0.50",
                "sigma_value": 0.5,
                "photons": 2,
                "prediction": [0.75, 0.35, 0.1],
            },
        ],
    }
    output_path = tmp_path / "learned_vs_target.png"
    plot_gaussian_fits_from_payload(payload, output_path)
    assert output_path.exists()


def test_plot_dataset_examples_from_payload_saves(tmp_path: Path) -> None:
    payload = {
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
    output_path = tmp_path / "classification_datasets.png"
    plot_dataset_examples_from_payload(payload, output_path)
    assert output_path.exists()


def test_plot_accuracy_bars_saves(tmp_path: Path) -> None:
    quantum = [
        {
            "sigma_label": "sigma=1.00",
            "n_photons": 2,
            "circular_acc": 0.7,
            "moon_acc": 0.6,
            "blob_acc": 0.8,
        }
    ]
    classical = [
        {
            "sigma_label": "sigma=1.00",
            "circular_acc": 0.75,
            "moon_acc": 0.65,
            "blob_acc": 0.85,
        }
    ]
    output_path = tmp_path / "svm_accuracy.png"
    plot_accuracy_bars(quantum, classical, output_path)
    assert output_path.exists()
