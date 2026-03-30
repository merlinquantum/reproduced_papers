from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.fock_state_expressivity.VQC_fourier_series.utils.plotting import (  # noqa: E402
    plot_learned_functions_from_predictions,
    plot_training_curves,
)


def test_plot_training_curves_saves(tmp_path: Path) -> None:
    results = {
        "VQC_[1, 0, 0]": {
            "runs": [
                {"losses": [1.0, 0.8, 0.6]},
                {"losses": [1.1, 0.9, 0.7]},
            ],
            "color": "#1f77b4",
        }
    }
    output_path = tmp_path / "training_curves.png"
    plot_training_curves(results, save_path=output_path, show=False)
    assert output_path.exists()


def test_plot_learned_functions_from_predictions_saves(tmp_path: Path) -> None:
    payload = {
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
    output_path = tmp_path / "learned_vs_target.png"
    plot_learned_functions_from_predictions(payload, save_path=output_path, show=False)
    assert output_path.exists()
