from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.shared.fock_state_expressivity.VQC_fourier_series.data import (  # noqa: E402
    FourierCoefficient,
    generate_dataset,
)


def test_fourier_coefficient_complex_value() -> None:
    coeff = FourierCoefficient(n=2, real=0.5, imag=-0.25)
    assert coeff.complex_value == complex(0.5, -0.25)


def test_generate_dataset_shapes_and_keys() -> None:
    cfg = {
        "x_start": 0.0,
        "x_end": 0.2,
        "step": 0.1,
        "coefficients": [
            {"n": 0, "real": 0.2, "imag": 0.0},
            {"n": 1, "real": 0.5, "imag": 0.1},
        ],
    }
    dataset = generate_dataset(cfg)
    assert set(dataset.keys()) == {"x", "y", "x_numpy", "y_numpy", "coefficients"}
    assert isinstance(dataset["x"], torch.Tensor)
    assert isinstance(dataset["y"], torch.Tensor)
    assert dataset["x"].shape == (3, 1)
    assert dataset["y"].shape == (3,)
    assert dataset["coefficients"] == cfg["coefficients"]
