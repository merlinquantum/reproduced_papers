"""Shared Fourier-series dataset generator for VQC_fourier_series."""

import json
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FourierCoefficient:
    n: int
    real: float
    imag: float = 0.0

    @property
    def complex_value(self) -> complex:
        return complex(self.real, self.imag)


def _build_coefficients(specs: list[dict]) -> dict[int, complex]:
    """Expand positive-frequency specs into +/-n complex coefficients."""
    coeffs: dict[int, complex] = {}
    for spec in specs:
        coeff = FourierCoefficient(
            n=int(spec["n"]),
            real=float(spec.get("real", 0.0)),
            imag=float(spec.get("imag", 0.0)),
        )
        coeffs[coeff.n] = coeff.complex_value
        if coeff.n > 0:
            coeffs[-coeff.n] = np.conj(coeff.complex_value)
    return coeffs


def generate_dataset(cfg: dict) -> dict[str, torch.Tensor]:
    """
    Generate inputs and target outputs for the Fourier-series regression task.

    Args:
        cfg (Dict): Configuration with the following keys:
            - x_start (float): Lower bound of the domain.
            - x_end (float): Upper bound of the domain.
            - step (float): Sampling step size.
            - coefficients (List[Dict]): Fourier coefficients specification.

    Returns:
        Dict[str, torch.Tensor]: Contains `x`, `y`, and numpy counterparts for plotting.
    """
    start = float(cfg.get("x_start", -3 * np.pi))
    end = float(cfg.get("x_end", 3 * np.pi))
    step = float(cfg.get("step", 0.05))
    specs = cfg.get("coefficients", [])

    coeff_map = _build_coefficients(specs)
    xs = np.arange(start, end + step / 2.0, step)
    values = np.zeros_like(xs, dtype=np.complex128)

    for n, coeff in coeff_map.items():
        values += coeff * np.exp(1j * n * xs)

    ys = values.real  # Imaginary residuals are numerical noise only.
    assert np.allclose(
        values.imag, 0, atol=1e-6
    ), "Fourier series should evaluate to real values."

    x_tensor = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(ys, dtype=torch.float32)

    return {
        "x": x_tensor,
        "y": y_tensor,
        "x_numpy": xs,
        "y_numpy": ys,
        "coefficients": json.loads(json.dumps(specs)),
    }


__all__ = ["FourierCoefficient", "generate_dataset"]
