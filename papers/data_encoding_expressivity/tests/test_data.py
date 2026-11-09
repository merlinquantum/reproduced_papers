"""The seven target-generator tests required by LOG.md "Phase 2 generator specification"."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from common import PROJECT_DIR  # noqa: F401  (puts the repository and paper on sys.path)
from lib.baselines import least_squares_oracle
from lib.data import build_dataset, build_grid, evaluate_target
from lib.fourier import fourier_coefficients, periodic_grid

GRID = {"start": -6.0, "stop": 6.0, "num": 70}


def _coefficients(values):
    return [{"re": c.real, "im": c.imag} for c in values]


T1 = {
    "c0": 0.1,
    "coefficients": _coefficients([0.15 - 0.15j]),
    "sign": -1,
    "scaling": 1.0,
}
T3 = {
    "c0": 0.1,
    "coefficients": _coefficients([0.15 - 0.15j] * 2),
    "sign": -1,
    "scaling": 1.0,
}
T4 = {
    "c0": 0.0,
    "coefficients": _coefficients([0.05 - 0.05j] * 5),
    "sign": 1,
    "scaling": 2.0,
}
T4B = {**T4, "scaling": 1.0}

EXPECTED_STATISTICS = {
    "T1": (T1, -0.3242, 0.5243, 0.0904, 0.09808),
    "T3": (T3, -0.5243, 0.9224, 0.0809, 0.18041),
    "T4": (T4, -0.3959, 0.6671, -0.0127, 0.04419),
    "T4b": (T4B, -0.4077, 0.6599, -0.0152, 0.04372),
}


def _targets(target):
    return build_dataset({"grid": GRID, "target": target})


def test_shape_dtype_and_grid_endpoints():
    inputs, targets = _targets(T1)
    assert inputs.shape == targets.shape == (70,)
    assert inputs.dtype == targets.dtype == torch.float64
    assert inputs[0] == -6.0 and inputs[-1] == 6.0


def test_complex_sum_is_real():
    inputs = build_grid(GRID)
    complex_values = torch.full((70,), 0.1, dtype=torch.complex128)
    for n in (1, 2):
        phase = torch.exp(-1j * n * inputs.to(torch.complex128))
        complex_values = (
            complex_values + (0.15 - 0.15j) * phase + (0.15 + 0.15j) * phase.conj()
        )
    assert complex_values.imag.abs().max() < 1e-12
    assert torch.allclose(complex_values.real, evaluate_target(inputs, T3), atol=1e-15)


@pytest.mark.parametrize("label", sorted(EXPECTED_STATISTICS))
def test_statistics_match_specification(label):
    target, minimum, maximum, mean, mean_square = EXPECTED_STATISTICS[label]
    _, values = _targets(target)
    values = values.numpy()
    assert values.min() == pytest.approx(minimum, abs=1e-4)
    assert values.max() == pytest.approx(maximum, abs=1e-4)
    assert values.mean() == pytest.approx(mean, abs=1e-4)
    assert np.mean(values**2) == pytest.approx(mean_square, abs=1e-4)


def test_spectral_structure_on_whole_period_grid():
    grid = torch.tensor(periodic_grid(64), dtype=torch.float64)
    frequencies, coefficients = fourier_coefficients(evaluate_target(grid, T3).numpy())
    support = sorted(int(n) for n in frequencies[np.abs(coefficients) > 1e-12])
    assert support == [-2, -1, 0, 1, 2]
    by_frequency = dict(zip(frequencies.tolist(), coefficients))
    # sign = -1: the coefficient c_n multiplies exp(-inx), so it sits at frequency -n.
    assert abs(by_frequency[-1] - (0.15 - 0.15j)) < 1e-12
    assert abs(by_frequency[-2] - (0.15 - 0.15j)) < 1e-12
    assert abs(by_frequency[0] - 0.1) < 1e-12

    frequencies, coefficients = fourier_coefficients(evaluate_target(grid, T4).numpy())
    support = sorted(int(n) for n in frequencies[np.abs(coefficients) > 1e-12])
    assert support == [-10, -8, -6, -4, -2, 2, 4, 6, 8, 10]
    by_frequency = dict(zip(frequencies.tolist(), coefficients))
    assert abs(by_frequency[2] - (0.05 - 0.05j)) < 1e-12


def test_paper_and_reference_code_conventions_agree():
    inputs = build_grid(GRID)
    code_t1 = {
        "c0": 0.1,
        "coefficients": _coefficients([0.15 + 0.15j]),
        "sign": 1,
        "scaling": 1.0,
    }
    assert torch.allclose(
        evaluate_target(inputs, T1), evaluate_target(inputs, code_t1), atol=1e-15
    )

    code_t4 = {
        "c0": 0.0,
        "coefficients": _coefficients([0.05 + 0.05j] * 5),
        "sign": 1,
        "scaling": 2.0,
    }
    assert torch.allclose(
        evaluate_target(inputs, T4), evaluate_target(-inputs, code_t4), atol=1e-15
    )


def test_oracle_errors_match_specification():
    inputs, t3 = _targets(T3)
    assert least_squares_oracle(inputs.numpy(), t3.numpy(), [1], 1.0)[
        "mse"
    ] == pytest.approx(0.08971, abs=1e-4)
    inputs, t1 = _targets(T1)
    assert least_squares_oracle(inputs.numpy(), t1.numpy(), [1], 2.0)[
        "mse"
    ] == pytest.approx(0.08971, abs=1e-4)
    inputs, t4 = _targets(T4)
    assert (
        least_squares_oracle(inputs.numpy(), t4.numpy(), [1, 2, 3, 4, 5], 2.0)["mse"]
        < 1e-20
    )


def test_missing_scaling_raises():
    target = copy.deepcopy(T1)
    del target["scaling"]
    with pytest.raises(KeyError):
        evaluate_target(build_grid(GRID), target)


def test_full_batch_train_loader_yields_the_whole_grid_every_step():
    from lib.data import build_dataloader

    case = {
        "grid": GRID,
        "target": T1,
        "training": {"steps": 3, "full_batch": True, "batch_size": 25},
    }
    batches = list(build_dataloader(case, "train", seed=0))
    assert len(batches) == 3
    full_inputs, _ = build_dataset(case)
    for inputs, _ in batches:
        assert torch.equal(inputs, full_inputs)


def test_train_loader_requires_the_full_batch_key():
    from lib.data import build_dataloader

    case = {"grid": GRID, "target": T1, "training": {"steps": 3, "batch_size": 25}}
    with pytest.raises(KeyError):
        build_dataloader(case, "train", seed=0)
