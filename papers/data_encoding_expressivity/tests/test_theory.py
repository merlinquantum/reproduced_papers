"""Spectrum prediction, bounds, dense spectrum models and the App. C construction."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repository and paper on sys.path)
from lib.fourier import (
    distinct_sum_size_bound,
    fourier_coefficients,
    paper_size_bound,
    periodic_grid,
    predicted_spectrum,
    spectrum_size,
    spectrum_support,
)
from lib.spectrum import (
    generator_with_spectrum,
    haar_unitary,
    layered_model_values,
    pauli_x_sum_generator,
    random_hermitian,
)
from lib.universality import (
    pauli_family_eigenvalues,
    square_wave_coefficients,
    square_wave_tail_rms,
    truncated_series_values,
    universal_model_values,
    universal_observable,
)


@pytest.mark.parametrize("repetitions", [1, 2, 3, 4, 5])
def test_serial_and_parallel_pauli_spectra_are_degree_r(repetitions):
    serial = predicted_spectrum([-0.5, 0.5], repetitions)
    parallel = predicted_spectrum(
        np.linalg.eigvalsh(pauli_x_sum_generator(repetitions)), 1
    )
    expected = np.arange(-repetitions, repetitions + 1)
    assert np.array_equal(serial, expected)
    assert np.array_equal(parallel, expected)
    assert spectrum_size(serial) == repetitions


def test_bounds_hold_and_golomb_ruler_saturates_the_distinct_sum_bound():
    # Eigenvalues {0, 1, 4, 6} have pairwise distinct differences.
    golomb = predicted_spectrum([0, 1, 4, 6], 1)
    assert spectrum_size(golomb) == distinct_sum_size_bound(4, 1) == 6
    assert spectrum_size(golomb) <= paper_size_bound(4, 1)
    for dimension, layers in [(2, 1), (2, 3), (3, 2), (4, 2)]:
        assert distinct_sum_size_bound(dimension, layers) <= paper_size_bound(
            dimension, layers
        )


@pytest.mark.parametrize("rotate", [False, True])
def test_dense_model_support_equals_predicted_spectrum(rotate):
    rng = np.random.default_rng(5)
    eigenvalues = [0.0, 1.0, 3.0]
    generator = generator_with_spectrum(eigenvalues, rotate, rng)
    predicted = predicted_spectrum(eigenvalues, 2)
    grid = periodic_grid(64)
    unitaries = [haar_unitary(3, rng) for _ in range(3)]
    values = layered_model_values(grid, generator, unitaries, random_hermitian(3, rng))
    frequencies, coefficients = fourier_coefficients(values)
    assert spectrum_support(frequencies, coefficients, 1e-9) == sorted(
        int(v) for v in predicted
    )


def test_haar_unitary_is_unitary():
    unitary = haar_unitary(5, np.random.default_rng(6))
    assert np.allclose(unitary.conj().T @ unitary, np.eye(5), atol=1e-12)


def test_pauli_family_eigenvalue_differences_cover_minus_m_to_m():
    eigenvalues = pauli_family_eigenvalues(4)
    differences = {round(a - b, 9) for a in eigenvalues for b in eigenvalues}
    assert differences == set(range(-4, 5))


@pytest.mark.parametrize("degree,extra", [(1, 0), (3, 0), (4, 2)])
def test_universal_construction_reproduces_truncated_series_exactly(degree, extra):
    rng = np.random.default_rng(degree)
    constant = 0.3
    coefficients = list(
        rng.uniform(-0.5, 0.5, degree) + 1j * rng.uniform(-0.5, 0.5, degree)
    )
    observable = universal_observable(constant, coefficients, degree + extra)
    assert np.allclose(observable, observable.conj().T)
    grid = periodic_grid(256)
    values = universal_model_values(grid, observable, degree + extra)
    assert (
        np.max(np.abs(values - truncated_series_values(grid, constant, coefficients)))
        < 1e-12
    )


def test_universal_construction_needs_enough_qubits():
    with pytest.raises(ValueError):
        universal_observable(0.0, [0.1, 0.1, 0.1], 2)


def test_square_wave_tail_matches_parseval_on_fine_grid():
    degree = 7
    grid = periodic_grid(8192) + np.pi / 8192
    constant, coefficients = square_wave_coefficients(degree)
    truncation = truncated_series_values(grid, constant, coefficients)
    rms = np.sqrt(np.mean((truncation - np.sign(np.sin(grid))) ** 2))
    assert rms == pytest.approx(square_wave_tail_rms(degree), rel=1e-3)
