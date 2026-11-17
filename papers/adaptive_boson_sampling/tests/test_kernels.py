from __future__ import annotations

import common  # noqa: F401
import numpy as np
import pytest
from lib.circuits import get_platform, random_u0, state_table
from lib.kernels import (
    designed_kernel,
    overlap_kernel,
    quantum_kernel,
    random_rank_d_kernel,
    select_u0,
    states_for_assignment,
    uhlmann_fidelity,
)
from lib.u0_fit import fit_u0_to_kernel


def test_overlap_equals_fidelity_for_pure_states():
    states, _ = state_table(get_platform("B3"), random_u0(8, 1))
    diagonal = states_for_assignment(states, np.arange(15))
    assert np.allclose(
        quantum_kernel(diagonal, "overlap"),
        quantum_kernel(diagonal, "fidelity"),
        atol=1e-6,
    )


def test_uhlmann_fidelity_of_mixed_states():
    maximally_mixed = np.eye(2) / 2
    pure = np.diag([1.0, 0.0])
    assert uhlmann_fidelity(maximally_mixed, pure) == pytest.approx(0.5)
    assert uhlmann_fidelity(maximally_mixed, maximally_mixed) == pytest.approx(1.0)


def test_designed_kernel_is_banded_like_fig4():
    kernel = designed_kernel(get_platform("B2"))
    # Similarity decays with index distance (Gaussian-like design).
    assert kernel[0, 1] > kernel[0, 7] > kernel[0, 14]
    assert kernel[0, 14] < 0.1


def test_u0_selection_is_deterministic_and_recorded():
    platform = get_platform("B2")
    first, record = select_u0(platform, 3, "max_design_agreement", 5)
    second, _ = select_u0(platform, 3, "max_design_agreement", 5)
    assert np.allclose(first, second)
    assert 15 <= record["chosen_haar_seed"] < 20
    with pytest.raises(ValueError, match="unknown u0 strategy"):
        select_u0(platform, 0, "bogus", 1)


def test_assignment_rejects_zero_probability_states():
    states = np.full((15, 15, 2), np.nan, dtype=complex)
    with pytest.raises(ValueError, match="zero-probability"):
        states_for_assignment(states, np.arange(15))


def test_random_rank_d_kernel_is_valid():
    kernel = random_rank_d_kernel(15, 3, seed=0)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10


def test_u0_fit_recovers_kernel_of_known_u0():
    # Known-answer check: the fit must reproduce the kernel of a known Haar U_0.
    platform = get_platform("B2")
    states, _ = state_table(platform, random_u0(8, 11))
    target = overlap_kernel(np.array([states[k, k] for k in range(15)]))
    u0, record = fit_u0_to_kernel(platform, target, [0, 1], 600, 0.05)
    assert np.allclose(u0.conj().T @ u0, np.eye(8), atol=1e-10)
    fitted_states, _ = state_table(platform, u0)
    fitted = overlap_kernel(np.array([fitted_states[k, k] for k in range(15)]))
    best = min(start["mean_abs_error"] for start in record["starts"])
    upper = np.triu_indices(15, 1)
    fitted_error = np.abs(fitted - target)[upper].mean()
    assert fitted_error == pytest.approx(best, abs=1e-6)
    assert best < 0.02
