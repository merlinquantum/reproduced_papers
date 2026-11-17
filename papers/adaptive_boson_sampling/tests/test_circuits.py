from __future__ import annotations

import common  # noqa: F401
import numpy as np
import pytest
from lib.circuits import (
    gaussian_like_index,
    get_platform,
    mzi_matrix,
    programmed_unitary,
    random_u0,
    state_table,
)
from lib.kernels import overlap_kernel

# Fig. 3b theoretical kernel of platform A: the calibration target that fixes
# the MZI convention and the outcome order.
FIG3B_THEORY = {(0, 1): 0.298, (0, 2): 0.966, (1, 2): 0.331}


def _diagonal_kernel(platform, u0=None):
    states, _ = state_table(platform, u0)
    return overlap_kernel(
        np.array([states[k, k] for k in range(len(platform.outcomes))])
    )


def test_mzi_is_unitary_and_identity_at_pi():
    for theta in (0.0, 0.3, np.pi / 4, np.pi):
        matrix = mzi_matrix(theta)
        assert np.allclose(matrix.conj().T @ matrix, np.eye(2))
    # theta = pi gives full transmission (up to a phase); theta = 0 swaps modes.
    assert abs(mzi_matrix(np.pi)[0, 0]) == pytest.approx(1.0)
    assert abs(mzi_matrix(0.0)[0, 1]) == pytest.approx(1.0)


def test_platform_a_reproduces_fig3b_theory_kernel():
    kernel = _diagonal_kernel(get_platform("A"))
    for (row, column), value in FIG3B_THEORY.items():
        assert kernel[row, column] == pytest.approx(value, abs=1.5e-3)


@pytest.mark.parametrize(
    "name, m, n, d, D",
    [("A", 6, 2, 2, 3), ("B1", 8, 3, 2, 15), ("B2", 8, 3, 2, 15), ("B3", 8, 3, 3, 15)],
)
def test_platform_labels_match_paper(name, m, n, d, D):
    platform = get_platform(name)
    assert platform.n_modes == m
    assert len(platform.input_modes) == n
    assert platform.output_dimension == d
    assert len(platform.outcomes) == D
    assert sorted(platform.detected_modes + platform.residual_modes) == list(range(m))


@pytest.mark.parametrize("name", ["B2", "B3"])
def test_gaussian_like_indices_are_one_to_fifteen_in_order(name):
    platform = get_platform(name)
    indices = [gaussian_like_index(outcome) for outcome in platform.outcomes]
    assert indices == list(range(1, 16))


@pytest.mark.parametrize("name", ["A", "B1", "B2", "B3"])
def test_programmed_unitaries_are_unitary_and_kernels_valid(name):
    platform = get_platform(name)
    u0 = random_u0(8, 0) if platform.needs_random_u0 else None
    unitary = programmed_unitary(platform, platform.outcomes[0], u0)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(platform.n_modes))
    kernel = _diagonal_kernel(platform, u0)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10


def test_u0_presence_is_checked():
    with pytest.raises(ValueError, match="needs a U_0"):
        programmed_unitary(get_platform("B2"), get_platform("B2").outcomes[0], None)
    with pytest.raises(ValueError, match="fixed mesh"):
        programmed_unitary(get_platform("A"), get_platform("A").outcomes[0], np.eye(6))
