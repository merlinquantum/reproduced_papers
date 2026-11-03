"""E5 (Lemma 4.1, Theorem 4.1 LPs) and E7 (measurement-based constructions, Sec. 7.2, App. D)."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts papers/ on sys.path)
from contextuality_inductive_bias.lib.theory import (
    lemma_lhs_for_eta,
    max_lemma_lhs,
    max_noncontextual_eta,
)


def test_lemma_bound_is_five_halves_and_attained():
    result = max_lemma_lhs(with_preparation_equivalence=True)
    assert result["optimum"] == pytest.approx(2.5, abs=1e-9)


def test_bound_without_preparation_equivalence_is_trivial():
    assert max_lemma_lhs(with_preparation_equivalence=False)[
        "optimum"
    ] == pytest.approx(3.0, abs=1e-9)


def test_lhs_on_u_eta_matches_section_4_3():
    for eta in (0.0, 0.5, 2 / 3, 0.9, 1.0):
        assert lemma_lhs_for_eta(eta) == pytest.approx(1.5 * (eta + 1.0))
    assert lemma_lhs_for_eta(2 / 3) == pytest.approx(2.5)


def test_noncontextual_eta_threshold_is_at_most_two_thirds():
    # Theorem 4.1: eta > 2/3 forces contextuality, so the noncontextual optimum is <= 2/3.
    assert max_noncontextual_eta()["eta_max"] <= 2 / 3 + 1e-9


def _pm_one_spectrum(matrix: np.ndarray) -> bool:
    return np.allclose(np.sort(np.abs(np.linalg.eigvalsh(matrix))), 1.0)


def test_trine_qubit_observables_sum_to_zero():
    sigma = [
        np.array([[0, 1], [1, 0]], complex),
        np.array([[0, -1j], [1j, 0]]),
        np.diag([1.0, -1.0]).astype(complex),
    ]
    observables = []
    for k in (1, 2, 3):
        theta = 2 * k * np.pi / 3
        observables.append(np.cos(theta) * sigma[0] + np.sin(theta) * sigma[1])
    np.testing.assert_allclose(sum(observables), 0, atol=1e-12)
    assert all(_pm_one_spectrum(o) for o in observables)


@pytest.mark.parametrize("dimension", range(2, 22, 2))
def test_appendix_d_construction_for_even_dimensions(dimension):
    half = dimension // 2
    eye, zero = np.eye(half), np.zeros((half, half))
    p = np.block([[eye, zero], [zero, zero]])
    q = 0.25 * np.block([[eye, np.sqrt(3) * eye], [np.sqrt(3) * eye, 3 * eye]])
    np.testing.assert_allclose(p @ p, p, atol=1e-12)
    np.testing.assert_allclose(q @ q, q, atol=1e-12)
    np.testing.assert_allclose((p - q) @ (p - q), 0.75 * np.eye(dimension), atol=1e-12)
    o1, o2 = 2 * p - np.eye(dimension), 2 * q - np.eye(dimension)
    o3 = -o1 - o2
    assert _pm_one_spectrum(o1) and _pm_one_spectrum(o2) and _pm_one_spectrum(o3)
