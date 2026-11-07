from __future__ import annotations

import numpy as np
import pytest
import torch
from common import PROJECT_DIR  # noqa: F401

pytest.importorskip("pyscf")

from lib.h2 import exact_energy_and_angle, h2_coefficients  # noqa: E402
from lib.vqe import PAULI_MATRICES, VQEModel  # noqa: E402
from scipy.linalg import expm  # noqa: E402

BOND_LENGTHS = (0.25, 0.7414, 1.5, 2.85)


def _hamiltonian(coefficients) -> np.ndarray:
    return sum(c * p for c, p in zip(coefficients, PAULI_MATRICES))


def test_equilibrium_ground_energy_matches_the_known_h2_value():
    energy, _ = exact_energy_and_angle(h2_coefficients(0.7414))
    assert energy == pytest.approx(-1.13727, abs=1e-4)  # hartree, STO-3G full CI


def test_ground_energy_equals_full_ci_for_all_bond_lengths():
    # h2_coefficients raises if the 2-qubit ground energy differs from PySCF full CI
    for bond_length in BOND_LENGTHS:
        coefficients = h2_coefficients(bond_length)
        ground = np.linalg.eigvalsh(_hamiltonian(coefficients)).min()
        assert exact_energy_and_angle(coefficients)[0] == pytest.approx(
            ground, abs=1e-9
        )


def test_closed_form_optimal_angle_matches_brute_force_minimisation():
    angles = np.linspace(0, np.pi, 20001)
    for bond_length in BOND_LENGTHS:
        coefficients = h2_coefficients(bond_length)
        hamiltonian = _hamiltonian(coefficients)
        states = np.stack(
            [
                np.zeros_like(angles),
                np.cos(angles),
                -np.sin(angles),
                np.zeros_like(angles),
            ],
            1,
        )
        energies = np.einsum("ni,ij,nj->n", states, hamiltonian, states)
        _, closed_form = exact_energy_and_angle(coefficients)
        assert closed_form == pytest.approx(angles[energies.argmin()], abs=2e-3)


def test_ucc_state_equals_exponential_of_x0y1_applied_to_hartree_fock():
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]])
    theta = 0.37
    hartree_fock = np.array([0, 1, 0, 0], dtype=complex)  # |01>
    state = expm(-1j * theta * np.kron(x, y)) @ hartree_fock
    np.testing.assert_allclose(state, [0, np.cos(theta), -np.sin(theta), 0], atol=1e-12)
    # the model's energy uses exactly this state
    coefficients = h2_coefficients(0.7414)
    model = VQEModel(0, torch.float64)
    model.weights.data.fill_(theta)
    energy = model(
        torch.ones(1, 1, dtype=torch.float64),
        torch.tensor([coefficients], dtype=torch.float64),
    )
    assert energy.item() == pytest.approx(
        (state.conj() @ _hamiltonian(coefficients) @ state).real
    )


def test_hartree_fock_energy_is_above_the_exact_energy():
    coefficients = h2_coefficients(1.5)
    model = VQEModel(0, torch.float64)  # weights start at zero: the HF state
    hf_energy = model(
        torch.ones(1, 1, dtype=torch.float64),
        torch.tensor([coefficients], dtype=torch.float64),
    )
    assert hf_energy.item() > exact_energy_and_angle(coefficients)[0] + 1e-3
