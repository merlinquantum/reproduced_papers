"""Two-qubit H2 Hamiltonian in the STO-3G basis (paper Eq. 20, ref. [37]).

The paper lists the form ``c0 I + c1 Z0 + c2 Z1 + c3 Z0 Z1 + c4 X0 X1 + c5 Y0 Y1`` but
not the coefficients, so they are derived here: PySCF gives the molecular-orbital
integrals, the fermionic Hamiltonian is built on four spin orbitals (Jordan-Wigner),
and its one-up-one-down sector (the four determinants that contain every state the
ansatz can reach) is mapped to two qubits so that the Hartree-Fock state is ``|01>`` and
the doubly excited determinant is ``|10>``.

The mapping is a gauge choice: any sign convention for the four determinants gives
the same spectrum, and the ground-state energy is checked against PySCF's full CI.
In this gauge ``c4`` (the XX coefficient) is zero and the exchange coupling sits in
``c5``; the paper's ansatz, which rotates ``|01>`` into ``|10>``, sees only the sum.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np

PAULI_ORDER = ("II", "ZI", "IZ", "ZZ", "XX", "YY")
_I2 = np.eye(2)
_PAULIS = {
    "I": _I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]),
}
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]])  # |0><1|; occupied is |1>
N_SPIN_ORBITALS = 4  # (sigma_g up, sigma_g down, sigma_u up, sigma_u down)
# Basis index of each determinant in the Jordan-Wigner space (mode 0 most significant).
DETERMINANT_INDEX = {
    "single_a": 0b1001,
    "hf": 0b1100,
    "double": 0b0011,
    "single_b": 0b0110,
}
# Two-qubit basis order |00>, |01>, |10>, |11> -> determinant.
QUBIT_BASIS = ("single_a", "hf", "double", "single_b")
STRUCTURE_TOLERANCE = 1e-8


def _kron(*matrices: np.ndarray) -> np.ndarray:
    result = np.array([[1.0]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def _annihilation_operators() -> list[np.ndarray]:
    z_matrix = _PAULIS["Z"]
    return [
        _kron(*([z_matrix] * mode + [_LOWER] + [_I2] * (N_SPIN_ORBITALS - mode - 1)))
        for mode in range(N_SPIN_ORBITALS)
    ]


def molecular_integrals(
    bond_length: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """RHF integrals of H2 / STO-3G.

    Parameters
    ----------
    bond_length : float
        H-H distance in angstrom.

    Returns
    -------
    tuple
        ``(h1, eri, nuclear_repulsion, fci_energy)`` with ``h1`` the one-body integrals
        in the molecular-orbital basis, ``eri`` the chemist-notation ``(pq|rs)``
        two-body integrals and ``fci_energy`` PySCF's full-CI total energy.
    """
    from pyscf import ao2mo, fci, gto, scf

    molecule = gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length}", basis="sto-3g", unit="Angstrom", verbose=0
    )
    mean_field = scf.RHF(molecule)
    mean_field.kernel()
    orbitals = mean_field.mo_coeff
    h1 = orbitals.T @ mean_field.get_hcore() @ orbitals
    eri = ao2mo.restore(1, ao2mo.kernel(molecule, orbitals), 2)
    fci_energy = fci.FCI(mean_field).kernel()[0]
    return h1, eri, float(molecule.energy_nuc()), float(fci_energy)


def _fermionic_hamiltonian(
    h1: np.ndarray, eri: np.ndarray, nuclear: float
) -> np.ndarray:
    annihilate = _annihilation_operators()
    create = [operator.conj().T for operator in annihilate]
    mode = lambda orbital, spin: 2 * orbital + spin  # noqa: E731
    dimension = 2**N_SPIN_ORBITALS
    hamiltonian = nuclear * np.eye(dimension, dtype=complex)
    for p, q in itertools.product(range(2), repeat=2):
        for spin in range(2):
            hamiltonian += h1[p, q] * create[mode(p, spin)] @ annihilate[mode(q, spin)]
    for p, q, r, s in itertools.product(range(2), repeat=4):
        for spin, other_spin in itertools.product(range(2), repeat=2):
            hamiltonian += (
                0.5
                * eri[p, q, r, s]
                * create[mode(p, spin)]
                @ create[mode(r, other_spin)]
                @ annihilate[mode(s, other_spin)]
                @ annihilate[mode(q, spin)]
            )
    return hamiltonian


def two_qubit_matrix(bond_length: float) -> tuple[np.ndarray, float]:
    """The ``4 x 4`` Hamiltonian in the ``|00>, |01>, |10>, |11>`` basis and PySCF's FCI energy."""
    h1, eri, nuclear, fci_energy = molecular_integrals(bond_length)
    full = _fermionic_hamiltonian(h1, eri, nuclear)
    indices = [DETERMINANT_INDEX[name] for name in QUBIT_BASIS]
    return full[np.ix_(indices, indices)].real, fci_energy


@cache
def h2_coefficients(bond_length: float) -> tuple[float, ...]:
    """Pauli coefficients ``(c0..c5)`` in the order ``II, ZI, IZ, ZZ, XX, YY``.

    ``ZI`` is ``Z`` on qubit 0. Raises ``RuntimeError`` if the mapped matrix has weight
    outside the six allowed Pauli strings or its ground energy differs from full CI.
    """
    matrix, fci_energy = two_qubit_matrix(bond_length)
    coefficients = {}
    for left, right in itertools.product("IXYZ", repeat=2):
        pauli = np.kron(_PAULIS[left], _PAULIS[right])
        coefficients[left + right] = float(np.trace(pauli @ matrix).real / 4)
    stray = {
        name: value
        for name, value in coefficients.items()
        if name not in PAULI_ORDER and abs(value) > STRUCTURE_TOLERANCE
    }
    if stray:
        raise RuntimeError(f"H2 mapping left weight outside the six-term form: {stray}")
    ground = float(np.linalg.eigvalsh(matrix).min())
    if abs(ground - fci_energy) > 1e-8:
        raise RuntimeError(
            f"2-qubit ground energy {ground} differs from full CI {fci_energy}"
        )
    return tuple(coefficients[name] for name in PAULI_ORDER)


def exact_energy_and_angle(coefficients: tuple[float, ...]) -> tuple[float, float]:
    """Exact ground energy and the optimal UCC angle for given coefficients.

    The ansatz state is ``cos(t)|01> - sin(t)|10>``, so its energy is
    ``E(t) = (a+b)/2 + (a-b)/2 cos(2t) - c5 sin(2t)`` with ``a = <01|H|01>`` and
    ``b = <10|H|10>``; the optimum is returned modulo ``pi``.
    """
    c0, c1, c2, c3, _c4, c5 = coefficients
    energy_hf = c0 + c1 - c2 - c3  # |01>: Z0 = +1, Z1 = -1
    energy_double = c0 - c1 + c2 - c3  # |10>: Z0 = -1, Z1 = +1
    half_gap = (energy_hf - energy_double) / 2
    amplitude = float(np.hypot(half_gap, c5))
    phase = float(np.arctan2(c5, half_gap))
    optimal_angle = ((np.pi - phase) / 2) % np.pi
    return (energy_hf + energy_double) / 2 - amplitude, float(optimal_angle)
