"""Minimum sampling overhead of a noise inverse (paper App. D, Eq. D3).

For a single-qubit channel ``N`` the optimal quasi-probability decomposition
``N^{-1} = c1 C1 - c2 C2`` over all CPTP maps solves

    min c1 + c2  s.t.  J_{N^-1} = J_{C1} - J_{C2},
                       J_{Ci} >= 0,  Tr_out J_{Ci} = c_i I_in,

with Choi operators ``J = sum_ij |i><j| ⊗ M(|i><j|)`` (input ⊗ output).
"""

from __future__ import annotations

import numpy as np

_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def noise_kraus(model: str, level: float) -> list[np.ndarray]:
    """Kraus operators of the single-qubit noise channel to be mitigated.

    Raises
    ------
    ValueError
        If ``model`` is unknown.
    """
    if model == "depolarizing":
        # D_p(rho) = (1 - p) rho + p I/2  <=>  weights 1 - 3p/4 and p/4.
        return [np.sqrt(1 - 3 * level / 4) * _PAULIS["I"]] + [
            np.sqrt(level / 4) * _PAULIS[name] for name in "XYZ"
        ]
    if model == "amplitude_damping":
        return [
            np.array([[1, 0], [0, np.sqrt(1 - level)]], dtype=complex),
            np.array([[0, np.sqrt(level)], [0, 0]], dtype=complex),
        ]
    raise ValueError(f"Unknown noise model {model!r}")


def _matrix_units() -> list[tuple[int, int, np.ndarray]]:
    units = []
    for i in range(2):
        for j in range(2):
            unit = np.zeros((2, 2), dtype=complex)
            unit[i, j] = 1.0
            units.append((i, j, unit))
    return units


def inverse_choi(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Choi operator of ``N^{-1}`` built from the inverted superoperator."""
    superoperator = sum(np.kron(op.conj(), op) for op in kraus_ops)
    inverse = np.linalg.inv(superoperator)
    choi = np.zeros((4, 4), dtype=complex)
    for _, _, unit in _matrix_units():
        image = (inverse @ unit.reshape(-1, order="F")).reshape(2, 2, order="F")
        choi += np.kron(unit, image)
    return choi


def min_sampling_overhead(model: str, level: float, solver: str = "CLARABEL") -> dict:
    """Solve the App. D SDP for the minimum sampling overhead of ``N^{-1}``.

    Parameters
    ----------
    model : str
        ``"depolarizing"`` or ``"amplitude_damping"``.
    level : float
        Noise level ``p`` or ``q``.
    solver : str
        CVXPY solver name. Default value is ``"CLARABEL"``.

    Returns
    -------
    dict
        ``gamma_min`` (optimal ``c1 + c2``), ``c1``, ``c2`` and ``status``.

    Raises
    ------
    RuntimeError
        If the solver does not report an optimal solution.
    """
    import cvxpy as cp

    target = inverse_choi(noise_kraus(model, level))
    choi_plus = cp.Variable((4, 4), hermitian=True)
    choi_minus = cp.Variable((4, 4), hermitian=True)
    c_plus = cp.Variable()
    c_minus = cp.Variable()
    identity = np.eye(2)
    constraints = [
        choi_plus - choi_minus == target,
        choi_plus >> 0,
        choi_minus >> 0,
        cp.partial_trace(choi_plus, [2, 2], axis=1) == c_plus * identity,
        cp.partial_trace(choi_minus, [2, 2], axis=1) == c_minus * identity,
    ]
    problem = cp.Problem(cp.Minimize(c_plus + c_minus), constraints)
    problem.solve(solver=solver)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(
            f"SDP for {model}({level}) ended with status {problem.status}"
        )
    return {
        "gamma_min": float(problem.value),
        "c1": float(c_plus.value),
        "c2": float(c_minus.value),
        "status": problem.status,
    }


def depolarizing_min_overhead_analytic(level: float) -> float:
    """Closed form ``(1 + p/2)/(1 - p)`` used in the authors' plotting notebook."""
    return (1 + level / 2) / (1 - level)
