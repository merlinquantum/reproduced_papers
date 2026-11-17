"""Quantum (ABS) and classical kernels over the 15 encodable outcomes.

Quantum kernels follow the paper: K(p, q) = |<psi_q|psi_p>|^2 for pure states
(Fig. 1c), or the Uhlmann fidelity F(rho_p, rho_q) = (Tr sqrt(sqrt(rho_p) rho_q
sqrt(rho_p)))^2 used on the tomographic states (SI Note 3). They coincide for
pure states.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .circuits import AbsPlatform, mzi_matrix, random_u0, state_table
from .u0_fit import fit_u0_to_kernel


def overlap_kernel(states: np.ndarray) -> np.ndarray:
    """Return ``|<psi_i|psi_j>|^2`` for rows of a ``(D, d)`` state array."""
    gram = states.conj() @ states.T
    return np.abs(gram) ** 2


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Return ``|psi><psi|`` for a normalised state vector."""
    return np.outer(state, state.conj())


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the PSD square root of a Hermitian PSD matrix via eigh.

    Unlike ``scipy.linalg.sqrtm`` this is exact for rank-deficient (pure-state)
    density matrices; tiny negative eigenvalues from round-off are clipped.
    """
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    root = np.sqrt(np.clip(eigenvalues, 0.0, None))
    return (eigenvectors * root) @ eigenvectors.conj().T


def uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return the Uhlmann fidelity ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2``."""
    sqrt_rho = psd_sqrt(rho)
    inner = psd_sqrt(sqrt_rho @ sigma @ sqrt_rho)
    return float(np.real(np.trace(inner)) ** 2)


def fidelity_kernel(density_matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Return the matrix of pairwise Uhlmann fidelities."""
    size = len(density_matrices)
    kernel = np.eye(size)
    for row in range(size):
        for column in range(row + 1, size):
            value = uhlmann_fidelity(density_matrices[row], density_matrices[column])
            kernel[row, column] = kernel[column, row] = value
    return kernel


def states_for_assignment(
    states_by_outcome_and_circuit: np.ndarray, assignment: Sequence[int]
) -> np.ndarray:
    """Pick one residual state per adaptive circuit under an outcome assignment.

    ``assignment[k]`` is the observed outcome whose events are used for data point
    ``k`` while circuit ``k`` is programmed. The identity assignment is the
    paper's kernel; other permutations are the alternative kernels of SI Note 4.

    Raises
    ------
    ValueError
        If a selected (outcome, circuit) pair has zero post-selection probability.
    """
    selected = np.array(
        [
            states_by_outcome_and_circuit[observed, circuit]
            for circuit, observed in enumerate(assignment)
        ]
    )
    if np.isnan(selected).any():
        raise ValueError("assignment selects a zero-probability outcome")
    return selected


def quantum_kernel(states: np.ndarray, kind: str) -> np.ndarray:
    """Return the ABS kernel of pure states with the configured estimator.

    Parameters
    ----------
    states : numpy.ndarray
        ``(D, d)`` normalised states, or ``(D, d, d)`` density matrices (from the
        noisy MerLin model, ``fidelity`` only).
    kind : str
        ``"overlap"`` or ``"fidelity"``.
    """
    if states.ndim == 3:
        if kind != "fidelity":
            raise ValueError("density-matrix states need kernel_kind 'fidelity'")
        return fidelity_kernel(list(states))
    if kind == "overlap":
        return overlap_kernel(states)
    if kind == "fidelity":
        return fidelity_kernel([density_matrix(state) for state in states])
    raise ValueError(f"unknown kernel kind {kind!r}; expected overlap or fidelity")


def designed_kernel(platform: AbsPlatform) -> np.ndarray:
    """Return the B2/B3 kernel for a fixed input state into the adaptive block.

    This is the kernel the adaptive rule phi_i = theta_i = i/5 alone would give,
    if every outcome left the residual photon in the first residual mode before
    the adaptive block. It is used only to rank Haar ``U_0`` candidates.
    """
    if not platform.needs_random_u0:
        raise ValueError("designed_kernel is defined only for B2 and B3")
    dimension = len(platform.residual_modes)
    if platform.output_dimension != dimension:
        raise ValueError("designed_kernel assumes one residual photon")
    states = []
    for outcome in platform.outcomes:
        local = np.eye(dimension, dtype=np.complex128)
        for kind, mode, value in platform.operations_for(outcome):
            step = np.eye(dimension, dtype=np.complex128)
            position = platform.residual_modes.index(mode)
            if kind == "ps":
                step[position, position] = np.exp(1j * value)
            else:
                step[np.ix_([position, position + 1], [position, position + 1])] = (
                    mzi_matrix(value)
                )
            local = step @ local
        states.append(local[:, 0])
    return overlap_kernel(np.array(states))


def select_u0(
    platform: AbsPlatform, seed: int, strategy: str, n_candidates: int
) -> tuple[np.ndarray, dict]:
    """Return the ``U_0`` used for B2/B3 and a record of how it was chosen.

    Parameters
    ----------
    platform : AbsPlatform
        B2 or B3.
    seed : int
        Haar seed (``strategy="seed"``) or base seed of the candidate pool.
    strategy : str
        ``"seed"``: the Haar matrix for ``seed``. ``"max_design_agreement"``:
        among ``n_candidates`` Haar matrices with seeds
        ``seed * n_candidates + c``, the one whose diagonal kernel has the smallest
        Frobenius distance to ``designed_kernel``. No paper data enter the
        selection.
    n_candidates : int
        Pool size for ``max_design_agreement``; ignored for ``seed``.
    """
    if strategy == "seed":
        return random_u0(platform.n_modes, seed), {"strategy": "seed", "seed": seed}
    if strategy != "max_design_agreement":
        raise ValueError(f"unknown u0 strategy {strategy!r}")
    target = designed_kernel(platform)
    best = None
    for candidate in range(n_candidates):
        candidate_seed = seed * n_candidates + candidate
        u0 = random_u0(platform.n_modes, candidate_seed)
        states, _ = state_table(platform, u0)
        diagonal = np.array([states[k, k] for k in range(len(platform.outcomes))])
        distance = float(np.linalg.norm(overlap_kernel(diagonal) - target))
        if best is None or distance < best[0]:
            best = (distance, candidate_seed, u0)
    distance, chosen_seed, u0 = best
    return u0, {
        "strategy": strategy,
        "pool_seed": seed,
        "n_candidates": n_candidates,
        "chosen_haar_seed": chosen_seed,
        "frobenius_distance_to_design": distance,
    }


def u0_for_config(
    platform: AbsPlatform, seed: int, u0_cfg: dict, digitized_path: Path
) -> tuple[np.ndarray, dict]:
    """Return ``U_0`` for one seed of a config ``u0`` section.

    ``"seed"`` and ``"max_design_agreement"`` go through :func:`select_u0`.
    ``"fit_digitized"`` fits ``U_0`` to the digitised kernel ``target_key`` with
    ``n_candidates`` random starts (seeds ``seed * n_candidates + c``),
    ``n_steps`` Adam steps and ``learning_rate`` (see ``lib/u0_fit.py``).

    Raises
    ------
    KeyError
        If a key required by the strategy is missing.
    """
    strategy, n_candidates = u0_cfg["strategy"], u0_cfg["n_candidates"]
    if strategy != "fit_digitized":
        return select_u0(platform, seed, strategy, n_candidates)
    target_key = u0_cfg["target_key"]
    u0, record = fit_u0_to_kernel(
        platform,
        load_digitized_kernel(digitized_path, target_key),
        [seed * n_candidates + c for c in range(n_candidates)],
        u0_cfg["n_steps"],
        u0_cfg["learning_rate"],
    )
    return u0, {"strategy": strategy, "target_key": target_key, **record}


def gaussian_kernel(features: np.ndarray, gamma: float) -> np.ndarray:
    """Return ``exp(-gamma ||x - y||^2)`` for rows of a 2D feature array."""
    squared = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * squared)


def median_heuristic_gamma(features: np.ndarray) -> float:
    """Return ``1 / median(squared pairwise distance)`` over distinct pairs."""
    squared = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    upper = squared[np.triu_indices(len(features), k=1)]
    median = float(np.median(upper[upper > 0]))
    return 1.0 / median


def random_rank_d_kernel(n_points: int, dimension: int, seed: int) -> np.ndarray:
    """Return ``|<v_i|v_j>|^2`` for Haar-random unit vectors in C^dimension.

    A classical control with the same output dimension as the ABS states but no
    circuit structure.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n_points, dimension)) + 1j * rng.normal(
        size=(n_points, dimension)
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return overlap_kernel(vectors)


def load_digitized_kernel(path: Path, key: str) -> np.ndarray:
    """Load one 15x15 kernel digitised from the paper figures.

    Raises
    ------
    FileNotFoundError
        If the digitised archive is missing (run ``utils/digitize_figures.py``).
    KeyError
        If ``key`` is not in the archive.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"digitised kernels not found at {path}; run "
            "papers/adaptive_boson_sampling/utils/digitize_figures.py first"
        )
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"{key!r} not in {path}; keys: {sorted(archive.files)}")
        return np.array(archive[key], dtype=float)


__all__ = [
    "density_matrix",
    "designed_kernel",
    "fidelity_kernel",
    "gaussian_kernel",
    "load_digitized_kernel",
    "median_heuristic_gamma",
    "overlap_kernel",
    "quantum_kernel",
    "random_rank_d_kernel",
    "select_u0",
    "u0_for_config",
    "states_for_assignment",
    "uhlmann_fidelity",
]
