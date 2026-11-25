"""Reservoir read-out: k-local Pauli features (paper Sec. 3.2.1).

Three estimators of ``Tr(P rho)`` for every Pauli string ``P`` of weight ``1..k``:

* ``exact``: noiseless expectations.
* ``per_observable_shots``: the authors' simulation (``CSFeatureMapsRetriever``):
  independent +-1 outcomes per observable, ``shots`` each, aggregated by median of
  means over ``min(16, sqrt(shots))`` groups. The random draws replay the authors'
  single ``rng.binomial`` call, so seed 12345 reproduces their released features.
* ``random_pauli_shadows``: classical shadows as the paper text describes them:
  every snapshot measures all qubits in uniformly random X/Y/Z bases and is shared
  by all observables; ``3**|P|`` inverse-channel weights; median of means.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from .reservoir import kron_qubits

LETTERS = ("X", "Y", "Z")
_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}
# Rotations mapping the X, Y, Z eigenbases onto the computational basis.
_BASIS_CHANGE = {
    0: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),  # H
    1: np.array([[1, -1j], [1, 1j]]) / np.sqrt(2),  # H S^dag
    2: np.eye(2, dtype=complex),
}


def k_local_paulis(locality: int, num_qubits: int) -> list[dict[int, str]]:
    """Pauli strings of weight ``1..locality``, in the authors' enumeration order.

    Returns
    -------
    list[dict[int, str]]
        Each observable as ``{qubit: letter}`` over its support.
    """
    paulis = []
    for weight in range(1, min(locality, num_qubits) + 1):
        for qubits in combinations(range(num_qubits), weight):
            for letters in product(LETTERS, repeat=weight):
                paulis.append(dict(zip(qubits, letters)))
    return paulis


def pauli_matrix(pauli: dict[int, str], num_qubits: int) -> np.ndarray:
    """Dense matrix of a Pauli string (qubit 0 least significant)."""
    return kron_qubits([_PAULI[pauli.get(q, "I")] for q in range(num_qubits)])


def exact_expectations(states: np.ndarray, paulis: list[dict[int, str]]) -> np.ndarray:
    """``Tr(P rho)`` for states of shape ``(N, R, D, D)``; returns ``(N, R, K)``."""
    num_qubits = int(np.log2(states.shape[-1]))
    matrices = np.stack([pauli_matrix(p, num_qubits) for p in paulis])
    return np.einsum("kij,nrji->nrk", matrices, states).real


def median_of_means_groups(shots: int) -> int:
    """Number of median-of-means groups used by the authors' retriever."""
    return max(1, min(16, int(np.sqrt(shots))))


def per_observable_shot_noise(mu: np.ndarray, *, shots: int, seed: int) -> np.ndarray:
    """The authors' shot-noise model applied to exact expectations ``mu`` (N, R, K).

    Uses ``shots // groups`` outcomes per group, so with 1000 shots and 16 groups
    992 outcomes per observable enter the estimate, as in the authors' code.
    """
    groups = median_of_means_groups(shots)
    batch = max(shots // groups, 1)
    mu = np.clip(mu, -1.0, 1.0)
    probability = np.clip((1.0 + mu) / 2.0, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    counts = rng.binomial(n=batch, p=probability[..., None], size=mu.shape + (groups,))
    group_means = (2.0 * counts - batch) / float(batch)
    return np.median(group_means, axis=-1)


def _basis_probabilities(states: np.ndarray, num_qubits: int) -> np.ndarray:
    """Outcome distributions in all ``3**n`` measurement settings, shape ``(B, 3**n, D)``."""
    settings = list(
        product(range(3), repeat=num_qubits)
    )  # setting[q] = basis of qubit q
    rotations = np.stack(
        [
            kron_qubits([_BASIS_CHANGE[s[q]] for q in range(num_qubits)])
            for s in settings
        ]
    )
    rotated = rotations[None] @ states[:, None]  # (B, S, D, D): U rho
    probs = np.einsum(
        "bsoj,soj->bso", rotated, rotations.conj()
    ).real  # diag(U rho U^dag)
    return np.clip(probs, 0.0, None)


def random_pauli_shadows(
    states: np.ndarray,
    paulis: list[dict[int, str]],
    *,
    shots: int,
    seed: int,
    chunk_size: int = 50,
) -> np.ndarray:
    """Classical-shadow estimates with shared random-Pauli snapshots.

    Parameters
    ----------
    states : numpy.ndarray
        Density matrices of shape ``(N, R, D, D)``.
    paulis : list[dict[int, str]]
        Observables from :func:`k_local_paulis`.
    shots : int
        Snapshots per state (shared by all observables).
    seed : int
        Seed of the snapshot stream.
    chunk_size : int
        States processed together (memory bound). Default value is 50.

    Returns
    -------
    numpy.ndarray
        Estimates of shape ``(N, R, K)``.
    """
    num_windows, num_reservoirs, dim, _ = states.shape
    num_qubits = int(np.log2(dim))
    groups = median_of_means_groups(shots)
    batch = max(shots // groups, 1)
    used = groups * batch
    # Each observable as two (qubit, letter) factors; weight-1 strings use a dummy
    # factor equal to one.
    factor_qubit = np.zeros((len(paulis), 2), dtype=int)
    factor_letter = np.full((len(paulis), 2), 3, dtype=int)  # 3 = identity slot
    for k, pauli in enumerate(paulis):
        for slot, (qubit, letter) in enumerate(pauli.items()):
            factor_qubit[k, slot] = qubit
            factor_letter[k, slot] = LETTERS.index(letter)

    flat = states.reshape(num_windows * num_reservoirs, dim, dim)
    rng = np.random.default_rng(seed)
    estimates = np.empty((flat.shape[0], len(paulis)))
    powers = (
        3 ** np.arange(num_qubits)[::-1]
    )  # setting index of product(range(3), repeat=n)
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        size = chunk.shape[0]
        probs = _basis_probabilities(chunk, num_qubits)  # (B, 3**n, D)
        bases = rng.integers(0, 3, size=(size, used, num_qubits))
        setting = bases @ powers
        cdf = np.cumsum(np.take_along_axis(probs, setting[..., None], axis=1), axis=-1)
        draws = rng.random(size=(size, used, 1)) * cdf[..., -1:]
        outcome = (draws > cdf).sum(axis=-1)  # basis index of the observed bit string
        bits = (outcome[..., None] >> np.arange(num_qubits)) & 1
        signs = 1.0 - 2.0 * bits  # (B, S, n)
        # weight[b, s, q, letter] = 3 * [basis == letter] * sign, plus an identity slot.
        weights = np.ones((size, used, num_qubits, 4))
        for letter in range(3):
            weights[..., letter] = 3.0 * (bases == letter) * signs
        snapshot = (
            weights[:, :, factor_qubit[:, 0], factor_letter[:, 0]]
            * weights[:, :, factor_qubit[:, 1], factor_letter[:, 1]]
        )  # (B, S, K)
        group_means = snapshot.reshape(size, groups, batch, len(paulis)).mean(axis=2)
        estimates[start : start + size] = np.median(group_means, axis=1)
    return estimates.reshape(num_windows, num_reservoirs, len(paulis))


def pauli_features(
    states: np.ndarray, *, locality: int, estimator: str, shots: int, seed: int
) -> np.ndarray:
    """Feature matrix of shape ``(N, R * K)``, reservoir-major as in the authors' code."""
    num_qubits = int(np.log2(states.shape[-1]))
    paulis = k_local_paulis(locality, num_qubits)
    if estimator == "exact":
        values = exact_expectations(states, paulis)
    elif estimator == "per_observable_shots":
        values = per_observable_shot_noise(
            exact_expectations(states, paulis), shots=shots, seed=seed
        )
    elif estimator == "random_pauli_shadows":
        values = random_pauli_shadows(states, paulis, shots=shots, seed=seed)
    else:
        raise ValueError(f"Unknown estimator {estimator!r}")
    return values.reshape(values.shape[0], -1)
