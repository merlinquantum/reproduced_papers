"""Kirby-Love contextuality of the Pauli support of g_T(H), extension E12.

Kirby & Love (PRL 123, 200501, 2019): a set S of Pauli strings admits a
noncontextual model iff, after removing the strings that commute with every
element of S, "commutes with" is an equivalence relation (transitive) on what
remains; equivalently the remaining commutation graph is a disjoint union of
cliques (the cliques then pairwise anticommute). The identity is dropped
first: it commutes with everything.

The activation observable ``g_T(H)`` is expanded in the Pauli basis,
``g_T(H) = sum_P c_P P`` with ``c_P = Tr[P g_T(H)] / 2**n``. The
epsilon-support ``S_eps(T) = {P != I : |c_P| > eps * max_P |c_P|}`` is
tested for every T. At high T, ``g_T(H) ~ H/T`` and the support is the term
set of H; lower T adds products of terms (Eq. (121)).

Graded measure: ``contextual_fraction`` is one minus the largest squared
weight ``sum c_P**2`` of a noncontextual subset found greedily (strings added
in decreasing |c_P| while the set stays noncontextual), divided by the total
squared weight of the support. Greedy gives a *lower* bound on the best
noncontextual weight, so the reported fraction is an upper bound on the true
minimal fraction.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .activations import make_activation, matrix_function
from .pauli import pauli_string

_SYMPLECTIC = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def all_pauli_labels(n_qubits: int) -> list[str]:
    """Return the 4**n Pauli labels in lexicographic I, X, Y, Z order."""
    return [
        "".join(characters) for characters in itertools.product("IXYZ", repeat=n_qubits)
    ]


@functools.lru_cache(maxsize=8)
def _pauli_basis(labels: tuple[str, ...]) -> np.ndarray:
    return np.stack([pauli_string(label) for label in labels])


def pauli_expansion(observable: np.ndarray, labels: list[str]) -> np.ndarray:
    """Return real coefficients ``c_P = Tr[P O] / d`` of a Hermitian observable."""
    dim = observable.shape[0]
    basis = _pauli_basis(tuple(labels))
    return np.real(np.einsum("pab,ba->p", basis, observable)) / dim


def _symplectic(labels: list[str]) -> np.ndarray:
    return np.array(
        [
            [bit for character in label for bit in _SYMPLECTIC[character]]
            for label in labels
        ],
        dtype=np.int64,
    )


def commutation_matrix(labels: list[str]) -> np.ndarray:
    """Return the boolean matrix ``A[a, b] = (P_a commutes with P_b)``.

    Two Pauli strings commute iff their symplectic product
    ``sum_i (x_a z_b + z_a x_b)_i`` is even.
    """
    vectors = _symplectic(labels)
    x, z = vectors[:, 0::2], vectors[:, 1::2]
    product = (x @ z.T + z @ x.T) % 2
    return product == 0


def is_noncontextual(labels: list[str]) -> bool:
    """Kirby-Love test: True iff the Pauli set admits a noncontextual model."""
    labels = [label for label in labels if set(label) != {"I"}]
    if len(labels) < 3:
        return True
    commutes = commutation_matrix(labels)
    remaining = ~commutes.all(axis=1)
    sub = commutes[np.ix_(remaining, remaining)]
    # Transitivity <=> each connected component of the commutation graph is a clique.
    n_components, component_of = connected_components(csr_matrix(sub), directed=False)
    for component in range(n_components):
        members = np.flatnonzero(component_of == component)
        if not sub[np.ix_(members, members)].all():
            return False
    return True


def greedy_contextual_fraction(labels: list[str], coefficients: np.ndarray) -> float:
    """Upper bound on the squared weight that must be removed to make the support noncontextual."""
    order = np.argsort(-np.abs(coefficients))
    kept: list[str] = []
    kept_weight = 0.0
    for index in order:
        candidate = [*kept, labels[index]]
        if is_noncontextual(candidate):
            kept = candidate
            kept_weight += float(coefficients[index] ** 2)
    total = float(np.sum(coefficients**2))
    return 0.0 if total == 0.0 else 1.0 - kept_weight / total


def temperature_scan(
    term_labels: list[str],
    theta: np.ndarray,
    temperatures: list[float],
    epsilons: list[float],
    activation: str = "tanh",
    graded_max_support: int | None = None,
) -> list[dict]:
    """Return support size, noncontextuality flag and contextual fraction per (T, eps).

    ``activation`` names the function applied to H (see
    :func:`lib.activations.make_activation`; default value is "tanh").
    ``graded_max_support`` bounds the support size for which the O(|S|^3)
    greedy contextual fraction is computed; larger supports report None.
    None computes it always. Default value is None.

    Parameters
    ----------
    term_labels : list of str
        Pauli strings H_j of the Hamiltonian H = sum_j theta_j H_j.
    theta : numpy.ndarray
        Coefficients.
    temperatures, epsilons : list of float
        Scan grid and relative support thresholds.
    """
    n_qubits = len(term_labels[0])
    terms = np.stack([pauli_string(label) for label in term_labels])
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("j,jab->ab", theta, terms))
    labels = all_pauli_labels(n_qubits)
    non_identity = np.array([set(label) != {"I"} for label in labels])
    records = []
    for temperature in temperatures:
        observable = matrix_function(
            make_activation(activation, temperature), eigenvalues, eigenvectors
        )
        coefficients = pauli_expansion(observable, labels)
        scale = float(np.max(np.abs(coefficients[non_identity])))
        for epsilon in epsilons:
            mask = non_identity & (np.abs(coefficients) > epsilon * scale)
            support = [label for label, keep in zip(labels, mask) if keep]
            records.append(
                {
                    "temperature": float(temperature),
                    "epsilon": float(epsilon),
                    "support_size": len(support),
                    "noncontextual": is_noncontextual(support),
                    "contextual_fraction": (
                        greedy_contextual_fraction(support, coefficients[mask])
                        if graded_max_support is None
                        or len(support) <= graded_max_support
                        else None
                    ),
                    "max_pauli_weight": max(
                        (n_qubits - label.count("I") for label in support), default=0
                    ),
                }
            )
    return records


def onset_temperature(records: list[dict], epsilon: float) -> float | None:
    """Return the highest T at which the epsilon-support is contextual (None if never)."""
    contextual = [
        record["temperature"]
        for record in records
        if record["epsilon"] == epsilon and not record["noncontextual"]
    ]
    return max(contextual) if contextual else None


def random_noncontextual_labels(
    n_qubits: int,
    size: int,
    locality: int,
    rng: np.random.Generator,
    max_tries: int = 20000,
) -> list[str]:
    """Rejection-sample a set of distinct k-local Pauli strings that is Kirby-Love noncontextual.

    Raises
    ------
    RuntimeError
        If no noncontextual set is found within ``max_tries``.
    """
    candidates = [
        label
        for label in all_pauli_labels(n_qubits)
        if 0 < sum(c != "I" for c in label) <= locality
    ]
    for _ in range(max_tries):
        chosen = [
            candidates[i] for i in rng.choice(len(candidates), size=size, replace=False)
        ]
        # Keep only sets with at least one anticommuting pair, so the scan is not trivially classical.
        if is_noncontextual(chosen) and not commutation_matrix(chosen).all():
            return chosen
    raise RuntimeError(
        f"No noncontextual {locality}-local set of size {size} on {n_qubits} qubits found"
    )


def random_contextual_labels(
    n_qubits: int,
    size: int,
    locality: int,
    rng: np.random.Generator,
    max_tries: int = 20000,
) -> list[str]:
    """Rejection-sample a set of distinct k-local Pauli strings that is Kirby-Love contextual.

    Raises
    ------
    RuntimeError
        If no contextual set is found within ``max_tries``.
    """
    candidates = [
        label
        for label in all_pauli_labels(n_qubits)
        if 0 < sum(c != "I" for c in label) <= locality
    ]
    for _ in range(max_tries):
        chosen = [
            candidates[i] for i in rng.choice(len(candidates), size=size, replace=False)
        ]
        if not is_noncontextual(chosen):
            return chosen
    raise RuntimeError(
        f"No contextual {locality}-local set of size {size} on {n_qubits} qubits found"
    )


_SYMPLECTIC_CODE = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}  # I, X, Y, Z


def _walsh_hadamard(rows: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform along the last axis (length 2^n)."""
    result = rows.astype(np.complex128, copy=True)
    size = result.shape[-1]
    half = 1
    while half < size:
        shaped = result.reshape(*result.shape[:-1], size // (2 * half), 2, half)
        left, right = shaped[..., 0, :].copy(), shaped[..., 1, :].copy()
        shaped[..., 0, :] = left + right
        shaped[..., 1, :] = left - right
        result = shaped.reshape(*result.shape[:-1], size)
        half *= 2
    return result


@functools.lru_cache(maxsize=8)
def _label_index_map(n_qubits: int) -> np.ndarray:
    """``index[x, z]`` = position of the Pauli string P(x, z) in ``all_pauli_labels``."""
    dim = 2**n_qubits
    x_bits, z_bits = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
    index = np.zeros((dim, dim), dtype=np.int64)
    for qubit in range(n_qubits):
        bit = n_qubits - 1 - qubit  # qubit 0 is the most significant bit
        x_q, z_q = (x_bits >> bit) & 1, (z_bits >> bit) & 1
        code = np.where(x_q & z_q, 2, np.where(x_q, 1, np.where(z_q, 3, 0)))
        index = index * 4 + code
    return index


def fast_pauli_expansion(observable: np.ndarray) -> np.ndarray:
    """Return ``c_P = Tr[P O] / d`` for all 4^n strings in ``all_pauli_labels`` order.

    Uses ``P(x, z) = i^{x.z} X^x Z^z``, for which
    ``Tr[P O] = i^{x.z} sum_l (-1)^{z.l} O[l, l xor x]``: one Walsh-Hadamard
    transform per x, O(n 4^n) instead of the O(16^n) dense traces of
    :func:`pauli_expansion` (agreement checked in tests).
    """
    dim = observable.shape[0]
    n_qubits = dim.bit_length() - 1
    rows = np.arange(dim)
    shifted = np.stack([observable[rows, rows ^ x] for x in range(dim)])
    transformed = _walsh_hadamard(shifted)
    x_bits, z_bits = np.meshgrid(rows, rows, indexing="ij")
    overlap = np.bitwise_count(x_bits & z_bits) % 4
    values = np.real((1j**overlap) * transformed) / dim
    coefficients = np.empty(dim * dim)
    coefficients[_label_index_map(n_qubits).ravel()] = values.ravel()
    return coefficients


def paulis_needed(coefficients: np.ndarray, epsilon: float) -> int:
    """Smallest number of non-identity Pauli strings whose dropped coefficients sum to <= epsilon.

    ``coefficients`` is in ``all_pauli_labels`` order (identity first). Since
    ``|Tr[P rho]| <= 1``, keeping these strings computes ``Tr[O rho]`` within
    ``epsilon`` for every state; the identity term is a known constant.
    """
    magnitudes = np.sort(np.abs(coefficients[1:]))
    droppable = int(np.searchsorted(np.cumsum(magnitudes), epsilon, side="right"))
    return len(magnitudes) - droppable


def random_noncontextual_two_local(
    n_qubits: int, size: int, rng: np.random.Generator, max_tries: int = 2000
) -> list[str]:
    """Construct a random Kirby-Love noncontextual set of distinct <=2-local strings.

    A core of 2 or 3 pairwise-anticommuting strings is completed with random
    strings that commute with the core and with each other (the commuting
    centre), so every clique is a singleton and the set is noncontextual by
    construction. Rejection sampling of whole sets fails at larger n, where
    almost every random set is contextual.

    Raises
    ------
    RuntimeError
        If no set of the requested size is found within ``max_tries``.
    """
    candidates = [
        label
        for label in all_pauli_labels(n_qubits)
        if 0 < sum(c != "I" for c in label) <= 2
    ]
    for _ in range(max_tries):
        order = [candidates[i] for i in rng.permutation(len(candidates))]
        core_size = int(rng.integers(2, 4))
        core = [order[0]]
        for label in order[1:]:
            if len(core) == core_size:
                break
            if not commutation_matrix([*core, label])[-1, :-1].any():
                core.append(label)
        if len(core) < 2:
            continue
        chosen = list(core)
        for label in order:
            if len(chosen) == size:
                break
            if (
                label not in chosen
                and commutation_matrix([*chosen, label])[-1, :-1].all()
            ):
                chosen.append(label)
        if (
            len(chosen) == size
            and is_noncontextual(chosen)
            and not commutation_matrix(chosen).all()
        ):
            return chosen
    raise RuntimeError(
        f"No noncontextual 2-local set of size {size} on {n_qubits} qubits constructed"
    )
