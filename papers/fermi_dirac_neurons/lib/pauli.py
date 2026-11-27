"""Pauli-string Hamiltonian terms used by the paper's models (Sec. VI B).

Qubit ordering: qubit 0 is the leftmost tensor factor, so a computational
basis index ``k`` has qubit 0 as its most significant bit. Every matrix is a
dense ``complex128`` array of shape ``(2**n, 2**n)``.

Term sets (paper equation, term count, matching notebook in the authors'
repository at commit 9c2094c):

* ``tfim``: Eq. (113), ``Z_i Z_{i+1}`` (n-1), ``X_i`` (n), identity (1) -> 2n.
* ``ising``: Eq. (114) (IM), same with ``Z_i`` fields -> 2n.
* ``heisenberg``: Eq. (115), ``P_i P_{i+1}`` for P in X, Y, Z (3(n-1)) and
  ``P_i`` (3n), no identity -> 6n - 3.
* ``fcim``: Eq. (116) restricted to i < j as in the notebook, ``Z_i Z_j``
  (n(n-1)/2) and ``Z_i`` (n), no identity -> n(n+1)/2.
"""

from __future__ import annotations

import itertools

import numpy as np

IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = {"I": IDENTITY, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}

TERM_SETS = (
    "tfim",
    "ising",
    "heisenberg",
    "fcim",
    "tfim_global",
    "ising_x0",
    "explicit",
)


def pauli_string(label: str) -> np.ndarray:
    """Return the dense matrix of a Pauli string such as ``"XZI"``.

    Parameters
    ----------
    label : str
        One character per qubit from ``I``, ``X``, ``Y``, ``Z``; qubit 0 first.

    Returns
    -------
    numpy.ndarray
        Complex matrix of shape ``(2**len(label), 2**len(label))``.
    """
    matrix = np.ones((1, 1), dtype=np.complex128)
    for character in label:
        matrix = np.kron(matrix, PAULIS[character])
    return matrix


def _label(n_qubits: int, operators: dict[int, str]) -> str:
    return "".join(operators.get(site, "I") for site in range(n_qubits))


def term_labels(
    term_set: str,
    n_qubits: int,
    include_identity: bool,
    explicit_labels: list[str] | None = None,
) -> list[str]:
    """Return the Pauli-string labels of one model Hamiltonian.

    Parameters
    ----------
    term_set : str
        One of ``"tfim"``, ``"ising"``, ``"heisenberg"``, ``"fcim"``, or
        ``"tfim_global"`` (TFIM plus the global strings X^n and Z^n, the
        control term set of extension E10), ``"ising_x0"`` (Ising couplings,
        X on qubit 0 and Z on the other qubits: non-commuting but Kirby-Love
        noncontextual, extension E15) or ``"explicit"`` (the strings given
        in ``explicit_labels``).
    n_qubits : int
        Number of qubits, at least 2.
    include_identity : bool
        Append the identity term (the bias ``b`` of Eqs. (113)-(114)). The
        paper uses it for the squared-loss experiments and removes it for the
        logistic-loss experiments (p. 21).

    Returns
    -------
    list of str
        Labels in the order used for the parameter vector theta.

    Raises
    ------
    ValueError
        If ``term_set`` is unknown or ``n_qubits < 2``.
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}")
    if (term_set == "explicit") != (explicit_labels is not None):
        raise ValueError(
            "explicit_labels must be given exactly when term_set is 'explicit'"
        )
    nearest_pairs = [(site, site + 1) for site in range(n_qubits - 1)]
    if term_set in ("tfim", "tfim_global"):
        labels = [_label(n_qubits, {i: "Z", j: "Z"}) for i, j in nearest_pairs]
        labels += [_label(n_qubits, {i: "X"}) for i in range(n_qubits)]
        if term_set == "tfim_global":
            labels += ["X" * n_qubits, "Z" * n_qubits]
    elif term_set == "ising":
        labels = [_label(n_qubits, {i: "Z", j: "Z"}) for i, j in nearest_pairs]
        labels += [_label(n_qubits, {i: "Z"}) for i in range(n_qubits)]
    elif term_set == "heisenberg":
        # Order follows logloss_nqubits.ipynb: all XX, then YY, then ZZ
        # couplings, then all X, Y, Z fields.
        labels = [
            _label(n_qubits, {i: pauli, j: pauli})
            for pauli in "XYZ"
            for i, j in nearest_pairs
        ]
        labels += [
            _label(n_qubits, {i: pauli}) for pauli in "XYZ" for i in range(n_qubits)
        ]
    elif term_set == "ising_x0":
        labels = [_label(n_qubits, {i: "Z", j: "Z"}) for i, j in nearest_pairs]
        labels += [_label(n_qubits, {0: "X"})]
        labels += [_label(n_qubits, {i: "Z"}) for i in range(1, n_qubits)]
    elif term_set == "explicit":
        labels = list(explicit_labels)
        invalid = [
            label
            for label in labels
            if len(label) != n_qubits or set(label) - set("IXYZ") or set(label) == {"I"}
        ]
        if invalid or len(set(labels)) != len(labels):
            raise ValueError(
                f"explicit_labels must be distinct non-identity {n_qubits}-qubit Pauli strings, got {labels}"
            )
    elif term_set == "fcim":
        labels = [
            _label(n_qubits, {i: "Z", j: "Z"})
            for i, j in itertools.combinations(range(n_qubits), 2)
        ]
        labels += [_label(n_qubits, {i: "Z"}) for i in range(n_qubits)]
    else:
        raise ValueError(f"term_set must be one of {TERM_SETS}, got {term_set!r}")
    if include_identity:
        labels.append("I" * n_qubits)
    return labels


def build_terms(
    term_set: str,
    n_qubits: int,
    include_identity: bool,
    explicit_labels: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return labels and stacked matrices ``H_j`` of a model Hamiltonian.

    Returns
    -------
    labels : list of str
        Pauli-string labels.
    terms : numpy.ndarray
        Array of shape ``(J, 2**n, 2**n)``.
    """
    labels = term_labels(term_set, n_qubits, include_identity, explicit_labels)
    return labels, np.stack([pauli_string(label) for label in labels])


def build_sparse_terms(
    term_set: str,
    n_qubits: int,
    include_identity: bool,
    explicit_labels: list[str] | None = None,
) -> tuple[list[str], list]:
    """Return labels and sparse CSR matrices ``H_j`` of a model Hamiltonian.

    Each Pauli string has exactly ``2**n`` non-zero entries, so the sparse
    form avoids the ``(J, 2**n, 2**n)`` dense stack (6 GB for the 12-qubit
    TFIM) in the extension studies.
    """
    from scipy import sparse

    labels = term_labels(term_set, n_qubits, include_identity, explicit_labels)
    matrices = []
    for label in labels:
        matrix = sparse.identity(1, dtype=np.complex128, format="csr")
        for character in label:
            matrix = sparse.kron(
                matrix, sparse.csr_matrix(PAULIS[character]), format="csr"
            )
        matrices.append(matrix)
    return labels, matrices


def hamiltonian(theta: np.ndarray, terms: np.ndarray) -> np.ndarray:
    """Return ``H(theta) = sum_j theta_j H_j`` (Eq. (16))."""
    return np.einsum("j,jab->ab", theta, terms)


def is_commuting(terms: np.ndarray, atol: float = 1e-12) -> bool:
    """Return whether all terms pairwise commute (the classical case, Eq. (12))."""
    for left, right in itertools.combinations(terms, 2):
        if not np.allclose(left @ right, right @ left, atol=atol):
            return False
    return True
