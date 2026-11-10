"""Gradient-variance (barren-plateau) study of the Fermi-Dirac neuron, extension E10.

For random parameters ``theta ~ U[-a, a]`` the gradient of the output
``f(theta, rho) = Tr[g_T(H(theta)) rho]`` with respect to ``theta_j`` is

    d_j f = Re sum_ab F_ab (V^dag H_j V)_ab (V^dag rho V)_ba     (Daleckii-Krein)

with ``H(theta) = V diag(l) V^dag`` and ``F`` the divided differences of
``g_T`` on ``l``. One eigendecomposition per draw serves every temperature
and every input; a pure input ``psi`` costs O(d^2) because
``d_j f = psi~^dag (F o H~_j) psi~`` with ``psi~ = V^dag psi``.

Input ensembles (one record per cell; random kinds are redrawn for every theta
draw, so their variance is over theta and the input ensemble together):

* ``zero``: |0...0> (fixed)
* ``ghz``: (|0...0> + |1...1>)/sqrt(2) (fixed)
* ``product``: random product of single-qubit Haar states (redrawn)
* ``haar``: Haar-random pure state on all qubits (redrawn)
* ``ginibre``: random mixed state A A^dag / Tr (redrawn)

Data-induced concentration check: at one fixed theta, the variance of
``d_j f`` over Haar-random pure states equals

    Var = (Tr G^2 / d - (Tr G / d)^2) / (d + 1),  G = V (F o H~_j) V^dag,

for any Hermitian G (second moment of the Haar measure); the code compares
this with the empirical variance over ``data_variance_states`` states.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg

from .activations import divided_differences, make_activation
from .pauli import build_sparse_terms

logger = logging.getLogger(__name__)

INPUT_KINDS = ("zero", "ghz", "product", "haar", "ginibre")


def _haar_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    vector = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return vector / np.linalg.norm(vector)


def _input_state(kind: str, n_qubits: int, rng: np.random.Generator):
    """Return ``("pure", psi)`` or ``("mixed", rho)`` for one input draw."""
    dim = 2**n_qubits
    if kind == "zero":
        vector = np.zeros(dim, dtype=np.complex128)
        vector[0] = 1.0
        return "pure", vector
    if kind == "ghz":
        vector = np.zeros(dim, dtype=np.complex128)
        vector[0] = vector[-1] = 1.0 / np.sqrt(2.0)
        return "pure", vector
    if kind == "product":
        vector = np.ones(1, dtype=np.complex128)
        for _ in range(n_qubits):
            vector = np.kron(vector, _haar_vector(2, rng))
        return "pure", vector
    if kind == "haar":
        return "pure", _haar_vector(dim, rng)
    if kind == "ginibre":
        matrix = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        rho = matrix @ matrix.conj().T
        return "mixed", rho / np.trace(rho)
    raise ValueError(f"input kind must be one of {INPUT_KINDS}, got {kind!r}")


def hermitian_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition with LAPACK's MRRR driver (ZHEEVR).

    ``numpy.linalg.eigh`` uses the divide-and-conquer driver ZHEEVD, which in
    Apple Accelerate rejects its own workspace size at d = 4096 ("parameter
    number 10 had an illegal value"); this failed both attempts of the
    12-qubit E10 cells. ZHEEVR has no such limit and agrees with ZHEEVD to
    round-off (``tests/test_extensions.py``).
    """
    return scipy.linalg.eigh(matrix, driver="evr")


def _dense_hamiltonian(coefficients: np.ndarray, terms: list) -> np.ndarray:
    return sum(
        coefficient * term for coefficient, term in zip(coefficients, terms)
    ).toarray()


def _rotate_term(term, eigenvectors: np.ndarray) -> np.ndarray:
    """Return ``V^dag H_j V`` with one sparse and one dense product."""
    return eigenvectors.conj().T @ (term @ eigenvectors)


def _rotate(state, eigenvectors: np.ndarray):
    """Return the input in the eigenbasis of H: a vector or a density matrix."""
    kind, value = state
    if kind == "pure":
        return kind, eigenvectors.conj().T @ value
    return kind, eigenvectors.conj().T @ value @ eigenvectors


def _output(values_of_g: np.ndarray, rotated_state) -> float:
    """Return ``Tr[g(H) rho]`` from g on the spectrum and the rotated input."""
    kind, value = rotated_state
    weights = np.abs(value) ** 2 if kind == "pure" else np.real(np.diag(value))
    return float(values_of_g @ weights)


def _gradient(divided: np.ndarray, rotated_term: np.ndarray, rotated_state) -> float:
    """Return ``d/dtheta_j Tr[g(H) rho]`` (Daleckii-Krein) in the eigenbasis."""
    kind, value = rotated_state
    weighted = divided * rotated_term
    if kind == "pure":
        return float(np.real(value.conj() @ weighted @ value))
    return float(np.real(np.sum(weighted * value.T)))


def _expectation(observable: np.ndarray, state) -> float:
    kind, value = state
    if kind == "pure":
        return float(np.real(value.conj() @ observable @ value))
    return float(np.real(np.sum(observable * value.T)))


def bootstrap_variance_ci(
    values: np.ndarray, rng: np.random.Generator, resamples: int = 1000
) -> tuple[float, float]:
    """Return the 95 % percentile bootstrap interval of the sample variance."""
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    variances = values[indices].var(axis=1, ddof=1)
    low, high = np.percentile(variances, [2.5, 97.5])
    return float(low), float(high)


def temperatures_for(n_qubits: int, temperature_cfg: dict) -> list[tuple[str, float]]:
    """Return ``(rule, T)`` pairs: fixed values and ``c * sqrt(n)`` scalings."""
    pairs = [(f"fixed_{value:g}", float(value)) for value in temperature_cfg["fixed"]]
    pairs += [
        (f"sqrt_n_{c:g}", float(c) * float(np.sqrt(n_qubits)))
        for c in temperature_cfg["sqrt_n_factors"]
    ]
    return pairs


def variance_cell_sweep(
    term_set: str,
    include_identity: bool,
    n_qubits: int,
    init_range: float,
    temperature_cfg: dict,
    input_kinds: list[str],
    parameter_indices: list[int],
    draws: int,
    rng: np.random.Generator,
    target_range: float = 2.0,
    on_draw=None,
    explicit_labels: list[str] | None = None,
) -> list[dict]:
    """Return gradient statistics for every (T, input, parameter) cell of one Hamiltonian family.

    Parameters
    ----------
    term_set, include_identity : str, bool
        Hamiltonian family, see :func:`lib.pauli.term_labels`.
    n_qubits : int
        Number of qubits.
    init_range : float
        Half-width ``a`` of the uniform parameter distribution U[-a, a].
    temperature_cfg : dict
        Keys ``fixed`` (list of T) and ``sqrt_n_factors`` (list of c).
    input_kinds : list of str
        Input ensembles, see ``INPUT_KINDS``.
    parameter_indices : list of int
        Indices j whose gradients are recorded; negative indices count from
        the end, as in Python.
    draws : int
        Number of theta draws.
    rng : numpy.random.Generator
        Random stream for the target, theta, random inputs and the bootstrap.
    target_range : float
        Half-width of the target coefficients zeta ~ U[-r, r], drawn once per
        call, that define ``y = Tr[g_T(H*) rho]`` for the single-input
        squared loss ``L = (f - y)**2``. Default value is 2.0 (the paper's
        target range).
    on_draw : callable or None
        Called with the 1-based draw index after each draw (progress
        logging). Default value is None.
    explicit_labels : list of str or None
        Pauli strings when ``term_set`` is ``"explicit"``. Default value is
        None.

    Returns
    -------
    list of dict
        One record per (temperature rule, input kind, parameter, quantity)
        with ``quantity`` in {"output", "loss"} (``d_j f`` and ``d_j L``):
        sample mean, variance, its 95 % bootstrap interval and mean square.

    Raises
    ------
    ValueError
        If ``draws < 2``.
    """
    if draws < 2:
        raise ValueError(f"draws must be >= 2 to estimate a variance, got {draws}")
    labels, terms = build_sparse_terms(
        term_set, n_qubits, include_identity, explicit_labels
    )
    indices = sorted({index % len(labels) for index in parameter_indices})
    temperatures = temperatures_for(n_qubits, temperature_cfg)
    activations = {rule: make_activation("tanh", value) for rule, value in temperatures}
    zeta = rng.uniform(-target_range, target_range, size=len(labels))
    target_values, target_vectors = hermitian_eigh(_dense_hamiltonian(zeta, terms))
    target_observables = {
        rule: (target_vectors * activations[rule].value(target_values))
        @ target_vectors.conj().T
        for rule, _ in temperatures
    }
    samples = {
        (rule, kind, j, quantity): []
        for rule, _ in temperatures
        for kind in input_kinds
        for j in indices
        for quantity in ("output", "loss")
    }
    fixed_states = {
        kind: _input_state(kind, n_qubits, rng)
        for kind in input_kinds
        if kind in ("zero", "ghz")
    }
    for draw in range(1, draws + 1):
        theta = rng.uniform(-init_range, init_range, size=len(labels))
        eigenvalues, eigenvectors = hermitian_eigh(_dense_hamiltonian(theta, terms))
        rotated_terms = {j: _rotate_term(terms[j], eigenvectors) for j in indices}
        states = {
            kind: fixed_states.get(kind) or _input_state(kind, n_qubits, rng)
            for kind in input_kinds
        }
        rotated_states = {
            kind: _rotate(state, eigenvectors) for kind, state in states.items()
        }
        for rule, _ in temperatures:
            activation = activations[rule]
            divided = divided_differences(activation, eigenvalues)
            values_of_g = activation.value(eigenvalues)
            for kind, rotated in rotated_states.items():
                residual = _output(values_of_g, rotated) - _expectation(
                    target_observables[rule], states[kind]
                )
                for j in indices:
                    output_gradient = _gradient(divided, rotated_terms[j], rotated)
                    samples[(rule, kind, j, "output")].append(output_gradient)
                    samples[(rule, kind, j, "loss")].append(
                        2.0 * residual * output_gradient
                    )
        if on_draw is not None:
            on_draw(draw)
    temperature_values = dict(temperatures)
    records = []
    for (rule, kind, j, quantity), values in samples.items():
        array = np.array(values)
        low, high = bootstrap_variance_ci(array, rng)
        records.append(
            {
                "term_set": term_set,
                "include_identity": include_identity,
                "n_qubits": n_qubits,
                "init_range": init_range,
                "temperature_rule": rule,
                "temperature": temperature_values[rule],
                "input": kind,
                "parameter_index": j,
                "parameter_label": labels[j],
                "quantity": quantity,
                "draws": draws,
                "mean": float(array.mean()),
                "variance": float(array.var(ddof=1)),
                "variance_ci95": [low, high],
                "mean_square": float(np.mean(array**2)),
            }
        )
    return records


def data_variance_check(
    term_set: str,
    include_identity: bool,
    n_qubits: int,
    init_range: float,
    temperature: float,
    parameter_index: int,
    n_states: int,
    rng: np.random.Generator,
    explicit_labels: list[str] | None = None,
) -> dict:
    """Compare the empirical Haar-data variance of ``d_j f`` with the exact formula at one theta."""
    labels, terms = build_sparse_terms(
        term_set, n_qubits, include_identity, explicit_labels
    )
    j = parameter_index % len(labels)
    dim = 2**n_qubits
    theta = rng.uniform(-init_range, init_range, size=len(labels))
    eigenvalues, eigenvectors = hermitian_eigh(_dense_hamiltonian(theta, terms))
    divided = divided_differences(make_activation("tanh", temperature), eigenvalues)
    generator = divided * (_rotate_term(terms[j], eigenvectors))
    trace = float(np.real(np.trace(generator)))
    trace_square = float(np.real(np.sum(np.abs(generator) ** 2)))
    predicted = (trace_square / dim - (trace / dim) ** 2) / (dim + 1)
    vectors = rng.standard_normal((n_states, dim)) + 1j * rng.standard_normal(
        (n_states, dim)
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rotated = vectors @ eigenvectors.conj()
    values = np.real(np.einsum("ma,ab,mb->m", rotated.conj(), generator, rotated))
    return {
        "term_set": term_set,
        "n_qubits": n_qubits,
        "temperature": temperature,
        "parameter_label": labels[j],
        "n_states": n_states,
        "predicted_mean": trace / dim,
        "empirical_mean": float(values.mean()),
        "predicted_variance": predicted,
        "empirical_variance": float(values.var(ddof=1)),
    }
