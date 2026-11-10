"""Numerical checks of the paper's identities and algorithms (experiments E0a-E0d).

Every formula below is transcribed from the page screenshots in
``data/fermi_dirac_neurons/sources/screenshots/pages/`` (page numbers given
next to each function). The check compares the printed right-hand side with
an exact left-hand side computed by diagonalization.

Notation: ``U_x(rho) = exp(-i H x) rho exp(i H x)`` (Eq. (22));
``E_{t~p, s~U[0,1]}[...]`` integrals over the kernels

* ``mu(t) = t / (2 sinh(pi t / 2))`` (Eq. (21)),
* ``gamma(t) = (2/pi) ln|coth(pi t / 2)|`` (Eq. (65)),
* ``xi(t) = (gamma(t) + mu(t)) / 2`` (Eq. (78)).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate
from scipy.special import erf, expit
from scipy.stats import norm

from .activations import divided_differences, make_activation, matrix_function
from .pauli import build_terms, pauli_string

_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(64)
# Map Gauss-Legendre nodes from [-1, 1] to [0, 1].
UNIT_NODES = 0.5 * (_GAUSS_NODES + 1.0)
UNIT_WEIGHTS = 0.5 * _GAUSS_WEIGHTS


def mu_density(t):
    """Eq. (21): ``mu(t) = t / (2 sinh(pi t / 2))``, with ``mu(0) = 1/pi``."""
    t = np.asarray(t, dtype=float)
    small = np.abs(t) < 1e-12
    x = np.pi * np.abs(np.where(small, 1.0, t)) / 2.0
    # t / (2 sinh x) = |t| e^{-x} / (1 - e^{-2x}); no overflow for large |t|.
    value = np.abs(t) * np.exp(-x) / (-np.expm1(-2.0 * x))
    return np.where(small, 1.0 / np.pi, value)


def gamma_density(t):
    """Eq. (65): ``gamma(t) = (2/pi) ln|coth(pi t / 2)|`` (log-singular at 0)."""
    x = np.abs(np.asarray(t, dtype=float)) * np.pi / 2.0
    # ln coth(x) = -ln tanh(x), written to stay accurate for large x.
    return (2.0 / np.pi) * -np.log(np.tanh(x))


def xi_density(t):
    """Eq. (78): ``xi(t) = (gamma(t) + mu(t)) / 2``."""
    return 0.5 * (gamma_density(t) + mu_density(t))


DENSITIES = {"mu": mu_density, "gamma": gamma_density, "xi": xi_density}


def density_normalization(name: str) -> float:
    """Return ``int p(t) dt`` over the real line (should be 1)."""
    pdf = DENSITIES[name]
    half, _ = integrate.quad(lambda t: float(pdf(t)), 0.0, np.inf, limit=400)
    return 2.0 * half


def _trace_kernel(
    generator: np.ndarray,
    left_operator: np.ndarray,
    rho: np.ndarray,
    time_scale: float,
    density: str,
    s_power: int,
) -> float:
    """Return ``E_{t~density, s~U}[s**p Re Tr[A e^{i G t/tau} U^G_{s t/tau}(rho)]]``.

    ``A = left_operator``, ``G = generator``, ``tau = time_scale``. In the
    eigenbasis of G the trace is
    ``sum_ab A_ba rho_ab exp(i t/tau (l_a (1 - s) + l_b s))``; s is integrated
    with 64-point Gauss-Legendre and t with adaptive quadrature.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    rotated_a = eigenvectors.conj().T @ left_operator @ eigenvectors
    rotated_rho = eigenvectors.conj().T @ rho @ eigenvectors
    coefficients = rotated_a.T * rotated_rho  # C_ab = A_ba rho_ab
    left = eigenvalues[:, None, None]
    right = eigenvalues[None, :, None]
    s = UNIT_NODES[None, None, :]
    frequencies = (left * (1.0 - s) + right * s) / time_scale  # (d, d, n_s)
    s_weights = UNIT_WEIGHTS * UNIT_NODES**s_power
    pdf = DENSITIES[density]

    def integrand(t: float) -> float:
        phases = np.exp(1j * t * frequencies) @ s_weights
        return float(np.real(np.sum(coefficients * phases)) * pdf(t))

    options = {"limit": 400, "epsabs": 1e-13, "epsrel": 1e-11}
    negative, _ = integrate.quad(integrand, -np.inf, 0.0, **options)
    positive, _ = integrate.quad(integrand, 0.0, np.inf, **options)
    return negative + positive


def exact_trace_gradient(
    activation_name: str,
    temperature: float,
    theta: np.ndarray,
    terms: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Return ``d/dtheta_j Tr[phi(H(theta)) rho]`` for every j (Daleckii-Krein)."""
    activation = make_activation(activation_name, temperature)
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("j,jab->ab", theta, terms))
    divided = divided_differences(activation, eigenvalues)
    rotated_terms = eigenvectors.conj().T @ terms @ eigenvectors
    rotated_rho = eigenvectors.conj().T @ rho @ eigenvectors
    return np.real(np.einsum("ab,jab,ba->j", divided, rotated_terms, rotated_rho))


def exact_trace(
    activation_name: str, temperature: float, hamiltonian: np.ndarray, rho: np.ndarray
) -> float:
    """Return ``Tr[phi(H) rho]``."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    observable = matrix_function(
        make_activation(activation_name, temperature), eigenvalues, eigenvectors
    )
    return float(np.real(np.trace(observable @ rho)))


def random_problem(
    n_qubits: int, term_set: str, include_identity: bool, rng: np.random.Generator
):
    """Return random ``theta ~ U[-1, 1]``, the terms, and a random mixed state."""
    _, terms = build_terms(term_set, n_qubits, include_identity)
    theta = rng.uniform(-1.0, 1.0, size=len(terms))
    dim = 2**n_qubits
    matrix = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = matrix @ matrix.conj().T
    return theta, terms, rho / np.trace(rho)


def _record(lhs, rhs) -> dict:
    lhs = np.atleast_1d(np.asarray(lhs, dtype=float))
    rhs = np.atleast_1d(np.asarray(rhs, dtype=float))
    return {
        "lhs": lhs.tolist(),
        "rhs": rhs.tolist(),
        "max_abs_error": float(np.max(np.abs(lhs - rhs))),
    }


# --- E0a: classical reduction (Eqs. (7)-(8), (12)-(15), p. 3 and p. 6) -------


def check_classical_reduction(
    n_qubits: int, activation_name: str, temperature: float, rng
) -> dict:
    """Check ``Tr[phi(H_C) rho] = sum_z p(z) phi(z^T W z + w^T z + b)`` (Eq. (15))."""
    dim = 2**n_qubits
    weights = rng.uniform(-1.0, 1.0, size=(n_qubits, n_qubits))
    fields = rng.uniform(-1.0, 1.0, size=n_qubits)
    bias = float(rng.uniform(-1.0, 1.0))
    hamiltonian = bias * np.eye(dim, dtype=np.complex128)
    for i in range(n_qubits):
        hamiltonian += fields[i] * pauli_string(
            "".join("Z" if k == i else "I" for k in range(n_qubits))
        )
        for j in range(n_qubits):
            label = "".join(
                "Z" if k in (i, j) and i != j else "I" for k in range(n_qubits)
            )
            hamiltonian += weights[i, j] * pauli_string(label)
    _, _, rho = random_problem(n_qubits, "ising", False, rng)
    lhs = exact_trace(activation_name, temperature, hamiltonian, rho)
    activation = make_activation(activation_name, temperature)
    rhs = 0.0
    for index in range(dim):
        # Basis index bit k (qubit 0 most significant) 0 -> z = +1, 1 -> z = -1.
        z = np.array(
            [1.0 - 2.0 * ((index >> (n_qubits - 1 - k)) & 1) for k in range(n_qubits)]
        )
        energy = z @ weights @ z + fields @ z + bias
        rhs += float(np.real(rho[index, index])) * float(
            activation.value(np.array(energy))
        )
    return _record(lhs, rhs)


# --- E0b: integral formulas ------------------------------------------------


def check_theorem1(theta, terms, rho, temperature) -> dict:
    """Theorem 1, Eq. (20), p. 7:
    ``d_j Tr[g_T(H) rho] = (1/T) E_{t~mu, s}[Re Tr[H_j e^{iHt/T} U_{st/T}(rho)]]``."""
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    lhs = exact_trace_gradient("tanh", temperature, theta, terms, rho)
    rhs = [
        _trace_kernel(hamiltonian, term, rho, temperature, "mu", 0) / temperature
        for term in terms
    ]
    return _record(lhs, rhs)


def _truncated_parameters(theta: np.ndarray, j: int, lam: float) -> np.ndarray:
    """Eq. (26): ``theta^(j)(lambda) = (0, ..., 0, lambda theta_j, theta_{j+1}, ...)``."""
    truncated = np.array(theta, dtype=float)
    truncated[:j] = 0.0
    truncated[j] = lam * theta[j]
    return truncated


def check_theorem2(theta, terms, rho, temperature, lambda_nodes: int = 24) -> dict:
    """Theorem 2, Eq. (23), p. 7: the telescoping objective formula.

    ``Tr[g_T(H) rho] = (|theta|_1 / T) E_{j~q, t~mu, s, lambda}[sgn(theta_j)
    Re Tr[H_j e^{iH(j,lambda)t/T} U^{H(j,lambda)}_{st/T}(rho)]]``; with
    ``q(j) = |theta_j| / |theta|_1`` this equals
    ``(1/T) sum_j theta_j int_0^1 dlambda E_{t,s}[...]``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(lambda_nodes)
    nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    lhs = exact_trace("tanh", temperature, hamiltonian, rho)
    rhs = 0.0
    for j, term in enumerate(terms):
        for lam, weight in zip(nodes, weights):
            truncated = np.einsum(
                "k,kab->ab", _truncated_parameters(theta, j, lam), terms
            )
            rhs += (
                theta[j]
                * weight
                * _trace_kernel(truncated, term, rho, temperature, "mu", 0)
            )
    return _record(lhs, rhs / temperature)


def check_theorem5(theta, terms, rho, temperature, label: float) -> dict:
    """Theorem 5, Eq. (63), p. 12: logistic-loss gradient.

    ``d_j T Tr[ln(I + e^{-y H/T}) rho] = -(y/2) Tr[H_j rho] + (1/(2T))
    E_{s, t~gamma}[s Re Tr[H H_j e^{i y H t/T} U^{yH}_{st/T}(rho)]]``
    (``|theta|_1 E_k sgn(theta_k) H_k = H``).
    """
    name = "logistic_pos" if label > 0 else "logistic_neg"
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    lhs = exact_trace_gradient(name, temperature, theta, terms, rho)
    rhs = [
        -0.5 * label * float(np.real(np.trace(term @ rho)))
        + _trace_kernel(
            label * hamiltonian, hamiltonian @ term, rho, temperature, "gamma", 1
        )
        / (2.0 * temperature)
        for term in terms
    ]
    return _record(lhs, rhs)


def check_theorem6(theta, terms, rho, temperature) -> dict:
    """Theorem 6, Eq. (71), p. 13: softplus gradient.

    ``d_j Tr[r_T(H) rho] = (1/2) Tr[H_j rho] + (1/(2T)) E_{s, t~gamma}[s Re
    Tr[H H_j e^{iHt/T} U_{st/T}(rho)]]``.
    """
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    lhs = exact_trace_gradient("softplus", temperature, theta, terms, rho)
    rhs = [
        0.5 * float(np.real(np.trace(term @ rho)))
        + _trace_kernel(hamiltonian, hamiltonian @ term, rho, temperature, "gamma", 1)
        / (2.0 * temperature)
        for term in terms
    ]
    return _record(lhs, rhs)


def check_theorem8_as_printed(theta, terms, rho, temperature) -> dict:
    """Theorem 8, Eq. (76), p. 14, as printed: SiLU gradient.

    ``d_j Tr[SiLU_T(H) rho] = (1/2) Tr[H_j rho] + (1/(2T)) E_{t~xi, s}[s Re
    Tr[H H_j e^{iHt/(2T)} U_{st/(2T)}(rho)]]``. LOG.md D14 predicts a mismatch.
    """
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    lhs = exact_trace_gradient("silu", temperature, theta, terms, rho)
    rhs = [
        0.5 * float(np.real(np.trace(term @ rho)))
        + _trace_kernel(
            hamiltonian, hamiltonian @ term, rho, 2.0 * temperature, "xi", 1
        )
        / (2.0 * temperature)
        for term in terms
    ]
    return _record(lhs, rhs)


def check_theorem13(
    theta, terms, rho, temperature, label: float, lambda_nodes: int = 24
) -> dict:
    """Theorem 13, Eq. (C44), p. 46: logistic-loss objective, printed and corrected.

    Printed: ``T Tr[ln(I + e^{-yH/T}) rho] = -(y/2) Tr[H rho] + (J/(2T))
    E_{j~[J], lambda, s, t~gamma}[|theta^(j)(lambda)|_1 E_{k~q_{j,lambda}}[s Re
    Tr[sgn(theta_k) H_k H_j e^{i y H^(j,lambda) t/T} U^{yH^(j,lambda)}_{st/T}(rho)]]]``.
    Because ``|theta^(j)(lambda)|_1 E_k sgn(theta_k) H_k = H^(j,lambda)``, the
    second term is ``(1/(2T)) sum_j int dlambda K_j(lambda)`` with
    ``K_j = E_{s,t~gamma}[s Re Tr[H^(j,lambda) H_j ...]]``.

    Corrected (chain rule ``d/dlambda H^(j,lambda) = theta_j H_j`` and the
    constant ``T Tr[ln(2 I) rho] = T ln 2`` at ``theta = 0``):
    ``T ln 2 - (y/2) Tr[H rho] + (1/(2T)) sum_j theta_j int dlambda K_j``.
    LOG.md D16.
    """
    nodes, weights = np.polynomial.legendre.leggauss(lambda_nodes)
    nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    name = "logistic_pos" if label > 0 else "logistic_neg"
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    lhs = exact_trace(name, temperature, hamiltonian, rho)
    linear_term = -0.5 * label * float(np.real(np.trace(hamiltonian @ rho)))
    printed, corrected_integral = 0.0, 0.0
    for j, term in enumerate(terms):
        for lam, weight in zip(nodes, weights):
            truncated = np.einsum(
                "k,kab->ab", _truncated_parameters(theta, j, lam), terms
            )
            kernel = _trace_kernel(
                label * truncated, truncated @ term, rho, temperature, "gamma", 1
            )
            printed += weight * kernel
            corrected_integral += theta[j] * weight * kernel
    printed_rhs = linear_term + printed / (2.0 * temperature)
    corrected_rhs = (
        temperature * math.log(2.0)
        + linear_term
        + corrected_integral / (2.0 * temperature)
    )
    return {
        "printed": _record(lhs, printed_rhs),
        "corrected": _record(lhs, corrected_rhs),
    }


def check_corollary1(theta, terms, lambda_nodes: int = 24) -> dict:
    """Corollary 1, Eq. (169), p. 28: log-partition function.

    ``ln Z(theta) = ln d - sum_j theta_j int_0^1 dlambda <H_j>_{rho(theta^(j)(lambda))}``
    with ``rho(theta) = e^{-H(theta)} / Z(theta)`` (Eqs. (156)-(157)).
    """
    nodes, weights = np.polynomial.legendre.leggauss(lambda_nodes)
    nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    dim = terms.shape[-1]
    eigenvalues = np.linalg.eigvalsh(np.einsum("j,jab->ab", theta, terms))
    lhs = float(np.log(np.sum(np.exp(-eigenvalues))))
    rhs = math.log(dim)
    for j, term in enumerate(terms):
        for lam, weight in zip(nodes, weights):
            values, vectors = np.linalg.eigh(
                np.einsum("k,kab->ab", _truncated_parameters(theta, j, lam), terms)
            )
            boltzmann = np.exp(-(values - values.min()))
            thermal = (vectors * (boltzmann / boltzmann.sum())) @ vectors.conj().T
            rhs -= theta[j] * weight * float(np.real(np.trace(term @ thermal)))
    return _record(lhs, rhs)


# --- E0d: convolution identities (scalar) ------------------------------------


def logistic_pdf(p, scale: float):
    """Eq. (38): ``l_T1(p) = e^{p/T1} / (T1 (e^{p/T1} + 1)**2)``."""
    return expit(p / scale) * expit(-p / scale) / scale


def check_convolution_identities(t1: float, t2: float, points: np.ndarray) -> dict:
    """Check Eqs. (41), (73), (90), (94) at ``points`` for both temperature conventions.

    Returns one record per identity and convention: ``T = 2 T1 T2`` (Alg. 4,
    p. 10; App. F3) and ``T = T1 T2`` (main-text sentences, pp. 10, 16; App.
    D40, F37).
    """
    records = {}
    logistic = lambda u: logistic_pdf(u, t1)  # noqa: E731
    gaussian = lambda u: norm.pdf(u, scale=t1)  # noqa: E731

    def sign_convolution(pdf, x):
        below, _ = integrate.quad(pdf, -np.inf, x, limit=400)
        above, _ = integrate.quad(pdf, x, np.inf, limit=400)
        return below - above

    def relu_convolution(pdf, x):
        value, _ = integrate.quad(lambda u: pdf(u) * (x - u), -np.inf, x, limit=400)
        return value

    lhs41 = [sign_convolution(logistic, p / t2) for p in points]
    lhs90 = [sign_convolution(gaussian, p / t2) for p in points]
    lhs73 = [t2 * relu_convolution(logistic, p / t2) for p in points]
    lhs94 = [t2 * relu_convolution(gaussian, p / t2) for p in points]
    for convention, temperature in (("T=2*T1*T2", 2 * t1 * t2), ("T=T1*T2", t1 * t2)):
        records[f"eq41_tanh_{convention}"] = _record(
            lhs41, np.tanh(points / temperature)
        )
        records[f"eq90_erf_{convention}"] = _record(
            lhs90, erf(np.sqrt(2.0) * points / temperature)
        )
        records[f"eq73_softplus_{convention}"] = _record(
            lhs73, temperature * np.logaddexp(0.0, points / temperature)
        )
        records[f"eq94_grelu_{convention}"] = _record(
            lhs94,
            points * norm.cdf(points / temperature)
            + temperature * norm.pdf(points / temperature),
        )
    return records


# --- E0c/E0d: Monte Carlo of the sampling and single-shot algorithms ----------


class TabulatedDensity:
    """Inverse-CDF sampler for a symmetric kernel density on ``[-limit, limit]``."""

    def __init__(self, name: str, limit: float = 60.0) -> None:
        pdf = DENSITIES[name]
        positive = np.concatenate(
            [np.geomspace(1e-14, 1e-2, 4000), np.linspace(1e-2, limit, 400000)[1:]]
        )
        grid = np.concatenate([-positive[::-1], [0.0], positive])
        values = pdf(np.where(grid == 0.0, 1e-300, grid)) if name != "mu" else pdf(grid)
        values = np.nan_to_num(values, posinf=0.0)
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))]
        )
        self.normalization = float(cdf[-1])
        self.grid = grid
        self.cdf = cdf / cdf[-1]

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` samples."""
        return np.interp(rng.uniform(size=size), self.cdf, self.grid)


def _hadamard_test_outcome(
    hamiltonian: np.ndarray,
    measured: np.ndarray,
    rho: np.ndarray,
    t: float,
    s: float,
    temperature: float,
    rng,
) -> float:
    """Sample ``Z * X`` from the circuit of Fig. 2(a).

    The ancilla starts in |0>, gets a Hadamard, controls ``U = e^{iHt/T}`` on
    ``sigma = U_{st/T}(rho)``, gets a Hadamard and is measured in Z; the
    system is measured in the eigenbasis of ``H_j``. For ancilla outcome z the
    unnormalized system state is ``(I + zU) sigma (I + zU)^dag / 4``, so
    ``E[Z X] = Re Tr[H_j U sigma]``.
    """
    values, vectors = np.linalg.eigh(hamiltonian)
    evolve_back = (
        vectors * np.exp(-1j * values * s * t / temperature)
    ) @ vectors.conj().T
    sigma = evolve_back @ rho @ evolve_back.conj().T
    unitary = (vectors * np.exp(1j * values * t / temperature)) @ vectors.conj().T
    measured_values, measured_vectors = np.linalg.eigh(measured)
    identity = np.eye(len(rho))
    outcomes, probabilities = [], []
    for z in (1.0, -1.0):
        branch = (
            (identity + z * unitary) @ sigma @ (identity + z * unitary).conj().T / 4.0
        )
        diagonal = np.real(
            np.einsum("ai,ab,bi->i", measured_vectors.conj(), branch, measured_vectors)
        )
        outcomes.extend(z * measured_values)
        probabilities.extend(np.clip(diagonal, 0.0, None))
    probabilities = np.array(probabilities) / np.sum(probabilities)
    return float(rng.choice(outcomes, p=probabilities))


def simulate_algorithm1(
    theta, terms, rho, temperature, j: int, shots: int, rng, sampler: TabulatedDensity
) -> np.ndarray:
    """Algorithm 1, p. 8: samples ``Y_k = (1/T) Z_k X_k``; their mean estimates d_j Tr[g_T(H) rho]."""
    hamiltonian = np.einsum("k,kab->ab", theta, terms)
    times = sampler.sample(shots, rng)
    s_values = rng.uniform(size=shots)
    return np.array(
        [
            _hadamard_test_outcome(hamiltonian, terms[j], rho, t, s, temperature, rng)
            / temperature
            for t, s in zip(times, s_values)
        ]
    )


def simulate_algorithm2(
    theta, terms, rho, temperature, shots: int, rng, sampler: TabulatedDensity
) -> np.ndarray:
    """Algorithm 2, p. 8: samples ``W_k = (|theta|_1/T) sgn(theta_j) Z_k X_k``."""
    norm1 = float(np.sum(np.abs(theta)))
    probabilities = np.abs(theta) / norm1
    samples = np.empty(shots)
    indices = rng.choice(len(theta), size=shots, p=probabilities)
    times = sampler.sample(shots, rng)
    s_values = rng.uniform(size=shots)
    lambdas = rng.uniform(size=shots)
    for k, (j, t, s, lam) in enumerate(zip(indices, times, s_values, lambdas)):
        truncated = np.einsum("i,iab->ab", _truncated_parameters(theta, j, lam), terms)
        outcome = _hadamard_test_outcome(
            truncated, terms[j], rho, t, s, temperature, rng
        )
        samples[k] = norm1 / temperature * np.sign(theta[j]) * outcome
    return samples


def _sample_eigenvalues(hamiltonian, rho, shots, rng) -> np.ndarray:
    """Sample eigenvalues of H with probabilities ``<v|rho|v>``.

    Measuring the control qumode(s) after ``exp(i x (x) H / T2)`` projects the
    data register on an eigenspace of H; each eigenvalue E occurs with
    probability ``Tr[Pi_E rho]`` and shifts the control momentum by ``E / T2``
    (``e^{-iax} p e^{iax} = p + a``).
    """
    values, vectors = np.linalg.eigh(hamiltonian)
    probabilities = np.clip(
        np.real(np.einsum("ai,ab,bi->i", vectors.conj(), rho, vectors)), 0.0, None
    )
    return rng.choice(values, size=shots, p=probabilities / probabilities.sum())


def simulate_single_shot_neurons(hamiltonian, rho, t1, t2, shots, rng) -> dict:
    """Simulate Algorithms 4, 5 and 7 (pp. 10, 13-15) and compare with exact traces.

    * Alg. 4: control |l_T1>, output sign(p) -> ``Tr[g_T(H) rho]``, ``T = 2 T1 T2``.
    * Alg. 5: output ``T2 max(p, 0)`` -> ``Tr[r_T(H) rho]``, ``T = T1 T2``.
    * Alg. 7: vacuum control (momentum density ``e^{-p^2}/sqrt(pi)``, Eq. (84)
      context) coupled by ``exp(i x (x) H)`` and |l_T1> by
      ``exp(i x (x) H / T2)``; output ``p1 1[p2 >= 0]`` ->
      ``Tr[SiLU_T(H) rho]``, ``T = T1 T2``.
    """
    energies = _sample_eigenvalues(hamiltonian, rho, shots, rng)
    logistic_noise = rng.logistic(0.0, t1, size=shots)
    momentum = logistic_noise + energies / t2
    vacuum_momentum = rng.normal(0.0, 1.0 / np.sqrt(2.0), size=shots) + energies
    second_momentum = rng.logistic(0.0, t1, size=shots) + energies / t2
    estimates = {
        "algorithm4_tanh": (
            np.where(momentum >= 0, 1.0, -1.0),
            exact_trace("tanh", 2 * t1 * t2, hamiltonian, rho),
        ),
        "algorithm5_softplus": (
            t2 * np.maximum(momentum, 0.0),
            exact_trace("softplus", t1 * t2, hamiltonian, rho),
        ),
        "algorithm7_silu": (
            vacuum_momentum * (second_momentum >= 0),
            exact_trace("silu", t1 * t2, hamiltonian, rho),
        ),
    }
    return {
        name: {
            "estimate": float(np.mean(samples)),
            "standard_error": float(np.std(samples, ddof=1) / np.sqrt(shots)),
            "exact": exact,
            "abs_error": float(abs(np.mean(samples) - exact)),
            "shots": int(shots),
        }
        for name, (samples, exact) in estimates.items()
    }
