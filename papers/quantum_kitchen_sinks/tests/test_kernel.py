"""Check the simulator against the paper's closed-form implicit kernel.

arXiv:1806.08321 derives, for the two-qubit CNOT ansatz of Fig. 2(a),

    k(u, v) = 1/2
              + (1/8)  exp(-(1/2) sigma^2 ||u^(1) - v^(1)||^2)
              + (1/16) exp(-(1/2) sigma^2 ||u      - v     ||^2)

where u^(1) is the input component driving the *control* qubit.  The kernel is
    k(u, v) = E_{Omega, beta} [ phi(u) . phi(v) ]
with phi the measured bit vector, so with exact outcome probabilities it is
    k(u, v) = E_{Omega, beta} [ sum_q P(bit_q = 1 | u) P(bit_q = 1 | v) ].

This ties the whole gate-model path (RX layer, entangler, Z-basis readout) to a
quantity the paper states in closed form, which no accuracy number can do.
"""

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (sets sys.path for `lib`)
from lib.circuits import make_ansatz_probs, qubit_marginals

SIGMA_CASES = [
    ((0.3, -0.8), (-0.5, 0.4), 1.0),
    ((1.2, 0.1), (0.9, -1.1), 0.5),
    ((-0.2, 0.7), (0.7, 0.7), 2.0),
]


def paper_kernel_cnot2(u: np.ndarray, v: np.ndarray, sigma: float) -> float:
    """Closed form of Eq. for Fig. 2(a); control qubit is qubit 0."""
    d_control = (u[0] - v[0]) ** 2
    d_all = float(((u - v) ** 2).sum())
    return (
        0.5
        + 0.125 * np.exp(-0.5 * sigma**2 * d_control)
        + (1.0 / 16.0) * np.exp(-0.5 * sigma**2 * d_all)
    )


def mc_kernel(
    name: str,
    n_qubits: int,
    u: np.ndarray,
    v: np.ndarray,
    sigma: float,
    n_draws: int = 50_000,
    seed: int = 0,
) -> float:
    """Monte-Carlo estimate of the implicit kernel over draws of (Omega, beta).

    Uses the split encoding with the identity assignment (input dim q drives
    qubit q), which is the assignment the paper's closed form assumes.
    """
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, sigma, size=(n_draws, n_qubits))
    beta = rng.uniform(0.0, 2.0 * np.pi, size=(n_draws, n_qubits))
    probs_of = make_ansatz_probs(name, n_qubits)

    def marginals(x: np.ndarray) -> np.ndarray:
        theta = omega * np.asarray(x)[None, :] + beta
        return qubit_marginals(probs_of(theta, 1), n_qubits)

    return float((marginals(u) * marginals(v)).sum(axis=1).mean())


@pytest.mark.parametrize("u,v,sigma", SIGMA_CASES)
def test_cnot2_implicit_kernel_matches_paper(u, v, sigma):
    u = np.array(u)
    v = np.array(v)
    got = mc_kernel("cnot2", 2, u, v, sigma)
    want = paper_kernel_cnot2(u, v, sigma)
    # 50k draws give a standard error around 1.2e-3; 6e-3 is a safe band.
    assert got == pytest.approx(want, abs=6e-3), (
        f"cnot2 implicit kernel {got:.5f} != paper closed form {want:.5f}"
    )


def test_kernel_is_one_half_at_maximal_separation():
    """Both exponentials vanish as ||u - v|| -> infinity, leaving 1/2."""
    u = np.array([0.0, 0.0])
    v = np.array([40.0, 40.0])
    assert mc_kernel("cnot2", 2, u, v, 1.0) == pytest.approx(0.5, abs=6e-3)


def test_marginals_sum_consistently():
    probs_of = make_ansatz_probs("cnot4", 4)
    theta = np.random.default_rng(3).uniform(0, 2 * np.pi, size=(64, 4))
    probs = probs_of(theta, 1)
    assert np.allclose(probs.sum(axis=1), 1.0)
    marg = qubit_marginals(probs, 4)
    assert marg.shape == (64, 4)
    assert np.all((marg >= -1e-12) & (marg <= 1 + 1e-12))


def test_cz2_implicit_kernel_is_constant_one_half():
    """Fig. 2(b): the paper derives the constant kernel k(u, v) = 1/2.

    ``CZ|++>`` is maximally entangled, so each qubit's reduced state is
    maximally mixed and the subsequent ``RX(theta_i)`` cannot make the outcome
    distribution depend on the input at all.  Every bit is a fair coin, hence
    the constant kernel and the paper's "no better than random".
    """
    for u, v, sigma in SIGMA_CASES:
        got = mc_kernel("cz2", 2, np.array(u), np.array(v), sigma)
        assert got == pytest.approx(0.5, abs=6e-3), (
            f"cz2 implicit kernel {got:.5f} != 1/2 for u={u}, v={v}, sigma={sigma}"
        )


def test_cz2_marginals_are_exactly_one_half():
    """The mechanism behind Fig. 2(b), checked exactly rather than by sampling.

    Every single-qubit marginal is exactly 1/2 for every theta, so each QKS
    feature bit is a fair coin carrying no information about the input.  The
    *joint* distribution does still depend on theta -- the input dependence
    survives in the bit-bit correlations -- but the QKS feature vector is the
    raw bits and the classifier on top is linear (the paper's Linear Baseline
    rule), so nothing downstream can reach that correlation.  This is the
    precise sense in which the CZ ansatz is "no better than random".
    """
    probs_of = make_ansatz_probs("cz2", 2)
    theta = np.random.default_rng(11).uniform(-6.0, 6.0, size=(256, 2))
    probs = probs_of(theta, 1)
    marg = qubit_marginals(probs, 2)
    assert np.allclose(marg, 0.5, atol=1e-12)
    # ... while the joint distribution is *not* uniform, i.e. the information
    # is present but inaccessible to a linear model on the raw bits.
    assert not np.allclose(probs, 0.25, atol=1e-6)


def test_cz2_ordering_is_what_makes_it_degenerate():
    """Guard against silently reverting to `RX then CZ`, which is informative.

    With the rotations first a diagonal entangler cannot change Z-basis
    marginals at all, so the ansatz would collapse to two independent
    single-qubit QKS circuits and would *not* reproduce Fig. 2(b).
    """
    from lib.circuits import entangler_precedes_rotations

    assert entangler_precedes_rotations("cz2")
    assert not entangler_precedes_rotations("cnot2")
