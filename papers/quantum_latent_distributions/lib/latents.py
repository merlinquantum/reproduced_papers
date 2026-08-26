"""Latent distributions for deep generative models.

Reproduces the four latent sources compared in Bacarreza et al.,
arXiv:2508.19857:

============================  ==========================================
``BosonSamplerLatent``        indistinguishable photons -> quantum, in Q
``DistinguishableLatent``     same circuit, no interference -> classical
``BernoulliLatent``           uniform bit strings on {0,1}^L
``GaussianLatent``            N(0, I), the industry baseline
============================  ==========================================

All four subclass :class:`merlin.LatentDistribution`, so they are drop-in for
MerLin's :class:`merlin.PhotonicGenerator` and for any classical generator that
just needs ``sample(batch_size)``.

Design note
-----------
The paper pre-generates a large bank of quantum samples (1M for the size-48
runs) and draws latent batches from it during training, because sampling is the
expensive step and the latent distribution is *fixed* -- no gradients ever flow
into the photonic circuit.  We keep that structure: every sampler here is a
:class:`SampleBankLatent` over a pre-drawn bank, which also makes the simulated
and the real-hardware paths interchangeable (see :mod:`lib.hardware`).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import perceval as pcvl
import torch
from lib.circuits import delay_line_unitary, haar_unitary, to_circuit
from merlin import LatentDistribution

__all__ = [
    "CopulaBosonLatent",
    "DirichletMultinomialLatent",
    "NegativeBinomialLatent",
    "ShuffledBosonLatent",
    "CHALLENGER_LATENT_KINDS",
    "LATENT_KINDS",
    "PAPER_LATENT_KINDS",
    "SampleBankLatent",
    "BosonSamplerLatent",
    "DistinguishableLatent",
    "BernoulliLatent",
    "GaussianLatent",
    "build_latent",
    "sample_boson",
    "sample_distinguishable",
    "exact_distribution",
]


# --------------------------------------------------------------------------- #
# raw samplers
# --------------------------------------------------------------------------- #
def _input_state(m: int, n_photons: int) -> pcvl.BasicState:
    """``n_photons`` single photons in the first ``n_photons`` modes."""
    return pcvl.BasicState([1] * n_photons + [0] * (m - n_photons))


#: Directory used to cache latent banks. Perceval's Clifford & Clifford
#: sampler is not seed-reproducible (see ``README`` -> Limitations), so exact
#: reproducibility comes from reusing a cached bank rather than from the seed.
BANK_CACHE = Path(__file__).resolve().parent.parent / "models" / "banks"


def _cache_path(
    tag: str, unitary: np.ndarray, n_photons: int, n_samples: int, seed: int
) -> Path:
    """Cache filename keyed by everything that defines a bank."""
    digest = hashlib.sha1(
        np.ascontiguousarray(unitary).tobytes()
        + f"|{tag}|{n_photons}|{n_samples}|{seed}".encode()
    ).hexdigest()[:16]
    return (
        BANK_CACHE / f"{tag}_m{unitary.shape[0]}_n{n_photons}_s{n_samples}_{digest}.npy"
    )


def sample_boson(
    unitary: np.ndarray,
    n_photons: int,
    n_samples: int,
    seed: int = 0,
    chunk: int = 50_000,
    cache: bool = True,
) -> np.ndarray:
    """Sample photon-count patterns from a boson sampler.

    Uses Perceval's Clifford & Clifford 2017 sampler, which draws exact boson
    sampling outcomes in time polynomial in the number of modes without ever
    forming the full ``C(m + n - 1, n)``-dimensional distribution.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_samples, m)``: photons per output mode.
    """
    m = unitary.shape[0]
    path = _cache_path("boson", unitary, n_photons, n_samples, seed)
    if cache and path.exists():
        return np.load(path)

    proc = pcvl.Processor("CliffordClifford2017", m)
    proc.set_circuit(to_circuit(unitary))
    proc.with_input(_input_state(m, n_photons))
    proc.min_detected_photons_filter(0)
    pcvl.random_seed(seed)
    sampler = pcvl.algorithm.Sampler(proc)

    out = np.empty((n_samples, m), dtype=np.int16)
    done = 0
    while done < n_samples:
        k = min(chunk, n_samples - done)
        res = sampler.samples(k)["results"]
        out[done : done + k] = np.asarray([list(s) for s in res[:k]], dtype=np.int16)
        done += k

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
    return out


def sample_distinguishable(
    unitary: np.ndarray,
    n_photons: int,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample from the same circuit with *distinguishable* photons.

    Without interference each photon independently picks an output mode with
    probability ``|U[j, k]|^2`` for input mode ``k``.  This is the paper's key
    control: identical circuit, identical *mean occupancy*, no quantum
    interference, and classically samplable in linear time.

    Note that "identical marginals" -- the paper's framing -- is too strong.
    Only the first moment matches: this sampler anti-bunches where the boson
    sampler bunches, by a Fano-factor ratio of roughly 1.4 at half filling
    (1.28 against 0.89 at the Table I size of 16 modes and 8 photons). The two
    therefore differ in their single-mode marginals as well as in their
    correlations. See the classical challenger latents below.
    """
    m = unitary.shape[0]
    rng = np.random.default_rng(seed)
    probs = np.abs(unitary[:, :n_photons]) ** 2  # column k -> output distribution
    probs /= probs.sum(axis=0, keepdims=True)
    out = np.zeros((n_samples, m), dtype=np.int16)
    for k in range(n_photons):
        modes = rng.choice(m, size=n_samples, p=probs[:, k])
        np.add.at(out, (np.arange(n_samples), modes), 1)
    return out


def shuffle_columns(bank: np.ndarray, seed: int = 0) -> np.ndarray:
    """Independently permute each column of a sample bank.

    The result has *exactly* the same single-mode marginal distribution as the
    input in every mode -- not just the same mean -- while every cross-mode
    correlation is destroyed. It is a product distribution, so it is trivially
    samplable classically once the marginals are known.

    Parameters
    ----------
    bank : numpy.ndarray
        Array of shape ``(n_samples, n_modes)``.
    seed : int
        Seed for the permutations.

    Returns
    -------
    numpy.ndarray
        Array of the same shape, column-wise shuffled.
    """
    rng = np.random.default_rng(seed)
    out = np.array(bank, copy=True)
    for column in range(out.shape[1]):
        rng.shuffle(out[:, column])
    return out


def gaussian_copula_resample(
    bank: np.ndarray, n_samples: int, seed: int = 0
) -> np.ndarray:
    """Resample a bank through a Gaussian copula.

    Preserves every single-mode marginal exactly (values are drawn from the
    empirical quantiles) and reproduces the rank-correlation matrix, but imposes
    Gaussian dependence -- so all structure beyond second order is discarded.
    Fitting and sampling need only a covariance matrix and a sort, both cheap
    and entirely classical.

    Parameters
    ----------
    bank : numpy.ndarray
        Array of shape ``(n_bank, n_modes)``.
    n_samples : int
        Number of samples to draw.
    seed : int
        Seed for the generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_samples, n_modes)``.
    """
    rng = np.random.default_rng(seed)
    n_bank, n_modes = bank.shape

    # Rank-transform the bank to uniforms, then to Gaussians.
    ranks = np.argsort(np.argsort(bank, axis=0), axis=0) + 1.0
    uniforms = ranks / (n_bank + 1.0)
    normals = _ndtri(uniforms)

    corr = np.corrcoef(normals, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    # Nearest positive-definite repair, needed because rank ties make the
    # empirical correlation of a discrete bank slightly indefinite.
    values, vectors = np.linalg.eigh(corr)
    corr = vectors @ np.diag(np.clip(values, 1e-8, None)) @ vectors.T
    scale = np.sqrt(np.diag(corr))
    corr = corr / np.outer(scale, scale)

    draws = rng.multivariate_normal(np.zeros(n_modes), corr, size=n_samples)
    quantiles = _ndtr(draws)

    # Map back through each column's empirical quantile function.
    out = np.empty((n_samples, n_modes))
    for column in range(n_modes):
        ordered = np.sort(bank[:, column])
        index = np.clip((quantiles[:, column] * n_bank).astype(int), 0, n_bank - 1)
        out[:, column] = ordered[index]
    return out


def _ndtri(x: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    x = np.clip(x, 1e-12, 1 - 1e-12)
    lower, upper = x < 0.02425, x > 1 - 0.02425
    central = ~(lower | upper)
    out = np.empty_like(x)

    q = np.sqrt(-2 * np.log(x[lower]))
    out[lower] = (
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = np.sqrt(-2 * np.log(1 - x[upper]))
    out[upper] = -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = x[central] - 0.5
    r = q * q
    out[central] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
    return out


def _ndtr(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, via ``math.erf`` so scipy is not a dependency."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def exact_distribution(unitary: np.ndarray, n_photons: int):
    """Exact output distribution via MerLin's ``QuantumLayer`` (small systems).

    Used to validate the stochastic sampler; the full distribution has
    ``C(m + n - 1, n)`` entries so this is only tractable for small ``m``/``n``.

    Returns
    -------
    tuple[list, numpy.ndarray]
        Fock-state keys and their probabilities.
    """
    import merlin as ML

    m = unitary.shape[0]
    layer = ML.QuantumLayer(
        circuit=to_circuit(unitary),
        input_state=[1] * n_photons + [0] * (m - n_photons),
        trainable_parameters=[],
        input_parameters=[],
        # the default computation space is UNBUNCHED; boson sampling outcomes
        # include bunched events, so the full Fock space is required
        measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
    )
    with torch.no_grad():
        probs = layer().squeeze(0).numpy()
    return list(layer.output_keys), probs


# --------------------------------------------------------------------------- #
# LatentDistribution implementations
# --------------------------------------------------------------------------- #
class SampleBankLatent(LatentDistribution):
    """Latent distribution backed by a fixed, pre-drawn bank of samples.

    Parameters
    ----------
    bank : numpy.ndarray | torch.Tensor
        Array of shape ``(n_samples, dim)``.
    normalize : {"center", "standardize", "none"}
        ``"center"`` (default) subtracts the per-coordinate mean, which is what
        the paper does: *"all distributions were centered to have a mean value
        of 0 before being injected into the generator"* (Appendix C).
        ``"standardize"`` additionally divides by the per-coordinate std.
        Both are affine invertible maps, so neither changes the complexity
        class of the distribution and Theorem 1 still applies.
    name : str
        Human-readable label used in plots and result tables.
    """

    def __init__(
        self,
        bank,
        *,
        normalize: str = "center",
        standardize: bool | None = None,
        name: str = "bank",
    ):
        if standardize is not None:  # backwards-compatible alias
            normalize = "standardize" if standardize else "none"
        if normalize not in {"center", "standardize", "none"}:
            raise ValueError(f"unknown normalize mode: {normalize}")
        bank_t = torch.as_tensor(np.asarray(bank), dtype=torch.get_default_dtype())
        if bank_t.ndim != 2:
            raise ValueError("bank must be 2-dimensional (n_samples, dim)")
        super().__init__(int(bank_t.shape[1]))
        self.name = name
        self.raw_bank = bank_t
        self.normalize = normalize
        mean = bank_t.mean(dim=0, keepdim=True)
        std = bank_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        if normalize == "standardize":
            self.bank, self.mean, self.std = (bank_t - mean) / std, mean, std
        elif normalize == "center":
            self.bank, self.mean, self.std = (
                bank_t - mean,
                mean,
                torch.ones(1, self.dim),
            )
        else:
            self.bank = bank_t
            self.mean, self.std = torch.zeros(1, self.dim), torch.ones(1, self.dim)
        self._g = torch.Generator().manual_seed(0)

    def sample(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        if type(batch_size) is not int:
            raise TypeError("batch_size must have int type.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        idx = torch.randint(0, self.bank.shape[0], (batch_size,), generator=self._g)
        out = self.bank[idx]
        if dtype is not None:
            out = out.to(dtype)
        return out.to(device) if device is not None else out

    def save(self, path: str | Path) -> None:
        np.save(Path(path), self.raw_bank.numpy())

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim}, bank={tuple(self.bank.shape)})"


class BosonSamplerLatent(SampleBankLatent):
    """Quantum latent: indistinguishable photons in a random interferometer.

    Parameters
    ----------
    dim : int
        Latent dimension == number of optical modes / time bins.
    n_photons : int | None
        Defaults to ``dim // 2``, the half-filling used throughout the paper.
    architecture : {"haar", "1-1", "1-3-9"}
        ``"haar"`` gives an unstructured interferometer; the others give the
        experimentally realistic loop cascades.
    n_samples : int
        Size of the pre-drawn bank.
    seed : int
        Seeds both the circuit draw and the sampler.
    """

    def __init__(
        self,
        dim: int,
        n_photons: int | None = None,
        architecture: str = "haar",
        n_samples: int = 200_000,
        seed: int = 0,
        normalize: str = "center",
    ):
        n_photons = dim // 2 if n_photons is None else n_photons
        rng = np.random.default_rng(seed)
        u = (
            haar_unitary(dim, rng)
            if architecture == "haar"
            else delay_line_unitary(dim, rng, architecture)
        )
        self.unitary = u
        self.n_photons = n_photons
        self.architecture = architecture
        bank = sample_boson(u, n_photons, n_samples, seed=seed)
        super().__init__(bank, normalize=normalize, name="boson sampler")


class DistinguishableLatent(SampleBankLatent):
    """Classical control: the same circuit without quantum interference."""

    def __init__(
        self,
        dim: int,
        n_photons: int | None = None,
        architecture: str = "haar",
        n_samples: int = 200_000,
        seed: int = 0,
        normalize: str = "center",
    ):
        n_photons = dim // 2 if n_photons is None else n_photons
        rng = np.random.default_rng(seed)
        u = (
            haar_unitary(dim, rng)
            if architecture == "haar"
            else delay_line_unitary(dim, rng, architecture)
        )
        self.unitary = u
        self.n_photons = n_photons
        self.architecture = architecture
        bank = sample_distinguishable(u, n_photons, n_samples, seed=seed)
        super().__init__(bank, normalize=normalize, name="dist. sampler")


class ShuffledBosonLatent(SampleBankLatent):
    """**Classical challenger.** Boson-sampling marginals, zero correlation.

    Takes a boson-sampling bank and independently permutes each column. Every
    single-mode marginal is preserved *exactly* -- not just its mean, as with
    :class:`DistinguishableLatent`, but the whole photon-number distribution,
    bunching tail included -- while every trace of the multi-photon correlation
    structure is destroyed. The result is a product distribution.

    This is the control the paper is missing. Its distinguishable-photon
    baseline matches only the first moment of each mode; boson sampling is
    super-Poissonian (Fano factor about 1.28 against 0.89 for distinguishable
    photons in the same circuit), so the two latents differ in their marginals
    as well as in their correlations, and any gap between them is not
    attributable to interference alone.

    If a GAN trained on this latent matches one trained on the true boson
    sampler, the advantage never needed the joint distribution -- only
    over-dispersed discrete marginals, which are classically constructible.

    Parameters
    ----------
    dim : int
        Latent dimension == number of optical modes.
    n_photons : int | None
        Defaults to ``dim // 2``.
    architecture : {"haar", "1-1", "1-3-9"}
        Interferometer family used to produce the bank being shuffled.
    n_samples : int
        Size of the bank.
    seed : int
        Seeds the circuit, the sampler and the column permutations.
    normalize : {"center", "standardize", "none"}
        Preprocessing applied to the bank.
    """

    def __init__(
        self,
        dim: int,
        n_photons: int | None = None,
        architecture: str = "haar",
        n_samples: int = 200_000,
        seed: int = 0,
        normalize: str = "center",
    ):
        n_photons = dim // 2 if n_photons is None else n_photons
        rng = np.random.default_rng(seed)
        unitary = (
            haar_unitary(dim, rng)
            if architecture == "haar"
            else delay_line_unitary(dim, rng, architecture)
        )
        self.unitary = unitary
        self.n_photons = n_photons
        self.architecture = architecture
        bank = sample_boson(unitary, n_photons, n_samples, seed=seed)
        super().__init__(
            shuffle_columns(bank, seed=seed), normalize=normalize, name="shuffled boson"
        )


class CopulaBosonLatent(SampleBankLatent):
    """**Classical challenger.** Boson marginals plus pairwise correlations.

    One rung above :class:`ShuffledBosonLatent`: the exact single-mode marginals
    are kept *and* the rank-correlation matrix is reproduced through a Gaussian
    copula, so only structure beyond second order is discarded. Fitting it needs
    a covariance matrix; sampling it needs a Cholesky factor and a sort.

    If this matches the true boson sampler, the useful content of a boson
    sampling distribution -- for this task -- is entirely captured by its first
    two orders, both of which are classically accessible.

    Parameters
    ----------
    dim : int
        Latent dimension == number of optical modes.
    n_photons : int | None
        Defaults to ``dim // 2``.
    architecture : {"haar", "1-1", "1-3-9"}
        Interferometer family used to produce the reference bank.
    n_samples : int
        Size of the bank.
    seed : int
        Seeds the circuit, the sampler and the copula draws.
    normalize : {"center", "standardize", "none"}
        Preprocessing applied to the bank.
    """

    def __init__(
        self,
        dim: int,
        n_photons: int | None = None,
        architecture: str = "haar",
        n_samples: int = 200_000,
        seed: int = 0,
        normalize: str = "center",
    ):
        n_photons = dim // 2 if n_photons is None else n_photons
        rng = np.random.default_rng(seed)
        unitary = (
            haar_unitary(dim, rng)
            if architecture == "haar"
            else delay_line_unitary(dim, rng, architecture)
        )
        self.unitary = unitary
        self.n_photons = n_photons
        self.architecture = architecture
        reference = sample_boson(unitary, n_photons, n_samples, seed=seed)
        super().__init__(
            gaussian_copula_resample(reference, n_samples, seed=seed),
            normalize=normalize,
            name="copula boson",
        )


class NegativeBinomialLatent(SampleBankLatent):
    """**Classical challenger.** Over-dispersed counts, no quantum data at all.

    Draws each mode independently from a negative binomial matched to the mean
    and variance of the corresponding boson-sampling mode. Unlike the two
    challengers above it does not resample a quantum bank: it needs only the
    first two moments per mode, which are low-order marginals of the boson
    sampling distribution and therefore classically computable. Everything else
    is a textbook count distribution.

    This is the strongest form of the sceptical question -- if it works, the
    experiment says "use over-dispersed discrete latents", not "use a quantum
    computer".

    Parameters
    ----------
    dim : int
        Latent dimension == number of optical modes.
    n_photons : int | None
        Defaults to ``dim // 2``.
    architecture : {"haar", "1-1", "1-3-9"}
        Interferometer family whose moments are matched.
    n_samples : int
        Size of the bank.
    seed : int
        Seeds the circuit, the moment estimate and the draws.
    normalize : {"center", "standardize", "none"}
        Preprocessing applied to the bank.
    moment_samples : int
        Number of boson samples used to estimate the per-mode mean and
        variance. Default value is 20000.
    """

    def __init__(
        self,
        dim: int,
        n_photons: int | None = None,
        architecture: str = "haar",
        n_samples: int = 200_000,
        seed: int = 0,
        normalize: str = "center",
        moment_samples: int = 20_000,
    ):
        n_photons = dim // 2 if n_photons is None else n_photons
        rng = np.random.default_rng(seed)
        unitary = (
            haar_unitary(dim, rng)
            if architecture == "haar"
            else delay_line_unitary(dim, rng, architecture)
        )
        self.unitary = unitary
        self.n_photons = n_photons
        self.architecture = architecture

        reference = sample_boson(unitary, n_photons, moment_samples, seed=seed)
        mean = reference.mean(axis=0).astype(float)
        var = reference.var(axis=0).astype(float)
        self.target_moments = (mean, var)

        bank = np.empty((n_samples, dim), dtype=np.int16)
        for mode in range(dim):
            m, v = max(mean[mode], 1e-6), var[mode]
            if v <= m:  # not over-dispersed: fall back to Poisson
                bank[:, mode] = rng.poisson(m, size=n_samples)
                continue
            # NB parametrised by mean m and variance v > m
            r = m * m / (v - m)
            prob = r / (r + m)
            bank[:, mode] = rng.negative_binomial(r, prob, size=n_samples)
        super().__init__(bank, normalize=normalize, name="negative binomial")


class DirichletMultinomialLatent(SampleBankLatent):
    """**Classical challenger.** Over-dispersed counts at a *fixed* photon total.

    The other challengers here (:class:`ShuffledBosonLatent`,
    :class:`CopulaBosonLatent`, :class:`NegativeBinomialLatent`) reproduce boson
    sampling's over-dispersed marginals but break a property the real
    distribution has: every boson sampling shot contains exactly ``n_photons``
    photons, so the coordinates are constrained to a simplex. Breaking that is a
    confound -- a latent could lose out for that reason rather than for lack of
    quantum correlations.

    This latent removes the confound. Photons are dealt by a Polya urn: draw
    ``p ~ Dirichlet(alpha * q)`` once per sample, where ``q`` is the boson
    sampler's mean occupancy vector, then draw ``multinomial(n_photons, p)``.
    The total is exactly ``n_photons``, the mean occupancies match, and
    ``alpha`` tunes the over-dispersion -- at ``alpha -> infinity`` this reduces
    to the paper's distinguishable-photon control, and lowering it inflates the
    variance until the Fano factor matches the boson sampler's.

    It is a two-line classical sampler with no quantum structure whatsoever.

    Parameters
    ----------
    dim : int
        Latent dimension == number of optical modes.
    n_photons : int | None
        Defaults to ``dim // 2``.
    architecture : {"haar", "1-1", "1-3-9"}
        Interferometer family whose occupancies and dispersion are matched.
    n_samples : int
        Size of the bank.
    seed : int
        Seeds the circuit, the moment estimate and the draws.
    normalize : {"center", "standardize", "none"}
        Preprocessing applied to the bank.
    moment_samples : int
        Boson samples used to estimate the target moments. Default value
        is 20000.
    """

    def __init__(
        self,
        dim: int,
        n_photons: int | None = None,
        architecture: str = "haar",
        n_samples: int = 200_000,
        seed: int = 0,
        normalize: str = "center",
        moment_samples: int = 20_000,
    ):
        n_photons = dim // 2 if n_photons is None else n_photons
        rng = np.random.default_rng(seed)
        unitary = (
            haar_unitary(dim, rng)
            if architecture == "haar"
            else delay_line_unitary(dim, rng, architecture)
        )
        self.unitary = unitary
        self.n_photons = n_photons
        self.architecture = architecture

        reference = sample_boson(unitary, n_photons, moment_samples, seed=seed)
        occupancy = reference.mean(axis=0).astype(float)
        occupancy = occupancy / occupancy.sum()
        target_fano = float(
            np.mean(reference.var(axis=0) / np.clip(reference.mean(axis=0), 1e-9, None))
        )
        self.alpha = _fit_dirichlet_alpha(
            occupancy, n_photons, target_fano, rng=np.random.default_rng(seed + 1)
        )
        self.target_fano = target_fano

        probs = rng.dirichlet(self.alpha * occupancy, size=n_samples)
        bank = np.empty((n_samples, dim), dtype=np.int16)
        for row in range(n_samples):
            bank[row] = rng.multinomial(n_photons, probs[row])
        super().__init__(bank, normalize=normalize, name="dirichlet-multinomial")


def _fit_dirichlet_alpha(
    occupancy: np.ndarray,
    n_photons: int,
    target_fano: float,
    rng: np.random.Generator,
    trial_samples: int = 4000,
) -> float:
    """Find the Dirichlet concentration reproducing a target Fano factor.

    The Dirichlet-multinomial Fano factor decreases monotonically in ``alpha``,
    so a bisection on ``log(alpha)`` suffices.

    Parameters
    ----------
    occupancy : numpy.ndarray
        Mean occupancy per mode, normalised to sum to 1.
    n_photons : int
        Photons dealt per sample.
    target_fano : float
        Fano factor to match, averaged over modes.
    rng : numpy.random.Generator
        Source of randomness for the trial draws.
    trial_samples : int
        Samples drawn per bisection step. Default value is 4000.

    Returns
    -------
    float
        Fitted concentration parameter.
    """

    def fano(alpha: float) -> float:
        probs = rng.dirichlet(alpha * occupancy, size=trial_samples)
        draws = np.stack([rng.multinomial(n_photons, p) for p in probs]).astype(float)
        return float(np.mean(draws.var(0) / np.clip(draws.mean(0), 1e-9, None)))

    low, high = -2.0, 6.0  # log10(alpha)
    for _ in range(18):
        mid = 0.5 * (low + high)
        if fano(10.0**mid) > target_fano:
            low = mid  # too dispersed -> need larger alpha
        else:
            high = mid
    return 10.0 ** (0.5 * (low + high))


class BernoulliLatent(SampleBankLatent):
    """Discrete uniform bit strings on ``{0, 1}^dim``."""

    def __init__(
        self,
        dim: int,
        n_samples: int = 200_000,
        seed: int = 0,
        normalize: str = "center",
    ):
        rng = np.random.default_rng(seed)
        bank = rng.integers(0, 2, size=(n_samples, dim)).astype(np.int16)
        super().__init__(bank, normalize=normalize, name="bernoulli")


class GaussianLatent(LatentDistribution):
    """The standard ``N(0, I)`` baseline."""

    name = "gaussian"

    def __init__(self, dim: int, seed: int = 0):
        super().__init__(dim)
        self._g = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        out = torch.randn(
            batch_size,
            self.dim,
            generator=self._g,
            dtype=dtype or torch.get_default_dtype(),
        )
        return out.to(device) if device is not None else out


#: The paper's four latents, in the order used by its tables.
PAPER_LATENT_KINDS = ("gaussian", "bernoulli", "distinguishable", "boson")

#: Classical challengers added by this reproduction. They are *not* in the
#: paper. Each one keeps more of the boson sampling distribution's structure
#: than the paper's distinguishable-photon control does, while remaining
#: classically samplable, so together they localise how much of the reported
#: advantage actually requires quantum correlations. See the README section
#: "Is the distinguishable control strong enough?".
CHALLENGER_LATENT_KINDS = (
    "shuffled_boson",
    "copula_boson",
    "negative_binomial",
    "dirichlet_multinomial",
)

#: Everything ``build_latent`` accepts.
LATENT_KINDS = PAPER_LATENT_KINDS + CHALLENGER_LATENT_KINDS

_PHOTONIC_KINDS = {
    "boson": BosonSamplerLatent,
    "distinguishable": DistinguishableLatent,
    "shuffled_boson": ShuffledBosonLatent,
    "copula_boson": CopulaBosonLatent,
    "negative_binomial": NegativeBinomialLatent,
    "dirichlet_multinomial": DirichletMultinomialLatent,
}


def build_latent(
    kind: str,
    dim: int,
    seed: int,
    *,
    n_photons: int | None = None,
    architecture: str = "haar",
    bank_size: int = 200_000,
    normalize: str = "center",
) -> LatentDistribution:
    """Instantiate one of the paper's four latent distributions.

    Parameters
    ----------
    kind : str
        One of :data:`LATENT_KINDS`: the paper's ``"gaussian"``,
        ``"bernoulli"``, ``"distinguishable"``, ``"boson"``, or one of this
        reproduction's classical challengers ``"shuffled_boson"``,
        ``"copula_boson"``, ``"negative_binomial"``.
    dim : int
        Latent dimension; for the photonic latents this is the number of optical
        modes (or time bins).
    seed : int
        Seeds both the random circuit draw and the sampler.
    n_photons : int | None
        Photon number for the photonic latents. Default value is None, which
        means ``dim // 2`` -- the half-filling used throughout the paper.
    architecture : {"haar", "1-1", "1-3-9"}
        Interferometer family for the photonic latents. Default value is "haar".
    bank_size : int
        Number of samples pre-drawn into the bank. Default value is 200000.
    normalize : {"center", "standardize", "none"}
        Preprocessing applied to the bank. Default value is "center", which is
        what the paper describes.

    Returns
    -------
    merlin.LatentDistribution
        Ready-to-sample latent distribution.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the four supported latents.
    """
    if kind in _PHOTONIC_KINDS:
        return _PHOTONIC_KINDS[kind](
            dim,
            n_photons=n_photons,
            architecture=architecture,
            n_samples=bank_size,
            seed=seed,
            normalize=normalize,
        )
    if kind == "bernoulli":
        return BernoulliLatent(dim, n_samples=bank_size, seed=seed, normalize=normalize)
    if kind == "gaussian":
        return GaussianLatent(dim, seed=seed)
    raise ValueError(f"unknown latent kind: {kind!r} (expected one of {LATENT_KINDS})")
