"""The four latent distributions and the circuits behind them."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401 - puts the project on sys.path
from lib.circuits import DELAY_CONFIGS, delay_line_unitary, haar_unitary
from lib.latents import LATENT_KINDS, build_latent


@pytest.mark.parametrize("architecture", ["haar", *DELAY_CONFIGS])
def test_circuits_are_unitary(architecture):
    rng = np.random.default_rng(0)
    m = 8
    u = (
        haar_unitary(m, rng)
        if architecture == "haar"
        else delay_line_unitary(m, rng, architecture)
    )
    assert np.allclose(u.conj().T @ u, np.eye(m), atol=1e-10)


@pytest.mark.parametrize("kind", LATENT_KINDS)
def test_every_latent_samples_the_right_shape(kind):
    latent = build_latent(kind, dim=6, seed=0, bank_size=500)
    z = latent.sample(16)
    assert z.shape == (16, 6)
    assert latent.dim == 6


def test_unknown_latent_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown latent kind"):
        build_latent("squeezed", dim=4, seed=0)


def test_centering_is_the_default_and_leaves_zero_mean():
    latent = build_latent("boson", dim=6, seed=0, bank_size=4000)
    assert latent.normalize == "center"
    assert abs(float(latent.bank.mean())) < 1e-5


def test_photon_number_is_conserved_by_the_boson_sampler():
    latent = build_latent("boson", dim=6, seed=0, bank_size=1000)
    assert latent.n_photons == 3
    assert np.all(latent.raw_bank.numpy().sum(axis=1) == 3)


def test_distinguishable_control_matches_mean_occupancy_but_nothing_else():
    """Pin down exactly how far the paper's control goes.

    It matches the boson sampler's *mean occupancy* (first moment) and differs
    in the joint distribution -- which is what the paper relies on. But it also
    differs in the *second* moment: the boson sampler bunches relative to the
    control. So it does not have identical single-mode marginals, and a gap
    between the two is not attributable to interference alone. See the
    classical challenger study.

    The Fano ratio (about 1.4 at half filling) is the config-independent part.
    The absolute values are not: at the Table I size of 16 modes / 8 photons
    they are 1.28 against 0.89 -- super- against sub-Poissonian -- while at the
    small size used here the fixed photon total pushes both below 1.
    """
    quantum = build_latent("boson", dim=6, seed=1, bank_size=60_000)
    classical = build_latent("distinguishable", dim=6, seed=1, bank_size=60_000)
    q, c = quantum.raw_bank.numpy(), classical.raw_bank.numpy()

    occupancy_gap = np.abs(q.mean(0) - c.mean(0)).max()
    assert occupancy_gap < 0.05, "same circuit must give the same mean occupancy"

    fano_q = q.var(axis=0).mean() / q.mean(axis=0).mean()
    fano_c = c.var(axis=0).mean() / c.mean(axis=0).mean()
    assert fano_q > 1.2 * fano_c, (
        "boson sampling must bunch relative to the distinguishable control "
        f"(got Fano {fano_q:.3f} against {fano_c:.3f})"
    )

    bunching_gap = float(
        (quantum.raw_bank.numpy().max(1) > 1).mean()
        - (classical.raw_bank.numpy().max(1) > 1).mean()
    )
    assert abs(bunching_gap) > 0.02, "interference must change the bunching statistics"


def test_shuffle_columns_preserves_every_marginal_exactly():
    """Column shuffling is a per-mode permutation, so marginals are untouched."""
    import numpy as np
    from lib.latents import shuffle_columns

    rng = np.random.default_rng(0)
    bank = rng.poisson(0.7, size=(4000, 6)).astype(np.int16)
    shuffled = shuffle_columns(bank, seed=1)

    for column in range(bank.shape[1]):
        assert np.array_equal(np.sort(bank[:, column]), np.sort(shuffled[:, column]))
    assert not np.array_equal(bank, shuffled)


def test_challengers_match_boson_marginals_but_not_its_correlations():
    """The point of the challengers: same marginals, different joint structure.

    The paper's distinguishable-photon control matches only the *mean* of each
    mode. These classical challengers match the whole single-mode marginal, so
    they isolate the contribution of the multi-photon correlations.
    """
    import numpy as np
    from lib.latents import build_latent

    def stats(kind):
        bank = build_latent(kind, 8, 0, bank_size=8000).raw_bank.numpy().astype(float)
        corr = np.corrcoef(bank.T)
        off = corr[~np.eye(bank.shape[1], dtype=bool)]
        return bank.mean(0), bank.var(0), np.abs(np.nan_to_num(off)).max()

    mean_q, var_q, corr_q = stats("boson")
    mean_s, var_s, corr_s = stats("shuffled_boson")
    mean_d, var_d, _ = stats("distinguishable")

    # shuffled boson reproduces the boson marginals (exactly for a given bank;
    # to sampling precision here, since each call draws its own bank)
    assert np.abs(mean_q - mean_s).max() < 0.05
    assert np.abs(var_q - var_s).max() < 0.05
    # but carries essentially no cross-mode correlation
    assert corr_s < corr_q

    # the paper's control matches the means yet is measurably under-dispersed
    assert np.abs(mean_q - mean_d).max() < 0.05
    assert var_d.mean() < var_q.mean()


def test_boson_sampling_is_super_poissonian_and_the_control_is_not():
    """Fano factor > 1 for boson sampling, < 1 for distinguishable photons."""
    import numpy as np
    from lib.latents import build_latent

    def fano(kind):
        bank = build_latent(kind, 16, 0, bank_size=20000).raw_bank.numpy().astype(float)
        return float(np.mean(bank.var(0) / np.clip(bank.mean(0), 1e-9, None)))

    assert fano("boson") > 1.05
    assert fano("distinguishable") < 0.95
    assert fano("negative_binomial") > 1.05
