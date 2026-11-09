"""Click sources: the boson sampler and the interference-free controls.

Every source has the same contract

    draw(theta, shots, rng) -> ndarray of shape (shots, m), entries in {0, 1}

so the solver never learns which one it is using. All of them depend on theta
through |U(theta)|^2, so the interferometer still trains under every source;
what changes is which statistics of |U(theta)|^2 survive.

The controls exist because the paper's own ablations remove *training* (freezing
theta, freezing the bit flips) but never remove *quantumness*, so nothing in the
paper isolates the contribution of the boson sampler itself.
"""

from __future__ import annotations

import numpy as np
import perceval as pcvl
from lib.tbi import mode_count, perceval_unitary
from perceval.backends import Clifford2017Backend

SOURCES = ("boson", "shuffled_boson", "distinguishable", "bernoulli")


def threshold(occupations):
    """Threshold detection: a mode clicks when it holds at least one photon."""
    return (np.asarray(occupations) > 0).astype(np.int8)


def _column_probabilities(theta, m, delays, input_modes, topology):
    """|U(theta)|^2 restricted to the occupied input columns and detected rows."""
    u = perceval_unitary(theta, m, delays, topology)
    return u[:m, list(input_modes)] ** 2


class ClickSource:
    """Draw threshold-detection patterns from one of the sources in SOURCES.

    Parameters
    ----------
    m : int
        Number of modes.
    delays : sequence of int
        Delay-line lengths.
    input_state : sequence of int
        Photon occupation of each input mode; the paper uses |1,0,1,0,...>.
    source : str
        One of :data:`SOURCES`. Default value is ``"boson"``.
    topology : str
        Circuit model, see :func:`lib.tbi.beamsplitter_layout`. Default value is
        ``"chain"``.
    rate_scale : float
        Multiplies the per-mode click probability of the ``bernoulli`` source.
        1.0 reproduces the distinguishable-photon marginals exactly; other values
        move the mean click density without touching anything else, which is how
        the density control is run. Default value is 1.0.
    """

    def __init__(
        self, m, delays, input_state, source="boson", rate_scale=1.0, topology="chain"
    ):
        if source not in SOURCES:
            raise ValueError(f"unknown click source {source!r}; available: {SOURCES}")
        self.m = m
        self.topology = topology
        self.n_modes = mode_count(m, delays, topology)
        self.delays = tuple(delays)
        self.input_state = tuple(input_state)
        self.input_modes = [i for i, n in enumerate(self.input_state) if n]
        self.source = source
        self.rate_scale = rate_scale
        self._backend = (
            Clifford2017Backend() if source in ("boson", "shuffled_boson") else None
        )
        self._pcvl_input = pcvl.BasicState(list(self.input_state))

    def draw(self, theta, shots, rng):
        """Return ``(shots, m)`` threshold patterns for beamsplitter angles ``theta``."""
        if self.source in ("boson", "shuffled_boson"):
            clicks = self._boson(theta, shots)
            if self.source == "shuffled_boson":
                # Permute each mode's column independently: every per-mode marginal
                # is preserved exactly and the joint distribution is destroyed. This
                # separates "the marginals of |U|^2 matter" from "the multi-photon
                # correlations matter".
                clicks = np.stack(
                    [rng.permutation(clicks[:, j]) for j in range(self.m)], axis=1
                )
            return clicks

        probabilities = _column_probabilities(
            theta, self.m, self.delays, self.input_modes, self.topology
        )

        if self.source == "distinguishable":
            # Each photon lands independently in a mode drawn from its own column of
            # |U|^2: photon number is conserved, multi-photon interference is not.
            cumulative = np.cumsum(
                probabilities / probabilities.sum(axis=0, keepdims=True), axis=0
            )
            draws = rng.random((shots, probabilities.shape[1]))
            landed = (draws[:, None, :] > cumulative[None, :, :]).sum(axis=1)
            clicks = np.zeros((shots, self.m), dtype=np.int8)
            np.put_along_axis(clicks, np.clip(landed, 0, self.m - 1), 1, axis=1)
            return clicks

        # bernoulli: keep only the per-mode click rate, drop every correlation.
        rate = 1.0 - np.prod(1.0 - probabilities, axis=1)
        rate = np.clip(self.rate_scale * rate, 0.0, 1.0)
        return (rng.random((shots, self.m)) < rate[None, :]).astype(np.int8)

    def _boson(self, theta, shots):
        matrix = pcvl.Matrix(
            perceval_unitary(theta, self.m, self.delays, self.topology)
        )
        self._backend.set_circuit(pcvl.Unitary(matrix))
        self._backend.set_input_state(self._pcvl_input)
        drawn = self._backend.samples(shots)
        # only the m time bins are detected; photons left circulating in a rail
        # of the loop topology leave after the train and are never measured
        return np.stack([threshold(list(state)[: self.m]) for state in drawn])
