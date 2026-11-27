"""The Bosonic Binary Solver of arXiv:2510.08274.

One update step, following Algorithm 1 of the paper:

1. sample S threshold patterns Y from the interferometer at the current angles;
2. flip each bit independently with probability p_i, giving candidates X = Y xor B;
3. estimate dE[C(X)]/dtheta_k with the parameter-shift rule, using 2K extra
   circuit evaluations;
4. estimate dE[C(X)]/dalpha_i analytically from 2m extra cost evaluations;
5. take a plain SGD step (no momentum) on both parameter groups.

The best candidate ever evaluated is returned. That includes the strings drawn
while estimating gradients: Appendix B of the paper bounds "the number of
possible candidate solutions" by N S (2 sum(m - l_i) + 2m + 1), which only counts
correctly if the shifted evaluations are included, and ORCA's released solver
does exactly this (its ``readout`` updates ``E_min_encountered`` on every call,
including the parameter-shift calls). ``evaluations`` in the returned dict is
checked against that bound in tests/test_bbs.py.

Parameter-shift convention
--------------------------
The paper prints dE/dtheta ~= (E[C(X_{theta+phi})] - E[C(X_{theta-phi})]) / sin(phi).
Under this package's R = cos^2(theta/2) convention E(theta) carries harmonics of
theta, for which that expression equals *twice* the true derivative; ORCA's
released code divides instead by sin(2 phi) under its own R = cos^2(theta)
convention, where it is exact. The factor is constant, so it is exactly
equivalent to doubling the learning rate, and ``shift_scale`` selects which
convention to follow: 1.0 reproduces the paper as printed (the default), 0.5
gives the true gradient in this package's convention.
"""

from __future__ import annotations

import numpy as np
from lib.sampler import ClickSource
from lib.tbi import alternating_input, beamsplitter_layout, candidate_budget


class BosonicBinarySolver:
    """Train a time-bin interferometer and a bit-flip layer to minimise E[C(X)].

    Parameters
    ----------
    m : int
        Number of binary variables, equal to the number of time bins/modes.
    delays : sequence of int
        Delay-line lengths. The paper's simulations use ``(1, 3, 9)``.
    updates : int
        Gradient updates N. The paper uses 200. Default value is 200.
    samples : int
        Shots S per expectation value. The paper uses 50. Default value is 50.
    lr_theta, lr_alpha : float
        Learning rates for the beamsplitter angles and the bit-flip logits. The
        paper uses 0.01 and 0.05. Default values are 0.01 and 0.05.
    shift : float
        Parameter-shift magnitude phi. Default value is pi/2.
    shift_scale : float
        See the module docstring. Default value is 1.0 (the paper as printed).
    source : str
        Click source, see :data:`lib.sampler.SOURCES`. Default value is
        ``"boson"``.
    rate_scale : float
        Click-rate multiplier for the ``bernoulli`` source. Default value is 1.0.
    topology : str
        Circuit model, see :func:`lib.tbi.beamsplitter_layout`. Default value is
        ``"chain"``, the model consistent with the paper's own parameter count.
    seed : int
        Seed for angle initialisation and for all sampling. Default value is 0.
    """

    def __init__(
        self,
        m,
        delays=(1, 3, 9),
        updates=200,
        samples=50,
        lr_theta=0.01,
        lr_alpha=0.05,
        shift=np.pi / 2,
        shift_scale=1.0,
        source="boson",
        rate_scale=1.0,
        topology="chain",
        seed=0,
    ):
        self.m = m
        self.delays = tuple(delays)
        self.updates = updates
        self.samples = samples
        self.lr_theta = lr_theta
        self.lr_alpha = lr_alpha
        self.shift = shift
        self.shift_scale = shift_scale
        self.seed = seed
        self.topology = topology

        self.n_angles = len(beamsplitter_layout(m, self.delays, topology))
        self.input_state = alternating_input(m, self.delays, topology)
        self.rng = np.random.default_rng(seed)
        # paper: angles uniform on (0, 2 pi); bit-flip probabilities start at 1/2,
        # which is alpha = 0 under p = sigmoid(alpha)
        self.theta = self.rng.random(self.n_angles) * 2 * np.pi
        self.alpha = np.zeros(m)
        self.source = ClickSource(
            m,
            self.delays,
            self.input_state,
            source=source,
            rate_scale=rate_scale,
            topology=topology,
        )

    @property
    def flip_probabilities(self):
        return 1.0 / (1.0 + np.exp(-self.alpha))

    def budget(self):
        """Appendix B's bound for this configuration."""
        return candidate_budget(self.m, self.delays, self.updates, self.samples)

    def solve(self, cost_batch):
        """Run the full optimisation for one problem instance.

        Parameters
        ----------
        cost_batch : callable
            Maps a ``(n_candidates, m)`` array of bits to a vector of costs.

        Returns
        -------
        dict
            ``best_cost``, ``best_bits``, ``evaluations``, ``budget``,
            ``mean_clicks`` and the per-update ``history`` of E[C(X)].
        """
        best_cost = np.inf
        best_bits = None
        evaluations = 0
        clicks_seen = []
        history = []

        for _ in range(self.updates):
            clicks = self.source.draw(self.theta, self.samples, self.rng)
            clicks_seen.append(clicks.sum(axis=1).mean())
            flips = (
                self.rng.random((self.samples, self.m))
                < self.flip_probabilities[None, :]
            ).astype(np.int8)
            candidates = clicks ^ flips
            costs = cost_batch(candidates)
            evaluations += len(costs)
            best_cost, best_bits = self._keep_best(
                candidates, costs, best_cost, best_bits
            )
            history.append(float(costs.mean()))

            theta_gradient = np.zeros(self.n_angles)
            for k in range(self.n_angles):
                plus, minus = self.theta.copy(), self.theta.copy()
                plus[k] += self.shift
                minus[k] -= self.shift
                energies = []
                for angles in (plus, minus):
                    shifted_clicks = self.source.draw(angles, self.samples, self.rng)
                    shifted_flips = (
                        self.rng.random((self.samples, self.m))
                        < self.flip_probabilities[None, :]
                    ).astype(np.int8)
                    shifted_candidates = shifted_clicks ^ shifted_flips
                    shifted_costs = cost_batch(shifted_candidates)
                    evaluations += len(shifted_costs)
                    best_cost, best_bits = self._keep_best(
                        shifted_candidates, shifted_costs, best_cost, best_bits
                    )
                    energies.append(shifted_costs.mean())
                theta_gradient[k] = (
                    self.shift_scale * (energies[0] - energies[1]) / np.sin(self.shift)
                )

            # Analytic gradient for the bit-flip logits: forcing bit i to flipped
            # and to unflipped, holding every other draw fixed, gives
            # dE/dp_i = E[C | B_i = 1] - E[C | B_i = 0], and p = sigmoid(alpha)
            # contributes the chain-rule factor p (1 - p).
            alpha_gradient = np.zeros(self.m)
            probabilities = self.flip_probabilities
            for i in range(self.m):
                forced = []
                for value in (1, 0):
                    forced_flips = flips.copy()
                    forced_flips[:, i] = value
                    forced_candidates = clicks ^ forced_flips
                    forced_costs = cost_batch(forced_candidates)
                    evaluations += len(forced_costs)
                    best_cost, best_bits = self._keep_best(
                        forced_candidates, forced_costs, best_cost, best_bits
                    )
                    forced.append(forced_costs.mean())
                alpha_gradient[i] = (
                    (forced[0] - forced[1]) * probabilities[i] * (1 - probabilities[i])
                )

            # plain SGD, no momentum, as stated in the paper
            self.theta = self.theta - self.lr_theta * theta_gradient
            self.alpha = self.alpha - self.lr_alpha * alpha_gradient

        return {
            "best_cost": float(best_cost),
            "best_bits": best_bits.tolist(),
            "evaluations": int(evaluations),
            "budget": int(self.budget()),
            "mean_clicks": float(np.mean(clicks_seen)),
            "history": history,
        }

    @staticmethod
    def _keep_best(candidates, costs, best_cost, best_bits):
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            return float(costs[index]), candidates[index].copy()
        return best_cost, best_bits
