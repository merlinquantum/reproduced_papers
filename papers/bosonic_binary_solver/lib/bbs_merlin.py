"""MerLin photonic variant: the same solver with exact gradients.

The paper estimates dE[C(X)]/dtheta with a photonic parameter-shift rule, which
costs ``2 sum_i (m - l_i)`` extra circuit evaluations per update and dominates
its own candidate budget: at m=30, 154 of the 155 circuit evaluations per update
exist only to estimate a gradient.

MerLin computes the output distribution of the interferometer as a differentiable
function of the angles, so the same objective can be minimised with an exact
gradient and no shifted evaluations at all. The physics is unchanged -- same
unrolled time-bin circuit, same input state, same threshold detectors, same
trainable bit-flip layer -- only the derivative estimator differs.

This is exact only while the Fock space is small enough to enumerate
(C(m + n - 1, n) states for n = m/2 photons), so it is a small-m study by
construction: m = 12 has 12 376 states, m = 16 has 490 314.
"""

from __future__ import annotations

import merlin
import numpy as np
import perceval as pcvl
import torch
from lib.tbi import alternating_input, beamsplitter_layout, mode_count


def build_circuit(m, delays, topology="chain"):
    """Perceval circuit for the unrolled time-bin interferometer.

    Returns
    -------
    tuple
        ``(circuit, parameter_names)``. Beamsplitters use Perceval's convention
        R = cos^2(theta / 2), which is the one :func:`lib.tbi.unitary` uses, so
        the two agree matrix for matrix (asserted in tests/test_merlin.py).
    """
    pairs = beamsplitter_layout(m, delays, topology)
    circuit = pcvl.Circuit(mode_count(m, delays, topology))
    names = []
    for index, (a, b) in enumerate(pairs):
        name = f"theta{index}"
        names.append(name)
        # BS.Ry is the real rotation [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]],
        # matching lib.tbi.unitary exactly; the default BS.Rx would introduce
        # factors of i and silently change every interference term.
        block = pcvl.BS.Ry(theta=pcvl.P(name))
        if b == a + 1:
            circuit.add(a, block)
        else:
            # Delay lines of length 3 and 9 act on non-adjacent bins, so the pair
            # is brought together by a permutation, mixed, and sent back.
            width = b - a + 1
            forward = [0] + list(range(2, width)) + [1]
            inverse = [0] * width
            for source, target in enumerate(forward):
                inverse[target] = source
            circuit.add(a, pcvl.PERM(forward))
            circuit.add(a, block)
            circuit.add(a, pcvl.PERM(inverse))
    return circuit, names


class MerlinBinarySolver(torch.nn.Module):
    """Exact-gradient counterpart of :class:`lib.bbs.BosonicBinarySolver`.

    Parameters
    ----------
    m : int
        Number of binary variables.
    delays : sequence of int
        Delay-line lengths. Default value is ``(1, 3, 9)``.
    updates : int
        Gradient updates. Default value is 200.
    flip_samples : int
        Monte-Carlo samples used for the expectation over the bit-flip layer.
        The click distribution itself is exact. Default value is 50.
    lr_theta, lr_alpha : float
        Learning rates, as in the paper. Default values are 0.01 and 0.05.
    topology : str
        Circuit model. Default value is ``"chain"``.
    seed : int
        Seed for initialisation and flip sampling. Default value is 0.
    """

    def __init__(
        self,
        m,
        delays=(1, 3, 9),
        updates=200,
        flip_samples=50,
        lr_theta=0.01,
        lr_alpha=0.05,
        topology="chain",
        seed=0,
    ):
        super().__init__()
        self.m = m
        self.delays = tuple(delays)
        self.updates = updates
        self.flip_samples = flip_samples
        self.lr_theta = lr_theta
        self.lr_alpha = lr_alpha
        self.topology = topology

        circuit, names = build_circuit(m, self.delays, topology)
        self.n_angles = len(names)
        input_state = alternating_input(m, self.delays, topology)
        self.layer = merlin.QuantumLayer(
            circuit=circuit,
            input_state=list(input_state),
            trainable_parameters=names,
            measurement_strategy=merlin.MeasurementStrategy.probs(
                merlin.ComputationSpace.FOCK
            ),
            dtype=torch.float64,
        )

        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.layer.parameters():
                parameter.copy_(
                    torch.rand(
                        parameter.shape, generator=generator, dtype=torch.float64
                    )
                    * 2
                    * np.pi
                )
        self.alpha = torch.nn.Parameter(torch.zeros(m, dtype=torch.float64))
        self.rng = np.random.default_rng(seed)
        self._click_index = None

    def click_probabilities(self):
        """Exact probability of every threshold-detection pattern, shape (2^m,).

        Fock outcomes are mapped to click patterns by an index built once: a mode
        clicks when it holds at least one photon, and the patterns of the m time
        bins are read as binary numbers.
        """
        probabilities = self.layer().reshape(-1)
        if self._click_index is None:
            keys = self.layer.computation_process.simulation_graph.final_keys
            weights = (1 << np.arange(self.m - 1, -1, -1)).astype(np.int64)
            index = [
                int(((np.asarray(key)[: self.m] > 0).astype(np.int64) * weights).sum())
                for key in keys
            ]
            self._click_index = torch.tensor(index, dtype=torch.long)
        out = torch.zeros(2**self.m, dtype=probabilities.dtype)
        return out.index_add(0, self._click_index, probabilities)

    def solve(self, cost_batch, all_bits=None):
        """Minimise E[C(X)] with exact gradients on the angles.

        Two counters are reported and they mean different things.

        ``candidates_sampled`` is ``updates * flip_samples``: the strings actually
        drawn from the trained distribution and evaluated, and the only number
        comparable with the paper's Appendix B budget, because it is what a
        hardware run would cost. The best candidate is tracked over exactly these,
        so "% optimal" means the same thing as on the sampled path.

        ``simulator_cost_evaluations`` is ``updates * 2^m``: forming the exact
        expectation touches the whole cost table. That is a simulation-only
        capability which does **not** transfer to hardware, and it is why this
        variant is reported separately rather than as a faster BBS.

        Parameters
        ----------
        cost_batch : callable
            Maps ``(n_candidates, m)`` bit arrays to costs.
        all_bits : numpy.ndarray or None
            All 2^m bit strings; built once when omitted.

        Returns
        -------
        dict
            ``best_cost``, ``best_bits``, ``candidates_sampled``,
            ``simulator_cost_evaluations``, ``circuit_evaluations`` and the
            per-update ``history``.
        """
        if all_bits is None:
            all_bits = (
                (np.arange(2**self.m)[:, None] >> np.arange(self.m - 1, -1, -1)) & 1
            ).astype(np.int8)
        costs = torch.tensor(cost_batch(all_bits), dtype=torch.float64)

        optimizer = torch.optim.SGD(
            [
                {"params": list(self.layer.parameters()), "lr": self.lr_theta},
                {"params": [self.alpha], "lr": self.lr_alpha},
            ]
        )
        best_cost, best_index = float("inf"), None
        history = []
        for _ in range(self.updates):
            optimizer.zero_grad()
            click = self.click_probabilities()
            # expectation over the flip layer, exactly: X = Y xor B with
            # independent B_i, so P(X = x) = sum_y P(y) prod_i p_i^{|x_i - y_i|}
            # (1 - p_i)^{1 - |x_i - y_i|}. Computed as a sequence of m two-point
            # mixtures, each a cheap reshape, rather than a 2^m x 2^m matrix.
            probability = click.reshape([2] * self.m)
            flip = torch.sigmoid(self.alpha)
            for axis in range(self.m):
                probability = probability.movedim(axis, 0)
                kept, flipped = probability[0], probability[1]
                probability = torch.stack(
                    [
                        (1 - flip[axis]) * kept + flip[axis] * flipped,
                        flip[axis] * kept + (1 - flip[axis]) * flipped,
                    ]
                ).movedim(0, axis)
            flat = probability.reshape(-1)
            energy = (flat * costs).sum()
            energy.backward()
            optimizer.step()
            history.append(float(energy.detach()))

            # candidates: draw from the trained distribution and keep the best,
            # so the reported metric is comparable with the sampled path
            with torch.no_grad():
                drawn = torch.multinomial(
                    flat.detach().clamp_min(0), self.flip_samples, replacement=True
                )
            batch = costs[drawn]
            index = int(batch.argmin())
            if float(batch[index]) < best_cost:
                best_cost, best_index = float(batch[index]), int(drawn[index])

        return {
            "best_cost": best_cost,
            "best_bits": all_bits[best_index].tolist(),
            "evaluations": self.updates * self.flip_samples,
            "budget": self.updates * self.flip_samples,
            "candidates_sampled": self.updates * self.flip_samples,
            "simulator_cost_evaluations": self.updates * 2**self.m,
            "circuit_evaluations": self.updates,
            "mean_clicks": float("nan"),
            "history": history,
        }
