"""Batched cost functions on a torch device.

The scalar implementations in :mod:`lib.problems` are the readable definition and
stay the reference; these evaluate ``(instances, candidates, m)`` at once so that
one update step of every instance can be scored in a single kernel. Both are
checked against each other in tests/test_problems_torch.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lib.problems import knapsack_instance, locations_for_modes, tsp_instance


class KnapsackBatch:
    """Knapsack costs for a fixed set of instances.

    Parameters
    ----------
    instances : sequence of dict
        Instances from :func:`lib.problems.knapsack_instance`.
    device : torch.device
        Device holding the weight and value tables.
    """

    def __init__(self, instances, device):
        self.weights = torch.tensor(np.stack([i["weights"] for i in instances]), dtype=torch.float64, device=device)
        self.values = torch.tensor(np.stack([i["values"] for i in instances]), dtype=torch.float64, device=device)
        self.capacity = torch.tensor([i["capacity"] for i in instances], dtype=torch.float64, device=device)

    def __call__(self, bits):
        """``bits`` ``(I, Q, m)`` -> costs ``(I, Q)``; infeasible packings cost 0.

        The cast to float64 is what makes the einsum below legal on CUDA; the
        integer weights and values are small enough that float64 is exact.
        """
        bits = bits.to(torch.float64)
        weight = torch.einsum("iqm,im->iq", bits, self.weights)
        value = torch.einsum("iqm,im->iq", bits, self.values)
        return torch.where(weight > self.capacity[:, None], torch.zeros_like(value), -value)


class TSPBatch:
    """TSP tour lengths for a fixed set of instances.

    The bit string is read as an integer, reduced modulo ``(n-1)!`` and
    Lehmer-decoded, exactly as in :func:`lib.problems.tsp_cost`; the decode is
    vectorised over candidates by peeling one factorial block at a time.
    """

    def __init__(self, instances, m, device):
        points = np.stack([i["points"] for i in instances])
        self.n = points.shape[1]
        self.m = m
        self.device = device
        self.distance = torch.tensor(
            np.linalg.norm(points[:, :, None, :] - points[:, None, :, :], axis=-1),
            dtype=torch.float64, device=device,
        )
        self.block = math.factorial(self.n - 1)
        self.powers = (2 ** torch.arange(m - 1, -1, -1, dtype=torch.int64, device=device))

    def __call__(self, bits):
        # Elementwise multiply and sum, not einsum: torch.einsum dispatches to
        # baddbmm, which has no int64 CUDA kernel ("baddbmm_cuda not implemented
        # for 'Long'"), while working fine on CPU. Staying in int64 matters --
        # m can be 29, and the index must be exact before the modulo.
        index = (bits.to(torch.int64) * self.powers).sum(dim=-1) % self.block
        instances, candidates = index.shape
        remaining = torch.arange(self.n - 1, device=self.device).expand(instances, candidates, self.n - 1).clone()
        alive = torch.ones_like(remaining, dtype=torch.bool)
        order = torch.empty(instances, candidates, self.n - 1, dtype=torch.int64, device=self.device)
        current = index.clone()
        for position in range(self.n - 1, 0, -1):
            block = math.factorial(position - 1)
            which = current // block
            current = current % block
            # pick the which-th still-available item, without materialising a list
            ranks = torch.cumsum(alive.to(torch.int64), dim=-1) - 1
            hit = (ranks == which[..., None]) & alive
            picked = torch.argmax(hit.to(torch.int64), dim=-1)
            order[..., self.n - 1 - position] = remaining.gather(-1, picked[..., None]).squeeze(-1)
            alive.scatter_(-1, picked[..., None], False)
        tour = torch.cat([torch.zeros(instances, candidates, 1, dtype=torch.int64, device=self.device), order + 1], dim=-1)
        nxt = torch.roll(tour, -1, dims=-1)
        edges = self.distance.reshape(instances, -1).gather(1, (tour * self.n + nxt).reshape(instances, -1))
        return edges.reshape(instances, candidates, self.n).sum(-1)


def build_batch(family, m, seeds, device, **kwargs):
    """Return ``(batched_cost, optima)`` for a list of instance seeds."""
    from lib.problems import knapsack_optimum, tsp_optimum

    if family == "knapsack":
        instances = [knapsack_instance(m, seed, **kwargs) for seed in seeds]
        return KnapsackBatch(instances, device), [knapsack_optimum(i) for i in instances]
    if family == "tsp":
        instances = [tsp_instance(m, seed, **kwargs) for seed in seeds]
        return TSPBatch(instances, m, device), [tsp_optimum(i) for i in instances]
    raise ValueError(f"unknown problem family: {family!r}")
