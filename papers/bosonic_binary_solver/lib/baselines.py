"""Classical baselines at a matched cost-evaluation budget.

The paper compares the solver against simulated annealing and hill climbing but
does not state how many cost evaluations those baselines are allowed. The only
comparison that tests the paper's claim is one where every method may evaluate
the objective the same number of times, so both baselines here take the budget
as an explicit argument and the runner passes Appendix B's bound.
"""

from __future__ import annotations

import numpy as np


def _cost_scale(cost_batch, m, rng, probes=256):
    """Rough spread of the objective, used to set the annealing temperature.

    Reading the maximum off the full 2^m table is not possible at the sizes that
    matter, so it is estimated from random candidates.
    """
    sample = rng.integers(0, 2, (probes, m))
    costs = cost_batch(sample)
    spread = float(costs.max() - costs.min())
    return spread if spread > 0 else 1.0


def simulated_annealing(cost_batch, m, budget, rng, initial_temperature=None, final_temperature=None):
    """Single-flip Metropolis annealing with a geometric temperature schedule.

    Parameters
    ----------
    cost_batch : callable
        Maps ``(n, m)`` bit arrays to costs.
    m : int
        Number of binary variables.
    budget : int
        Number of cost evaluations allowed, matched to the solver's.
    rng : numpy.random.Generator
        Source of randomness.
    initial_temperature, final_temperature : float or None
        Endpoints of the geometric schedule. When omitted they are set from the
        observed cost spread. Default value is None.

    Returns
    -------
    dict
        ``best_cost``, ``best_bits``, ``evaluations``.
    """
    scale = _cost_scale(cost_batch, m, rng)
    hot = initial_temperature if initial_temperature is not None else 0.25 * scale
    cold = final_temperature if final_temperature is not None else 1e-3 * scale
    steps = max(budget, 1)
    decay = (cold / hot) ** (1.0 / steps)

    current = rng.integers(0, 2, m).astype(np.int8)
    current_cost = float(cost_batch(current[None, :])[0])
    best_cost, best_bits = current_cost, current.copy()
    temperature = hot
    evaluations = 1

    while evaluations < budget:
        index = int(rng.integers(0, m))
        current[index] ^= 1
        candidate_cost = float(cost_batch(current[None, :])[0])
        evaluations += 1
        if candidate_cost <= current_cost or rng.random() < np.exp(-(candidate_cost - current_cost) / temperature):
            current_cost = candidate_cost
            if candidate_cost < best_cost:
                best_cost, best_bits = candidate_cost, current.copy()
        else:
            current[index] ^= 1
        temperature *= decay

    return {"best_cost": best_cost, "best_bits": best_bits.tolist(), "evaluations": evaluations}


def hill_climbing(cost_batch, m, budget, rng):
    """Steepest-ascent hill climbing with random restarts.

    Each sweep evaluates all m single-bit neighbours at once and moves to the
    best improving one; when no neighbour improves, the search restarts from a
    fresh random point. The paper does not specify its restart policy, which is
    the most likely reason its hill-climbing column is hard to match exactly.
    """
    best_cost, best_bits = np.inf, None
    evaluations = 0

    while evaluations < budget:
        current = rng.integers(0, 2, m).astype(np.int8)
        current_cost = float(cost_batch(current[None, :])[0])
        evaluations += 1
        if current_cost < best_cost:
            best_cost, best_bits = current_cost, current.copy()

        while evaluations < budget:
            neighbours = np.repeat(current[None, :], m, axis=0)
            neighbours[np.arange(m), np.arange(m)] ^= 1
            take = min(m, budget - evaluations)
            if take <= 0:
                break
            costs = cost_batch(neighbours[:take])
            evaluations += take
            index = int(np.argmin(costs))
            if costs[index] >= current_cost:
                break
            current = neighbours[index].copy()
            current_cost = float(costs[index])
            if current_cost < best_cost:
                best_cost, best_bits = current_cost, current.copy()

    return {"best_cost": float(best_cost), "best_bits": best_bits.tolist(), "evaluations": int(evaluations)}


def random_search(cost_batch, m, budget, rng, block=8192):
    """Uniform random sampling at the same budget: the weakest honest reference.

    Any method that fails to beat this is not searching at all.
    """
    best_cost, best_bits = np.inf, None
    evaluations = 0
    while evaluations < budget:
        take = min(block, budget - evaluations)
        candidates = rng.integers(0, 2, (take, m)).astype(np.int8)
        costs = cost_batch(candidates)
        evaluations += take
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best_cost, best_bits = float(costs[index]), candidates[index].copy()
    return {"best_cost": float(best_cost), "best_bits": best_bits.tolist(), "evaluations": int(evaluations)}


BASELINES = {
    "simulated_annealing": simulated_annealing,
    "hill_climbing": hill_climbing,
    "random_search": random_search,
}
