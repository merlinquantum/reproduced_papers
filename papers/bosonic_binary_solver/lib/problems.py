"""Optimisation problems used by the paper: knapsack and TSP.

Every problem exposes the same contract:

    cost(bits) -> float          # lower is better, the paper minimises E[C(X)]
    optimum() -> float           # exact optimum, computed classically

The paper (Sec. IV) reports knapsack at m = 10, 15, 20, 25, 30 binary variables,
tactical deconfliction at the same sizes, and TSP at m = 10, 19, 29. Instances are
generated rather than downloaded; the paper does not publish its generator, so the
generators here are documented assumptions -- see the "Data" section of README.md.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------- knapsack


def knapsack_instance(m, seed, value_max=20, weight_max=20, capacity_fraction=0.5):
    """Generate one uncorrelated 0/1 knapsack instance with ``m`` items.

    Parameters
    ----------
    m : int
        Number of items, which is also the number of binary variables.
    seed : int
        Instance seed. The same seed always yields the same instance.
    value_max, weight_max : int
        Inclusive upper bounds for the uniform integer draws. Default value is 20.
    capacity_fraction : float
        Capacity as a fraction of the total weight. Default value is 0.5.

    Returns
    -------
    dict
        Keys ``values``, ``weights``, ``capacity``.
    """
    rng = np.random.default_rng(seed)
    values = rng.integers(1, value_max + 1, size=m)
    weights = rng.integers(1, weight_max + 1, size=m)
    return {
        "values": values,
        "weights": weights,
        "capacity": int(capacity_fraction * weights.sum()),
    }


def knapsack_cost(instance, bits):
    """Cost of one candidate: minus the packed value, or 0 when over capacity.

    Infeasible candidates are given cost 0, which is worse than any feasible
    packing because every value is positive. That keeps the objective a plain
    function of the bit string, as the paper's E[C(X)] requires.
    """
    bits = np.asarray(bits)
    weight = float(bits @ instance["weights"])
    if weight > instance["capacity"]:
        return 0.0
    return -float(bits @ instance["values"])


def knapsack_cost_batch(instance, bits):
    """Vectorised :func:`knapsack_cost` over a ``(n_candidates, m)`` array.

    The paper's own budget is over a million cost evaluations per instance, so
    the batched path is what makes the reproduction affordable; the scalar
    version above stays as the readable definition and is checked against this
    one in tests/test_problems.py.
    """
    bits = np.asarray(bits)
    weight = bits @ instance["weights"]
    value = bits @ instance["values"]
    return np.where(weight > instance["capacity"], 0.0, -value.astype(float))


def knapsack_optimum(instance):
    """Exact optimum by dynamic programming over capacity (pseudo-polynomial)."""
    capacity = instance["capacity"]
    best = np.zeros(capacity + 1, dtype=np.int64)
    for weight, value in zip(instance["weights"], instance["values"]):
        if weight > capacity:
            continue
        best[weight:] = np.maximum(best[weight:], best[: capacity + 1 - weight] + value)
    return -float(best[capacity])


# -------------------------------------------------------------------------- TSP


def modes_for_locations(n_locations):
    """Binary variables needed to index every tour: ceil(log2((n-1)!))."""
    return math.ceil(math.log2(math.factorial(n_locations - 1)))


def locations_for_modes(m):
    """Largest location count whose tours still fit in ``m`` binary variables.

    The paper's TSP sizes 10, 19 and 29 inverted this way give 7, 10 and 13
    locations respectively.
    """
    n = 2
    while modes_for_locations(n + 1) <= m:
        n += 1
    return n


def tsp_instance(m, seed, grid=100):
    """Uniform random points on a ``grid`` x ``grid`` square."""
    n = locations_for_modes(m)
    rng = np.random.default_rng(seed)
    return {"points": rng.integers(0, grid, size=(n, 2)).astype(float), "m": m}


def lehmer_to_permutation(index, n_items):
    """Decode an integer into a permutation of ``n_items`` via its Lehmer code."""
    items = list(range(n_items))
    permutation = []
    for position in range(n_items, 0, -1):
        block = math.factorial(position - 1)
        which, index = divmod(index, block)
        permutation.append(items.pop(which))
    return permutation


def tsp_cost(instance, bits):
    """Tour length of the cycle that the bit string indexes.

    The paper cites Schnaus et al. for the binary-to-permutation map without
    writing it out. This is the natural surjection: read the bits as an integer,
    reduce modulo (n-1)!, and Lehmer-decode. Every tour is reachable and the
    optimum is always representable; the map is many-to-one when 2^m > (n-1)!.
    """
    points = instance["points"]
    n = len(points)
    index = 0
    for bit in np.asarray(bits):
        index = (index << 1) | int(bit)
    order = lehmer_to_permutation(index % math.factorial(n - 1), n - 1)
    tour = [0] + [i + 1 for i in order]
    return float(
        sum(
            np.linalg.norm(points[tour[i]] - points[tour[(i + 1) % n]])
            for i in range(n)
        )
    )


def tsp_cost_batch(instance, bits):
    """Vectorised :func:`tsp_cost`: decode every row, then sum tour edges."""
    bits = np.asarray(bits)
    points = instance["points"]
    n = len(points)
    block = math.factorial(n - 1)
    weights = (1 << np.arange(bits.shape[1] - 1, -1, -1)).astype(object)
    indices = (bits.astype(object) @ weights) % block
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    out = np.empty(len(bits))
    for row, index in enumerate(indices):
        order = lehmer_to_permutation(int(index), n - 1)
        tour = [0] + [i + 1 for i in order]
        out[row] = sum(distance[tour[i], tour[(i + 1) % n]] for i in range(n))
    return out


def tsp_optimum(instance):
    """Exact optimum by Held-Karp: 2^(n-1) (n-1)^2 rather than (n-1)! tours."""
    points = instance["points"]
    n = len(points)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    others = list(range(1, n))
    size = len(others)
    best = {(1 << i, i): distance[0, others[i]] for i in range(size)}
    for _ in range(2, size + 1):
        nxt = {}
        for (mask, last), cost in best.items():
            for j in range(size):
                if mask & (1 << j):
                    continue
                key = (mask | (1 << j), j)
                candidate = cost + distance[others[last], others[j]]
                if key not in nxt or candidate < nxt[key]:
                    nxt[key] = candidate
        best.update(nxt)
    full = (1 << size) - 1
    return min(
        cost + distance[others[last], 0]
        for (mask, last), cost in best.items()
        if mask == full
    )


# ------------------------------------------------------------------ dispatch


def build_problem(family, m, seed, **kwargs):
    """Return ``(instance, cost_function, optimum)`` for one problem instance."""
    if family == "knapsack":
        instance = knapsack_instance(m, seed, **kwargs)
        return (
            instance,
            (lambda bits: knapsack_cost_batch(instance, bits)),
            knapsack_optimum(instance),
        )
    if family == "tsp":
        instance = tsp_instance(m, seed, **kwargs)
        return (
            instance,
            (lambda bits: tsp_cost_batch(instance, bits)),
            tsp_optimum(instance),
        )
    raise ValueError(f"unknown problem family: {family!r}")
