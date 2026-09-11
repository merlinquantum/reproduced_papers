"""Max-Cut: QUBO formulation, exact optimum, and Q-score beta."""

from __future__ import annotations

from collections import defaultdict

from networkx import Graph
from networkx.algorithms.approximation.maxcut import one_exchange

#: Above this size the exact cut is replaced by a greedy approximation.
EXACT_MAX_CUT_LIMIT = 20


def create_qubo_max_cut(G: Graph) -> defaultdict:
    """Create the QUBO formulation of a Max-Cut instance.

    Args:
        G: problem instance graph.

    Returns:
        QUBO as a ``{(i, j): coeff}`` dict.
    """
    Q: defaultdict = defaultdict(int)
    for i, j in G.edges:
        Q[(i, i)] += -1
        Q[(j, j)] += -1
        Q[(i, j)] += 2

    return Q


def exact_max_cut(G: Graph) -> int:
    """Maximum cut size of ``G``, by exhaustive search over the 2**n partitions.

    Falls back to the greedy ``one_exchange`` approximation above
    :data:`EXACT_MAX_CUT_LIMIT` nodes, where enumeration stops being viable.
    """
    n = len(G)
    if n > EXACT_MAX_CUT_LIMIT:
        return int(one_exchange(G)[0])

    nodes = list(G.nodes())
    index = {u: i for i, u in enumerate(nodes)}

    best = 0
    # Iterate all assignments; partition A/B indicated by bit i of mask.
    for mask in range(1 << n):
        cut = 0
        for u, v in G.edges():
            if ((mask >> index[u]) & 1) != ((mask >> index[v]) & 1):
                cut += 1
        best = max(best, cut)
    return best


def calculate_beta_max_cut(graph: Graph | int, max_cut_result: float) -> float:
    """Q-score beta for a Max-Cut result.

    Two normalizations, selected by the type of ``graph``:

    * ``Graph`` -- *exact* beta, against this instance's own optimum.
    * ``int``   -- *asymptotic* beta, using the Q-score standard's closed forms
      (random cut ``N**2/8``, optimum ``0.178 * N**1.5``).

    Both baselines are deterministic, so no seed is needed here (unlike
    Max-Clique, whose random baseline is sampled).

    Args:
        graph: problem instance graph (exact) or graph size (asymptotic).
        max_cut_result: objective found by the solver.

    Returns:
        beta -- 0 means no better than random, 1 means optimal.
    """
    if isinstance(graph, Graph):  # only suitable for small graph sizes.
        n = len(graph)
        random_score = n * (n - 1) / 8
        exact_score = exact_max_cut(graph)
        if exact_score == random_score:
            return 1
        return (max_cut_result - random_score) / (exact_score - random_score)

    random_score = graph**2 / 8
    return (max_cut_result - random_score) / (0.178 * pow(graph, 3 / 2))
