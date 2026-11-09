"""Max-Clique: QUBO formulation, exact optimum, and Q-score beta.

The QUBO and the beta formulas are the upstream Q-score definitions; the only
change here is that the random-search baselines accept an optional ``seed`` so
the reference score is reproducible (see :mod:`lib.seeding`).
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from networkx import Graph

#: Asymptotic average clique size of a naive search on G(N, 1/2):
#: sum_i^N i * (1 - p^i) * p^(i(i-1)/2). Used when no graph is available.
ASYMPTOTIC_RANDOM_CLIQUE = 1.6416325


def create_qubo_max_clique(G: Graph) -> defaultdict:
    """Create the QUBO formulation of a Max-Clique instance.

    Selecting a vertex is rewarded (-1 on the diagonal) and every pair of
    selected vertices that is *not* an edge of ``G`` is penalized (+2), so the
    minimum-energy assignment is the largest clique.

    Args:
        G: problem instance graph.

    Returns:
        QUBO as a ``{(i, j): coeff}`` dict.
    """
    G_C = nx.complement(G)
    Q: defaultdict = defaultdict(int)
    for i in G.nodes:
        Q[(i, i)] -= 1
    for i, j in G_C.edges:
        Q[(i, j)] += 2

    return Q


def naive_clique_size(G: Graph, seed: int | None = None) -> float:
    """Clique size found by one naive search pass over ``G``.

    Vertices are drawn in random order and kept while the induced subgraph stays
    complete; the search stops at the first vertex that breaks the clique.

    ``seed`` makes the vertex order reproducible. Averaging several passes (see
    :func:`average_naive_clique_size`) gives the Q-score random baseline.
    """
    generator = np.random.default_rng(seed)
    nodes = list(G.nodes())
    random_nodes: list = []
    while len(nodes) > 0:
        random_node = nodes.pop(int(generator.integers(len(nodes))))
        random_nodes.append(random_node)
        H = G.subgraph(random_nodes)
        n = len(random_nodes)
        if H.size() != n * (n - 1) / 2:
            return n - 1
    return len(random_nodes)


def average_naive_clique_size(
    G: Graph, seed: int | None = None, trials: int = 1000
) -> float:
    """Mean naive-search clique size over ``trials`` passes -- the random baseline.

    One generator drives every pass, so the whole average is reproducible from a
    single seed.
    """
    generator = np.random.default_rng(seed)
    sizes = [
        naive_clique_size(G, seed=int(generator.integers(2**31 - 1)))
        for _ in range(trials)
    ]
    return float(np.mean(sizes))


def calculate_beta_max_clique(
    graph: Graph | int,
    max_clique_result: float,
    seed: int | None = None,
) -> float:
    """Q-score beta for a Max-Clique result.

    Two normalizations, selected by the type of ``graph``:

    * ``Graph`` -- *exact* beta: the instance is scored against its own graph
      (naive random search vs. the true maximum clique). Correct at the small
      ``N`` used here, and the more honest number.
    * ``int``   -- *asymptotic* beta: the Q-score standard's closed-form random
      baseline and clique-size asymptote. Deflates beta noticeably below
      ``N ~ 10``, so a solver can look far worse than it is.

    Args:
        graph: problem instance graph (exact) or graph size (asymptotic).
        max_clique_result: objective found by the solver.
        seed: seeds the random baseline on the exact path.

    Returns:
        beta -- 0 means no better than random, 1 means optimal.
    """
    if isinstance(graph, Graph):  # only suitable for small graph sizes.
        random_score = average_naive_clique_size(graph, seed=seed)
        exact_score = int(nx.max_weight_clique(graph, weight=None)[1])
        if exact_score == random_score:
            return 1
        return (max_clique_result - random_score) / (exact_score - random_score)

    random_score = ASYMPTOTIC_RANDOM_CLIQUE
    asymptote = (
        2 * np.log2(graph) - 2 * np.log2(np.log2(graph)) + 2 * np.log2(np.e / 2) + 1
    )
    return (max_clique_result - random_score) / (asymptote - random_score)
