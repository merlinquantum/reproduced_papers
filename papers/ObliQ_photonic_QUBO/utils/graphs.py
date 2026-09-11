"""Problem-instance generation: the benchmark's Erdos-Renyi graphs.

The instance seed *is* the graph seed -- no derivation, no offset -- so anything
that knows a run's seed sequence can regenerate its exact instances. Solver
randomness is split off separately (see :mod:`lib.seeding`).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from networkx import Graph

#: Edge probability of the benchmark's Erdos-Renyi instances, G(N, 1/2).
EDGE_PROBABILITY = 1 / 2

#: Attempts allowed when resampling an edgeless draw.
MAX_GRAPH_ATTEMPTS = 100


def sample_instance_graph(
    size: int, seed: int | None, probability: float = EDGE_PROBABILITY
) -> Graph:
    """Sample an Erdos-Renyi graph with at least one edge.

    An edgeless draw has no meaningful cut or clique, so the seed is advanced by
    one and the draw retried. Retries do *not* consume the next instance's seed --
    the sweep precomputes its seed list -- so this stays a pure function of
    ``seed``.

    Anything that needs to regenerate a run's instances (notably the exact-beta
    path in :mod:`plotter`) must call *this*, not ``nx.erdos_renyi_graph``
    directly: at small ``N`` a large fraction of raw draws are edgeless (about
    half at ``N = 2``), and scoring those against the wrong graph silently
    corrupts beta.

    Args:
        size: number of nodes.
        seed: RNG seed; ``None`` draws a fresh graph each call.
        probability: edge probability; defaults to the benchmark's 1/2 so the
            sweep, the plotter and the tests cannot drift apart on it.

    Returns:
        A graph with at least one edge.

    Raises:
        ValueError: if no non-empty graph was found in
            :data:`MAX_GRAPH_ATTEMPTS` attempts.
    """
    attempt_seed = seed

    for _ in range(MAX_GRAPH_ATTEMPTS):
        graph = nx.erdos_renyi_graph(size, probability, seed=attempt_seed)
        if graph.number_of_edges() > 0:
            return graph
        if attempt_seed is None:
            attempt_seed = int(np.random.randint(0, 1000000))
        else:
            attempt_seed += 1

    raise ValueError("Failed to sample a graph with at least one edge.")
