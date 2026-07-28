"""Instance generation: the seed sequence anyone can replay."""

import networkx as nx
import pytest

from utils.graphs import MAX_GRAPH_ATTEMPTS, sample_instance_graph


def test_instances_are_never_edgeless():
    """An edgeless graph has no meaningful cut or clique."""
    for seed in range(101200, 101260):
        assert sample_instance_graph(2, seed).number_of_edges() > 0


def test_same_seed_gives_the_same_graph():
    a = sample_instance_graph(7, 101700)
    b = sample_instance_graph(7, 101700)
    assert nx.utils.graphs_equal(a, b)


def test_different_seeds_generally_give_different_graphs():
    graphs = [sample_instance_graph(7, 101700 + i) for i in range(10)]
    edge_sets = {frozenset(map(frozenset, g.edges)) for g in graphs}
    assert len(edge_sets) > 1


def test_retry_diverges_from_a_raw_networkx_draw():
    """The retry-on-empty behaviour is exactly why the plotter must use this.

    At N=2 about half of the raw draws are edgeless; scoring a solver's answer
    against a raw draw would compare it to a *different* graph's optimum. This
    pins that the two really do differ, so the divergence stays a caught bug.
    """
    mismatches = [
        seed
        for seed in range(101200, 101300)
        if nx.erdos_renyi_graph(2, 1 / 2, seed=seed).number_of_edges() == 0
    ]
    assert mismatches, "expected some edgeless raw draws at N=2"
    for seed in mismatches[:5]:
        assert sample_instance_graph(2, seed).number_of_edges() == 1


def test_retry_does_not_consume_the_next_instances_seed():
    """Retrying must not shift the sequence, or every later instance would move."""
    before = sample_instance_graph(2, 101201)
    _ = sample_instance_graph(2, 101200)  # this one retries
    assert nx.utils.graphs_equal(sample_instance_graph(2, 101201), before)


def test_unseeded_sampling_still_returns_a_usable_graph():
    assert sample_instance_graph(4, None).number_of_edges() > 0


def test_gives_up_after_the_attempt_budget():
    """A probability of 0 can never produce an edge; it must raise, not hang."""
    with pytest.raises(ValueError, match="at least one edge"):
        sample_instance_graph(5, 1, probability=0.0)
    assert MAX_GRAPH_ATTEMPTS == 100
