from __future__ import annotations

import torch
from power_of_QNN.lib.classical import (
    FeedForwardNetwork,
    enumerate_topologies,
    parameter_count,
)


def test_parameter_counts_of_the_paper_networks():
    assert parameter_count([4, 4, 4, 2], bias=False) == 40
    assert parameter_count([6, 7, 2, 2], bias=False) == 60
    assert parameter_count([8, 8, 2], bias=False) == 80
    assert parameter_count([10, 8, 1, 4, 2], bias=False) == 100
    assert parameter_count([6, 110, 2], bias=False) == 880


def test_enumeration_is_complete_and_exact():
    d8 = enumerate_topologies(4, 2, 8, max_hidden_layers=3)
    assert d8 == [[4, 2], [4, 1, 1, 1, 2]]
    d40 = enumerate_topologies(4, 2, 40, max_hidden_layers=3)
    assert [4, 4, 4, 2] in d40
    assert all(parameter_count(sizes, bias=False) == 40 for sizes in d40)
    # Brute-force cross-check of the pruned search for up to two hidden layers.
    brute = set()  # the linear model [4, 2] has d=8, not 40
    for h1 in range(1, 11):
        if 4 * h1 + 2 * h1 == 40:
            brute.add((4, h1, 2))
        for h2 in range(1, 40):
            if 4 * h1 + h1 * h2 + 2 * h2 == 40:
                brute.add((4, h1, h2, 2))
    assert brute <= {tuple(sizes) for sizes in d40 if len(sizes) <= 4}
    assert {tuple(sizes) for sizes in d40 if len(sizes) <= 4} <= brute


def test_flat_parameters_follow_nn_linear_order():
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(
        [torch.nn.Linear(4, 3, bias=True), torch.nn.Linear(3, 2, bias=True)]
    ).double()
    flat = torch.cat(
        [parameter.reshape(-1) for parameter in layers.parameters()]
    ).detach()
    x = torch.randn(5, 4, dtype=torch.float64)
    reference = torch.softmax(
        layers[1](torch.nn.functional.leaky_relu(layers[0](x))), dim=-1
    )
    network = FeedForwardNetwork([4, 3, 2], "leaky_relu", "softmax", bias=True)
    ours = network.probabilities(x, flat.expand(5, -1))
    assert torch.allclose(ours, reference)
