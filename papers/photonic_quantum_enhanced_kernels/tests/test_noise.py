import random

import numpy as np
import pytest
import torch
from perceval import BasicState, Circuit, Matrix, NoiseModel, Processor, Unitary
from utils.noise import NoisySLOSComputeGraph


@pytest.mark.parametrize(
    "m, n, indistinguishability",
    [
        (2, 2, 0.0),
        (2, 2, 0.5),
        (2, 2, 1.0),
        (4, 2, 0.0),
        (4, 3, 0.7),
        (4, 4, 1.0),
        (6, 2, 0.3),
        (6, 3, 0.9),
    ],
)
def test_partial_distinguishability(m, n, indistinguishability):
    unitary = Matrix.random_unitary(m)
    circuit = Circuit(m) // Unitary(unitary)
    input_state = [1] * n + [0] * (m - n)
    random.shuffle(input_state)

    # Calculate probabilities using Perceval
    processor = Processor("SLOS")
    processor.noise = NoiseModel(indistinguishability=indistinguishability)
    processor.set_circuit(circuit)
    processor.min_detected_photons_filter(0)
    processor.with_input(BasicState(input_state))

    probs_perceval = list(processor.probs()["results"].values())
    probs_perceval.sort()
    probs_perceval = torch.tensor(probs_perceval)

    # Calculate probabilities using noise.py
    unitary_torch = torch.tensor(np.array(unitary), dtype=torch.complex64)

    noisy_slos_graph = NoisySLOSComputeGraph(indistinguishability)

    _, probs_torch = noisy_slos_graph.compute_probs(unitary_torch, input_state)

    # Due to MerLin instability
    probs_torch, _ = torch.sort(probs_torch)

    probs_torch.squeeze()
    probs_perceval = probs_perceval.to(probs_torch.dtype)

    assert torch.allclose(probs_perceval, probs_torch, rtol=5e-4)
