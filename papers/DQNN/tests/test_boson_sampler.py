import sys
from pathlib import Path

import merlin as ML
import numpy as np
import pytest
import torch

try:
    from papers.DQNN.lib.boson_sampler import BosonSampler
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    REPO_ROOT = Path(__file__).resolve().parents[3]
    for path in (REPO_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from papers.DQNN.lib.boson_sampler import BosonSampler


@pytest.fixture
def bs_1():
    return BosonSampler(m=9, n=4)


@pytest.fixture
def bs_2():
    return BosonSampler(m=8, n=4)


def test_init(bs_1, bs_2):
    # bs_1
    bs = bs_1
    assert bs.nb_parameters == 108
    assert bs.nb_parameters >= bs.num_effective_params
    assert len(list(bs.quantum_layer.parameters()))
    assert isinstance(bs.quantum_layer, ML.QuantumLayer)

    # bs_2
    bs = bs_2
    assert bs.nb_parameters == 84
    assert bs.nb_parameters >= bs.num_effective_params
    assert len(list(bs.quantum_layer.parameters()))
    assert isinstance(bs.quantum_layer, ML.QuantumLayer)


def test_set_params(bs_1):
    bs = bs_1

    # Setting all params to 0
    bs.set_params(np.zeros(bs_1.num_effective_params))

    bs_params = []

    for i in bs.quantum_layer.parameters():
        bs_params.extend(i.detach().cpu().flatten().tolist())

    np.testing.assert_allclose(
        bs_params, np.zeros(bs_1.num_effective_params), rtol=0.000001
    )

    # Randomly assign parameters 5 times
    for _ in range(5):
        rand_params = np.random.random(bs_1.num_effective_params)
        bs.set_params(rand_params)

        bs_params = []

        for i in bs.quantum_layer.parameters():
            bs_params.extend(i.detach().cpu().flatten().tolist())

        np.testing.assert_allclose(bs_params, rand_params, rtol=0.000001)


def test_embedding_size(bs_1):
    bs = bs_1
    assert bs.embedding_size == 126
    assert len(torch.flatten(bs.quantum_layer())) == 126
