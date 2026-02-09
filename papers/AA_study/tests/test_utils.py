from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch
from papers.AA_study.utils.utils import (
    state_vector_to_density_matrix,
    normalize_features,
    find_mode_photon_config,
)
from papers.AA_study.utils.datasets import generate_fig_2_dataset


def test_state_vector_to_density_matrix():
    plus_state = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    assert np.allclose(
        [[0.5, 0.5], [0.5, 0.5]], state_vector_to_density_matrix(plus_state)
    )
    minus_state = torch.tensor([1 / np.sqrt(2), -1 / np.sqrt(2)])
    assert np.allclose(
        [[0.5, -0.5], [-0.5, 0.5]], state_vector_to_density_matrix(minus_state)
    )
    minus_i_state = torch.tensor([1 / np.sqrt(2), -1.0j / np.sqrt(2)])
    assert np.allclose(
        [[0.5, 0.5j], [-0.5j, 0.5]], state_vector_to_density_matrix(minus_i_state)
    )


def test_normalize_features():
    dataset = generate_fig_2_dataset()
    norm_dataset = normalize_features(dataset, [-5, -5], [5, 5])

    for i in norm_dataset:
        assert i >= 0
        assert i <= 1


def test_find_mode_photon_config():
    m, n = find_mode_photon_config(num_features=2004)
    assert m == 10
    assert n == 5
    m, n = find_mode_photon_config(num_features=2001)
    assert m == 9
    assert n == 4
    m, n = find_mode_photon_config(num_features=56)
    assert m == 6
    assert n == 3
