import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from papers.AA_study.utils.datasets import generate_fig_2_dataset  # noqa: E402
from papers.AA_study.utils.utils import (  # noqa: E402
    find_mode_photon_config,
    normalize_features,
    state_vector_to_density_matrix,
)


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
    norm_dataset = normalize_features(dataset, [-5, -5], [5, 5]).tensors[0]

    for i in norm_dataset:
        assert i[0] >= 0
        assert i[0] <= 1
        assert i[1] >= 0
        assert i[1] <= 1


def test_find_mode_photon_config():
    m, n = find_mode_photon_config(num_features=2004)
    assert m == 11
    assert n == 5
    m, n = find_mode_photon_config(num_features=2001)
    assert m == 10
    assert n == 5
    m, n = find_mode_photon_config(num_features=56)
    assert m == 6
    assert n == 3
