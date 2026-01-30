from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch
from papers.AA_study.utils.utils import state_vector_to_density_matrix


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
