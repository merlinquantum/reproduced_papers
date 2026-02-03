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
from papers.AA_study.utils.qiskit_utils import reshape_input


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


def test_reshape_input():
    one_input_tensor = torch.rand((21))
    assert reshape_input(one_input_tensor).shape == (1, 32)

    perfect_input_tensor = torch.rand((13, 8))
    assert reshape_input(perfect_input_tensor).shape == (13, 8)

    only_one_dim_input_tensor = torch.rand((8))
    assert reshape_input(only_one_dim_input_tensor).shape == (1, 8)

    bad_input_tensor = torch.rand((22, 1, 3, 4, 1, 8))
    assert reshape_input(bad_input_tensor).shape == (22, 128)

    bad_input_tensor = torch.rand((22, 1, 32, 32))
    assert reshape_input(bad_input_tensor).shape == (22, 1024)
