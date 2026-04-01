import merlin as ml
import torch.nn as nn

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.utils.utils import (
    randomize_trainable_parameters,
)


def create_gate_based_fig_2_3_models():
    # PCA 8
    classical_model_8 = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
    )
    randomize_trainable_parameters(classical_model_8)

    # Full classical_model
    classical_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
    )
    randomize_trainable_parameters(classical_model)
    return classical_model_8, classical_model


def create_gate_based_fig_5_models():
    # PCA 8
    classical_model_4 = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
    )
    randomize_trainable_parameters(classical_model_4)

    # Full classical_model
    classical_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
    )
    randomize_trainable_parameters(classical_model)
    return classical_model_4, classical_model
