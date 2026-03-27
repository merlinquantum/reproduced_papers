import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.lib.gate_based_model import create_paper_models
from papers.nn_embedding.utils.data import data_load_and_process


def basic_train_and_evaluate(
    dataset="mnist",
    batch_size: int = 25,
    num_epochs_training_embedding: int = 100,
    num_epochs_training_classifier: int = 100,
    lr: float = 0.01,
):
    gate_based_model_1, gate_based_model_2, gate_based_model_3 = create_paper_models()

    print("Model 1")
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=dataset, feature_reduction="PCA8", classes=[0, 1]
    )
    loss_list_embedding, train_distance, test_distance = (
        gate_based_model_1.train_embedding(
            x_train,
            y_train,
            x_test,
            y_test,
            batch_size=batch_size,
            num_epochs=num_epochs_training_embedding,
            lr=lr,
            return_data=True,
        )
    )

    loss_list_classier, train_acc, test_acc = gate_based_model_1.train_classifier(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=batch_size,
        num_epochs=num_epochs_training_classifier,
        lr=lr,
        return_data=True,
    )
