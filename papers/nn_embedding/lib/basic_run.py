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
from papers.nn_embedding.utils.plotting import (
    plot_trace_distance,
    quick_loss_plot,
    plot_accuracies,
)


def basic_train_and_evaluate(
    dataset="mnist",
    batch_size: int = 25,
    num_epochs_training_embedding: int = 50,
    num_epochs_training_classifier: int = 50,
    lr: float = 0.01,
):
    print(f"Creating models")
    gate_based_model_1, gate_based_model_2, gate_based_model_3 = create_paper_models()
    print(f"Models created")

    print("Model 1")
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=dataset, feature_reduction="PCA8", classes=[0, 1], samples_per_class=150
    )
    print("Data loaded")

    print("Training the embedding")
    loss_list_embedding, train_distances, test_distances = (
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

    print("Training the classifier")
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

    plot_trace_distance(train_distances, test_distances)
    quick_loss_plot(loss_list_embedding)
    quick_loss_plot(loss_list_classier)
    plot_accuracies(train_acc, test_acc)


basic_train_and_evaluate()
