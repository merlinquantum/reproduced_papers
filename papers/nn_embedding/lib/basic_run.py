import json
import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.lib.merlin_based_model import create_basic_merlin_model
from papers.nn_embedding.lib.gate_based_model import create_paper_models
from papers.nn_embedding.utils.data import data_load_and_process
from papers.nn_embedding.utils.plotting import (
    plot_trace_distance,
    quick_loss_plot,
    plot_accuracies,
)


def to_serializable_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, np.generic):
        return values.item()
    if isinstance(values, (list, tuple)):
        return [to_serializable_list(value) for value in values]
    return values


def basic_train_and_evaluate(
    dataset="mnist",
    batch_size: int = 100,
    num_epochs_training_embedding: int = 50,
    num_epochs_training_classifier: int = 50,
    lr: float = 0.01,
):
    print(f"Creating models")
    gate_based_model_1, gate_based_model_2, gate_based_model_3 = create_paper_models()

    merlin_based_model = create_basic_merlin_model()
    print(f"Models created")

    print("Model 1")
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=dataset, feature_reduction="PCA8", classes=[0, 1], samples_per_class=150
    )
    print("Data loaded")

    print("Training the embedding")
    (
        loss_list_embedding,
        train_distances,
        test_distances,
        train_lower_bound,
        test_lower_bound,
    ) = merlin_based_model.train_embedding(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=batch_size,
        num_epochs=num_epochs_training_embedding,
        lr=lr,
        return_data=True,
    )

    print("Training the classifier")
    loss_list_classier, train_acc, test_acc = merlin_based_model.train_classifier(
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
    quick_loss_plot(loss_list_embedding, filename="quick_loss_plot_embedding.pdf")
    quick_loss_plot(loss_list_classier, filename="quick_loss_plot_classifier.pdf")
    plot_accuracies(train_acc, test_acc)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "basic_run_results.json"
    payload = {
        "loss_list_embedding": to_serializable_list(loss_list_embedding),
        "train_distances": to_serializable_list(train_distances),
        "test_distances": to_serializable_list(test_distances),
        "train_lower_bound": train_lower_bound,
        "test_lower_bound": test_lower_bound,
        "loss_list_classier": to_serializable_list(loss_list_classier),
        "train_acc": to_serializable_list(train_acc),
        "test_acc": to_serializable_list(test_acc),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {output_path}")


basic_train_and_evaluate()
