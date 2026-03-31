import sys
from pathlib import Path
import torch.nn as nn
import torch
import merlin as ml
import json
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from papers.nn_embedding.lib.merlin_based_model import NeuralEmbeddingMerLinModel
from papers.nn_embedding.lib.gate_based_model import NeuralEmbeddingGateBasedModel
from papers.nn_embedding.utils.gate_based_embedding import (
    Four_QuantumEmbedding2,
    FourQCNN,
)
from papers.nn_embedding.lib.training_without_nqe import (
    train_gate_based,
    train_merlin_based,
)
from papers.nn_embedding.utils.data import data_load_and_process
from papers.nn_embedding.utils.utils import to_serializable_list
from papers.nn_embedding.utils.plotting import plot_figure_2_bc


def _randomize_trainable_parameters(module: nn.Module) -> None:
    """Force a fresh random initialization for each repetition."""
    for submodule in module.modules():
        if submodule is module:
            continue
        if hasattr(submodule, "reset_parameters"):
            submodule.reset_parameters()

    for param in module.parameters():
        if param.requires_grad:
            with torch.no_grad():
                param.uniform_(-torch.pi, torch.pi)


def reproduce_figure_2(
    dataset: str = "mnist",
    use_merlin: bool = False,
    batch_size: int = 100,
    num_epochs_training_embedding: int = 50,
    num_epochs_training_classifier: int = 50,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_class: int = 150,
    num_classes: int = 2,
    num_repetitions: int = 5,
):
    keys = ("pca_nqe", "nqe", "without_nqe")
    embedding_keys = ("pca_nqe", "nqe")

    results = {
        "loss_lists_embedding": {key: [] for key in embedding_keys},
        "training_distances": {key: [] for key in keys},
        "testing_distances": {key: [] for key in keys},
        "train_lower_bounds": {key: [] for key in keys},
        "test_lower_bounds": {key: [] for key in keys},
        "loss_lists_classifier": {key: [] for key in keys},
        "train_accuracies": {key: [] for key in keys},
        "test_accuracies": {key: [] for key in keys},
    }

    # load the data
    x_train_PCA8, x_test_PCA8, y_train_PCA8, y_test_PCA8 = data_load_and_process(
        dataset=dataset,
        feature_reduction="PCA8",
        classes=[0, 1],
        samples_per_class=samples_per_class,
    )
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=dataset,
        feature_reduction=False,
        classes=[0, 1],
        samples_per_class=samples_per_class,
    )
    for i in range(num_repetitions):
        if use_merlin:
            # Quantum embedding
            circ = ml.CircuitBuilder(n_modes=8)
            circ.add_entangling_layer()
            embedder = ml.QuantumLayer(
                input_size=0,
                builder=circ,
                n_photons=4,
                measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            )
            _randomize_trainable_parameters(embedder)

            # Quantum classifier
            circ = ml.CircuitBuilder(n_modes=8)
            circ.add_entangling_layer()
            classifier = ml.QuantumLayer(
                builder=circ,
                n_photons=4,
                amplitude_encoding=True,
                measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            )
            _randomize_trainable_parameters(classifier)

            # PCA 8
            classical_model_8 = nn.Sequential(
                nn.Linear(8, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, sum([i.numel() for i in embedder.parameters()])),
            )
            _randomize_trainable_parameters(classical_model_8)

            # Full classical_model
            classical_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, sum([i.numel() for i in embedder.parameters()])),
            )
            _randomize_trainable_parameters(classical_model)

            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingMerLinModel(
                classical_model=classical_model_8,
                quantum_embedding_layer=deepcopy(embedder),
                quantum_classifier=deepcopy(classifier),
                num_classes=num_classes,
            )
            _randomize_trainable_parameters(model)
            print("Training embedding")
            (
                loss_list_embedding,
                train_distances,
                test_distances,
                train_lower_bound,
                test_lower_bound,
            ) = model.train_embedding(
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=True,
            )
            print("Training classifier")
            loss_list_classifier, train_acc, test_acc = model.train_classifier(
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                batch_size=batch_size,
                num_epochs=num_epochs_training_classifier,
                lr=lr,
                return_data=True,
            )

            results["loss_lists_embedding"]["pca_nqe"].append(loss_list_embedding)
            results["training_distances"]["pca_nqe"].append(train_distances)
            results["testing_distances"]["pca_nqe"].append(test_distances)
            results["train_lower_bounds"]["pca_nqe"].append(train_lower_bound)
            results["test_lower_bounds"]["pca_nqe"].append(test_lower_bound)
            results["loss_lists_classifier"]["pca_nqe"].append(loss_list_classifier)
            results["train_accuracies"]["pca_nqe"].append(train_acc)
            results["test_accuracies"]["pca_nqe"].append(test_acc)

            # NQE
            print("NQE")
            model = NeuralEmbeddingMerLinModel(
                classical_model=classical_model,
                quantum_embedding_layer=deepcopy(embedder),
                quantum_classifier=deepcopy(classifier),
                num_classes=num_classes,
            )
            _randomize_trainable_parameters(model)
            print("Training embedding")
            (
                loss_list_embedding,
                train_distances,
                test_distances,
                train_lower_bound,
                test_lower_bound,
            ) = model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=True,
            )
            print("Training classifier")
            loss_list_classifier, train_acc, test_acc = model.train_classifier(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                batch_size=batch_size,
                num_epochs=num_epochs_training_classifier,
                lr=lr,
                return_data=True,
            )

            results["loss_lists_embedding"]["nqe"].append(loss_list_embedding)
            results["training_distances"]["nqe"].append(train_distances)
            results["testing_distances"]["nqe"].append(test_distances)
            results["train_lower_bounds"]["nqe"].append(train_lower_bound)
            results["test_lower_bounds"]["nqe"].append(test_lower_bound)
            results["loss_lists_classifier"]["nqe"].append(loss_list_classifier)
            results["train_accuracies"]["nqe"].append(train_acc)
            results["test_accuracies"]["nqe"].append(test_acc)

            # No NQE
            print("No NQE")
            print("Training classifier")
            (
                loss_list,
                train_accs,
                test_accs,
                train_distance,
                test_distance,
                train_lower_bound,
                test_lower_bound,
            ) = train_merlin_based(
                quantum_embedding_layer=deepcopy(embedder),
                quantum_classifier=deepcopy(classifier),
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                batch_size=batch_size,
                num_epochs=num_epochs_training_classifier,
                lr=lr,
                distance=distance,
                return_data=True,
                num_classes=num_classes,
            )
            results["training_distances"]["without_nqe"].append([train_distance])
            results["testing_distances"]["without_nqe"].append([test_distance])
            results["train_lower_bounds"]["without_nqe"].append(train_lower_bound)
            results["test_lower_bounds"]["without_nqe"].append(test_lower_bound)
            results["loss_lists_classifier"]["without_nqe"].append(loss_list)
            results["train_accuracies"]["without_nqe"].append(train_accs)
            results["test_accuracies"]["without_nqe"].append(test_accs)
        else:

            # PCA 8
            classical_model_8 = nn.Sequential(
                nn.Linear(8, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 8),
            )
            _randomize_trainable_parameters(classical_model_8)

            # Full classical_model
            classical_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 8),
            )
            _randomize_trainable_parameters(classical_model)
            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingGateBasedModel(
                num_qubits=4,
                classical_model=classical_model_8,
                quantum_embedding_layer=Four_QuantumEmbedding2,
                quantum_classifier=FourQCNN,
                quantum_classifier_params_shape=(30),
                num_classes=num_classes,
            )
            _randomize_trainable_parameters(model)
            print("Training embedding")
            (
                loss_list_embedding,
                train_distances,
                test_distances,
                train_lower_bound,
                test_lower_bound,
            ) = model.train_embedding(
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=True,
            )
            print("Training classifier")
            loss_list_classifier, train_acc, test_acc = model.train_classifier(
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                batch_size=batch_size,
                num_epochs=num_epochs_training_classifier,
                lr=lr,
                return_data=True,
            )

            results["loss_lists_embedding"]["pca_nqe"].append(loss_list_embedding)
            results["training_distances"]["pca_nqe"].append(train_distances)
            results["testing_distances"]["pca_nqe"].append(test_distances)
            results["train_lower_bounds"]["pca_nqe"].append(train_lower_bound)
            results["test_lower_bounds"]["pca_nqe"].append(test_lower_bound)
            results["loss_lists_classifier"]["pca_nqe"].append(loss_list_classifier)
            results["train_accuracies"]["pca_nqe"].append(train_acc)
            results["test_accuracies"]["pca_nqe"].append(test_acc)

            # NQE
            print("NQE")
            model = NeuralEmbeddingGateBasedModel(
                num_qubits=4,
                classical_model=classical_model,
                quantum_embedding_layer=Four_QuantumEmbedding2,
                quantum_classifier=FourQCNN,
                quantum_classifier_params_shape=(30),
                num_classes=num_classes,
            )
            _randomize_trainable_parameters(model)
            print("Training embedding")
            (
                loss_list_embedding,
                train_distances,
                test_distances,
                train_lower_bound,
                test_lower_bound,
            ) = model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=True,
            )
            print("Training classifier")
            loss_list_classifier, train_acc, test_acc = model.train_classifier(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                batch_size=batch_size,
                num_epochs=num_epochs_training_classifier,
                lr=lr,
                return_data=True,
            )

            results["loss_lists_embedding"]["nqe"].append(loss_list_embedding)
            results["training_distances"]["nqe"].append(train_distances)
            results["testing_distances"]["nqe"].append(test_distances)
            results["train_lower_bounds"]["nqe"].append(train_lower_bound)
            results["test_lower_bounds"]["nqe"].append(test_lower_bound)
            results["loss_lists_classifier"]["nqe"].append(loss_list_classifier)
            results["train_accuracies"]["nqe"].append(train_acc)
            results["test_accuracies"]["nqe"].append(test_acc)

            # No NQE
            print("No NQE")
            print("Training classifier")
            (
                loss_list,
                train_accs,
                test_accs,
                train_distance,
                test_distance,
                train_lower_bound,
                test_lower_bound,
            ) = train_gate_based(
                num_qubits=4,
                quantum_embedding_layer=Four_QuantumEmbedding2,
                quantum_classifier=FourQCNN,
                quantum_classifier_params_shape=(30),
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                batch_size=batch_size,
                num_epochs=num_epochs_training_classifier,
                lr=lr,
                distance=distance,
                return_data=True,
                num_classes=num_classes,
            )
            results["training_distances"]["without_nqe"].append([train_distance])
            results["testing_distances"]["without_nqe"].append([test_distance])
            results["train_lower_bounds"]["without_nqe"].append(train_lower_bound)
            results["test_lower_bounds"]["without_nqe"].append(test_lower_bound)
            results["loss_lists_classifier"]["without_nqe"].append(loss_list)
            results["train_accuracies"]["without_nqe"].append(train_accs)
            results["test_accuracies"]["without_nqe"].append(test_accs)

        print(f"Repetition {i+1} done")

    payload = {
        "loss_lists_embedding": to_serializable_list(results["loss_lists_embedding"]),
        "training_distances": to_serializable_list(results["training_distances"]),
        "testing_distances": to_serializable_list(results["testing_distances"]),
        "train_lower_bounds": to_serializable_list(results["train_lower_bounds"]),
        "test_lower_bounds": to_serializable_list(results["test_lower_bounds"]),
        "loss_lists_classifier": to_serializable_list(results["loss_lists_classifier"]),
        "train_accuracies": to_serializable_list(results["train_accuracies"]),
        "test_accuracies": to_serializable_list(results["test_accuracies"]),
        "config": {
            "dataset": dataset,
            "use_merlin": use_merlin,
            "batch_size": batch_size,
            "num_epochs_training_embedding": num_epochs_training_embedding,
            "num_epochs_training_classifier": num_epochs_training_classifier,
            "lr": lr,
            "distance": distance,
            "samples_per_class": samples_per_class,
            "num_classes": num_classes,
            "num_repetitions": num_repetitions,
        },
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    backend = "merlin" if use_merlin else "gate_based"
    output_path = results_dir / f"figure_2_{backend}_results.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {output_path}")

    plot_figure_2_bc(
        train_pca_nqe=results["training_distances"]["pca_nqe"],
        train_nqe=results["training_distances"]["nqe"],
        test_pca_nqe=results["testing_distances"]["pca_nqe"],
        test_nqe=results["testing_distances"]["nqe"],
        baseline_trace_distance=results["training_distances"]["without_nqe"],
        pca_nqe_losses=results["loss_lists_classifier"]["pca_nqe"],
        nqe_losses=results["loss_lists_classifier"]["nqe"],
        without_nqe_losses=results["loss_lists_classifier"]["without_nqe"],
        lower_bound_pca_nqe=results["train_lower_bounds"]["pca_nqe"],
        lower_bound_nqe=results["train_lower_bounds"]["nqe"],
        lower_bound_without_nqe=results["train_lower_bounds"]["without_nqe"],
        accuracy_rows=[
            (
                "Without NQE",
                [curve[-1] for curve in results["test_accuracies"]["without_nqe"]],
            ),
            ("PCA-NQE", [curve[-1] for curve in results["test_accuracies"]["pca_nqe"]]),
            ("NQE", [curve[-1] for curve in results["test_accuracies"]["nqe"]]),
        ],
        run_dir=results_dir,
        filename=f"figure_2_bc_{backend}.pdf",
    )

    return payload


reproduce_figure_2(use_merlin=True)
