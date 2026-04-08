import sys
from pathlib import Path
import torch
import merlin as ml
import numpy as np
import json
import gc
from copy import deepcopy
from math import comb
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from papers.nn_embedding.lib.merlin_based_model import (
    NeuralEmbeddingMerLinModel,
    NeuralEmbeddingMerLinKernel,
)
from papers.nn_embedding.lib.gate_based_model import (
    NeuralEmbeddingGateBasedModel,
    NeuralEmbeddingGateBasedKernel,
)
from papers.nn_embedding.utils.merlin_models import (
    create_merlin_fig_2_models,
    create_merlin_fig_3_models,
    create_trainable_merlin_layer_fig_3,
    create_merlin_fig_4_models,
    create_merlin_fig_5_models,
)
from papers.nn_embedding.utils.gate_based_models import (
    create_gate_based_fig_2_3_models,
    create_gate_based_fig_5_models,
)
from papers.nn_embedding.utils.gate_based_embedding import (
    EmbeddingCallable,
    FourQCNN,
    QCNN,
)
from papers.nn_embedding.lib.training_without_nqe import (
    train_gate_based,
    train_merlin_based,
)
from papers.nn_embedding.utils.data import data_load_and_process
from papers.nn_embedding.utils.utils import (
    to_serializable_list,
    get_error_bound,
    TransparentModel,
    assign_params,
    state_vector_to_density_matrix,
    two_design_deviation_gate_based,
    two_design_deviation_photonics,
    kernel_variance,
    get_local_dimension,
)
from papers.nn_embedding.utils.plotting import (
    plot_figure_2_bc,
    plot_figure_3,
    plot_figure_4,
    plot_figure_5,
    plot_figure_6,
)


def reproduce_figure_2(
    dataset: str = "mnist",
    use_merlin: bool = False,
    batch_size: int = 100,
    num_epochs_training_embedding: int = 50,
    num_epochs_training_classifier: int = 1000,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_class: int = 150,
    num_classes: int = 2,
    num_repetitions: int = 5,
    run_dir: Path = None,
    generate_graph: bool = True,
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

    for i in range(num_repetitions):
        # load the data
        x_train_PCA8, x_test_PCA8, y_train_PCA8, y_test_PCA8 = data_load_and_process(
            dataset=dataset,
            feature_reduction=8,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )
        x_train, x_test, y_train, y_test = data_load_and_process(
            dataset=dataset,
            feature_reduction=False,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )

        ################################################################# MerLin-based
        if use_merlin:
            embedder, classifier, classical_model_8, classical_model = (
                create_merlin_fig_2_models()
            )

            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingMerLinModel(
                classical_model=classical_model_8,
                quantum_embedding_layer=deepcopy(embedder),
                quantum_classifier=deepcopy(classifier),
                num_classes=num_classes,
            )
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
            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingMerLinModel(
                classical_model=classical_model,
                quantum_embedding_layer=deepcopy(embedder),
                quantum_classifier=deepcopy(classifier),
                num_classes=num_classes,
            )
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
            del model

            # No NQE
            print("No NQE")

            # Adapt the dataset to the number of parameters of the embedder
            number_of_params = 0
            for param in embedder.parameters():
                number_of_params += param.numel()

            if number_of_params > 28 * 28:
                raise ValueError(
                    "Not implemented yet for more params that number of features"
                )
            elif number_of_params < 28 * 28:
                x_train_no_nqe, x_test_no_nqe, y_train_no_nqe, y_test_no_nqe = (
                    data_load_and_process(
                        dataset=dataset,
                        feature_reduction=number_of_params,
                        classes=[0, 1],
                        samples_per_class=samples_per_class,
                    )
                )
            else:
                x_train_no_nqe, x_test_no_nqe, y_train_no_nqe, y_test_no_nqe = (
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                )
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
                x_train=x_train_no_nqe,
                y_train=y_train_no_nqe,
                x_test=x_test_no_nqe,
                y_test=y_test_no_nqe,
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
            del embedder, classifier, classical_model_8, classical_model

        ################################################################# Gate-based
        else:
            classical_model_8, classical_model = create_gate_based_fig_2_3_models()
            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingGateBasedModel(
                num_qubits=4,
                classical_model=classical_model_8,
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding2,
                quantum_classifier=FourQCNN,
                quantum_classifier_params_shape=(30),
                num_classes=num_classes,
            )
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
            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingGateBasedModel(
                num_qubits=4,
                classical_model=classical_model,
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding2,
                quantum_classifier=FourQCNN,
                quantum_classifier_params_shape=(30),
                num_classes=num_classes,
            )
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
            del model

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
                quantum_embedding_circuit=EmbeddingCallable().Four_QuantumEmbedding2,
                quantum_classifier_circuit=FourQCNN,
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
            del classical_model_8, classical_model

        print(f"Repetition {i+1} done")
        gc.collect()

        ################################################################# Writing the results
        payload = {
            "loss_lists_embedding": to_serializable_list(
                results["loss_lists_embedding"]
            ),
            "training_distances": to_serializable_list(results["training_distances"]),
            "testing_distances": to_serializable_list(results["testing_distances"]),
            "train_lower_bounds": to_serializable_list(results["train_lower_bounds"]),
            "test_lower_bounds": to_serializable_list(results["test_lower_bounds"]),
            "loss_lists_classifier": to_serializable_list(
                results["loss_lists_classifier"]
            ),
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

        results_dir = run_dir if run_dir is not None else PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        backend = "merlin" if use_merlin else "gate_based"
        output_path = results_dir / f"figure_2_{backend}_results.json"
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved results to {output_path}")

    if generate_graph:
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
                (
                    "PCA-NQE",
                    [curve[-1] for curve in results["test_accuracies"]["pca_nqe"]],
                ),
                ("NQE", [curve[-1] for curve in results["test_accuracies"]["nqe"]]),
            ],
            run_dir=results_dir,
            filename=f"figure_2_bc_{backend}.pdf",
        )

    return payload


def reproduce_figure_3(
    dataset: str = "mnist",
    use_merlin: bool = False,
    batch_size: int = 100,
    num_epochs_training_embedding: int = 50,
    num_epochs_training_classifier: int = 1000,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_class: int = 150,
    num_classes: int = 2,
    num_repetitions: int = 5,
    layers_to_test: list[int] = [1, 2, 3],
    run_dir: Path = None,
    generate_graph: bool = True,
):
    keys = ["pca_nqe", "nqe"]
    for i in layers_to_test:
        keys.append(f"layer_{i}")
    keys = tuple(keys)

    results = {
        "loss_lists_classifier": {key: [] for key in keys},
        "train_accuracies": {key: [] for key in keys},
        "test_accuracies": {key: [] for key in keys},
    }

    for i in range(num_repetitions):
        # load the data
        x_train_PCA8, x_test_PCA8, y_train_PCA8, y_test_PCA8 = data_load_and_process(
            dataset=dataset,
            feature_reduction=8,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )
        x_train, x_test, y_train, y_test = data_load_and_process(
            dataset=dataset,
            feature_reduction=False,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )
        ################################################################# MerLin-based
        if use_merlin:
            embedder, classifier, classical_model_8, classical_model = (
                create_merlin_fig_3_models()
            )
            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingMerLinModel(
                classical_model=classical_model_8,
                quantum_embedding_layer=deepcopy(embedder),
                quantum_classifier=deepcopy(classifier),
                num_classes=num_classes,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
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
            results["loss_lists_classifier"]["pca_nqe"].append(loss_list_classifier)
            results["train_accuracies"]["pca_nqe"].append(train_acc)
            results["test_accuracies"]["pca_nqe"].append(test_acc)
            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingMerLinModel(
                classical_model=classical_model,
                quantum_embedding_layer=deepcopy(embedder),
                quantum_classifier=deepcopy(classifier),
                num_classes=num_classes,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
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
            results["loss_lists_classifier"]["nqe"].append(loss_list_classifier)
            results["train_accuracies"]["nqe"].append(train_acc)
            results["test_accuracies"]["nqe"].append(test_acc)
            del model

            # No NQE
            print("No NQE")
            for layer in layers_to_test:
                print(f"Doing layer {layer}")
                trainable_embedder = create_trainable_merlin_layer_fig_3(layer)
                print("Training classifier")
                (
                    loss_list,
                    train_accs,
                    test_accs,
                    _,
                    _,
                    _,
                    _,
                ) = train_merlin_based(
                    quantum_embedding_layer=trainable_embedder,
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
                    trainable_embedding=True,
                    num_layers=layer,
                )

                results["loss_lists_classifier"][f"layer_{layer}"].append(loss_list)
                results["train_accuracies"][f"layer_{layer}"].append(train_accs)
                results["test_accuracies"][f"layer_{layer}"].append(test_accs)
            del embedder, classifier, classical_model_8, classical_model

        ################################################################# Gate-based
        else:
            classical_model_8, classical_model = create_gate_based_fig_2_3_models()
            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingGateBasedModel(
                num_qubits=8,
                classical_model=classical_model_8,
                quantum_embedding_layer=EmbeddingCallable().QuantumEmbedding1,
                quantum_classifier=QCNN,
                quantum_classifier_params_shape=(60),
                num_classes=num_classes,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train_PCA8,
                y_train=y_train_PCA8,
                x_test=x_test_PCA8,
                y_test=y_test_PCA8,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
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
            results["loss_lists_classifier"]["pca_nqe"].append(loss_list_classifier)
            results["train_accuracies"]["pca_nqe"].append(train_acc)
            results["test_accuracies"]["pca_nqe"].append(test_acc)
            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingGateBasedModel(
                num_qubits=8,
                classical_model=classical_model,
                quantum_embedding_layer=EmbeddingCallable().QuantumEmbedding1,
                quantum_classifier=QCNN,
                quantum_classifier_params_shape=(60),
                num_classes=num_classes,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
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
            results["loss_lists_classifier"]["nqe"].append(loss_list_classifier)
            results["train_accuracies"]["nqe"].append(train_acc)
            results["test_accuracies"]["nqe"].append(test_acc)
            del model
            del classical_model_8, classical_model
            # No NQE
            print("No NQE")
            print("Training classifier")
            for layer in layers_to_test:
                print(f"Doing layer {layer}")
                (
                    loss_list,
                    train_accs,
                    test_accs,
                    _,
                    _,
                    _,
                    _,
                ) = train_gate_based(
                    num_qubits=8,
                    quantum_embedding_circuit=EmbeddingCallable(
                        N_layers=layer
                    ).QuantumEmbedding1Trainable,
                    quantum_classifier_circuit=QCNN,
                    quantum_classifier_params_shape=(60),
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
                    trainable_embedding=True,
                    embedding_params_shape=(36 * layer),
                )
                results["loss_lists_classifier"][f"layer_{layer}"].append(loss_list)
                results["train_accuracies"][f"layer_{layer}"].append(train_accs)
                results["test_accuracies"][f"layer_{layer}"].append(test_accs)

        print(f"Repetition {i+1} done")
        gc.collect()

        ################################################################# Writing the data

        payload = {
            "loss_lists_classifier": to_serializable_list(
                results["loss_lists_classifier"]
            ),
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
                "layers_to_test": layers_to_test,
            },
        }

        results_dir = run_dir if run_dir is not None else PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        backend = "merlin" if use_merlin else "gate_based"
        output_path = results_dir / f"figure_3_{backend}_results.json"
        output_path.write_text(json.dumps(payload))

    if generate_graph:
        plot_figure_3(
            loss_lists_classifier=payload["loss_lists_classifier"],
            test_accuracies=payload["test_accuracies"],
            layers_to_test=payload["config"]["layers_to_test"],
            run_dir=results_dir,
            filename=f"figure_3_{backend}.pdf",
        )


def reproduce_figure_4(
    use_merlin: bool = False,
    batch_size: int = 25,
    num_epochs_training_embedding: int = 100,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_datatset: int = 400,
    num_datasets: int = 10,
    num_repetitions_per_dataset: int = 20,
    epsilon: float = 0.01,
    num_samples_int: int = 100,
    run_dir: Path = None,
    generate_graph: bool = True,
):
    keys = ("nqe", "without_nqe")

    results = {
        "effective_dimension": {key: [] for key in keys},
    }
    if use_merlin:
        dim = create_merlin_fig_4_models()[3]
        print(f"The Merlin dimension is {dim}")
    else:
        dim = 4

    for i in range(num_datasets):

        X, Y = make_classification(
            n_samples=int(1e6),
            n_features=dim,
            n_informative=4,
            n_clusters_per_class=4,
            return_X_y=True,
        )
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        for j in range(num_repetitions_per_dataset):
            ################################################################# MerLin-based
            if use_merlin:
                embedder, classifier, classical_model, _ = create_merlin_fig_4_models()
                # Create a dataset that has the correct dimension for the encoder

                # NQE
                print("NQE")
                model = NeuralEmbeddingMerLinModel(
                    classical_model=classical_model,
                    quantum_embedding_layer=deepcopy(embedder),
                    quantum_classifier=classifier,
                )
                print("Training embedding")
                model.train_embedding(
                    x_train=X[samples_per_datatset:],
                    y_train=Y[samples_per_datatset:],
                    distance=distance,
                    batch_size=batch_size,
                    num_epochs=num_epochs_training_embedding,
                    lr=lr,
                    return_data=False,
                )
                print("Calculating the dimension")

                n_vals, led_vals = get_local_dimension(
                    model.model, X, Y, epsilon=epsilon, num_samples=num_samples_int
                )
                results["effective_dimension"]["nqe"].append(led_vals)
                results["n_values"] = n_vals
                del model

                # No NQE
                print("No NQE")
                model = NeuralEmbeddingMerLinModel(
                    classical_model=TransparentModel(),
                    quantum_embedding_layer=deepcopy(embedder),
                    quantum_classifier=classifier,
                )
                print("Calculating the dimension")

                n_vals, led_vals = get_local_dimension(
                    model.model, X, Y, epsilon=epsilon, num_samples=num_samples_int
                )
                results["effective_dimension"]["without_nqe"].append(led_vals)

                del model, embedder, classifier, classical_model

            ################################################################# Gate-based
            else:
                classical_model_4, _ = create_gate_based_fig_5_models()
                # NQE
                print("NQE")
                model = NeuralEmbeddingGateBasedModel(
                    num_qubits=4,
                    classical_model=classical_model_4,
                    quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
                    quantum_classifier=FourQCNN,
                    quantum_classifier_params_shape=(30),
                )
                print("Training embedding")
                model.train_embedding(
                    x_train=X,
                    y_train=Y,
                    distance=distance,
                    batch_size=batch_size,
                    num_epochs=num_epochs_training_embedding,
                    lr=lr,
                    return_data=False,
                )
                print("Calculating the dimension")

                n_vals, led_vals = get_local_dimension(
                    model.model, X, Y, epsilon=epsilon, num_samples=num_samples_int
                )
                results["effective_dimension"]["nqe"].append(led_vals)
                results["n_values"] = n_vals
                del model

                # No NQE
                print("No NQE")
                model = NeuralEmbeddingGateBasedModel(
                    num_qubits=4,
                    classical_model=TransparentModel,
                    quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
                    quantum_classifier=FourQCNN,
                    quantum_classifier_params_shape=(30),
                )
                print("Training embedding")
                model.train_embedding(
                    x_train=X,
                    y_train=Y,
                    distance=distance,
                    batch_size=batch_size,
                    num_epochs=num_epochs_training_embedding,
                    lr=lr,
                    return_data=False,
                )
                print("Calculating the dimension")

                n_vals, led_vals = get_local_dimension(
                    model.model, X, Y, epsilon=epsilon, num_samples=num_samples_int
                )
                results["effective_dimension"]["without_nqe"].append(led_vals)

                del model, embedder, classical_model_4

            print(f"Repetition {i+1} done")
            gc.collect()

            ################################################################# Writing the data

            payload = {
                "effective_dimension": to_serializable_list(
                    results["effective_dimension"]
                ),
                "config": {
                    "use_merlin": use_merlin,
                    "batch_size": batch_size,
                    "num_epochs_training_embedding": num_epochs_training_embedding,
                    "lr": lr,
                    "distance": distance,
                    "samples_per_datatset": samples_per_datatset,
                    "num_datasets": num_datasets,
                    "num_repetitions_per_dataset": num_repetitions_per_dataset,
                    "epsilon": epsilon,
                    "num_samples_int": num_samples_int,
                },
            }

            results_dir = run_dir if run_dir is not None else PROJECT_ROOT / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            backend = "merlin" if use_merlin else "gate_based"
            output_path = results_dir / f"figure_4_{backend}_results.json"
            output_path.write_text(json.dumps(payload))

            print(
                f"Repetition {j+1}/{num_repetitions_per_dataset} done for datatset {i+1}/{num_datasets}"
            )

    if generate_graph:
        plot_figure_4(
            effective_dimension=payload["effective_dimension"],
            n_values=results.get("n_values"),
            run_dir=results_dir,
            filename=f"figure_4_{backend}.pdf",
        )


# Following the paper even though the code does not say this,
# The goal is to study the generalization error upper bound changing with nqe
def reproduce_figure_5(
    dataset: str = "mnist",
    use_merlin: bool = False,
    batch_size: int = 100,
    num_epochs_training_embedding: int = 1000,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_class: int = 500,
    num_repetitions: int = 5,
    weights: list[float] = np.arange(0.1, 1, 0.1).tolist(),
    run_dir: Path = None,
    generate_graph: bool = True,
):
    keys = ("pca_nqe", "nqe", "without_nqe")

    results = {
        "generalization_error": {key: [] for key in keys},
    }

    for i in range(num_repetitions):
        # load the data
        x_train_PCA4, x_test_PCA4, y_train_PCA4, y_test_PCA4 = data_load_and_process(
            dataset=dataset,
            feature_reduction=4,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )
        y_train_PCA4_minus_one = np.array([-1 if y == 0 else 1 for y in y_train_PCA4])
        x_train, x_test, y_train, y_test = data_load_and_process(
            dataset=dataset,
            feature_reduction=False,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )
        y_train_minus_one = np.array([-1 if y == 0 else 1 for y in y_train])
        ################################################################# MerLin-based
        if use_merlin:
            embedder, classical_model_4, classical_model, dim = (
                create_merlin_fig_5_models()
            )

            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingMerLinKernel(
                classical_model=deepcopy(classical_model_4),
                quantum_embedding_layer=deepcopy(embedder),
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train_PCA4,
                y_train=y_train_PCA4,
                x_test=x_test_PCA4,
                y_test=y_test_PCA4,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            errors = get_error_bound(
                weights, kernel_matrix.detach().numpy(), y_train_PCA4_minus_one
            )

            results["generalization_error"]["pca_nqe"].append(errors)
            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingMerLinKernel(
                classical_model=classical_model,
                quantum_embedding_layer=deepcopy(embedder),
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train)
            errors = get_error_bound(
                weights, kernel_matrix.detach().numpy(), y_train_minus_one
            )

            results["generalization_error"]["nqe"].append(errors)
            del model

            # No NQE — use PCA-4 data with a fresh (untrained) classical encoder
            # that maps 4 features to embedder param count, same architecture as
            # PCA-NQE but without training the embedding.
            print("No NQE")
            model = NeuralEmbeddingMerLinKernel(
                classical_model=deepcopy(classical_model_4),
                quantum_embedding_layer=deepcopy(embedder),
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            errors = get_error_bound(
                weights, kernel_matrix.detach().numpy(), y_train_PCA4_minus_one
            )

            results["generalization_error"]["without_nqe"].append(errors)

            del model, embedder, classical_model_4, classical_model

        ################################################################# Gate-based
        else:
            classical_model_4, classical_model = create_gate_based_fig_5_models()
            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=classical_model_4,
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train_PCA4,
                y_train=y_train_PCA4,
                x_test=x_test_PCA4,
                y_test=y_test_PCA4,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            errors = get_error_bound(
                weights, kernel_matrix.detach().numpy(), y_train_PCA4_minus_one
            )

            results["generalization_error"]["pca_nqe"].append(errors)
            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=classical_model,
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train)
            errors = get_error_bound(
                weights, kernel_matrix.detach().numpy(), y_train_minus_one
            )

            results["generalization_error"]["nqe"].append(errors)
            del model

            # No NQE
            print("No NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=deepcopy(classical_model_4),
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            errors = get_error_bound(
                weights, kernel_matrix.detach().numpy(), y_train_PCA4_minus_one
            )
            results["generalization_error"]["without_nqe"].append(errors)

            del model, classical_model_4, classical_model

        print(f"Repetition {i+1} done")
        gc.collect()

        ################################################################# Writing the data

        payload = {
            "generalization_error": to_serializable_list(
                results["generalization_error"]
            ),
            "config": {
                "dataset": dataset,
                "use_merlin": use_merlin,
                "batch_size": batch_size,
                "num_epochs_training_embedding": num_epochs_training_embedding,
                "lr": lr,
                "distance": distance,
                "samples_per_class": samples_per_class,
                "num_repetitions": num_repetitions,
                "weights": weights,
            },
        }

        results_dir = run_dir if run_dir is not None else PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        backend = "merlin" if use_merlin else "gate_based"
        output_path = results_dir / f"figure_5_{backend}_results.json"
        output_path.write_text(json.dumps(payload))

    if generate_graph:
        plot_figure_5(
            generalization_error=payload["generalization_error"],
            weights=weights,
            run_dir=results_dir,
            filename=f"figure_5_{backend}.pdf",
        )


def reproduce_figure_6(
    dataset: str = "mnist",
    use_merlin: bool = False,
    batch_size: int = 100,
    num_epochs_training_embedding: int = 50,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_class: int = 150,
    num_repetitions: int = 5,
    run_dir: Path = None,
    generate_graph: bool = True,
):
    keys = ("pca_nqe", "nqe", "without_nqe")

    results = {
        "train_deviation": {key: [] for key in keys},
        "test_deviation": {key: [] for key in keys},
        "train_kernel_var": {key: [] for key in keys},
        "test_kernel_var": {key: [] for key in keys},
    }

    for i in range(num_repetitions):
        # load the data
        x_train_PCA4, x_test_PCA4, y_train_PCA4, y_test_PCA4 = data_load_and_process(
            dataset=dataset,
            feature_reduction=4,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )
        x_train, x_test, y_train, y_test = data_load_and_process(
            dataset=dataset,
            feature_reduction=False,
            classes=[0, 1],
            samples_per_class=samples_per_class,
        )
        ################################################################# MerLin-based
        if use_merlin:
            embedder, classical_model_4, classical_model, dim = (
                create_merlin_fig_5_models()
            )
            # Create a dataset that has the correct dimension for the encoder
            (
                x_train_PCA_no_NQE,
                x_test_PCA_no_NQE,
                _,
                _,
            ) = data_load_and_process(
                dataset=dataset,
                feature_reduction=dim,
                classes=[0, 1],
                samples_per_class=samples_per_class,
            )

            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingMerLinKernel(
                classical_model=classical_model_4,
                quantum_embedding_layer=deepcopy(embedder),
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train_PCA4,
                y_train=y_train_PCA4,
                x_test=x_test_PCA4,
                y_test=y_test_PCA4,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the metrics")

            classical_data = model.classical_encoder(x_train_PCA4)
            states = assign_params(model.quantum_embedding_layer, classical_data)
            rhos_train = state_vector_to_density_matrix(states)

            classical_data = model.classical_encoder(x_test_PCA4)
            states = assign_params(model.quantum_embedding_layer, classical_data)
            rhos_test = state_vector_to_density_matrix(states)

            train_kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            test_kernel_matrix = model.compute_kernel_matrix(x_test_PCA4)

            results["train_deviation"]["pca_nqe"].append(
                two_design_deviation_photonics(rhos_train, comb(6, 2), x_train.shape[0])
            )
            results["test_deviation"]["pca_nqe"].append(
                two_design_deviation_photonics(rhos_test, comb(6, 2), x_test.shape[0])
            )
            results["train_kernel_var"]["pca_nqe"].append(
                kernel_variance(train_kernel_matrix)
            )
            results["test_kernel_var"]["pca_nqe"].append(
                kernel_variance(test_kernel_matrix)
            )

            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingMerLinKernel(
                classical_model=classical_model,
                quantum_embedding_layer=deepcopy(embedder),
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the metrics")

            classical_data = model.classical_encoder(x_train)
            states = assign_params(model.quantum_embedding_layer, classical_data)
            rhos_train = state_vector_to_density_matrix(states)

            classical_data = model.classical_encoder(x_test)
            states = assign_params(model.quantum_embedding_layer, classical_data)
            rhos_test = state_vector_to_density_matrix(states)

            train_kernel_matrix = model.compute_kernel_matrix(x_train)
            test_kernel_matrix = model.compute_kernel_matrix(x_test)
            results["train_deviation"]["nqe"].append(
                two_design_deviation_photonics(rhos_train, comb(6, 2), x_train.shape[0])
            )
            results["test_deviation"]["nqe"].append(
                two_design_deviation_photonics(rhos_test, comb(6, 2), x_test.shape[0])
            )
            results["train_kernel_var"]["nqe"].append(
                kernel_variance(train_kernel_matrix)
            )
            results["test_kernel_var"]["nqe"].append(
                kernel_variance(test_kernel_matrix)
            )

            del model

            # No NQE
            print("No NQE")
            model = NeuralEmbeddingMerLinKernel(
                classical_model=TransparentModel(),
                quantum_embedding_layer=deepcopy(embedder),
            )
            print("Calculating the metrics")

            classical_data = model.classical_encoder(x_train_PCA_no_NQE)
            states = assign_params(model.quantum_embedding_layer, classical_data)
            rhos_train = state_vector_to_density_matrix(states)

            classical_data = model.classical_encoder(x_test_PCA_no_NQE)
            states = assign_params(model.quantum_embedding_layer, classical_data)
            rhos_test = state_vector_to_density_matrix(states)

            train_kernel_matrix = model.compute_kernel_matrix(x_train_PCA_no_NQE)
            test_kernel_matrix = model.compute_kernel_matrix(x_test_PCA_no_NQE)
            results["train_deviation"]["without_nqe"].append(
                two_design_deviation_photonics(
                    rhos_train, comb(6, 2), x_train_PCA_no_NQE.shape[0]
                )
            )
            results["test_deviation"]["without_nqe"].append(
                two_design_deviation_photonics(
                    rhos_test, comb(6, 2), x_test_PCA_no_NQE.shape[0]
                )
            )
            results["train_kernel_var"]["without_nqe"].append(
                kernel_variance(train_kernel_matrix)
            )
            results["test_kernel_var"]["without_nqe"].append(
                kernel_variance(test_kernel_matrix)
            )

            del model, embedder, classical_model_4, classical_model

        ################################################################# Gate-based
        else:
            classical_model_4, classical_model = create_gate_based_fig_5_models()
            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=classical_model_4,
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train_PCA4,
                y_train=y_train_PCA4,
                x_test=x_test_PCA4,
                y_test=y_test_PCA4,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the metrics")
            rhos_train = torch.stack(
                tuple(
                    model.state_embedding_circuit(sample)
                    for sample in model.classical_encoder(x_train_PCA4)
                ),
                dim=0,
            )

            rhos_test = torch.stack(
                tuple(
                    model.state_embedding_circuit(sample)
                    for sample in model.classical_encoder(x_test_PCA4)
                ),
                dim=0,
            )

            train_kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            test_kernel_matrix = model.compute_kernel_matrix(x_test_PCA4)
            results["train_deviation"]["pca_nqe"].append(
                two_design_deviation_gate_based(rhos_train, 4, x_train.shape[0])
            )
            results["test_deviation"]["pca_nqe"].append(
                two_design_deviation_gate_based(rhos_test, 4, x_test.shape[0])
            )
            results["train_kernel_var"]["pca_nqe"].append(
                kernel_variance(train_kernel_matrix)
            )
            results["test_kernel_var"]["pca_nqe"].append(
                kernel_variance(test_kernel_matrix)
            )

            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=classical_model,
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
            )
            print("Training embedding")
            model.train_embedding(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                distance=distance,
                batch_size=batch_size,
                num_epochs=num_epochs_training_embedding,
                lr=lr,
                return_data=False,
            )
            print("Calculating the metrics")

            rhos_train = torch.stack(
                tuple(
                    model.state_embedding_circuit(sample)
                    for sample in model.classical_encoder(x_train)
                ),
                dim=0,
            )

            rhos_test = torch.stack(
                tuple(
                    model.state_embedding_circuit(sample)
                    for sample in model.classical_encoder(x_test)
                ),
                dim=0,
            )

            train_kernel_matrix = model.compute_kernel_matrix(x_train)
            test_kernel_matrix = model.compute_kernel_matrix(x_test)
            results["train_deviation"]["nqe"].append(
                two_design_deviation_gate_based(rhos_train, 4, x_train.shape[0])
            )
            results["test_deviation"]["nqe"].append(
                two_design_deviation_gate_based(rhos_test, 4, x_test.shape[0])
            )
            results["train_kernel_var"]["nqe"].append(
                kernel_variance(train_kernel_matrix)
            )
            results["test_kernel_var"]["nqe"].append(
                kernel_variance(test_kernel_matrix)
            )

            # No NQE
            print("No NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=TransparentModel(),
                quantum_embedding_layer=EmbeddingCallable().Four_QuantumEmbedding1,
            )
            print("Calculating the metrics")
            rhos_train = torch.stack(
                tuple(
                    model.state_embedding_circuit(sample)
                    for sample in model.classical_encoder(x_train_PCA4)
                ),
                dim=0,
            )

            rhos_test = torch.stack(
                tuple(
                    model.state_embedding_circuit(sample)
                    for sample in model.classical_encoder(x_test_PCA4)
                ),
                dim=0,
            )

            train_kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            test_kernel_matrix = model.compute_kernel_matrix(x_test_PCA4)
            results["train_deviation"]["without_nqe"].append(
                two_design_deviation_gate_based(rhos_train, 4, x_train.shape[0])
            )
            results["test_deviation"]["without_nqe"].append(
                two_design_deviation_gate_based(rhos_test, 4, x_test.shape[0])
            )
            results["train_kernel_var"]["without_nqe"].append(
                kernel_variance(train_kernel_matrix)
            )
            results["test_kernel_var"]["without_nqe"].append(
                kernel_variance(test_kernel_matrix)
            )

            del model, embedder, classical_model_4, classical_model

        print(f"Repetition {i+1} done")
        gc.collect()

        ################################################################# Writing the data

        payload = {
            "train_deviation": to_serializable_list(results["train_deviation"]),
            "test_deviation": to_serializable_list(results["test_deviation"]),
            "train_kernel_var": to_serializable_list(results["train_kernel_var"]),
            "test_kernel_var": to_serializable_list(results["test_kernel_var"]),
            "config": {
                "dataset": dataset,
                "use_merlin": use_merlin,
                "batch_size": batch_size,
                "num_epochs_training_embedding": num_epochs_training_embedding,
                "lr": lr,
                "distance": distance,
                "samples_per_class": samples_per_class,
                "num_repetitions": num_repetitions,
            },
        }

        results_dir = run_dir if run_dir is not None else PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        backend = "merlin" if use_merlin else "gate_based"
        output_path = results_dir / f"figure_6_{backend}_results.json"
        output_path.write_text(json.dumps(payload))

    if generate_graph:
        plot_figure_6(
            train_deviation=payload["train_deviation"],
            test_deviation=payload["test_deviation"],
            train_kernel_var=payload["train_kernel_var"],
            test_kernel_var=payload["test_kernel_var"],
            run_dir=results_dir,
            filename=f"figure_6_{backend}.pdf",
        )
