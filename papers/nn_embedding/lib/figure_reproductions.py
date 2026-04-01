import sys
from pathlib import Path
import torch.nn as nn
import merlin as ml
import json
import gc
from copy import deepcopy

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
from papers.nn_embedding.utils.utils import to_serializable_list, get_error_bound
from papers.nn_embedding.utils.plotting import (
    plot_figure_2_bc,
    plot_figure_3,
    plot_figure_5,
)


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
    for i in range(num_repetitions):

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


def reproduce_figure_3(
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
    layers_to_test: list[int] = [1, 2, 3],
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

    for i in range(num_repetitions):
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

        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        backend = "merlin" if use_merlin else "gate_based"
        output_path = results_dir / f"figure_3_{backend}_results.json"
        output_path.write_text(json.dumps(payload))

    plot_figure_3(
        loss_lists_classifier=payload["loss_lists_classifier"],
        test_accuracies=payload["test_accuracies"],
        layers_to_test=payload["config"]["layers_to_test"],
        run_dir=results_dir,
        filename=f"figure_3_{backend}.pdf",
    )


# Following the paper even though the code does not say this, The goal is to study the generalization error upper bound changing with nqe
def reproduce_figure_5(
    dataset: str = "mnist",
    use_merlin: bool = False,
    batch_size: int = 100,
    num_epochs_training_embedding: int = 50,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_class: int = 150,
    num_repetitions: int = 5,
    weights: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
):
    keys = ("pca_nqe", "nqe", "without_nqe")

    results = {
        "generalization_error": {key: [] for key in keys},
    }

    # load the data
    x_train_PCA4, x_test_PCA4, y_train_PCA4, y_test_PCA4 = data_load_and_process(
        dataset=dataset,
        feature_reduction=8,
        classes=[0, 1],
        samples_per_class=samples_per_class,
    )
    y_train_PCA4_minus_one = [-1 if y == 0 else 1 for y in y_train_PCA4]
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=dataset,
        feature_reduction=False,
        classes=[0, 1],
        samples_per_class=samples_per_class,
    )
    y_train_minus_one = [-1 if y == 0 else 1 for y in y_train]

    for i in range(num_repetitions):
        ################################################################# MerLin-based
        if use_merlin:
            embedder, classical_model_4, classical_model = create_merlin_fig_5_models()
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
            errors = get_error_bound(weights, kernel_matrix, y_train_PCA4_minus_one)

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
            errors = get_error_bound(weights, kernel_matrix, y_train_minus_one)

            results["generalization_error"]["nqe"].append(errors)
            del model

            # No NQE
            model = NeuralEmbeddingMerLinKernel(
                classical_model=deepcopy(classical_model_4),
                quantum_embedding_layer=deepcopy(embedder),
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            errors = get_error_bound(weights, kernel_matrix, y_train_PCA4_minus_one)
            results["generalization_error"]["without_nqe"].append(errors)

            del model, embedder, classical_model_4, classical_model

        ################################################################# Gate-based
        else:
            classical_model_4, classical_model = create_gate_based_fig_5_models()
            # PCA_NQE
            print("PCA_NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=deepcopy(classical_model_4),
                quantum_embedding_layer=EmbeddingCallable.Four_QuantumEmbedding1,
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
            errors = get_error_bound(weights, kernel_matrix, y_train_PCA4_minus_one)

            results["generalization_error"]["pca_nqe"].append(errors)
            del model

            # NQE
            print("NQE")
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=classical_model,
                quantum_embedding_layer=EmbeddingCallable.Four_QuantumEmbedding1,
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
            errors = get_error_bound(weights, kernel_matrix, y_train_minus_one)

            results["generalization_error"]["nqe"].append(errors)
            del model

            # No NQE
            model = NeuralEmbeddingGateBasedKernel(
                num_qubits=4,
                classical_model=deepcopy(classical_model_4),
                quantum_embedding_layer=EmbeddingCallable.Four_QuantumEmbedding1,
            )
            print("Calculating the error")
            kernel_matrix = model.compute_kernel_matrix(x_train_PCA4)
            errors = get_error_bound(weights, kernel_matrix, y_train_PCA4_minus_one)
            results["generalization_error"]["without_nqe"].append(errors)

            del model, embedder, classical_model_4, classical_model

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
            },
        }

        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        backend = "merlin" if use_merlin else "gate_based"
        output_path = results_dir / f"figure_5_{backend}_results.json"
        output_path.write_text(json.dumps(payload))

    plot_figure_5(
        generalization_error=payload["generalization_error"],
        weights=weights,
        run_dir=results_dir,
        filename=f"figure_5_{backend}.pdf",
    )


reproduce_figure_3(use_merlin=False)
