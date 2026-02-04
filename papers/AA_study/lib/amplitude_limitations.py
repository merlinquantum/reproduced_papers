import json
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.lib.qlayers import (
    angle_encoding_simple,
    amplitude_encoding_simple,
    PhotonicQCNN,
)
from papers.AA_study.lib.qiskit_models import single_qubit_model, qiskit_QCNN
from papers.AA_study.utils.datasets import (
    generate_fig_1_dataset,
    generate_fig_2_dataset,
    generate_fig_3_dataset,
    get_data_loader,
    get_binary_dataset,
)
from papers.AA_study.utils.states import superposition_state, mixed_state
from papers.AA_study.utils.plots import (
    plot_amplitude_encoding_limitations,
    plot_fig_4,
    plot_fig_5,
    plot_fig_7,
)
from papers.AA_study.utils.utils import (
    trace_distance,
    state_vector_to_density_matrix,
    basic_model_training,
)
from typing import List


def reproduce_fig_1(
    num_max_samples: int = 2000,
    run_dir: Path = None,
):
    SUP_STATE = superposition_state(1)
    distance_from_sup_state = [[], []]
    for sample_per_class in range(1, num_max_samples + 1):
        dataset = generate_fig_1_dataset(
            num_samples_per_class=sample_per_class, shuffle=False
        )
        class_1_features = dataset.tensors[0][:sample_per_class]
        normalized_class_1_features = [
            features / np.linalg.norm(features) for features in class_1_features
        ]
        class_1_density_matrices = [
            state_vector_to_density_matrix(x) for x in normalized_class_1_features
        ]
        class_1_expected_state = np.mean(class_1_density_matrices, axis=0)
        distance_from_sup_state[0].append(
            trace_distance(class_1_expected_state, SUP_STATE)
        )

        class_2_features = dataset.tensors[0][sample_per_class:]
        normalized_class_2_features = [
            features / np.linalg.norm(features) for features in class_2_features
        ]
        class_2_density_matrices = [
            state_vector_to_density_matrix(x) for x in normalized_class_2_features
        ]
        class_2_expected_state = np.mean(class_2_density_matrices, axis=0)
        distance_from_sup_state[1].append(
            trace_distance(class_2_expected_state, SUP_STATE)
        )
    plot_amplitude_encoding_limitations(
        distances=distance_from_sup_state,
        dataset_unshuffled=dataset,
        num_samples_per_class=num_max_samples,
        fig_simulated=1,
        run_dir=run_dir,
    )
    return distance_from_sup_state


def reproduce_fig_2(
    num_max_samples: int = 2000,
    run_dir: Path = None,
):
    MIXED_STATE = mixed_state(1)
    distance_from_mixed_state = [[], []]
    for sample_per_class in range(1, num_max_samples + 1):
        dataset = generate_fig_2_dataset(
            num_samples_per_class=sample_per_class, shuffle=False
        )
        class_1_features = dataset.tensors[0][:sample_per_class]
        normalized_class_1_features = [
            features / np.linalg.norm(features) for features in class_1_features
        ]
        class_1_density_matrices = [
            state_vector_to_density_matrix(x) for x in normalized_class_1_features
        ]
        class_1_expected_state = np.mean(class_1_density_matrices, axis=0)
        distance_from_mixed_state[0].append(
            trace_distance(class_1_expected_state, MIXED_STATE)
        )

        class_2_features = dataset.tensors[0][sample_per_class:]
        normalized_class_2_features = [
            features / np.linalg.norm(features) for features in class_2_features
        ]
        class_2_density_matrices = [
            state_vector_to_density_matrix(x) for x in normalized_class_2_features
        ]
        class_2_expected_state = np.mean(class_2_density_matrices, axis=0)
        distance_from_mixed_state[1].append(
            trace_distance(class_2_expected_state, MIXED_STATE)
        )
    plot_amplitude_encoding_limitations(
        distances=distance_from_mixed_state,
        dataset_unshuffled=dataset,
        num_samples_per_class=num_max_samples,
        fig_simulated=2,
        run_dir=run_dir,
    )
    return distance_from_mixed_state


def reproduce_fig_3(
    num_max_samples: int = 2000,
    run_dir: Path = None,
):
    distance_between_classes = []
    for sample_per_class in range(1, num_max_samples + 1):
        dataset = generate_fig_3_dataset(
            num_samples_per_class=sample_per_class, shuffle=False
        )
        class_1_features = dataset.tensors[0][:sample_per_class]
        normalized_class_1_features = [
            features / np.linalg.norm(features) for features in class_1_features
        ]
        class_1_density_matrices = [
            state_vector_to_density_matrix(x) for x in normalized_class_1_features
        ]
        class_1_expected_state = np.mean(class_1_density_matrices, axis=0)

        class_2_features = dataset.tensors[0][sample_per_class:]
        normalized_class_2_features = [
            features / np.linalg.norm(features) for features in class_2_features
        ]
        class_2_density_matrices = [
            state_vector_to_density_matrix(x) for x in normalized_class_2_features
        ]
        class_2_expected_state = np.mean(class_2_density_matrices, axis=0)
        distance_between_classes.append(
            trace_distance(
                class_1_expected_state,
                class_2_expected_state,
            )
        )
    plot_amplitude_encoding_limitations(
        distances=distance_between_classes,
        dataset_unshuffled=dataset,
        num_samples_per_class=num_max_samples,
        fig_simulated=3,
        run_dir=run_dir,
    )
    return distance_between_classes


def reproduce_fig_4(
    batch_size: int = 50,
    num_epochs: int = 1000,
    lr: float = 0.01,
    num_samples_per_class: int = 2000,
    run_dir: Path = None,
):
    qiskit_losses = [[], [], []]
    amplitude_losses = [[], [], []]
    angle_losses = [[], [], []]

    qiskit_accuracies = [[], [], []]
    amplitude_accuracies = [[], [], []]
    angle_accuracies = [[], [], []]

    dataset_loader_1 = get_data_loader(
        generate_fig_1_dataset(num_samples_per_class=num_samples_per_class),
        batch_size=batch_size,
    )
    dataset_loader_2 = get_data_loader(
        generate_fig_2_dataset(num_samples_per_class=num_samples_per_class),
        batch_size=batch_size,
    )
    dataset_loader_3 = get_data_loader(
        generate_fig_3_dataset(num_samples_per_class=num_samples_per_class),
        batch_size=batch_size,
    )

    for L in [1, 10, 30]:
        print(f"Testing {L} layers")
        for i, loader in enumerate(
            [dataset_loader_1, dataset_loader_2, dataset_loader_3]
        ):
            qiskit_model = single_qubit_model(num_layers=L)
            amplitude_model = amplitude_encoding_simple(num_features=2, num_layers=L)
            angle_model = angle_encoding_simple(num_features=2, num_layers=L)

            print("Qiskit model:")
            _, accuracies, losses = basic_model_training(
                qiskit_model, loader, lr=lr, num_epochs=num_epochs
            )
            qiskit_accuracies[i].append(accuracies)
            qiskit_losses[i].append(losses)

            print("Amplitude model:")
            _, accuracies, losses = basic_model_training(
                amplitude_model, loader, lr=lr, num_epochs=num_epochs
            )
            amplitude_accuracies[i].append(accuracies)
            amplitude_losses[i].append(losses)

            print("Angle model:")
            _, accuracies, losses = basic_model_training(
                angle_model, loader, lr=lr, num_epochs=num_epochs
            )
            angle_accuracies[i].append(accuracies)
            angle_losses[i].append(losses)

        json_payload = {
            "qiskit_accuracies": [
                [[float(v) for v in t] for t in l] for l in qiskit_accuracies
            ],
            "amplitude_accuracies": [
                [[float(v) for v in t] for t in l] for l in amplitude_accuracies
            ],
            "angle_accuracies": [
                [[float(v) for v in t] for t in l] for l in angle_accuracies
            ],
            "qiskit_losses": [
                [[float(v) for v in t] for t in l] for l in qiskit_losses
            ],
            "amplitude_losses": [
                [[float(v) for v in t] for t in l] for l in amplitude_losses
            ],
            "angle_losses": [[[float(v) for v in t] for t in l] for l in angle_losses],
        }

        json_str = json.dumps(json_payload, indent=4)
        current_dir = str(Path(__file__).parent.parent.resolve()) + "/results/"
        with open(current_dir + "fig_4_data.json", "w") as f:
            f.write(json_str)
    plot_fig_4(
        layers_tested=[1, 10, 30],
        qiskit_accuracies=qiskit_accuracies,
        amplitude_accuracies=amplitude_accuracies,
        angle_accuracies=angle_accuracies,
        qiskit_losses=qiskit_losses,
        amplitude_losses=amplitude_losses,
        angle_losses=angle_losses,
        run_dir=run_dir,
    )
    return qiskit_losses, amplitude_losses, angle_losses


def reproduce_fig_5(
    num_max_samples: int = 750,
    run_dir: Path = None,
):
    distance_between_classes = [[], [], [], []]
    for sample_per_class in range(1, num_max_samples + 1):

        for dataset_name in ["MNIST", "CIFAR-10", "PathMNIST", "EuroSAT"]:
            print(f"Doing the {dataset_name} analysis")
            dataset = get_binary_dataset(
                name=dataset_name, num_samples_per_class=sample_per_class, shuffle=False
            )[0]
            class_1_features = dataset.tensors[0][:sample_per_class]
            normalized_class_1_features = [
                features.flatten() / np.linalg.norm(features.flatten())
                for features in class_1_features
            ]
            class_1_density_matrices = [
                state_vector_to_density_matrix(x) for x in normalized_class_1_features
            ]
            class_1_expected_state = np.mean(class_1_density_matrices, axis=0)

            class_2_features = dataset.tensors[0][sample_per_class:]
            normalized_class_2_features = [
                features.flatten() / np.linalg.norm(features.flatten())
                for features in class_2_features
            ]
            class_2_density_matrices = [
                state_vector_to_density_matrix(x) for x in normalized_class_2_features
            ]
            class_2_expected_state = np.mean(class_2_density_matrices, axis=0)
            distance_between_classes.append(
                trace_distance(
                    class_1_expected_state,
                    class_2_expected_state,
                )
            )
    plot_fig_5(
        sample_sizes=[i for i in range(1, num_max_samples + 1)],
        MNIST_trace_distances=distance_between_classes[0],
        CIFAR_10_trace_distances=distance_between_classes[1],
        PathMNIST_trace_distances=distance_between_classes[2],
        EuroSAT_trace_distances=distance_between_classes[3],
        run_dir=run_dir,
    )
    return distance_between_classes


def reproduce_fig_7(
    dataset_to_run: str = "MNIST",
    sample_size_per_class: List[int] = [1, 10, 100, 1000],
    batch_size: int = 50,
    num_epochs: int = 1000,
    lr: float = 0.01,
    run_dir: Path = None,
):
    qiskit_accuracies = []
    qiskit_losses = []
    qiskit_gen_error = []

    merlin_accuracies = []
    merlin_losses = []
    merlin_gen_error = []

    for sampler_size in sample_size_per_class:

        train_dataset, test_dataset = get_binary_dataset(
            name=dataset_to_run,
            num_samples_per_class=sampler_size,
        )

        train_loader = get_data_loader(train_dataset, batch_size=batch_size)
        test_loader = get_data_loader(test_dataset, batch_size=batch_size)

        qiskit_model = qiskit_QCNN()
        merlin_model = PhotonicQCNN(
            dims=(32, 32),
            conv_circuit="MZI",
            dense_circuit="MZI",
            dense_added_modes=0,
            output_proba_type="state",
            output_formatting="Lex_grouping",
            num_classes=2,
        )

        print("Qiskit model:")
        _, accuracy, loss, gen_error = basic_model_training(
            qiskit_model,
            train_loader,
            lr=lr,
            num_epochs=num_epochs,
            test_loader=test_loader,
        )
        qiskit_accuracies.append(accuracy)
        qiskit_losses.append(loss)
        qiskit_gen_error.append(gen_error)

        print("MerLin model")
        _, accuracy, loss, gen_error = basic_model_training(
            merlin_model,
            train_loader,
            lr=lr,
            num_epochs=num_epochs,
            test_loader=test_loader,
        )
        qiskit_accuracies.append(accuracy)
        qiskit_losses.append(loss)
        qiskit_gen_error.append(gen_error)

        json_payload = {
            "qiskit_accuracies": [[float(v) for v in t] for t in qiskit_accuracies],
            "qiskit_losses": [[float(v) for v in t] for t in qiskit_losses],
            "qiskit_gen_error": [[float(v) for v in t] for t in qiskit_gen_error],
            "merlin_accuracies": [[float(v) for v in t] for t in merlin_accuracies],
            "merlin_losses": [[float(v) for v in t] for t in merlin_losses],
            "merlin_gen_error": [[float(v) for v in t] for t in merlin_gen_error],
        }

        json_str = json.dumps(json_payload, indent=4)
        current_dir = str(Path(__file__).parent.parent.resolve()) + "/results/"
        with open(current_dir + "fig_7_data.json", "w") as f:
            f.write(json_str)

    plot_fig_7(
        sample_sizes=sample_size_per_class,
        training_losses=qiskit_losses,
        generalization_errors=qiskit_gen_error,
        testing_accuracies=qiskit_accuracies,
        model_name="qiskit",
        run_dir=run_dir,
    )
    plot_fig_7(
        sample_sizes=sample_size_per_class,
        training_losses=merlin_losses,
        generalization_errors=merlin_gen_error,
        testing_accuracies=merlin_accuracies,
        model_name="merlin",
        run_dir=run_dir,
    )
    return (
        qiskit_accuracies,
        qiskit_losses,
        qiskit_gen_error,
        merlin_accuracies,
        merlin_losses,
        merlin_gen_error,
    )


# reproduce_fig_1()
# reproduce_fig_2()
# reproduce_fig_3()
# reproduce_fig_4(num_epochs=5)
reproduce_fig_5()
# reproduce_fig_7()
