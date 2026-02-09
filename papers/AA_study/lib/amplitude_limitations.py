import json
import numpy as np
import sys
from pathlib import Path
from copy import deepcopy

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
    normalize_features,
)
from typing import List


def reproduce_fig_1(
    num_max_samples: int = 2000,
    run_dir: Path = None,
):
    """
    Reproduce Fig. 1 by computing trace distance of the amplitude encoding of dataset 1
    to the superposition state.

    Parameters
    ----------
    num_max_samples : int, optional
        Maximum number of samples per class to evaluate.
    run_dir : pathlib.Path, optional
        Output directory for the plot.

    Returns
    -------
    list[list[float]]
        Trace-distance curves for class 1 and class 2.
    """
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
    """
    Reproduce Fig. 2 by computing trace distance of the amplitude encoding of dataset 2
    to the mixed state.

    Parameters
    ----------
    num_max_samples : int, optional
        Maximum number of samples per class to evaluate.
    run_dir : pathlib.Path, optional
        Output directory for the plot.

    Returns
    -------
    list[list[float]]
        Trace-distance curves for class 1 and class 2.
    """
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
    """
    Reproduce Fig. 3 by computing inter-class amplitude encoding trace distance of dataset 3.

    Parameters
    ----------
    num_max_samples : int, optional
        Maximum number of samples per class to evaluate.
    run_dir : pathlib.Path, optional
        Output directory for the plot.

    Returns
    -------
    list[float]
        Trace-distance curve between class-averaged states.
    """
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
    """
    Reproduce Fig. 4 by training Qiskit, amplitude, and angle models on datasets 1,2 and 3.

    Parameters
    ----------
    batch_size : int, optional
        Training batch size.
    num_epochs : int, optional
        Number of training epochs.
    lr : float, optional
        Learning rate.
    num_samples_per_class : int, optional
        Samples per class for synthetic datasets.
    run_dir : pathlib.Path, optional
        Output directory for plots and JSON data.

    Returns
    -------
    tuple[list, list, list]
        (qiskit_losses, amplitude_losses, angle_losses).
    """
    qiskit_losses = [[], [], []]
    amplitude_losses = [[], [], []]
    angle_losses = [[], [], []]

    qiskit_accuracies = [[], [], []]
    amplitude_accuracies = [[], [], []]
    angle_accuracies = [[], [], []]

    fig_1_dataset = generate_fig_1_dataset(num_samples_per_class=num_samples_per_class)
    fig_2_dataset = generate_fig_2_dataset(num_samples_per_class=num_samples_per_class)
    fig_3_dataset = generate_fig_3_dataset(num_samples_per_class=num_samples_per_class)

    amplitude_loaders = [
        get_data_loader(deepcopy(fig_1_dataset), batch_size=batch_size),
        get_data_loader(deepcopy(fig_2_dataset), batch_size=batch_size),
        get_data_loader(deepcopy(fig_3_dataset), batch_size=batch_size),
    ]
    angle_loaders = [
        get_data_loader(
            normalize_features(fig_1_dataset, [4, 4], [6.5, 6.5]), batch_size=batch_size
        ),
        get_data_loader(
            normalize_features(fig_2_dataset, [-5, -5], [5, 5]), batch_size=batch_size
        ),
        get_data_loader(
            normalize_features(fig_3_dataset, [-3, -5], [3, 5]), batch_size=batch_size
        ),
    ]

    for L in [1, 10, 30]:
        print(f"Testing {L} layers")
        for i in range(3):
            qiskit_model = single_qubit_model(num_layers=L)
            amplitude_model = amplitude_encoding_simple(num_features=2, num_layers=L)
            angle_model = angle_encoding_simple(num_features=2, num_layers=L)

            print("Qiskit model:")
            _, accuracies, losses = basic_model_training(
                qiskit_model, amplitude_loaders[i], lr=lr, num_epochs=num_epochs
            )
            qiskit_accuracies[i].append(accuracies)
            qiskit_losses[i].append(losses)

            print("Amplitude model:")
            _, accuracies, losses = basic_model_training(
                amplitude_model, amplitude_loaders[i], lr=lr, num_epochs=num_epochs
            )
            amplitude_accuracies[i].append(accuracies)
            amplitude_losses[i].append(losses)

            print("Angle model:")
            _, accuracies, losses = basic_model_training(
                angle_model, angle_loaders[i], lr=lr, num_epochs=num_epochs
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
    num_max_samples: int = 250,
    run_dir: Path = None,
):
    """
    Reproduce Fig. 5 by computing inter-class trace distances with the features endoded as amplitudes across datasets.

    Parameters
    ----------
    num_max_samples : int, optional
        Maximum number of samples per class to evaluate.
    run_dir : pathlib.Path, optional
        Output directory for the plot.

    Returns
    -------
    list[list[float]]
        Trace-distance curves for MNIST, CIFAR-10, PathMNIST, and EuroSAT.
    """
    distance_between_classes = [[], [], [], []]

    dataset_mnist = get_binary_dataset(
        name="MNIST", num_samples_per_class=num_max_samples, shuffle=False
    )[0]
    dataset_cifar_10 = get_binary_dataset(
        name="CIFAR-10", num_samples_per_class=num_max_samples, shuffle=False
    )[0]
    dataset_pathmnist = get_binary_dataset(
        name="PathMNIST", num_samples_per_class=num_max_samples, shuffle=False
    )[0]
    dataset_eurosat = get_binary_dataset(
        name="EuroSAT", num_samples_per_class=num_max_samples, shuffle=False
    )[0]

    datasets = [dataset_mnist, dataset_cifar_10, dataset_pathmnist, dataset_eurosat]

    for dataset_index, dataset in enumerate(datasets):
        print()
        print(f"Doing dataset {dataset_index+1}/{4}")
        print("Printing the dataset")

        class_1_features = dataset.tensors[0][:num_max_samples]
        class_2_features = dataset.tensors[0][num_max_samples:]

        # Initialize expected states lazily to avoid huge allocations up front.
        class_1_expected_state = None
        class_2_expected_state = None
        eps = 1e-12
        for sample_per_class in range(1, num_max_samples + 1):
            print(f"Doing sample {sample_per_class} / {num_max_samples}.")
            f1 = (
                class_1_features[sample_per_class - 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            f2 = (
                class_2_features[sample_per_class - 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            v1 = f1.flatten()
            v2 = f2.flatten()
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            v1 = v1 / (n1 + eps)
            v2 = v2 / (n2 + eps)
            rho1 = state_vector_to_density_matrix(v1)
            rho2 = state_vector_to_density_matrix(v2)

            if class_1_expected_state is None:
                class_1_expected_state = np.zeros_like(rho1)
                class_2_expected_state = np.zeros_like(rho2)
            class_1_expected_state = (
                class_1_expected_state * (sample_per_class - 1) + rho1
            ) / sample_per_class
            class_2_expected_state = (
                class_2_expected_state * (sample_per_class - 1) + rho2
            ) / sample_per_class
            distance_between_classes[dataset_index].append(
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
    """
    Reproduce Fig. 7 by training Qiskit and Merlin QCNN models across different sample sizes.

    Parameters
    ----------
    dataset_to_run : str, optional
        Dataset name (e.g., "MNIST").
    sample_size_per_class : list[int], optional
        Sample sizes per class to evaluate.
    batch_size : int, optional
        Training batch size.
    num_epochs : int, optional
        Number of training epochs.
    lr : float, optional
        Learning rate.
    run_dir : pathlib.Path, optional
        Output directory for plots and JSON data.

    Returns
    -------
    tuple
        Metrics for qiskit and merlin models.
    """
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
            measure_subset=None,
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
        merlin_accuracies.append(accuracy)
        merlin_losses.append(loss)
        merlin_gen_error.append(gen_error)

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
# reproduce_fig_4()
# reproduce_fig_5()
reproduce_fig_7()
