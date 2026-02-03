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
)
from papers.AA_study.lib.qiskit_models import single_qubit_model
from papers.AA_study.utils.datasets import (
    generate_fig_1_dataset,
    generate_fig_2_dataset,
    generate_fig_3_dataset,
    get_data_loader,
)
from papers.AA_study.utils.states import superposition_state, mixed_state
from papers.AA_study.utils.plots import plot_amplitude_encoding_limitations
from papers.AA_study.utils.utils import trace_distance, state_vector_to_density_matrix


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


reproduce_fig_1()
print("One done")
reproduce_fig_2()
print("Two done")
reproduce_fig_3()
print("Three done")
