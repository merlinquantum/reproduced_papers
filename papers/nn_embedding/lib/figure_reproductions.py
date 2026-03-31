import sys
from pathlib import Path
import torch.nn as nn
import merlin as ml
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


def reproduce_figure_2(
    dataset: str = "MNIST",
    use_merlin: bool = False,
    batch_size: int = 100,
    num_epochs_training_embedding: int = 50,
    num_epochs_training_classifier: int = 50,
    lr: float = 0.01,
    distance: str = "Trace",
    samples_per_class: int = 150,
    num_classes: int = 2,
):
    # Results lists
    loss_lists_embedding = []
    training_distances = []
    testing_distances = []
    train_lower_bounds = []
    test_lower_bounds = []
    loss_lists_classifier = []
    train_accuracies = []
    test_accuracies = []

    # load the data
    x_train_PCA8, x_test_PCA8, y_train_PCA8, y_test_PCA8 = data_load_and_process(
        dataset=dataset,
        feature_reduction="PCA8",
        classes=[0, 1],
        samples_per_class=samples_per_class,
    )
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=dataset,
        classes=[0, 1],
        samples_per_class=samples_per_class,
    )

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

        # Quantum classifier
        circ = ml.CircuitBuilder(n_modes=8)
        circ.add_entangling_layer()
        classifier = ml.QuantumLayer(
            builder=circ,
            n_photons=4,
            amplitude_encoding=True,
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
        )

        # PCA 8
        classical_model_8 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, sum([i.numel() for i in embedder.parameters()])),
        )

        # Full classical_model
        classical_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, sum([i.numel() for i in embedder.parameters()])),
        )

        # PCA_NQE
        model = NeuralEmbeddingMerLinModel(
            classical_model=classical_model_8,
            quantum_embedding_layer=deepcopy(embedder),
            quantum_classifier=deepcopy(classifier),
            num_classes=num_classes,
        )
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

        loss_lists_embedding.append(loss_list_embedding)
        training_distances.append(train_distances)
        testing_distances.append(test_distances)
        train_lower_bounds.append(train_lower_bound)
        test_lower_bounds.append(test_lower_bound)
        loss_lists_classifier.append(loss_list_classifier)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # NQE
        model = NeuralEmbeddingMerLinModel(
            classical_model=classical_model,
            quantum_embedding_layer=deepcopy(embedder),
            quantum_classifier=deepcopy(classifier),
            num_classes=num_classes,
        )
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

        loss_lists_embedding.append(loss_list_embedding)
        training_distances.append(train_distances)
        testing_distances.append(test_distances)
        train_lower_bounds.append(train_lower_bound)
        test_lower_bounds.append(test_lower_bound)
        loss_lists_classifier.append(loss_list_classifier)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # No NQE
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
        training_distances.append([train_distance])
        testing_distances.append([test_distance])
        train_lower_bounds.append(train_lower_bound)
        test_lower_bounds.append(test_lower_bound)
        loss_lists_classifier.append(loss_list)
        train_accuracies.append(train_accs)
        test_accuracies.append(test_accs)
    else:

        # PCA 8
        classical_model_8 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
        )

        # Full classical_model
        classical_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )
        # PCA_NQE
        model = NeuralEmbeddingGateBasedModel(
            num_qubits=4,
            classical_model=classical_model_8,
            quantum_embedding_layer=Four_QuantumEmbedding2,
            quantum_classifier=FourQCNN,
            quantum_classifier_params_shape=(30),
            num_classes=num_classes,
        )
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

        loss_lists_embedding.append(loss_list_embedding)
        training_distances.append(train_distances)
        testing_distances.append(test_distances)
        train_lower_bounds.append(train_lower_bound)
        test_lower_bounds.append(test_lower_bound)
        loss_lists_classifier.append(loss_list_classifier)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # NQE
        model = NeuralEmbeddingGateBasedModel(
            num_qubits=4,
            classical_model=classical_model,
            quantum_embedding_layer=Four_QuantumEmbedding2,
            quantum_classifier=FourQCNN,
            quantum_classifier_params_shape=(30),
            num_classes=num_classes,
        )
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

        loss_lists_embedding.append(loss_list_embedding)
        training_distances.append(train_distances)
        testing_distances.append(test_distances)
        train_lower_bounds.append(train_lower_bound)
        test_lower_bounds.append(test_lower_bound)
        loss_lists_classifier.append(loss_list_classifier)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # No NQE
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
        training_distances.append([train_distance])
        testing_distances.append([test_distance])
        train_lower_bounds.append(train_lower_bound)
        test_lower_bounds.append(test_lower_bound)
        loss_lists_classifier.append(loss_list)
        train_accuracies.append(train_accs)
        test_accuracies.append(test_accs)
