import sys
from pathlib import Path

import merlin as ml
import pennylane as qml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.utils.merlin_model_utils import (  # noqa: E402
    assign_params,
)
from papers.nn_embedding.utils.utils import (  # noqa: E402
    LinearLoss,
    calculate_distance,
    create_basic_gate_based_model,
    create_basic_merlin_model,
    create_trainable_embedding_gate_based_model,
    create_trainable_embedding_merlin_model,
    loss_lower_bound,
    pick_random_data,
    state_vector_to_density_matrix,
)


def train_gate_based(
    num_qubits: int,
    quantum_embedding_circuit: callable,
    quantum_classifier_circuit: callable,
    quantum_classifier_params_shape: tuple[int, ...],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 25,
    num_epochs: int = 100,
    lr: float = 0.01,
    opt: torch.optim = torch.optim.Adam,
    return_data: bool = False,
    num_classes: int = 2,
    distance: str = "Trace",
    trainable_embedding: bool = False,
    embedding_params_shape: tuple[int, ...] | None = None,
) -> list[list[float]] | None:
    """Train a gate-based baseline without the two-stage NQE procedure.

    This helper is used for the comparison models where the classifier is
    trained directly, with either a fixed embedding or a jointly trainable
    embedding circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used by the quantum circuit.
    quantum_embedding_circuit : callable
        Gate-based embedding circuit applied before classification.
    quantum_classifier_circuit : callable
        Gate-based classifier circuit applied after the embedding.
    quantum_classifier_params_shape : tuple[int, ...]
        Shape of the trainable classifier parameters.
    x_train : torch.Tensor
        Training inputs.
    y_train : torch.Tensor
        Training labels.
    x_test : torch.Tensor
        Test inputs.
    y_test : torch.Tensor
        Test labels.
    batch_size : int, optional
        Number of training examples drawn per optimization step
        (default: ``25``).
    num_epochs : int, optional
        Number of training epochs (default: ``100``).
    lr : float, optional
        Learning rate used by the optimizer (default: ``0.01``).
    opt : torch.optim, optional
        Optimizer class used for training (default: ``torch.optim.Adam``).
    return_data : bool, optional
        If ``True``, also return loss, accuracy, distance, and lower-bound
        metrics for train and test data (default: ``False``).
    num_classes : int, optional
        Number of output classes (default: ``2``).
    distance : str, optional
        Distance metric used when reporting encoded-state separation
        (default: ``"Trace"``).
    trainable_embedding : bool, optional
        If ``True``, include trainable embedding parameters in the model
        (default: ``False``).
    embedding_params_shape : tuple[int, ...] | None, optional
        Shape of the trainable embedding parameters when
        ``trainable_embedding`` is enabled (default: ``None``).

    Returns
    -------
    list[list[float]] | tuple | None
        Returns ``None`` when ``return_data`` is ``False``. Otherwise returns
        the loss history, train/test accuracies, train/test distances, and the
        train/test empirical lower bounds.
    """

    ### Creating the model
    if trainable_embedding:
        model = create_trainable_embedding_gate_based_model(
            num_qubits,
            quantum_embedding_circuit,
            quantum_classifier_circuit,
            embedding_params_shape,
            quantum_classifier_params_shape,
            num_classes,
        )
    else:
        model = create_basic_gate_based_model(
            num_qubits,
            quantum_embedding_circuit,
            quantum_classifier_circuit,
            quantum_classifier_params_shape,
            num_classes,
        )

    ### Optimizing the model
    optimizer = opt(model.complete_circuit_layer.parameters(), lr=lr)
    criterion = LinearLoss()

    train_accs = []
    test_accs = []
    loss_list = []

    for epoch in range(num_epochs):
        ## Training loop
        model.train()

        X_batch, Y_batch = pick_random_data(batch_size, x_train, y_train)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(Y_batch, outputs[:, 1])

        loss.backward()
        optimizer.step()

        loss_list.append(loss.cpu().detach().numpy())

        print(f"Epoch {epoch + 1} had a loss of {loss_list[-1]}")

        if return_data:
            ### Evaluate the accuracy
            model.eval()
            with torch.no_grad():
                # Check on the training set
                outputs = model(x_train)
                predicted = torch.argmax(outputs, dim=1)
                correct = 0
                correct += (predicted == y_train).sum().item()
                acc = 100 * correct / x_train.size(dim=0)
                train_accs.append(acc)

                # Check on the training set
                outputs = model(x_test)
                predicted = torch.argmax(outputs, dim=1)
                correct = 0
                correct += (predicted == y_test).sum().item()
                acc = 100 * correct / x_test.size(dim=0)
                test_accs.append(acc)

    if return_data:
        ## Calculate the distance between encoded states
        device = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(device, interface="torch")
        def embedding_state(inputs):
            if trainable_embedding:
                quantum_embedding_circuit(
                    inputs, model.complete_circuit_layer.embedding_params
                )
            else:
                quantum_embedding_circuit(inputs)
            return qml.density_matrix(wires=range(num_qubits))

        X1_test = torch.stack([x_test[i] for i in range(len(x_test)) if y_test[i] == 1])
        X0_test = torch.stack([x_test[i] for i in range(len(x_test)) if y_test[i] != 1])

        # Separating the train value classes
        X1_train = torch.stack(
            [x_train[i] for i in range(len(x_train)) if y_train[i] == 1]
        )
        X0_train = torch.stack(
            [x_train[i] for i in range(len(x_train)) if y_train[i] != 1]
        )
        # Training distances
        with torch.no_grad():
            rhos0_train = torch.stack(
                tuple(embedding_state(sample) for sample in X0_train),
                dim=0,
            )
            rhos1_train = torch.stack(
                tuple(embedding_state(sample) for sample in X1_train),
                dim=0,
            )
            rho0 = torch.sum(rhos0_train, dim=0) / len(X0_train)
            rho1 = torch.sum(rhos1_train, dim=0) / len(X1_train)
            train_distance = calculate_distance(rho0, rho1, distance=distance)

            # Test distances
            rhos0_test = torch.stack(
                tuple(embedding_state(sample) for sample in X0_test),
                dim=0,
            )
            rhos1_test = torch.stack(
                tuple(embedding_state(sample) for sample in X1_test),
                dim=0,
            )
            rho0 = torch.sum(rhos0_test, dim=0) / len(X0_test)
            rho1 = torch.sum(rhos1_test, dim=0) / len(X1_test)
            test_distance = calculate_distance(rho0, rho1, distance=distance)

        return (
            loss_list,
            train_accs,
            test_accs,
            train_distance,
            test_distance,
            loss_lower_bound(rhos0_train, rhos1_train),
            loss_lower_bound(rhos0_test, rhos1_test),
        )


def train_merlin_based(
    quantum_embedding_layer: ml.QuantumLayer,
    quantum_classifier: ml.QuantumLayer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 25,
    num_epochs: int = 100,
    lr: float = 0.01,
    opt: torch.optim = torch.optim.Adam,
    return_data: bool = False,
    num_classes: int = 2,
    distance: str = "Trace",
    trainable_embedding: bool = False,
):
    """Train a MerLin photonic baseline without the two-stage NQE procedure.

    This helper is used for the comparison models where the photonic
    classifier is trained directly, with either a fixed embedding or a jointly
    trainable embedding layer.

    Parameters
    ----------
    quantum_embedding_layer : ml.QuantumLayer
        Photonic embedding layer used to encode the inputs.
    quantum_classifier : ml.QuantumLayer
        Photonic classifier layer optimized during training.
    x_train : torch.Tensor
        Training inputs.
    y_train : torch.Tensor
        Training labels.
    x_test : torch.Tensor
        Test inputs.
    y_test : torch.Tensor
        Test labels.
    batch_size : int, optional
        Number of training examples drawn per optimization step
        (default: ``25``).
    num_epochs : int, optional
        Number of training epochs (default: ``100``).
    lr : float, optional
        Learning rate used by the optimizer (default: ``0.01``).
    opt : torch.optim, optional
        Optimizer class used for training (default: ``torch.optim.Adam``).
    return_data : bool, optional
        If ``True``, also return loss, accuracy, distance, and lower-bound
        metrics for train and test data (default: ``False``).
    num_classes : int, optional
        Number of output classes (default: ``2``).
    distance : str, optional
        Distance metric used when reporting encoded-state separation
        (default: ``"Trace"``).
    trainable_embedding : bool, optional
        If ``True``, optimize a model with a trainable embedding layer
        (default: ``False``).

    Returns
    -------
    list[list[float]] | tuple | None
        Returns ``None`` when ``return_data`` is ``False``. Otherwise returns
        the loss history, train/test accuracies, train/test distances, and the
        train/test empirical lower bounds.
    """
    if trainable_embedding:
        model = create_trainable_embedding_merlin_model(
            quantum_embedding_layer, quantum_classifier, num_classes
        )
    else:
        model = create_basic_merlin_model(
            quantum_embedding_layer,
            quantum_classifier,
            num_classes,
        )

    ### Optimizing the model
    optimizer = opt(quantum_classifier.parameters(), lr=lr)
    criterion = LinearLoss()

    train_accs = []
    test_accs = []
    loss_list = []

    for epoch in range(num_epochs):
        ## Training loop
        model.train()

        X_batch, Y_batch = pick_random_data(batch_size, x_train, y_train)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(Y_batch, outputs[:, 1])

        loss.backward()
        optimizer.step()

        loss_list.append(loss.cpu().detach().numpy())

        print(f"Epoch {epoch + 1} had a loss of {loss_list[-1]}")

        if return_data:
            ### Evaluate the accuracy
            model.eval()
            with torch.no_grad():
                # Check on the training set
                outputs = model(x_train)
                predicted = torch.argmax(outputs, dim=1)
                correct = 0
                correct += (predicted == y_train).sum().item()
                acc = 100 * correct / x_train.size(dim=0)
                train_accs.append(acc)

                # Check on the training set
                outputs = model(x_test)
                predicted = torch.argmax(outputs, dim=1)
                correct = 0
                correct += (predicted == y_test).sum().item()
                acc = 100 * correct / x_test.size(dim=0)
                test_accs.append(acc)

    if return_data:
        # Separating the test value classes
        X1_test = torch.stack([x_test[i] for i in range(len(x_test)) if y_test[i] == 1])
        X0_test = torch.stack([x_test[i] for i in range(len(x_test)) if y_test[i] != 1])

        # Separating the train value classes
        X1_train = torch.stack(
            [x_train[i] for i in range(len(x_train)) if y_train[i] == 1]
        )
        X0_train = torch.stack(
            [x_train[i] for i in range(len(x_train)) if y_train[i] != 1]
        )

        with torch.no_grad():
            if trainable_embedding:
                states = model.embedder(X0_train)
                rhos0_train = state_vector_to_density_matrix(states)
                states = model.embedder(X1_train)
                rhos1_train = state_vector_to_density_matrix(states)

                rho0 = torch.sum(rhos0_train, dim=0) / len(X0_train)
                rho1 = torch.sum(rhos1_train, dim=0) / len(X1_train)
                train_distance = calculate_distance(rho0, rho1, distance=distance)

                # Test distances
                states = model.embedder(X0_test)
                rhos0_test = state_vector_to_density_matrix(states)
                states = model.embedder(X1_test)
                rhos1_test = state_vector_to_density_matrix(states)

                rho0 = torch.sum(rhos0_test, dim=0) / len(X0_test)
                rho1 = torch.sum(rhos1_test, dim=0) / len(X1_test)
                test_distance = calculate_distance(rho0, rho1, distance=distance)
            else:
                states = assign_params(model.embedder, X0_train)
                rhos0_train = state_vector_to_density_matrix(states)
                states = assign_params(model.embedder, X1_train)
                rhos1_train = state_vector_to_density_matrix(states)

                rho0 = torch.sum(rhos0_train, dim=0) / len(X0_train)
                rho1 = torch.sum(rhos1_train, dim=0) / len(X1_train)
                train_distance = calculate_distance(rho0, rho1, distance=distance)

                # Test distances
                states = assign_params(model.embedder, X0_test)
                rhos0_test = state_vector_to_density_matrix(states)
                states = assign_params(model.embedder, X1_test)
                rhos1_test = state_vector_to_density_matrix(states)

                rho0 = torch.sum(rhos0_test, dim=0) / len(X0_test)
                rho1 = torch.sum(rhos1_test, dim=0) / len(X1_test)
                test_distance = calculate_distance(rho0, rho1, distance=distance)

        return (
            loss_list,
            train_accs,
            test_accs,
            train_distance,
            test_distance,
            loss_lower_bound(rhos0_train, rhos1_train),
            loss_lower_bound(rhos0_test, rhos1_test),
        )
