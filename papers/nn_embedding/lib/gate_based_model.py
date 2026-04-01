import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from merlin import LexGrouping

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.nn_embedding.utils.utils import (
    create_random_pairs,
    pick_random_data,
    calculate_distance,
    LinearLoss,
    loss_lower_bound,
)
from papers.nn_embedding.utils.gate_based_embedding import (
    EmbeddingCallable,
    QCNN,
)


class NeuralEmbeddingGateBasedModel(nn.Module):
    def __init__(
        self,
        num_qubits: int,
        classical_model: nn.Module,
        quantum_embedding_layer: callable,
        quantum_classifier: callable,
        quantum_classifier_params_shape: tuple[int, ...],
        num_classes: int = 2,
    ):
        """
        The quantum_embedding_layer must not have other
        trainable parameters that the ones in the input parameter of the function
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.classical_encoder = classical_model
        self.quantum_embedding_layer = quantum_embedding_layer
        self.quantum_classifier = quantum_classifier
        self.quantum_classifier_params_shape = quantum_classifier_params_shape
        self.output_grouper = LexGrouping(2**num_qubits, num_classes)
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Creating the torch pennylane layers
        self.distance_circuit_layer = self._create_distance_layer()
        self.complete_circuit_layer = self._create_complete_circuit_layer()
        self.state_embedding_circuit = self._create_state_embedding_circuit()

        # Creating the models
        self.embedding_training_model = self._TrainingModule(self)
        self.model = self._TrainedEmbeddingModel(self)

    def _create_distance_layer(self) -> torch.nn.Module:
        @qml.qnode(self.dev, interface="torch")
        def distance_qnode(inputs):
            split = len(inputs) // 2
            self.quantum_embedding_layer(inputs[0:split])
            qml.adjoint(self.quantum_embedding_layer)(inputs[split:])
            return qml.probs(wires=range(self.num_qubits))

        return qml.qnn.TorchLayer(distance_qnode, weight_shapes={})

    def _create_complete_circuit_layer(
        self,
    ) -> torch.nn.Module:
        @qml.qnode(self.dev, interface="torch")
        def complete_circuit(inputs, classifier_params):
            self.quantum_embedding_layer(inputs)
            self.quantum_classifier(classifier_params)
            return qml.probs(wires=range(self.num_qubits))

        return qml.qnn.TorchLayer(
            complete_circuit,
            weight_shapes={"classifier_params": self.quantum_classifier_params_shape},
        )

    ### No TorchLayer as it does not support complex data
    def _create_state_embedding_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def embedding_state(inputs):
            self.quantum_embedding_layer(inputs)
            return qml.density_matrix(wires=range(self.num_qubits))

        return embedding_state

    class _TrainingModule(nn.Module):
        def __init__(self, main_model):
            super().__init__()
            object.__setattr__(self, "main_model", main_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            separation_index = x.size(1) // 2
            data_1 = x[:, :separation_index]
            data_2 = x[:, separation_index:]

            data_1 = self.main_model.classical_encoder(data_1)
            data_2 = self.main_model.classical_encoder(data_2)

            data_1 = data_1.reshape(data_1.size(0), -1)
            data_2 = data_2.reshape(data_2.size(0), -1)

            x = torch.cat([data_1, data_2], dim=1)
            probs = torch.vstack(
                tuple(self.main_model.distance_circuit_layer(sample) for sample in x)
            )
            return probs[:, 0]

    class _TrainedEmbeddingModel(nn.Module):
        def __init__(self, main_model):
            super().__init__()
            object.__setattr__(self, "main_model", main_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                embedding_params = self.main_model.classical_encoder(x)
                embedding_params = embedding_params.reshape(
                    embedding_params.size(0), -1
                )

            probs = torch.stack(
                tuple(
                    self.main_model.complete_circuit_layer(sample)
                    for sample in embedding_params
                ),
                dim=0,
            )
            return self.main_model.output_grouper(probs)

    def train_embedding(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        distance: str = "Trace",
        batch_size: int = 25,
        num_epochs: int = 100,
        lr: float = 0.01,
        opt: torch.optim = torch.optim.Adam,
        return_data: bool = False,
    ) -> list[list[float]] | None:
        optimizer = opt(self.classical_encoder.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        train_distance = []
        test_distance = []
        loss_list = []

        if return_data:
            # Separating the test value classes
            X1_test = torch.stack(
                [x_test[i] for i in range(len(x_test)) if y_test[i] == 1]
            )
            X0_test = torch.stack(
                [x_test[i] for i in range(len(x_test)) if y_test[i] != 1]
            )

            # Separating the train value classes
            X1_train = torch.stack(
                [x_train[i] for i in range(len(x_train)) if y_train[i] == 1]
            )
            X0_train = torch.stack(
                [x_train[i] for i in range(len(x_train)) if y_train[i] != 1]
            )

        for epoch in range(num_epochs):

            # Training loop
            self.embedding_training_model.train()

            X1_batch, X2_batch, Y_batch = create_random_pairs(
                batch_size, x_train, y_train
            )

            x = torch.concatenate([X1_batch, X2_batch], dim=1)

            optimizer.zero_grad()
            outputs = self.embedding_training_model(x)
            loss = criterion(outputs, Y_batch)
            loss_list.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            self.embedding_training_model.eval()

            print(f"Epoch {epoch+1} had a loss of {loss_list[-1]}")

            # Distance evaluation
            if return_data:
                with torch.no_grad():
                    # Training distances
                    rhos0_train = torch.stack(
                        tuple(
                            self.state_embedding_circuit(sample)
                            for sample in self.classical_encoder(X0_train)
                        ),
                        dim=0,
                    )
                    rhos1_train = torch.stack(
                        tuple(
                            self.state_embedding_circuit(sample)
                            for sample in self.classical_encoder(X1_train)
                        ),
                        dim=0,
                    )
                    rho0 = torch.sum(rhos0_train, dim=0) / len(X0_train)
                    rho1 = torch.sum(rhos1_train, dim=0) / len(X1_train)
                    train_distance.append(
                        calculate_distance(rho0, rho1, distance=distance)
                    )

                    # Test distances
                    rhos0_test = torch.stack(
                        tuple(
                            self.state_embedding_circuit(sample)
                            for sample in self.classical_encoder(X0_test)
                        ),
                        dim=0,
                    )
                    rhos1_test = torch.stack(
                        tuple(
                            self.state_embedding_circuit(sample)
                            for sample in self.classical_encoder(X1_test)
                        ),
                        dim=0,
                    )
                    rho0 = torch.sum(rhos0_test, dim=0) / len(X0_test)
                    rho1 = torch.sum(rhos1_test, dim=0) / len(X1_test)
                    test_distance.append(
                        calculate_distance(rho0, rho1, distance=distance)
                    )

        if return_data:
            return (
                loss_list,
                train_distance,
                test_distance,
                loss_lower_bound(rhos0_train, rhos1_train),
                loss_lower_bound(rhos0_test, rhos1_test),
            )

    def train_classifier(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        batch_size: int = 25,
        num_epochs: int = 100,
        lr: float = 0.01,
        opt: torch.optim = torch.optim.Adam,
        return_data: bool = False,
    ) -> list[list[float]] | None:
        optimizer = opt(self.complete_circuit_layer.parameters(), lr=lr)
        criterion = LinearLoss()

        train_accs = []
        test_accs = []
        loss_list = []

        for epoch in range(num_epochs):

            ### Training loop
            self.model.train()

            X_batch, Y_batch = pick_random_data(batch_size, x_train, y_train)

            optimizer.zero_grad()
            outputs = self.model(X_batch)

            """
            Pourquoi pas donner les logits avec argmax: So the answer is:

            argmax throws away the gradient information
            the loss needs a differentiable quantity
            argmax is only for converting scores to final class labels at evaluation time
            """

            loss = criterion(Y_batch, outputs[:, 1])

            loss.backward()
            optimizer.step()

            loss_list.append(loss.cpu().detach().numpy())

            print(f"Epoch {epoch+1} had a loss of {loss_list[-1]}")

            if return_data:
                ### Evaluate the accuracy
                self.model.eval()
                with torch.no_grad():
                    # Check on the training set
                    outputs = self.model(x_train)
                    predicted = torch.argmax(outputs, dim=1)
                    correct = 0
                    correct += (predicted == y_train).sum().item()
                    acc = 100 * correct / x_train.size(dim=0)
                    train_accs.append(acc)

                    # Check on the training set
                    outputs = self.model(x_test)
                    predicted = torch.argmax(outputs, dim=1)
                    correct = 0
                    correct += (predicted == y_test).sum().item()
                    acc = 100 * correct / x_test.size(dim=0)
                    test_accs.append(acc)

        if return_data:
            return loss_list, train_accs, test_accs


class NeuralEmbeddingGateBasedKernel(nn.Module):
    def __init__(self):
        super().__init__()


def create_paper_models() -> tuple[
    NeuralEmbeddingGateBasedModel,
    NeuralEmbeddingGateBasedModel,
    NeuralEmbeddingGateBasedModel,
]:
    """
    Hybrid Model 1 transforms 8 dimensional features to 8 dimensional features using Fully connected classical NN.

    Hybrid Model 2 transforms 8 dimensional features to 16 dimensional features.

    Hybrid Model 3 transforms 28 * 28 dimensional features to 16 dimensional features using CNN.
    16 dimensional features are used as a rotation angle of the ZZ feature embedding.
    """
    ###Model 1
    classical_model = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
    )

    model_1 = NeuralEmbeddingGateBasedModel(
        num_qubits=8,
        classical_model=classical_model,
        quantum_embedding_layer=EmbeddingCallable().QuantumEmbedding1,
        quantum_classifier=QCNN,
        quantum_classifier_params_shape=(45),
    )

    ###Model 2
    classical_model = nn.Sequential(
        nn.Linear(8, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 16)
    )

    model_2 = NeuralEmbeddingGateBasedModel(
        num_qubits=8,
        classical_model=classical_model,
        quantum_embedding_layer=EmbeddingCallable().QuantumEmbedding2,
        quantum_classifier=QCNN,
        quantum_classifier_params_shape=(45),
    )

    ###Model 3
    classical_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Linear(7 * 7, 16, bias=True),
    )

    model_3 = NeuralEmbeddingGateBasedModel(
        num_qubits=8,
        classical_model=classical_model,
        quantum_embedding_layer=EmbeddingCallable().QuantumEmbedding2,
        quantum_classifier=QCNN,
        quantum_classifier_params_shape=(45),
    )

    return model_1, model_2, model_3
