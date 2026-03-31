import torch
import torch.nn as nn
import merlin as ml
from copy import deepcopy

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from papers.nn_embedding.utils.merlin_model_utils import (
    rename_params_in_current_order,
    count_parameters_with_prefixes,
    strip_simple_negation_expressions,
    compute_x2_permutation,
    assign_params,
)
from papers.nn_embedding.utils.utils import (
    create_random_pairs,
    pick_random_data,
    calculate_distance,
    state_vector_to_density_matrix,
    LinearLoss,
    loss_lower_bound,
)


class NeuralEmbeddingMerLinModel(nn.Module):
    def __init__(
        self,
        classical_model: nn.Module,
        quantum_embedding_layer: ml.QuantumLayer,
        quantum_classifier: ml.QuantumLayer,
        num_classes: int = 2,
    ):
        """
        The quantum classifier must have amplitude encoding on and return probs
        The embedding_layer must return amplitudes and have no input parameters, only trainable ones. The input state must also be a basis state
        """
        super().__init__()
        self.classical_encoder = classical_model
        self.quantum_embedding_layer = deepcopy(quantum_embedding_layer)
        for param in self.quantum_embedding_layer.parameters():
            param.requires_grad = False
        self.quantum_classifier = quantum_classifier
        self.output_grouper = ml.LexGrouping(
            self.quantum_classifier.output_size, num_classes
        )
        self.similarity_layer = self._SimilarityLayer(self)

        # Creating the models
        self.embedding_training_model = self._TrainingModule(self)
        self.model = self._TrainedEmbeddingModel(self)

    class _SimilarityLayer(nn.Module):
        def __init__(self, main_model):
            super().__init__()
            object.__setattr__(self, "main_model", main_model)
            encoder_1 = deepcopy(self.main_model.quantum_embedding_layer)
            encoder_2 = deepcopy(self.main_model.quantum_embedding_layer)

            rename_params_in_current_order(encoder_1.circuit, "x_1_")
            rename_params_in_current_order(encoder_2.circuit, "x_2_")

            encoder_2.circuit.inverse(h=True)
            strip_simple_negation_expressions(encoder_2.circuit)

            fidelity_circuit = encoder_1.circuit.add(0, encoder_2.circuit, merge=False)

            input_prefixes = ["x_1_", "x_2_"]
            input_size = count_parameters_with_prefixes(
                fidelity_circuit, input_prefixes
            )

            self.fidelity_layer = ml.QuantumLayer(
                input_size=input_size,
                circuit=fidelity_circuit,
                input_state=encoder_1.input_state,
                n_photons=encoder_1.n_photons,
                amplitude_encoding=False,
                computation_space=encoder_1.computation_space,
                measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
                device=encoder_1.device,
                dtype=encoder_1.dtype,
                input_parameters=input_prefixes,
            )

            self.perm_x2 = compute_x2_permutation(self.fidelity_layer)

            for param in self.fidelity_layer.parameters():
                param.requires_grad = False

            self.target_index = list(self.fidelity_layer.output_keys).index(
                tuple(self.fidelity_layer.input_state)
            )

        def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
            x_2 = -x_2
            x2_for_layer = x_2[..., self.perm_x2]

            probs = self.fidelity_layer(x_1, x2_for_layer)
            return probs[..., self.target_index]

    class _TrainingModule(nn.Module):
        def __init__(self, main_model):
            super().__init__()
            object.__setattr__(self, "main_model", main_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

            separation_index = x.size(1) // 2
            data_1 = x[:, :separation_index]
            data_2 = x[:, separation_index:]

            data_1 = self.main_model.classical_encoder(data_1)
            data_2 = self.main_model.classical_encoder(data_2)

            data_1 = data_1.reshape(data_1.size(0), -1)
            data_2 = data_2.reshape(data_2.size(0), -1)

            # # Not exact but similar
            # states_1 = self.main_model.quantum_embedding_layer(data_1)
            # states_2 = self.main_model.quantum_embedding_layer(data_2)
            # return torch.linalg.vecdot(states_2, states_1, dim=1)

            return self.main_model.similarity_layer(data_1, data_2)

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
                states = assign_params(
                    self.main_model.quantum_embedding_layer, embedding_params
                )

            probs = self.main_model.quantum_classifier(states)

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
                    classical_data = self.classical_encoder(X0_train)
                    states = assign_params(self.quantum_embedding_layer, classical_data)
                    rhos0_train = state_vector_to_density_matrix(states)
                    classical_data = self.classical_encoder(X1_train)
                    states = assign_params(self.quantum_embedding_layer, classical_data)
                    rhos1_train = state_vector_to_density_matrix(states)

                    rho0 = torch.sum(rhos0_train, dim=0) / len(X0_train)
                    rho1 = torch.sum(rhos1_train, dim=0) / len(X1_train)
                    train_distance.append(
                        calculate_distance(rho0, rho1, distance=distance)
                    )

                    # Test distances
                    classical_data = self.classical_encoder(X0_test)
                    states = assign_params(self.quantum_embedding_layer, classical_data)
                    rhos0_test = state_vector_to_density_matrix(states)
                    classical_data = self.classical_encoder(X1_test)
                    states = assign_params(self.quantum_embedding_layer, classical_data)
                    rhos1_test = state_vector_to_density_matrix(states)

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
        optimizer = opt(self.quantum_classifier.parameters(), lr=lr)
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


def create_basic_merlin_model() -> NeuralEmbeddingMerLinModel:
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

    classical_model = nn.Sequential(
        nn.Linear(8, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, sum([i.numel() for i in embedder.parameters()])),
    )

    return NeuralEmbeddingMerLinModel(classical_model, embedder, classifier)
