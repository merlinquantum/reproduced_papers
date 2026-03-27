import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class NeuralEmbeddingGateBasedModel(nn.Module):
    def __init__(
        self,
        num_qubits: int,
        classical_model: nn.Module,
        quantum_embedding_layer: callable,
        quantum_classifier: callable,
        classifier_params: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.classical_encoder = classical_model
        self.quantum_embedding_layer = quantum_embedding_layer
        self.quantum_classifier = quantum_classifier

        self.embedding_training_model = self._TrainingModule(self)
        self.classifier_training_model = self._TrainedEmbeddingModel(self)
        self.model = self._FullyTrainedModel(self)

        self.classifier_params = classifier_params

    def _distance_circuit(self, params: torch.Tensor) -> float:
        self.quantum_embedding_layer(params[: len(params) // 2])
        self.quantum_embedding_layer(params[len(params) // 2 :])
        return qml.probs(wires=range(self.num_qubits))[0]

    def _complete_circuit(
        self, encoding_params: torch.Tensor, classifier_params: torch.Tensor
    ) -> list[float]:
        self.quantum_embedding_layer(encoding_params)
        self.quantum_classifier(classifier_params)
        return qml.probs(wires=range(self.num_qubits))

    class _TrainingModule(nn.Module):
        def __init__(self, main_model):
            super().__init__(self)
            self.main_model = main_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seperation_index = x.size(dim=1) // 2
            data_1 = x[:, :seperation_index]
            data_2 = x[:, seperation_index:]
            data_1 = self.main_model.classical_encoder(data_1)
            data_2 = self.main_model.classical_encoder(data_2)

            data_1 = data_1.reshape(data_1.size(dim=0), np.prod(data_1.shape[1:]))
            data_2 = data_2.reshape(data_2.size(dim=0), np.prod(data_2.shape[1:]))

            params_to_apply = torch.concatenate([data_1, data_2], dim=1)

            output_tensor = torch.empty(x.size(dim=0))
            for i, params in enumerate(params_to_apply):
                output_tensor[i] = self.main_model._distance_circuit(params)
            return output_tensor

    class _TrainedEmbeddingModel(nn.Module):
        def __init__(self, main_model):
            super().__init__()
            self.main_model = main_model

        def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            with torch.no_grad:
                embedding_params = self.main_model.classical_encoder(x)

            output_tensor = torch.empty((x.size(dim=0), 2**self.main_model.num_qubits))
            for i, (encoding_params, qnn_params) in enumerate(
                zip(embedding_params, params)
            ):
                output_tensor[i] = self.main_model._complete_circuit(
                    encoding_params, qnn_params
                )
            return output_tensor

    class _FullyTrainedModel(nn.Module):
        def __init__(self, main_model):
            super().__init__()
            self.main_model = main_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad:
                embedding_params = self.main_model.classical_encoder(x)

            output_tensor = torch.empty((x.size(dim=0), 2**self.main_model.num_qubits))
            for i, params in enumerate(
                embedding_params,
            ):
                output_tensor[i] = self.main_model._complete_circuit(
                    params, self.main_model.classifier_params
                )
            return output_tensor

    def train_embedding(
        self,
        loss: str = "Loss",
        num_epochs: int = 100,
        lr: float = 0.01,
        opt: torch.optim = torch.optim.Adam,
        return_training_data: bool = True,
    ) -> nn.Module:
        optimizer = opt(self.classical_encoder.parameters(), lr=lr)
