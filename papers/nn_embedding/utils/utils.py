import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import merlin as ml
from copy import deepcopy

from papers.nn_embedding.utils.merlin_model_utils import assign_params


def create_random_pairs(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for _ in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)
    return (
        torch.stack(X1_new),
        torch.stack(X2_new),
        torch.as_tensor(Y_new, dtype=torch.float32),
    )


def pick_random_data(batch_size, X, Y):
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = torch.stack([X[i] for i in batch_index])
    Y_batch = torch.as_tensor([Y[i] for i in batch_index], dtype=torch.long)
    return X_batch, Y_batch


def calculate_distance(
    rho0: torch.Tensor, rho1: torch.Tensor, distance: str = "Trace"
) -> float:
    rho_diff = rho1 - rho0
    if distance == "Trace":
        eigvals = torch.linalg.eigvals(rho_diff)
        return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
    elif distance == "Hilbert-Schmidt":
        return 0.5 * torch.trace(rho_diff @ rho_diff)
    else:
        raise ValueError("No distance with that name")


class LinearLoss(nn.Module):
    def forward(self, labels, predictions):
        labels = labels.to(predictions.dtype)
        # Same as 1-(l * p) where the labels are -1 and 1
        return (labels + predictions - (2 * labels * predictions)).mean()


def state_vector_to_density_matrix(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return torch.outer(x, torch.conj(x))
    if x.ndim == 2:
        return x.unsqueeze(-1) * torch.conj(x).unsqueeze(-2)
    raise ValueError("x must have shape (state_dim,) or (batch_size, state_dim)")


def loss_lower_bound(rhos_0: torch.Tensor, rhos_1: torch.Tensor) -> float:
    """
    Empirical risk is lower bounded by this quantity
    """
    N = rhos_0.size(dim=0) + rhos_1.size(dim=1)

    return 0.5 - calculate_distance(
        torch.sum(rhos_0, dim=0) / N, torch.sum(rhos_1, dim=0) / N
    )


def create_basic_gate_based_model(
    num_qubits: int,
    quantum_embedding_circuit: callable,
    quantum_classifier_circuit: callable,
    quantum_classifier_params_shape: tuple[int, ...],
    num_classes: int = 2,
):
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device, interface="torch")
    def complete_circuit(inputs, classifier_params):
        quantum_embedding_circuit(inputs)
        quantum_classifier_circuit(classifier_params)
        return qml.probs(wires=range(num_qubits))

    complete_circuit_layer = qml.qnn.TorchLayer(
        complete_circuit,
        weight_shapes={"classifier_params": quantum_classifier_params_shape},
    )

    class BasicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_grouper = ml.LexGrouping(2**num_qubits, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            probs = torch.stack(
                tuple(complete_circuit_layer(sample) for sample in x),
                dim=0,
            )
            return self.output_grouper(probs)

    return BasicModel()


def create_basic_merlin_model(
    quantum_embedding_layer: ml.QuantumLayer,
    quantum_classifier: ml.QuantumLayer,
    num_classes: int = 2,
):
    class BasicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_grouper = ml.LexGrouping(
                quantum_classifier.output_size, num_classes
            )
            self.embedder = deepcopy(quantum_embedding_layer)
            for param in self.embedder.parameters():
                param.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                x = x.reshape(x.size(0), -1)
                states = assign_params(self.embedder, x)

            probs = quantum_classifier(states)

            return self.output_grouper(probs)

    return BasicModel()


def to_serializable_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, np.generic):
        return values.item()
    if isinstance(values, (list, tuple)):
        return [to_serializable_list(value) for value in values]
    return values
