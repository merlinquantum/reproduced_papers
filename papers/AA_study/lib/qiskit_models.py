import qiskit as qu
import torch
import torch.nn as nn
from qiskit.circuit import Parameter
import numpy as np
from typing import List, Optional

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.utils.qiskit_utils import (
    ParameterShiftFunction,
    U_gate,
    reshape_input,
)

# from qiskit_machine_learning.connectors import TorchConnector
# from qiskit_machine_learning.neural_networks import EstimatorQNN


# TorchConnector()
# OR FROM QLLM


class single_qubit_model(nn.Module):
    def __init__(self, num_layers: int = 10, output_strategy: str = "probabilities"):
        super().__init__()

        self.params = nn.Parameter(2 * np.pi * torch.rand(num_layers * 3))

        self.circuit, self.q_params = self._single_qubit_model_circuit(num_layers)
        self.output_strategy = output_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x)
        return ParameterShiftFunction.apply(x, self.params, self)

    # TODO Optimize angle encoding parametrizing
    def _single_qubit_model_circuit(self, num_layers: int = 10):
        """
        Must be exactly to features
        features=List[complex]
        """
        circuit = qu.QuantumCircuit(1)
        parameters = []
        for layer in range(num_layers):
            parameters.extend(
                [
                    Parameter(f"theta_{layer}"),
                    Parameter(f"gamma_{layer}"),
                    Parameter(f"phi_{layer}"),
                ]
            )
            circuit.rz(parameters[-3], 0)
            circuit.rx(parameters[-2], 0)
            circuit.rz(parameters[-1], 0)

        return circuit, parameters


class qiskit_QCNN(nn.Module):
    def __init__(
        self,
        num_qubits: int = 10,
    ):
        super().__init__()
        # TODO find the formula
        self.num_qubits = num_qubits

        self.num_params = 0
        num_qubits_alive = num_qubits
        while num_qubits_alive > 1:
            self.num_params += 9 * (num_qubits_alive - 1)
            self.num_params += num_qubits_alive // 2
            num_qubits_alive = num_qubits_alive - (num_qubits_alive // 2)

        self.params = nn.Parameter(2 * np.pi * torch.rand(self.num_params))

        self.output_strategy = "first_qubit_probabilities"
        self.circuit, self.q_params = self._CNN_circuit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x)
        return ParameterShiftFunction.apply(x, self.params, self)

    # TODO Optimize angle encoding parametrizing
    def _CNN_circuit(self):
        """
        Must be exactly to features
        features=List[complex]
        """
        circuit = qu.QuantumCircuit(self.num_qubits)
        width = len(str(self.num_params - 1))
        parameters = [Parameter(f"phi{i:0{width}d}") for i in range(self.num_params)]
        param_index = 0
        qubits_alive = [i for i in range(self.num_qubits)]
        while len(qubits_alive) > 1:
            for i in range(0, len(qubits_alive) - 1, 2):
                circuit = circuit.compose(
                    U_gate(parameters=parameters[param_index : param_index + 9]),
                    [qubits_alive[i], qubits_alive[i + 1]],
                )
                param_index += 9
            for i in range(1, len(qubits_alive) - 1, 2):
                circuit = circuit.compose(
                    U_gate(parameters=parameters[param_index : param_index + 9]),
                    [qubits_alive[i], qubits_alive[i + 1]],
                )
                param_index += 9

            for i in range(1, len(qubits_alive), 2):
                circuit.crx(
                    parameters[param_index], qubits_alive[i], qubits_alive[i - 1]
                )
                param_index += 1

            qubits_alive = qubits_alive[::2]

        return circuit, parameters


import torch.optim as optim
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from papers.AA_study.utils.datasets import create_known_datasets

test_model = qiskit_QCNN()
print("Model created")
optimizer = optim.Adam(test_model.parameters(), lr=0.1)

data_loader = create_known_datasets()[2]
criterion = nn.CrossEntropyLoss()


test_model.train()
for epoch in range(15):
    tot_loss = 0
    correct = 0
    total = 0
    for features, labels in data_loader:
        labels = labels.long()
        optimizer.zero_grad()
        logits = test_model(features)
        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()
        tot_loss += loss.item()
        preds = torch.clone(logits).detach().argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch} had a loss of {tot_loss} and accuracy of {accuracy}")
