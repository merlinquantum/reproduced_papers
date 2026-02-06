import qiskit as qu
import torch
import torch.nn as nn
from qiskit.circuit import Parameter
import numpy as np
import torch.nn.functional as F
from typing import List, Optional

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
LIB_ROOT = Path(__file__).resolve().parent
TORCHQUANTUM_ROOT = LIB_ROOT / "torchquantum"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TORCHQUANTUM_ROOT))


from papers.AA_study.utils.qiskit_utils import (
    ParameterShiftFunction,
    U_gate,
    reshape_input,
)

import torchquantum as tq
import torchquantum.functional as tqf


class single_qubit_model(tq.QuantumModule):
    class SingleQubitModelQLayer(tq.QuantumModule):
        def __init__(self, num_layers: int = 10):
            super().__init__()
            self.n_wires = 1

            # gates with trainable parameters
            self.gates = tq.QuantumModuleList(
                [
                    tq.RZ(has_params=True, trainable=True),
                    tq.RX(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                ]
                * num_layers
            )

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            # some trainable gates (instantiated ahead of time)
            for gates in self.gates:
                gates(self.q_device, wires=0)

    def __init__(self, num_layers: int = 10, return_probs: bool = True):
        super().__init__()
        self.n_wires = 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.SingleQubitModelQLayer(num_layers=num_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.return_probs = return_probs

    def forward(
        self,
        x,
    ):
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        # Convert single expectation value into 2-class logits
        logits = torch.cat([x, -x], dim=1)
        if self.return_probs:
            return torch.softmax(logits, dim=1)
        return logits


import torchquantum as tq


# TODO Add a num_classes parameter
class qiskit_QCNN(tq.QuantumModule):
    class QiskitQCNNQLayer(tq.QuantumModule):
        def __init__(self, num_qubits: int = 10):
            super().__init__()

            num_u_gates = 0
            num_v_gates = 0
            num_qubits_alive = num_qubits

            while num_qubits_alive > 1:
                num_u_gates += num_qubits_alive - 1
                num_v_gates += num_qubits_alive // 2
                num_qubits_alive = num_qubits_alive - (num_qubits_alive // 2)

            self.n_wires = num_qubits

            # gates with trainable parameters
            self.U_gates = tq.QuantumModuleList(
                [
                    tq.RXX(has_params=True, trainable=True),
                    tq.RYY(has_params=True, trainable=True),
                    tq.RZZ(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                    tq.RX(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                    tq.RX(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                ]
                * num_u_gates
            )
            self.V_gates = tq.QuantumModuleList(
                [tq.CRX(has_params=True, trainable=True)] * num_v_gates
            )

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            u_gate_index = 0
            v_gate_index = 0
            qubits_alive = [i for i in range(self.n_wires)]

            while len(qubits_alive) > 1:
                for i in range(0, len(qubits_alive) - 1, 2):
                    for local_idx, gate_to_apply in enumerate(
                        self.U_gates[u_gate_index : u_gate_index + 9]
                    ):
                        if local_idx < 3:
                            gate_to_apply(
                                self.q_device,
                                wires=[qubits_alive[i], qubits_alive[i + 1]],
                            )
                        elif local_idx < 6:
                            gate_to_apply(self.q_device, wires=qubits_alive[i])
                        else:
                            gate_to_apply(self.q_device, wires=qubits_alive[i + 1])
                    u_gate_index += 9
                for i in range(1, len(qubits_alive) - 1, 2):
                    for local_idx, gate_to_apply in enumerate(
                        self.U_gates[u_gate_index : u_gate_index + 9]
                    ):
                        if local_idx < 3:
                            gate_to_apply(
                                self.q_device,
                                wires=[qubits_alive[i], qubits_alive[i + 1]],
                            )
                        elif local_idx < 6:
                            gate_to_apply(self.q_device, wires=qubits_alive[i])
                        else:
                            gate_to_apply(self.q_device, wires=qubits_alive[i + 1])
                    u_gate_index += 9

                for i in range(1, len(qubits_alive), 2):
                    self.V_gates[v_gate_index](
                        self.q_device, wires=[qubits_alive[i], qubits_alive[i - 1]]
                    )
                    v_gate_index += 1

                qubits_alive = qubits_alive[::2]

    def __init__(self, num_qubits: int = 10, return_probs: bool = True):
        super().__init__()
        self.n_wires = num_qubits
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.QiskitQCNNQLayer(num_qubits=num_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.return_probs = return_probs

    def _preprocess_input(self, x):
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        max_dim = 2**self.n_wires
        if x.shape[1] > max_dim:
            x = x[:, :max_dim]
        return x

    def _first_qubit_probs(self, x):
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = x[:, 0]
        return torch.stack([(1.0 + z) / 2.0, (1.0 - z) / 2.0], dim=1)

    def forward(self, x):
        x = self._preprocess_input(x)
        return self._first_qubit_probs(x)

    def forward_logits(self, x):
        x = self._preprocess_input(x)
        probs = self._first_qubit_probs(x)
        z = probs[:, 0] - probs[:, 1]
        return torch.stack([z, -z], dim=1)


# class single_qubit_model(nn.Module):
#     def __init__(self, num_layers: int = 10, output_strategy: str = "probabilities"):
#         super().__init__()

#         self.params = nn.Parameter(2 * np.pi * torch.rand(num_layers * 3))

#         self.circuit, self.q_params = self._single_qubit_model_circuit(num_layers)
#         self.output_strategy = output_strategy

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = reshape_input(x)
#         return ParameterShiftFunction.apply(x, self.params, self)

#     # TODO Optimize angle encoding parametrizing
#     def _single_qubit_model_circuit(self, num_layers: int = 10):
#         """
#         Must be exactly to features
#         features=List[complex]
#         """
#         circuit = qu.QuantumCircuit(1)
#         parameters = []
#         for layer in range(num_layers):
#             parameters.extend(
#                 [
#                     Parameter(f"theta_{layer}"),
#                     Parameter(f"gamma_{layer}"),
#                     Parameter(f"phi_{layer}"),
#                 ]
#             )
#             circuit.rz(parameters[-3], 0)
#             circuit.rx(parameters[-2], 0)
#             circuit.rz(parameters[-1], 0)

#         return circuit, parameters

"""
class qiskit_QCNN(nn.Module):
    def __init__(self, num_qubits: int = 10, num_classes: int = 2):
        super().__init__()
        # TODO find the formula
        self.num_qubits = num_qubits
        self.num_classes = num_classes

        self.num_params = 0
        num_qubits_alive = num_qubits

        while num_qubits_alive > 1:
            self.num_params += 9 * (num_qubits_alive - 1)
            self.num_params += num_qubits_alive // 2
            num_qubits_alive = num_qubits_alive - (num_qubits_alive // 2)

        self.params = nn.Parameter(2 * np.pi * torch.rand(self.num_params))

        self.output_strategy = "first_qubit_probabilities"
        self.circuit, self.q_params = self._QCNN_circuit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x)
        return ParameterShiftFunction.apply(x, self.params, self)

    # TODO Optimize angle encoding parametrizing
    def _QCNN_circuit(self):

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

"""
