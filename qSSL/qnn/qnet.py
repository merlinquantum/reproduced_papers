# MIT License
#
# Copyright (c) 2021 Ben Jaderberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from config import Config
from gradient_calculator import calculate_gradient_list
from input.vector_data_handler import VectorDataHandler
from quantum_network_circuit import QuantumNetworkCircuit
from torch import cuda
from torch.autograd import Function


class QNetFunction(Function):
    @staticmethod
    def forward(
        ctx, input, weight, qnn: QuantumNetworkCircuit, shots, save_statevectors
    ):
        weight_vector = torch.flatten(weight).tolist()
        batch_size = input.size()[0]

        if (input > np.pi).any() or (input < 0).any():
            logging.info(
                "Input data to quantum neural network is outside range {0,π}. Consider using a bounded \
            activation function to prevent wrapping round of states within the Bloch sphere."
            )

        for i in range(batch_size):
            input_vector = torch.flatten(input[i, :]).tolist()

            if i == 0:
                logging.debug(f"First input vector of batch to QNN: {input_vector}")

            if type(qnn.config.data_handler is VectorDataHandler):
                if qnn.input_data is None:
                    qnn.construct_network(input_vector)
                ctx.QNN = qnn
            else:
                single_input_qnn = copy.deepcopy(qnn)
                single_input_qnn.construct_network(input_vector)
                ctx.QNN = single_input_qnn

            parameter_list = np.concatenate((np.array(input_vector), weight_vector))

            result = qnn.evaluate_circuit(parameter_list, shots=shots)
            vector = (
                torch.tensor(qnn.get_vector_from_results(result)).unsqueeze(0).float()
            )
            if save_statevectors and result.backend_name == "statevector_simulator":
                state = result.get_statevector(0)
                qnn.statevectors.append(state)

            if i == 0:
                output = vector
            else:
                output = torch.cat((output, vector), 0)

        ctx.shots = shots

        if cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        ctx.save_for_backward(input, weight)
        ctx.device = device
        output = output.to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        device = ctx.device
        weight_vector = torch.flatten(weight).tolist()
        batch_size = input.size()[0]

        for i in range(batch_size):
            input_vector = torch.flatten(input[i, :]).tolist()

            gradient = calculate_gradient_list(
                ctx.QNN,
                parameter_list=np.concatenate((input_vector, weight_vector)),
                method=ctx.QNN.config.grad_method,
                shots=ctx.shots,
            )

            ctx.QNN.gradients.append(gradient.tolist())

            single_vector_d_out_d_input = (
                torch.tensor(gradient[: len(input_vector)]).double().to(device)
            )
            single_vector_d_out_d_weight = (
                torch.tensor(gradient[len(input_vector) :]).double().to(device)
            )

            if i == 0:
                batched_d_out_d_input = single_vector_d_out_d_input.unsqueeze(0)
                batched_d_out_d_weight = single_vector_d_out_d_weight.unsqueeze(0)
            else:
                batched_d_out_d_input = torch.cat(
                    (batched_d_out_d_input, single_vector_d_out_d_input.unsqueeze(0)), 0
                )
                batched_d_out_d_weight = torch.cat(
                    (batched_d_out_d_weight, single_vector_d_out_d_weight.unsqueeze(0)),
                    0,
                )
        batched_d_loss_d_input = torch.bmm(
            batched_d_out_d_input, grad_output.unsqueeze(2).double()
        ).squeeze()
        batched_d_loss_d_weight = torch.bmm(
            batched_d_out_d_weight, grad_output.unsqueeze(2).double()
        ).squeeze()
        return (
            batched_d_loss_d_input.to(device),
            batched_d_loss_d_weight.to(device),
            None,
            None,
            None,
        )


class QNet(nn.Module):
    """
    Custom PyTorch module implementing neural network layer consisting on a parameterised quantum circuit. Forward and
    backward passes allow this to be directly integrated into a PyTorch network.
    For a "vector" input encoding, inputs should be restricted to the range [0,π) so that there is no wrapping of input
    states round the bloch sphere and extreme value of the input correspond to states with the smallest overlap. If
    inputs are given outside this range during the forward pass, info level logging will occur.
    """

    def __init__(
        self,
        n_qubits,
        encoding,
        ansatz_type,
        layers,
        sweeps_per_layer,
        activation_function_type,
        shots,
        backend_type="qasm_simulator",
        save_statevectors=False,
    ):
        super().__init__()

        config = Config(
            encoding=encoding,
            ansatz_type=ansatz_type,
            layers=layers,
            sweeps_per_layer=sweeps_per_layer,
            activation_function_type=activation_function_type,
            meas_method="all",
            backend_type=backend_type,
        )
        self.qnn = QuantumNetworkCircuit(config, n_qubits)

        self.shots = shots

        num_weights = len(list(self.qnn.ansatz_circuit_parameters))
        self.quantum_weight = nn.Parameter(torch.Tensor(num_weights))

        self.quantum_weight.data.normal_(std=1.0 / np.sqrt(n_qubits))

        self.save_statevectors = save_statevectors

        logging.debug(f"Quantum parameters initialised as {self.quantum_weight.data}")

    def forward(self, input_vector):
        return QNetFunction.apply(
            input_vector,
            self.quantum_weight,
            self.qnn,
            self.shots,
            self.save_statevectors,
        )
