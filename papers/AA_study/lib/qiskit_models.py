import qiskit as qu
import torch
import torch.nn as nn
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import numpy as np
from typing import List, Optional


class single_qubit_model(nn.Module):
    def __init__(self, num_layers: int = 10, output_strategy: str = "probabilities"):
        super().__init__()

        self.params = nn.Parameter(2 * np.pi * torch.rand(num_layers * 3))

        self.circuit, self.q_params = self._single_qubit_model_circuit(num_layers)
        self.output_strategy = output_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        """
        bind_dict = {p: float(v) for p, v in zip(self.q_params, self.params)}
        circuit_to_run = self.circuit.assign_parameters(bind_dict, inplace=False)

        outputs = []
        for input in x:
            circuit_per_input = circuit_to_run.copy()
            circuit_per_input.initialize(input, normalize=True)
            statevector = Statevector.from_instruction(circuit_per_input)
            if self.output_strategy == "statevector":
                outputs.append(statevector.data)
            elif self.output_strategy == "probabilities":
                outputs.append(statevector.probabilities())
            else:
                raise ValueError(
                    f"Unknown output '{self.output_strategy}'. Use 'statevector' or 'probabilities'."
                )
            return torch.tensor(np.array(outputs))
        """
        # TODO Modify so that gradients update
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


# TODO IMPLEMENT
class qiskit_QCNN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass


def _simulate_outputs(
    model: "single_qubit_model", params: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    bind_dict = {p: float(v) for p, v in zip(model.q_params, params)}
    circuit_to_run = model.circuit.assign_parameters(bind_dict, inplace=False)

    outputs = []
    for state in inputs:
        circuit_per_input = qu.QuantumCircuit(circuit_to_run.num_qubits)
        circuit_per_input.initialize(state.detach().cpu().numpy(), normalize=True)
        circuit_per_input.compose(circuit_to_run, inplace=True)
        statevector = Statevector.from_instruction(circuit_per_input)
        if model.output_strategy == "statevector":
            outputs.append(statevector.data)
        elif model.output_strategy == "probabilities":
            outputs.append(statevector.probabilities())
        else:
            raise ValueError(
                f"Unknown output '{model.output_strategy}'. Use 'statevector' or 'probabilities'."
            )

    out_np = np.stack(outputs, axis=0)
    if model.output_strategy == "statevector":
        return torch.from_numpy(out_np).to(dtype=torch.complex64)
    return torch.from_numpy(out_np).to(dtype=torch.float32)


def _parameter_shift_vjp(
    model: "single_qubit_model",
    params: torch.Tensor,
    inputs: torch.Tensor,
    grad_output: torch.Tensor,
    shift: float = np.pi / 2,
) -> torch.Tensor:
    grad_output_np = grad_output.detach().cpu().numpy()
    grad_params = np.zeros(params.numel(), dtype=np.float64)

    for i in range(params.numel()):
        params_plus = params.clone()
        params_minus = params.clone()
        params_plus[i] = params_plus[i] + shift
        params_minus[i] = params_minus[i] - shift

        out_plus = _simulate_outputs(model, params_plus, inputs).detach().cpu().numpy()
        out_minus = (
            _simulate_outputs(model, params_minus, inputs).detach().cpu().numpy()
        )

        grad_out = 0.5 * (out_plus - out_minus)
        vjp = np.sum(grad_output_np * grad_out)
        if np.iscomplexobj(vjp):
            vjp = np.real(vjp)
        grad_params[i] = vjp

    return torch.tensor(grad_params, dtype=params.dtype, device=params.device)


class ParameterShiftFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        params: torch.Tensor,
        model: "single_qubit_model",
    ) -> torch.Tensor:
        outputs = _simulate_outputs(model, params, inputs)
        ctx.save_for_backward(inputs, params)
        ctx.model = model
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, params = ctx.saved_tensors
        grad_params = _parameter_shift_vjp(ctx.model, params, inputs, grad_output)
        return None, grad_params, None


import torch.optim as optim
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from papers.AA_study.utils.datasets import generate_fig_1_dataset, get_data_loader

test_model = single_qubit_model()
optimizer = optim.Adam(test_model.parameters(), lr=0.1)

data_loader = get_data_loader(generate_fig_1_dataset(num_samples_per_class=400))
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
