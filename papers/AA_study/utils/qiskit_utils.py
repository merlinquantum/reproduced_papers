import qiskit as qu
import torch
import torch.nn as nn
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import numpy as np
from typing import List


def _simulate_outputs(
    model: nn.Module, params: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    """
    Run the Qiskit circuit for each input state and collect outputs.

    Parameters
    ----------
    model : torch.nn.Module
        Model carrying the Qiskit circuit, parameters, and output strategy.
    params : torch.Tensor
        Trainable parameters to bind into the circuit.
    inputs : torch.Tensor
        Input state vectors (one per sample).

    Returns
    -------
    torch.Tensor
        Batched outputs based on the model output strategy.
    """
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
        elif model.output_strategy == "first_qubit_probabilities":
            # Marginal probabilities for qubit 0; output shape (2,)
            probs = statevector.probabilities()
            p0 = 0.0
            for idx, p in enumerate(probs):
                bit = (idx >> 0) & 1
                if bit == 0:
                    p0 += p

            ### TODO CHECK IF LEGIT
            """
            Spliting the probability spectra into num_classes parts and calculating the distance between p0 and the center of the part
            
            """

            probability_class_vector = [
                1 - abs(((i / model.num_classes) + (1 / (2 * model.num_classes))) - p0)
                for i in range(model.num_classes)
            ]
            outputs.append(
                probability_class_vector / np.linalg.norm(probability_class_vector)
            )
        else:
            raise ValueError(
                f"Unknown output '{model.output_strategy}'. Use 'statevector', "
                "'probabilities' or 'first_qubit_probabilities'."
            )

    out_np = np.stack(outputs, axis=0)
    if model.output_strategy == "statevector":
        return torch.from_numpy(out_np).to(dtype=torch.complex64)
    return torch.from_numpy(out_np).to(dtype=torch.float32)


def _parameter_shift_vjp(
    model: nn.Module,
    params: torch.Tensor,
    inputs: torch.Tensor,
    grad_output: torch.Tensor,
    shift: float = np.pi / 2,
) -> torch.Tensor:
    """
    Compute vector-Jacobian product via parameter-shift rule.

    Parameters
    ----------
    model : torch.nn.Module
        Model carrying the Qiskit circuit.
    params : torch.Tensor
        Trainable parameters to differentiate with respect to.
    inputs : torch.Tensor
        Input state vectors.
    grad_output : torch.Tensor
        Upstream gradient.
    shift : float, optional
        Parameter-shift value in radians.

    Returns
    -------
    torch.Tensor
        Gradient with respect to params.
    """
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
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Forward pass with Qiskit simulation.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Autograd context.
        inputs : torch.Tensor
            Input state vectors.
        params : torch.Tensor
            Trainable parameters to bind into the circuit.
        model : torch.nn.Module
            Model carrying the Qiskit circuit.

        Returns
        -------
        torch.Tensor
            Batched outputs from Qiskit simulation.
        """
        outputs = _simulate_outputs(model, params, inputs)
        ctx.save_for_backward(inputs, params)
        ctx.model = model
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using parameter-shift VJP.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Autograd context with saved tensors.
        grad_output : torch.Tensor
            Upstream gradient.

        Returns
        -------
        tuple
            Gradients for (inputs, params, model). Only params has gradients.
        """
        inputs, params = ctx.saved_tensors
        grad_params = _parameter_shift_vjp(ctx.model, params, inputs, grad_output)
        return None, grad_params, None


def U_gate(parameters: List[Parameter]) -> qu.QuantumCircuit:
    """
    Build a 2-qubit U gate composed of fixed sub-gates.

    Parameters
    ----------
    parameters : list[qiskit.circuit.Parameter]
        Parameters for the 9 sub-gates.

    Returns
    -------
    qiskit.QuantumCircuit
        Two-qubit circuit implementing the U gate.
    """
    qcirc = qu.QuantumCircuit(2)
    qcirc.rxx(parameters[0], 0, 1)
    qcirc.ryy(parameters[1], 0, 1)
    qcirc.rzz(parameters[2], 0, 1)

    qcirc.rz(parameters[3], 0)
    qcirc.rx(parameters[4], 0)
    qcirc.rz(parameters[5], 0)

    qcirc.rz(parameters[6], 1)
    qcirc.rx(parameters[7], 1)
    qcirc.rz(parameters[8], 1)
    return qcirc


def reshape_input(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten and pad inputs to a power-of-two length.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Flattened and zero-padded tensor with feature dimension a power of two.
    """
    with torch.no_grad():
        if len(x.shape) > 2:
            x = x.squeeze()
            x = x.reshape((x.shape[0], np.prod(x.shape[1:])))

        elif len(x.shape) == 1:
            x = x.unsqueeze(0)

        if (x.shape[1] & (x.shape[1] - 1) != 0) or x.shape[1] == 0:
            num_features_to_add = int(2 ** np.ceil(np.log2(x.shape[1]))) - x.shape[1]
            x = torch.cat([x, torch.zeros((x.shape[0], num_features_to_add))], 1)
    return x
