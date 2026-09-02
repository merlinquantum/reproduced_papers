"""TorchQuantum implementation of the random 1D circuit in McClean et al."""

from __future__ import annotations

import random

import numpy as np
import torch


def sample_gradient_variance(
    number_of_qubits: int, number_of_layers: int, samples: int, cfg: dict
) -> float:
    """Estimate the variance of the first circuit-parameter gradient.

    Parameters
    ----------
    number_of_qubits : int
        Number of wires in the 1D circuit.
    number_of_layers : int
        Number of random rotation/CZ layers.
    samples : int
        Number of independently sampled parameterized circuits.
    cfg : dict
        Configuration containing the requested torch dtype.

    Returns
    -------
    float
        Sample variance of the first gradient component.
    """
    if number_of_qubits < 2:
        raise ValueError("The two-local ZZ objective requires at least two qubits")
    import torchquantum as tq

    dtype = torch.float64 if cfg.get("dtype") == "float64" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gradients: list[float] = []
    for _ in range(samples):
        parameters = torch.rand(
            number_of_qubits * number_of_layers, device=device, dtype=dtype
        ) * (2 * np.pi)
        parameters.requires_grad_(True)
        quantum_device = tq.QuantumDevice(n_wires=number_of_qubits, device=device)
        for wire in range(number_of_qubits):
            tq.ry(quantum_device, wires=wire, params=np.pi / 4)
        parameter_index = 0
        for _layer in range(number_of_layers):
            for wire in range(number_of_qubits):
                rotation = random.choice((tq.rx, tq.ry, tq.rz))
                rotation(quantum_device, wires=wire, params=parameters[parameter_index])
                parameter_index += 1
            for wire in range(number_of_qubits - 1):
                tq.cz(quantum_device, wires=[wire, wire + 1])
        probabilities = quantum_device.states.reshape(-1).abs().square()
        indices = torch.arange(probabilities.numel(), device=device)
        first_bit = (indices >> (number_of_qubits - 1)) & 1
        second_bit = (indices >> (number_of_qubits - 2)) & 1
        zz_values = 1.0 - 2.0 * (first_bit ^ second_bit).to(dtype)
        energy = torch.sum(probabilities * zz_values)
        energy.backward()
        gradients.append(float(parameters.grad[0].detach().cpu()))
    return float(np.var(np.asarray(gradients), ddof=1 if samples > 1 else 0))
