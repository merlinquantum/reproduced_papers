from __future__ import annotations

import merlin as ml
import perceval as pcvl
import torch
import torch.nn as nn
from math import pi

COEFFS = (0.1, 0.3, 0.5, 0.7, 1.0)
FIXED_PS_PHASES = {
    "PS_PI": pi,
    "PS_PI_2": pi / 2,
}


def build_token_pool(n_modes: int):
    """Enumerate the full token pool C. Returns list of (gate, q, data_idx, r)."""
    tokens = []
    for q in range(n_modes):
        for d in range(n_modes):
            for r in COEFFS:
                tokens.append(("PS", q, d, r))
        tokens.append(("PS_PI", q, 0, 0.0))
        tokens.append(("PS_PI_2", q, 0, 0.0))
    for q in range(n_modes - 1):  # CNOT (q, q+1)
        tokens.append(("BS", q, 0, 0.0))
    return tokens


def create_perceval_circuit(
    sequence, n_modes: int
) -> tuple[pcvl.Circuit, list[pcvl.Parameter], list[pcvl.Parameter]]:
    """"""

    circuit = pcvl.Circuit(m=n_modes)
    input_parameters = []
    trainable_parameters = []

    for gate, q, _, r in sequence:
        if gate == "PS":
            input_param = pcvl.Parameter(f"theta{len(input_parameters)}")
            trainable_param = pcvl.Parameter(
                f"phi{len(trainable_parameters)}", value=0.0
            )
            input_parameters.append(input_param)
            circuit.add(q, pcvl.PS(input_param * r + trainable_param))
            # D, the data index will be handled in the QuantumLayer
        elif gate in FIXED_PS_PHASES:
            circuit.add(q, pcvl.PS(FIXED_PS_PHASES[gate]))
        elif gate == "BS":
            circuit([q, q + 1], pcvl.BS())
        else:
            raise ValueError(f"Unsupported photonic gate token: {gate}")
    return circuit, input_parameters, trainable_parameters


def create_quantum_module(
    sequence,
    n_modes: int,
    num_photons: int = 2,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
):
    circuit, input_parameters, trainable_parameters = create_perceval_circuit(
        sequence, n_modes=n_modes
    )

    ps_data_indices = [data_idx for gate, _, data_idx, _ in sequence if gate == "PS"]

    class QuantumModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.ps_data_indices = ps_data_indices
            self.layer = ml.QuantumLayer(
                input_size=len(input_parameters),
                circuit=circuit,
                n_photons=num_photons,
                trainable_parameters=trainable_parameters,
                input_parameters=input_parameters,
                measurement_strategy=ml.MeasurementStrategy.amplitudes(
                    computation_space=computation_space
                ),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[-1] <= max(self.ps_data_indices, default=-1):
                raise ValueError(
                    "Input feature width is too small for the PS data indices in "
                    "the sequence."
                )

            layer_input = x[..., self.ps_data_indices]
            return self.layer(layer_input)

    return QuantumModule()
