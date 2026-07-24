from __future__ import annotations

from math import pi

import merlin as ml
import perceval as pcvl
import torch
import torch.nn as nn

COEFFS = (
    0.1 * torch.pi,
    0.3 * torch.pi,
    0.5 * torch.pi,
    0.7 * torch.pi,
    1.0 * torch.pi,
)
# Fixed beam-splitter angles selected by EGAS as part of the circuit architecture.
# Perceval's default BS convention maps these to 75%, 50%, and 25% reflectivity.
BS_ANGLES = (pi / 3, pi / 2, 2 * pi / 3)
FIXED_PS_PHASES = {
    "PS_PI": pi,
    "PS_PI_2": pi / 2,
}


def build_token_pool(n_modes: int, num_features: int):
    """Enumerate the full token pool C. Returns list of (gate, q, data_idx, r)."""
    tokens = []
    for q in range(n_modes):
        for d in range(num_features):
            for r in COEFFS:
                tokens.append(("PS", q, d, r))
        tokens.append(("PS_PI", q, 0, 0.0))
        tokens.append(("PS_PI_2", q, 0, 0.0))
    for q in range(n_modes - 1):
        for theta in BS_ANGLES:
            tokens.append(("BS", q, 0, theta))
    return tokens


def create_perceval_circuit(
    sequence, n_modes: int
) -> tuple[pcvl.Circuit, list[pcvl.Parameter], list[pcvl.Parameter]]:
    """"""

    circuit = pcvl.Circuit(m=n_modes)
    input_parameters = []
    trainable_parameters = []

    for gate, q, _, value in sequence:
        if gate == "PS":
            input_param = pcvl.Parameter(f"theta{len(input_parameters)}")
            trainable_param = pcvl.Parameter(f"phi{len(trainable_parameters)}")
            input_parameters.append(input_param)
            trainable_parameters.append(trainable_param)
            circuit.add(q, pcvl.PS(input_param))
            circuit.add(q, pcvl.PS(trainable_param))
            # D, the data index will be handled in the QuantumLayer
        elif gate in FIXED_PS_PHASES:
            circuit.add(q, pcvl.PS(FIXED_PS_PHASES[gate]))
        elif gate == "BS":
            # ``value`` is a fixed structural angle, not a data-dependent input.
            # Retain the original 50:50 splitter for legacy sequences stored with 0.0.
            theta = pi / 2 if value == 0.0 else value
            circuit.add([q, q + 1], pcvl.BS(theta=theta))
        else:
            raise ValueError(f"Unsupported photonic gate token: {gate}")
    return circuit, input_parameters, trainable_parameters


def create_quantum_module(
    sequence,
    num_features: int,
    n_modes: int,
    num_photons: int = 2,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
):
    circuit, input_parameters, trainable_parameters = create_perceval_circuit(
        sequence, n_modes=n_modes
    )

    ps_data_indices = [data_idx for gate, _, data_idx, _ in sequence if gate == "PS"]
    ps_r_factors = [r for gate, _, _, r in sequence if gate == "PS"]

    class BiasMLP(nn.Module):
        """Small MLP with a zero-initialised output head; output scaled by a fixed gain (=10)."""

        def __init__(
            self, n_in: int, output_size: int = 1, hidden: int = 32, gain: float = 10.0
        ):
            super().__init__()
            self.output_size = output_size
            self.gain = gain
            if output_size <= 0:
                self.net = None
            else:
                self.net = nn.Sequential(
                    nn.Linear(n_in, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, output_size),
                )
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            if self.output_size <= 0:
                return torch.zeros(X.shape[0], 0, dtype=torch.float32, device=X.device)
            return self.gain * self.net(X.to(torch.float32))

    class QuantumModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.ps_data_indices = ps_data_indices
            self.ps_r_factors = torch.tensor(ps_r_factors, dtype=torch.float32)
            self.layer = ml.QuantumLayer(
                input_size=2 * len(input_parameters),
                circuit=circuit,
                n_photons=num_photons,
                input_parameters=["theta", "phi"] if len(input_parameters) > 0 else [],
                measurement_strategy=ml.MeasurementStrategy.amplitudes(
                    computation_space=computation_space
                ),
            )
            self.bias = BiasMLP(n_in=num_features, output_size=len(input_parameters))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if len(self.ps_data_indices) > 0:
                if x.shape[-1] <= max(self.ps_data_indices, default=-1):
                    raise ValueError(
                        "Input feature width is too small for the PS data indices in "
                        "the sequence."
                    )
                theta = x[..., self.ps_data_indices] * self.ps_r_factors
                # Biases
                phi = self.bias(x)

                layer_input = torch.cat([theta, phi], dim=-1)
                return self.layer(layer_input)
            else:
                # No input parameters needed, create empty input with batch size
                layer_input = torch.zeros(x.shape[0], 0, dtype=x.dtype, device=x.device)
                return self.layer()

        def reset_bias(self, hidden: int | None = None, gain: float | None = None):
            if hidden is None:
                hidden = 32
            if gain is None:
                gain = 10.0
            # Get current device and dtype from the existing bias (or any parameter)
            try:
                param = next(self.parameters())
                device = param.device
                dtype = param.dtype
                self.bias = BiasMLP(
                    n_in=num_features,
                    output_size=len(input_parameters),
                    hidden=hidden,
                    gain=gain,
                ).to(device=device, dtype=dtype)
            except:
                self.bias = BiasMLP(
                    n_in=num_features,
                    output_size=len(input_parameters),
                    hidden=hidden,
                    gain=gain,
                )

    return QuantumModule()
