"""MerLin adaptation of the photonic Fig. 3 variance experiment.
From https://github.com/easonoob/Eason-2026-preasymptotic-trainability-pvqc/tree/main"""

from __future__ import annotations

import math

import numpy as np
import torch


def sample_photonic_variance(
    number_of_qubits: int,
    computation_space: str,
    initialization: str,
    samples: int,
    cfg: dict,
) -> float:
    """Estimate photonic gradient variance in one MerLin computation space.

    Parameters
    ----------
    number_of_qubits : int
        Number of dual-rail logical qubits.
    computation_space : str
        MerLin space: ``fock``, ``unbunched``, or ``dual_rail``.
    initialization : str
        ``arcsin`` or uniform phase initialization.
    samples : int
        Number of random parameter samples.
    cfg : dict
        Configuration containing dtype.

    Returns
    -------
    float
        Mean variance across trainable circuit parameters.
    """
    import merlin as ML
    import perceval as pcvl

    dtype = torch.float64 if cfg.get("dtype") == "float64" else torch.float32
    modes = 2 * number_of_qubits
    circuit = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS(pcvl.P(f"phi_{2 * i}")) // pcvl.PS(pcvl.P(f"phi_{2 * i + 1}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
        depth=modes,
    )
    initial_state = pcvl.BasicState([1, 0] * number_of_qubits)
    measurement_space = {
        "fock": ML.ComputationSpace.FOCK,
        "unbunched": ML.ComputationSpace.UNBUNCHED,
        "dual_rail": ML.ComputationSpace.DUAL_RAIL,
    }[computation_space]
    layer = ML.QuantumLayer(
        circuit=circuit,
        input_parameters=["phi"],
        input_state=initial_state,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        measurement_strategy=ML.MeasurementStrategy.probs(measurement_space),
        dtype=dtype,
    )
    parameter_count = math.ceil(modes / 2) * modes + (modes // 2) * (modes - 2)
    gradients: list[np.ndarray] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(samples):
        if initialization == "arcsin":
            parameters = torch.arcsin(
                torch.sqrt(torch.rand(parameter_count, device=device, dtype=dtype))
            )
        else:
            parameters = torch.rand(parameter_count, device=device, dtype=dtype) * (
                np.pi / 2
            )
        parameters.requires_grad_(True)
        probabilities = layer(parameters).squeeze(0).to(dtype)
        probabilities = probabilities / probabilities.sum()
        target = torch.rand_like(probabilities)
        target = target / target.sum()
        loss = (
            1
            - torch.sum(torch.sqrt(torch.clamp(probabilities, min=1e-20) * target)) ** 2
        )
        loss.backward()
        if torch.isfinite(parameters.grad).all():
            gradients.append(parameters.grad.detach().cpu().numpy())
    if len(gradients) < 2:
        raise RuntimeError(
            f"Fewer than two finite gradient samples remained for "
            f"{number_of_qubits} qubits, {computation_space}, {initialization}"
        )
    return float(np.mean(np.var(np.asarray(gradients), axis=0)))
