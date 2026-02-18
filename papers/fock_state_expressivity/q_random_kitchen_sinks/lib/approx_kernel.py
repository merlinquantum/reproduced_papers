from __future__ import annotations

import numpy as np
import perceval as pcvl
import torch
from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer


def sample_random_features(r: int, seed: int):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(r, 2))
    b = rng.uniform(0.0, 2.0 * np.pi, size=(r,))
    return w, b


def transform_inputs(
    points: np.ndarray, w: np.ndarray, b: np.ndarray, r: int, gamma: float
):
    projections = gamma * (points @ w.T + np.tile(b, (len(points), 1)))
    assert projections.shape == (len(points), r)
    return projections


def classical_features(x_proj: np.ndarray):
    r = x_proj.shape[1]
    return np.sqrt(2 / r) * np.cos(x_proj)


def build_mzi():
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P("data")))
    circuit.add(0, pcvl.BS())
    return circuit


def build_general():
    left = pcvl.GenericInterferometer(
        2,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(phi=pcvl.P(f"theta_psl1{i}"))
            // pcvl.BS()
            // pcvl.PS(phi=pcvl.P(f"theta_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    right = pcvl.GenericInterferometer(
        2,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(phi=pcvl.P(f"theta_psr1{i}"))
            // pcvl.BS()
            // pcvl.PS(phi=pcvl.P(f"theta_psr2{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit = pcvl.Circuit(2)
    circuit.add(0, left)
    circuit.add(0, pcvl.PS(pcvl.P("data")))
    circuit.add(0, right)
    return circuit


def build_quantum_model(cfg: dict):
    circuit_type = cfg.get("circuit", "mzi")
    num_photons = int(cfg.get("num_photon", 10))
    no_bunching = bool(cfg.get("no_bunching", False))
    computation_space = (
        ComputationSpace.UNBUNCHED if no_bunching else ComputationSpace.FOCK
    )
    if circuit_type == "mzi":
        circuit = build_mzi()
        trainable = []
    elif circuit_type == "general":
        circuit = build_general()
        trainable = ["theta"]
    else:
        raise ValueError(f"Unknown circuit: {circuit_type}")

    if num_photons % 2 == 0:
        input_state = [num_photons // 2, num_photons // 2]
    else:
        input_state = [num_photons // 2 + 1, num_photons // 2]

    layer = QuantumLayer(
        input_size=1,
        circuit=circuit,
        trainable_parameters=trainable,
        input_parameters=["data"],
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=computation_space
        ),
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 1))

    return model
