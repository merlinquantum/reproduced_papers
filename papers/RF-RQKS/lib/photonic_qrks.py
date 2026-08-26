"""Local photonic QRKS sampler."""

from __future__ import annotations

import math
import os
from pathlib import Path

import merlin as ml
import numpy as np
import perceval as pcvl
import torch
from torch import nn


class PhotonicQRKS(nn.Module):
    """Photonic quantum random kitchen sink simulator.

    Parameters
    ----------
    n : int
        Number of photons.
    m : int
        Number of optical modes.
    D : int
        Circuit depth.
    E : int
        Number of randomized episodes.
    L_strategy : str
        Encoding strategy, either ``L1`` or ``L2``.
    V_strategy : str | None
        Entangling strategy, either ``V1``, ``V2``, ``V3``, or None.
    data_size : int
        Flattened input feature count.
    v1_same_haar_dist : bool
        Whether every V1 layer reuses one Haar-random unitary.
    """

    def __init__(
        self,
        n: int,
        m: int,
        D: int,
        E: int,
        L_strategy: str,
        V_strategy: str | None,
        data_size: int,
        v1_same_haar_dist: bool = False,
        run_on_hardware: bool = False,
        hardware: str = "sim:slos",
        nsample: int = 5000,
        forward_saves_directory: str | None = None,
    ) -> None:
        super().__init__()
        if m < 2:
            raise ValueError("mode count must be at least 2")
        if n < 1 or n > m:
            raise ValueError("photon count must be between 1 and the mode count")
        if D < 1 or E < 1:
            raise ValueError("depth and episode count must be positive")
        if L_strategy not in {"L1", "L2"}:
            raise ValueError("encoding strategy must be L1 or L2")
        if V_strategy not in {None, "V1", "V2", "V3"}:
            raise ValueError("entangling strategy must be V1, V2, V3, or null")

        self.n = n
        self.m = m
        self.D = D
        self.E = E
        self.L_strategy = L_strategy
        self.V_strategy = V_strategy
        self.hardware = hardware
        self.nsample = nsample
        self.forward_saves_directory = forward_saves_directory
        self.remote_processor = None
        self.number_of_remote_forwards = 0
        self.compute_device = torch.device("cpu")
        self.layer_device = torch.device("cpu")
        self.haar_unitary = None
        if V_strategy == "V1" and v1_same_haar_dist:
            self.haar_unitary = np.asarray(
                pcvl.Matrix.random_unitary(m), dtype=np.complex128
            )

        input_size = D * (m // 2) if L_strategy == "L1" else D * m
        self.register_buffer(
            "weights", torch.normal(mean=0.0, std=2.0, size=(E, input_size, data_size))
        )
        self.register_buffer(
            "biases", torch.rand(E, input_size) * 2.0 * torch.pi
        )
        builder_or_circuit = self._build_circuit()
        measurement = ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.UNBUNCHED
        )
        if isinstance(builder_or_circuit, ml.CircuitBuilder):
            self.layer = ml.QuantumLayer(
                input_size=input_size,
                builder=builder_or_circuit,
                n_photons=n,
                measurement_strategy=measurement,
            )
        else:
            self.layer = ml.QuantumLayer(
                input_size=input_size,
                circuit=builder_or_circuit,
                n_photons=n,
                measurement_strategy=measurement,
                input_parameters=["x"],
            )
        self.layer = self.layer.to(self.layer_device)

        if run_on_hardware:
            self._enable_hardware()

        self.grouping = None
        if m >= 16:
            reduction = 10 if m < 18 else 100
            grouped_size = max(1, math.comb(m, n) // reduction)
            self.grouping = ml.ModGrouping(self.layer.output_size, grouped_size).to(
                self.layer_device
            )

    def to(self, *args, **kwargs):
        """Move classical buffers while retaining the photonic layer on CPU.

        Returns
        -------
        PhotonicQRKS
            Current module.
        """
        module = super().to(*args, **kwargs)
        self.compute_device = torch.empty(0).to(*args, **kwargs).device
        self.layer_device = torch.device("cpu")
        self.layer = self.layer.to(self.layer_device)
        if self.grouping is not None:
            self.grouping = self.grouping.to(self.layer_device)
        return module

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return concatenated output probabilities for every episode.

        Parameters
        ----------
        features : torch.Tensor
            Standardized DCT feature matrix.

        Returns
        -------
        torch.Tensor
            Photonic random features.
        """
        if features.ndim == 1:
            features = features.unsqueeze(0)
        features = features.to(self.compute_device)
        phases = torch.einsum("bd,eid->bei", features, self.weights) + self.biases
        phases = phases.reshape(features.shape[0] * self.E, -1)
        if self.remote_processor is None:
            output = self.layer(phases.to(self.layer_device))
        else:
            outputs = []
            remote_batch_size = 50
            for start in range(0, phases.shape[0], remote_batch_size):
                outputs.append(
                    self.remote_processor.forward(
                        self.layer.eval(),
                        phases[start : start + remote_batch_size].to(self.layer_device),
                        nsample=self.nsample,
                        timeout=0,
                    )
                )
            output = torch.cat(outputs, dim=0)
        if self.grouping is not None:
            output = self.grouping(output)
        output = output.to(self.compute_device).reshape(features.shape[0], -1)
        if self.remote_processor is not None and self.forward_saves_directory is not None:
            output_directory = Path(self.forward_saves_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / (
                f"result_m{self.m}_n{self.n}_E{self.E}_D{self.D}_"
                f"L-{self.L_strategy}_V-{self.V_strategy}_"
                f"num_{self.number_of_remote_forwards}.pt"
            )
            torch.save({"features": features.cpu(), "output": output.cpu()}, output_path)
            self.number_of_remote_forwards += 1
        return output

    def _enable_hardware(self) -> None:
        """Connect the sampler to the configured Quandela processor."""
        token = os.environ["QUANDELA_API_TOKEN"]
        from perceval.runtime import RemoteConfig

        RemoteConfig.set_token(token)
        remote_hardware = pcvl.RemoteProcessor(self.hardware)
        self.remote_processor = ml.MerlinProcessor(remote_hardware)
        self.layer.eval()

    def _build_circuit(self):
        """Build the configured encoding-entangling circuit.

        Returns
        -------
        merlin.CircuitBuilder | perceval.Circuit
            Parameterized photonic circuit.
        """
        builder = ml.CircuitBuilder(n_modes=self.m)
        if self.V_strategy is None:
            for _ in range(self.D):
                builder = _add_encoding(builder, self.L_strategy)
            return builder

        if self.V_strategy == "V1":
            return self._build_v1_circuit()

        if self.L_strategy == "L1":
            builder = _add_l1(builder)
            for _ in range(self.D - 1):
                builder = _add_entangler(builder, self.V_strategy)
                builder = _add_l1(builder)
            return builder

        for _ in range(self.D):
            builder = _add_entangler(builder, self.V_strategy)
            builder = _add_l2(builder)
        return _add_entangler(builder, self.V_strategy)

    def _build_v1_circuit(self) -> pcvl.Circuit:
        """Build a circuit containing explicit Haar-random unitaries.

        Returns
        -------
        perceval.Circuit
            Parameterized Perceval circuit.
        """
        circuit = pcvl.Circuit(m=self.m)
        parameter_index = 0
        if self.L_strategy == "L1":
            encoding, parameter_index = _l1_circuit(self.m, parameter_index)
            circuit.add(list(range(self.m)), encoding)
            for _ in range(self.D - 1):
                circuit.add(list(range(self.m)), _v1(self.m, self.haar_unitary))
                encoding, parameter_index = _l1_circuit(self.m, parameter_index)
                circuit.add(list(range(self.m)), encoding)
            return circuit

        for _ in range(self.D):
            circuit.add(list(range(self.m)), _v1(self.m, self.haar_unitary))
            encoding, parameter_index = _l2_circuit(self.m, parameter_index)
            circuit.add(list(range(self.m)), encoding)
        circuit.add(list(range(self.m)), _v1(self.m, self.haar_unitary))
        return circuit


def _add_encoding(builder: ml.CircuitBuilder, strategy: str) -> ml.CircuitBuilder:
    return _add_l1(builder) if strategy == "L1" else _add_l2(builder)


def _add_l1(builder: ml.CircuitBuilder) -> ml.CircuitBuilder:
    mode_pairs = [[2 * index, 2 * index + 1] for index in range(builder.n_modes // 2)]
    builder.add_superpositions(mode_pairs)
    builder.add_angle_encoding([pair[0] for pair in mode_pairs])
    builder.add_superpositions(mode_pairs)
    return builder


def _add_l2(builder: ml.CircuitBuilder) -> ml.CircuitBuilder:
    builder.add_angle_encoding()
    return builder


def _l1_circuit(mode_count: int, start: int) -> tuple[pcvl.Circuit, int]:
    circuit = pcvl.Circuit(m=mode_count)
    for pair_index in range(mode_count // 2):
        parameter = pcvl.Parameter(f"x{start + pair_index}")
        modes = [2 * pair_index, 2 * pair_index + 1]
        circuit.add(modes, pcvl.BS())
        circuit.add(modes[0], pcvl.PS(parameter))
        circuit.add(modes, pcvl.BS())
    return circuit, start + mode_count // 2


def _l2_circuit(mode_count: int, start: int) -> tuple[pcvl.Circuit, int]:
    circuit = pcvl.Circuit(m=mode_count)
    for mode_index in range(mode_count):
        circuit.add(mode_index, pcvl.PS(pcvl.Parameter(f"x{start + mode_index}")))
    return circuit, start + mode_count


def _v1(mode_count: int, unitary: np.ndarray | None) -> pcvl.Circuit:
    matrix = unitary
    if matrix is None:
        matrix = np.asarray(pcvl.Matrix.random_unitary(mode_count), dtype=np.complex128)
    circuit = pcvl.Circuit(m=mode_count)
    circuit.add(list(range(mode_count)), pcvl.Unitary(matrix))
    return circuit


def _add_entangler(
    builder: ml.CircuitBuilder, strategy: str
) -> ml.CircuitBuilder:
    mode_count = builder.n_modes
    if strategy == "V2":
        builder.add_superpositions([[index, index + 1] for index in range(mode_count - 1)])
        builder.add_superpositions(
            [[index, index + 1] for index in reversed(range(mode_count - 2))]
        )
        return builder
    builder.add_superpositions(
        [[index, index + 1] for index in range(0, (mode_count // 2) * 2, 2)]
    )
    offset = 1
    while offset < mode_count // 2 + mode_count % 2:
        builder.add_superpositions(
            [[index, index + 1] for index in range(offset, mode_count - offset, 2)]
        )
        offset += 1
    return builder
