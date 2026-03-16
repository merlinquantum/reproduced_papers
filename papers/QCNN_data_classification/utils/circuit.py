"""Circuit and photonic helpers for the QCNN reproduction."""

from __future__ import annotations

import random
from enum import Enum

import merlin as ML  # noqa: N812
import numpy as np
import perceval as pcvl
import torch.nn as nn


class StatePattern(Enum):
    DEFAULT = "default"
    SPACED = "spaced"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"


class StateGenerator:
    @staticmethod
    def generate_state(
        n_modes: int, n_photons: int, state_pattern: StatePattern
    ) -> list[int]:
        if n_photons < 0 or n_photons > n_modes:
            raise ValueError(f"Cannot place {n_photons} photons into {n_modes} modes.")
        if state_pattern == StatePattern.SPACED:
            return StateGenerator._generate_spaced_state(n_modes, n_photons)
        if state_pattern == StatePattern.SEQUENTIAL:
            return StateGenerator._generate_sequential_state(n_modes, n_photons)
        return StateGenerator._generate_periodic_state(n_modes, n_photons)

    @staticmethod
    def _generate_spaced_state(n_modes: int, n_photons: int) -> list[int]:
        if n_photons == 0:
            return [0] * n_modes
        if n_photons == 1:
            pos = n_modes // 2
            return [1 if i == pos else 0 for i in range(n_modes)]
        positions = [int(i * n_modes / n_photons) for i in range(n_photons)]
        positions = [min(pos, n_modes - 1) for pos in positions]
        occ = [0] * n_modes
        for pos in positions:
            occ[pos] += 1
        return occ

    @staticmethod
    def _generate_periodic_state(n_modes: int, n_photons: int) -> list[int]:
        bits = [1 if i % 2 == 0 else 0 for i in range(min(n_photons * 2, n_modes))]
        count = sum(bits)
        i = 0
        while count < n_photons and i < n_modes:
            if i >= len(bits):
                bits.append(0)
            if bits[i] == 0:
                bits[i] = 1
                count += 1
            i += 1
        padding = [0] * (n_modes - len(bits))
        return bits + padding

    @staticmethod
    def _generate_sequential_state(n_modes: int, n_photons: int) -> list[int]:
        return [1 if i < n_photons else 0 for i in range(n_modes)]


def _generate_interferometer(
    n_modes: int, stage_idx: int, reservoir_mode: bool = False
):
    if reservoir_mode:
        return pcvl.GenericInterferometer(
            n_modes,
            lambda idx: (
                pcvl.BS(theta=np.pi * 2 * random.random())
                // (0, pcvl.PS(phi=np.pi * 2 * random.random()))
            ),
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=2 * n_modes,
            phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
        )

    def mzi(P1, P2):
        return (
            pcvl.Circuit(2)
            .add((0, 1), pcvl.BS())
            .add(0, pcvl.PS(P1))
            .add((0, 1), pcvl.BS())
            .add(0, pcvl.PS(P2))
        )

    offset = stage_idx * (n_modes * (n_modes - 1) // 2)
    shape = pcvl.InterferometerShape.RECTANGLE
    return pcvl.GenericInterferometer(
        n_modes,
        fun_gen=lambda idx: mzi(
            pcvl.P(f"phi_0{offset + idx}"), pcvl.P(f"phi_1{offset + idx}")
        ),
        shape=shape,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(
            phi=pcvl.P(f"phi_02{stage_idx}_{idx}")
        ),
    )


def build_parallel_columns_circuit(
    n_modes: int, n_features: int, reservoir_mode: bool = False
):
    circuit = pcvl.Circuit(n_modes)
    ps_idx = 0
    for stage in range(n_features + 1):
        circuit.add(0, _generate_interferometer(n_modes, stage, reservoir_mode))
        if stage < n_features:
            for m_idx in range(n_modes):
                circuit.add(m_idx, pcvl.PS(pcvl.P(f"pl{ps_idx}x")))
                ps_idx += 1
    return circuit


def create_quantum_circuit(
    n_modes: int, n_features: int, amplitudes_encoding: bool = False
) -> pcvl.Circuit:
    wl = pcvl.GenericInterferometer(
        n_modes,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_li{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_lo{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    if not amplitudes_encoding:
        c_var = pcvl.Circuit(n_modes)
        for i in range(n_features):
            px = pcvl.P(f"px{i + 1}")
            c_var.add(i, pcvl.PS(px))

        wr = pcvl.GenericInterferometer(
            n_modes,
            lambda i: (
                pcvl.BS()
                // pcvl.PS(pcvl.P(f"theta_ri{i}"))
                // pcvl.BS()
                // pcvl.PS(pcvl.P(f"theta_ro{i}"))
            ),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        return wl // c_var // wr
    else:
        return wl


def required_input_params(n_modes: int, n_features: int) -> int:
    if n_features > n_modes:
        raise ValueError(
            "n_features cannot exceed n_modes when matching inputs one-to-one."
        )
    return n_features


def build_single_gi_layer(
    n_modes: int,
    n_features: int,
    shots: int,
    n_photons: int,
    reservoir_mode: bool,
    state_pattern: str,
):
    circ = create_quantum_circuit(n_modes, n_features)
    pattern = StatePattern[state_pattern.upper()]
    input_state = StateGenerator.generate_state(n_modes, n_photons, pattern)
    trainable_prefixes = [] if reservoir_mode else ["theta"]
    input_prefixes = ["px"]

    return ML.QuantumLayer(
        input_size=n_features,
        output_size=2,
        circuit=circ,
        trainable_parameters=trainable_prefixes,
        input_parameters=input_prefixes,
        input_state=input_state,
        output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
        shots=shots,
    )


def build_quantum_kernel_layer(
    kernel_modes: int,
    kernel_features: int,
    n_photons: int,
    state_pattern: str,
    reservoir_mode: bool,
    show_circuit: bool = False,
    amplitudes_encoding: bool = False,
) -> ML.QuantumLayer:
    circuit = create_quantum_circuit(kernel_modes, kernel_features, amplitudes_encoding)
    if show_circuit:
        print(
            f"\n ----- Created a quantum kernel circuit with {kernel_modes} modes and {kernel_features} features -----\n"
        )
        pcvl.pdisplay(circuit)

    if not amplitudes_encoding:
        # use "px" input parameters
        pattern = StatePattern[state_pattern.upper()]
        photon_state = StateGenerator.generate_state(
            kernel_modes,
            min(n_photons, kernel_modes),
            pattern,
        )

        trainable_prefixes = [] if reservoir_mode else ["theta"]
        input_prefixes = ["px"]
        q_layer = ML.QuantumLayer(
            input_size=kernel_features,
            circuit=circuit,
            trainable_parameters=trainable_prefixes,
            input_parameters=input_prefixes,
            input_state=photon_state,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
    else:
        trainable_prefixes = [] if reservoir_mode else ["theta"]
        q_layer = ML.QuantumLayer(
            circuit=circuit,
            trainable_parameters=trainable_prefixes,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            computation_space=ML.ComputationSpace.UNBUNCHED,
            amplitude_encoding=True,
        )

    complete = nn.Sequential(
        q_layer, ML.LexGrouping(input_size=q_layer.output_size, output_size=2)
    )
    return complete
