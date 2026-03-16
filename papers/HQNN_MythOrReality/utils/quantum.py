"""Helpers for constructing parametrised quantum circuits."""

from __future__ import annotations

import perceval as pcvl


def create_quantum_circuit(modes: int, size: int = 400) -> pcvl.Circuit:
    """Create the quantum circuit used by the HQNN architecture."""

    interferometer = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS(theta=pcvl.P(f"bs_1_{i}"))
            // pcvl.PS(pcvl.P(f"ps_1_{i}"))
            // pcvl.BS(theta=pcvl.P(f"bs_2_{i}"))
            // pcvl.PS(pcvl.P(f"ps_2_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    circuit = pcvl.Circuit(modes)
    circuit.add(0, interferometer, merge=True)

    variable_layers = pcvl.Circuit(modes)
    for index in range(size):
        px = pcvl.P(f"px-{index + 1}")
        variable_layers.add(index % modes, pcvl.PS(px))
    circuit.add(0, variable_layers, merge=True)

    readout = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"ps_3_{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"ps_4_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit.add(0, readout, merge=True)
    return circuit
