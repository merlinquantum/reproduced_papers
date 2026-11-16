"""Measure MerLin's distinguishability noise on the photonic swap test (why Fig. 3 is not translated).

Two identical registers must give P(even) = 1 without noise and close to 1
for nearly indistinguishable photons. The check compares a superposed register
state with a Fock basis state, and with the value an incoherent mixture of the
basis states would give: 1/2 + 1/2 * sum_i |a_i|^4.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/merlin_noise_check.py
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parent), str(PROJECT_DIR.parents[1])]

import merlin as ml  # noqa: E402
import numpy as np  # noqa: E402
import perceval as pcvl  # noqa: E402
import torch  # noqa: E402

from distributed_secure_kernels.lib.photonic_merlin import (  # noqa: E402
    PhotonicSwapTest,
)

MODES = 3
INDISTINGUISHABILITY = (1.0, 0.9999, 0.999, 0.99, 0.95)
STATES = {
    "superposed (0.6, 0, 0.8)": np.array([[0.6, 0.0, 0.8]]),
    "basis (1, 0, 0)": np.array([[1.0, 0.0, 0.0]]),
}


def _swap_test_circuit(modes: int) -> pcvl.Circuit:
    circuit = pcvl.Circuit(2 * modes)
    circuit.add(
        0, pcvl.PERM([2 * i for i in range(modes)] + [2 * i + 1 for i in range(modes)])
    )
    for i in range(modes):
        circuit.add(2 * i, pcvl.BS())
    return circuit


def main() -> None:
    swap_test = PhotonicSwapTest(MODES)
    print(
        # merlin.__version__ reports "0.4.1" in the 0.4.2 distribution, so read the installed version.
        f"MerLin {version('merlinquantum')}, Perceval {pcvl.__version__}, {MODES} modes per register, 2 photons each"
    )
    print("indistinguishability | " + " | ".join(STATES))
    for value in INDISTINGUISHABILITY:
        layer = ml.QuantumLayer(
            circuit=_swap_test_circuit(MODES),
            n_photons=4,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
            noise=pcvl.NoiseModel(indistinguishability=value),
            dtype=torch.float64,
        )
        even = torch.tensor(
            [
                sum(key[j] for j in range(1, 2 * MODES, 2)) % 2 == 0
                for key in layer.output_keys
            ],
            dtype=torch.float64,
        )
        row = []
        for state in STATES.values():
            with torch.no_grad():
                probs = layer(swap_test._joint_states(state, state))
            row.append(f"{float((probs * even).sum()):.4f}")
        print(f"{value} | " + " | ".join(row))
    amplitudes = STATES["superposed (0.6, 0, 0.8)"][0]
    print(
        f"incoherent-mixture value for the superposed state: {0.5 + 0.5 * np.sum(amplitudes**4):.4f}"
    )


if __name__ == "__main__":
    main()
