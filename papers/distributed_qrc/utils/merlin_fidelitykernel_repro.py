"""Minimal reproducer of the `merlin.FidelityKernel` deviation reported to the MerLin team.

On a frozen 5-mode, 2-photon re-uploading circuit (the kernel chip of the MerLin adaptation,
``lib.photonic_merlin.PhotonicKernelMap``), compare four Gram matrices on 6 points:

* ``exact``: |<psi(x')|psi(x)>|^2 from the builder-built ``QuantumLayer`` FOCK
  amplitudes (verified against Perceval SLOS in ``tests/test_photonic.py``);
* a ``QuantumLayer`` built from the same Perceval circuit (``circuit=``), raw
  phases ``scale * x``;
* ``merlin.FidelityKernel(FeatureMap(circuit=..., input_parameters="px"))`` on the
  same data, for ``computation_space`` None / "fock" / "unbunched";
* Perceval unitaries with |<s|U(x')^dagger U(x)|s>|^2 (and three wrong-adjoint
  variants), to test simple explanations of the discrepancy.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/merlin_fidelitykernel_repro.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import perceval as pcvl
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[3]),
    str(Path(__file__).resolve().parents[2]),
]
import merlin as ml  # noqa: E402
from distributed_qrc.lib.photonic_merlin import (  # noqa: E402
    PhotonicKernelMap,
    state_gram,
)

warnings.filterwarnings("ignore")
SCALE = 2.0


def main() -> int:
    km = PhotonicKernelMap(5, 5, 2, SCALE, 2, np.random.SeedSequence(3))
    x = np.random.default_rng(0).uniform(-1, 1, (6, 5))
    exact = state_gram(km.states(x)).numpy()
    inputs = torch.from_numpy(km.encode(x))
    circuit = km.layer.circuit
    print(
        f"merlin {ml.__version__}; modes 5, photons 2, input state {km.input_state}, {inputs.shape[1]} phase parameters"
    )

    layer = ml.QuantumLayer(
        input_size=inputs.shape[1],
        circuit=circuit,
        input_state=km.input_state,
        input_parameters=["px"],
        measurement_strategy=ml.MeasurementStrategy.amplitudes(
            computation_space=ml.ComputationSpace.FOCK
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        circuit_gram = state_gram(layer(SCALE * inputs).to(torch.complex128)).numpy()
    print(
        f"QuantumLayer(circuit=...) amplitudes, phases scale*x : max|K - exact| = {np.abs(circuit_gram - exact).max():.3e}"
    )

    for space in (None, "fock", "unbunched"):
        feature_map = ml.FeatureMap(
            circuit=circuit,
            input_size=inputs.shape[1],
            input_parameters="px",
            dtype=torch.float64,
        )
        kernel = ml.FidelityKernel(
            feature_map,
            input_state=km.input_state,
            computation_space=space,
            force_psd=False,
            dtype=torch.float64,
        )
        with torch.no_grad():
            gram = kernel(SCALE * inputs).detach().numpy()
        off = ~np.eye(6, dtype=bool)
        print(
            f"FidelityKernel computation_space={space!s:9s}: max|K - exact| = {np.abs(gram - exact).max():.3e}, "
            f"mean off-diagonal K/exact = {np.mean(gram[off] / exact[off]):.3f}"
        )

    params = sorted(
        (p for p in circuit.get_parameters(all_params=True) if p.name.startswith("px")),
        key=lambda p: int(p.name[2:]),
    )

    def unitary(values: np.ndarray) -> np.ndarray:
        for p, value in zip(params, values):
            p.set_value(float(value))
        return np.array(circuit.compute_unitary(use_symbolic=False))

    def amplitude(u: np.ndarray) -> complex:
        backend = pcvl.BackendFactory.get_backend("SLOS")
        backend.set_circuit(pcvl.Unitary(pcvl.Matrix(u)))
        backend.set_input_state(pcvl.BasicState(km.input_state))
        return backend.prob_amplitude(pcvl.BasicState(km.input_state))

    unitaries = [unitary(SCALE * row) for row in inputs.numpy()]
    feature_map = ml.FeatureMap(
        circuit=circuit,
        input_size=inputs.shape[1],
        input_parameters="px",
        dtype=torch.float64,
    )
    kernel = ml.FidelityKernel(
        feature_map,
        input_state=km.input_state,
        computation_space="fock",
        force_psd=False,
        dtype=torch.float64,
    )
    with torch.no_grad():
        gram = kernel(SCALE * inputs).detach().numpy()
    variants = {
        "U(x')^dagger U(x)": lambda a, b: b.conj().T @ a,
        "U(x')^T U(x)": lambda a, b: b.T @ a,
        "conj(U(x')) U(x)": lambda a, b: b.conj() @ a,
        "U(x') U(x)": lambda a, b: b @ a,
    }
    for name, compose in variants.items():
        values = np.array(
            [
                [
                    abs(amplitude(compose(unitaries[i], unitaries[j]))) ** 2
                    for j in range(6)
                ]
                for i in range(6)
            ]
        )
        print(
            f"Perceval |<s|{name}|s>|^2: max|FidelityKernel - V| = {np.abs(gram - values).max():.3e}, max|exact - V| = {np.abs(exact - values).max():.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
