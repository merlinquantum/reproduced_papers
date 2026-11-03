"""Time our exact numpy simulation against Qiskit on the paper's circuits (ideal, no noise).

Two comparisons, each on the same random inputs and checked for equal values:

* quantum neuron (Fig. 2): Qiskit ``StatevectorEstimator`` with one parameterized
  circuit and a batch of parameter values, against ``lib.quantum_neuron``;
* quantum kernel (Fig. 3, 20 features on 10 qubits): Qiskit statevectors computed
  once per point, then all fidelities by one matrix product, against ``lib.kernels``.

Costs are extrapolated linearly to one 60-neuron reservoir (60 neurons x 3 passes x
35,064 hourly windows) and to one Gram matrix on the full training split
(24,539 points). Timings depend on the machine; the output records the CPU.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/benchmark_simulation.py [--out FILE]
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import qiskit
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PAPER_DIR.parents[1]), str(PAPER_DIR.parent)]
from distributed_qrc.lib.kernels import bloch_vectors, kernel_block  # noqa: E402
from distributed_qrc.lib.noise import IDEAL  # noqa: E402
from distributed_qrc.lib.quantum_neuron import (  # noqa: E402
    neuron_expectations,
    neuron_functional,
)

WINDOWS = 35_064  # hourly windows in the dataset
PASSES = 3  # synchronous reservoir passes
RESERVOIR = 60  # largest reservoir size in the paper
TRAIN = 24_539  # training points of the kernel ridge
NEURON_BATCH = 2_000
KERNEL_POINTS = 2_000


def neuron_circuit(x, w) -> QuantumCircuit:
    # Fig. 2: RX encoding, then 2 blocks of RX(w) and a CX-RZ-CX ladder.
    qc = QuantumCircuit(4)
    for q in range(4):
        qc.rx(x[q], q)
    for _ in range(2):
        for q in range(4):
            qc.rx(w[q], q)
        for t in range(1, 4):
            qc.cx(t - 1, t)
            qc.rz(w[t - 1] - w[t], t)
            qc.cx(t - 1, t)
    return qc


def feature_map(x, n: int) -> QuantumCircuit:
    # Fig. 3: 2 blocks of L = ceil(d / n) layers of [H, P(2x)].
    layers = int(np.ceil(len(x) / n))
    pad = np.zeros(layers * n)
    pad[: len(x)] = x
    qc = QuantumCircuit(n)
    for _ in range(2):
        for layer in range(layers):
            for q in range(n):
                qc.h(q)
                qc.p(2 * pad[layer * n + q], q)
    return qc


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=PAPER_DIR / "results/checks/simulation_benchmark.txt",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    lines = []

    # Quantum neuron.
    w = rng.uniform(0, 2 * np.pi, 4)
    x = rng.uniform(-1, 1, (NEURON_BATCH, 4))
    xp, wp = ParameterVector("x", 4), ParameterVector("w", 4)
    # Qiskit orders circuit parameters by name: the w values come before the x values.
    values = np.hstack([np.tile(w, (NEURON_BATCH, 1)), x])

    def job():
        circuit = neuron_circuit(xp, wp)
        return (
            StatevectorEstimator()
            .run([(circuit, SparsePauliOp("ZZZZ"), values)])
            .result()[0]
            .data.evs
        )

    ref, t_qiskit = timed(job)
    functional = neuron_functional(w, IDEAL, "z_all")[None]
    ours, t_ours = timed(lambda: neuron_expectations(functional, x[:, None, :])[:, 0])
    _, t_ours_full = timed(
        lambda: neuron_expectations(functional, rng.uniform(-1, 1, (WINDOWS, 1, 4)))
    )
    per_qiskit, per_ours = t_qiskit / NEURON_BATCH, t_ours_full / WINDOWS
    evaluations = RESERVOIR * PASSES * WINDOWS
    lines += [
        f"quantum neuron, {NEURON_BATCH} windows: max |ours - Qiskit| = {np.abs(ours - ref).max():.1e}",
        f"  per window: Qiskit StatevectorEstimator {per_qiskit * 1e6:.0f} us, ours {per_ours * 1e6:.2f} us"
        f" (x{per_qiskit / per_ours:.0f})",
        f"  one {RESERVOIR}-neuron reservoir ({evaluations:,} evaluations): Qiskit {evaluations * per_qiskit / 60:.0f} min,"
        f" ours {evaluations * per_ours:.0f} s",
    ]

    # Quantum kernel.
    xk = rng.uniform(-1, 1, (KERNEL_POINTS, 20))
    states, t_states = timed(
        lambda: np.array([Statevector(feature_map(p, 10)).data for p in xk])
    )
    gram_ref, t_pairs = timed(lambda: np.abs(states.conj() @ states.T) ** 2)
    bloch, t_bloch = timed(lambda: torch.from_numpy(bloch_vectors(xk, 10)))
    gram, t_gram = timed(lambda: kernel_block(bloch, bloch, 1.0).numpy())
    pairs = KERNEL_POINTS**2
    full_qiskit = TRAIN * t_states / KERNEL_POINTS + TRAIN**2 * t_pairs / pairs
    full_ours = TRAIN * t_bloch / KERNEL_POINTS + TRAIN**2 * t_gram / pairs
    lines += [
        f"quantum kernel, {KERNEL_POINTS} points, 20 features on 10 qubits: max |ours - Qiskit| = {np.abs(gram - gram_ref).max():.1e}",
        f"  per point: Qiskit statevector {t_states / KERNEL_POINTS * 1e3:.2f} ms, ours (Bloch vectors) {t_bloch / KERNEL_POINTS * 1e6:.0f} us",
        f"  per pair: Qiskit (matrix product of states) {t_pairs / pairs * 1e9:.0f} ns, ours {t_gram / pairs * 1e9:.0f} ns",
        f"  full training Gram ({TRAIN:,} points): Qiskit {full_qiskit:.0f} s, ours {full_ours:.0f} s",
    ]

    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    lines.insert(
        0,
        f"CPU {cpu}; Python {platform.python_version()}, qiskit {qiskit.__version__}, numpy {np.__version__}, torch {torch.__version__}; ideal circuits, single run",
    )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
