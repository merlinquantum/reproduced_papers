"""Test two candidate causes of the B1 mismatch with Fig. 3e.

Usage::

    python utils/diagnose_b1_models.py --out results/b1_models.json

1. **Imperfect source.** The transparent peer-review file states that the
   platform-B "theoretical" model includes partial distinguishability (pairwise
   HOM visibility ~83 %, SI Note 1). This script uses the orthogonal-bad-bits
   model: photon i has internal state sqrt(x)|0> + sqrt(1 - x)|i>, so the Gram
   matrix off-diagonal is x and the HOM visibility is x^2. The residual qubit is
   then mixed; the kernel is the Uhlmann fidelity (SI Note 3). Multiphoton noise
   (g2 ~ 2 %) is not modelled.
2. **Circuit convention.** Every combination of beam-splitter matrix, phase arm,
   phase sign, meaning of theta, phase-shifter position, an extra pi/4 offset on
   the adaptive angles, and transposed unitary. Each is scored on the platform-A
   printed entries and on Pearson r of the ideal B1 kernel against Fig. 3e
   (identity outcome order).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
# The paper folder provides lib/; the repository root provides runtime_lib/.
sys.path[:0] = [str(PROJECT_DIR), str(PROJECT_DIR.parents[1])]

import lib.circuits as circuits  # noqa: E402
from lib.fock import conditional_state, transition_amplitude  # noqa: E402
from lib.kernels import uhlmann_fidelity  # noqa: E402

from runtime_lib.data_paths import paper_data_dir  # noqa: E402

UPPER = np.triu_indices(15, 1)
PLATFORM_A_PRINTED = np.array([0.298, 0.966, 0.331])


def compare(kernel: np.ndarray, target: np.ndarray) -> dict[str, float]:
    a, b = kernel[UPPER], target[UPPER]
    return {
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "mean_abs_error": float(np.abs(a - b).mean()),
        "kernel_min": float(a.min()),
        "kernel_mean": float(a.mean()),
    }


def partially_distinguishable_rho(unitary, platform, outcome, overlap):
    """Return the residual qubit density matrix for Gram off-diagonal ``overlap``.

    Spatial mode m with internal label l is extended mode ``m * L + l``; label 0
    is the shared internal state and label i + 1 is photon i's private one.
    """
    n_photons = len(platform.input_modes)
    n_labels = n_photons + 1
    extended = np.kron(unitary, np.eye(n_labels))
    detected = [
        mode
        for mode, count in zip(platform.detected_modes, outcome)
        for _ in range(count)
    ]
    rho = np.zeros((2, 2), dtype=np.complex128)
    for shared in itertools.product([True, False], repeat=n_photons):
        weight = math.prod(overlap if s else 1 - overlap for s in shared)
        if weight == 0:
            continue
        labels = [0 if s else index + 1 for index, s in enumerate(shared)]
        input_occupation = [0] * (unitary.shape[0] * n_labels)
        for mode, label in zip(platform.input_modes, labels):
            input_occupation[mode * n_labels + label] += 1
        for detected_labels in itertools.product(range(n_labels), repeat=2):
            if detected[0] == detected[1] and detected_labels[0] > detected_labels[1]:
                continue
            for residual_label in range(n_labels):
                amplitudes = []
                for residual_mode in platform.residual_modes:
                    output = [0] * len(input_occupation)
                    for mode, label in zip(detected, detected_labels):
                        output[mode * n_labels + label] += 1
                    output[residual_mode * n_labels + residual_label] += 1
                    amplitudes.append(
                        transition_amplitude(extended, input_occupation, output)
                    )
                amplitudes = np.array(amplitudes)
                rho += weight * np.outer(amplitudes, amplitudes.conj())
    return rho / np.trace(rho).real


def source_model_scan(target, experimental, overlaps):
    platform = circuits.get_platform("B1")
    rows = []
    for overlap in overlaps:
        rhos = [
            partially_distinguishable_rho(
                circuits.programmed_unitary(platform, outcome, None),
                platform,
                outcome,
                overlap,
            )
            for outcome in platform.outcomes
        ]
        kernel = np.array([[uhlmann_fidelity(a, b) for b in rhos] for a in rhos])
        rows.append(
            {
                "gram_overlap": overlap,
                "hom_visibility": overlap**2,
                "vs_theoretical": compare(kernel, target),
                "vs_experimental": compare(kernel, experimental),
            }
        )
    return rows


def convention_unitary_builder(splitter, arm, sign, theta_map, phase_position):
    half = 1 / math.sqrt(2)
    splitters = {
        "symmetric": np.array([[1, 1j], [1j, 1]]) * half,
        "rotation": np.array([[1, -1], [1, 1]]) * half,
        "hadamard": np.array([[1, 1], [1, -1]]) * half,
    }

    def cell(theta):
        theta = theta_map(theta)
        if splitter == "single_bs":
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, 1j * s], [1j * s, c]])
        phase = np.exp(sign * 1j * theta)
        inner = np.diag([phase, 1]) if arm == "top" else np.diag([1, phase])
        return splitters[splitter] @ inner @ splitters[splitter]

    def build(n_modes, operations):
        unitary = np.eye(n_modes, dtype=np.complex128)
        for kind, mode, value in operations:
            step = np.eye(n_modes, dtype=np.complex128)
            if kind == "ps":
                position = (
                    mode if phase_position == "top" else min(mode + 1, n_modes - 1)
                )
                step[position, position] = np.exp(sign * 1j * value)
            else:
                step[np.ix_([mode, mode + 1], [mode, mode + 1])] = cell(value)
            unitary = step @ unitary
        return unitary

    return build


def convention_kernel(name, build, angle_offset, transpose):
    platform = circuits.get_platform(name)
    original = circuits.cascaded_angles
    circuits.cascaded_angles = lambda outcome, n: [
        angle_offset + value for value in original(outcome, n)
    ]
    try:
        states = []
        for outcome in platform.outcomes:
            unitary = build(platform.n_modes, platform.operations_for(outcome))
            if transpose:
                unitary = unitary.T
            states.append(
                conditional_state(
                    unitary,
                    platform.input_modes,
                    platform.detected_modes,
                    outcome,
                    platform.residual_modes,
                )[0]
            )
    finally:
        circuits.cascaded_angles = original
    states = np.array(states)
    return np.abs(states.conj() @ states.T) ** 2


def convention_scan(target):
    theta_maps = {
        "theta": lambda t: t,
        "2theta": lambda t: 2 * t,
        "theta/2": lambda t: t / 2,
        "pi/2-theta": lambda t: math.pi / 2 - t,
    }
    rows = []
    for splitter, arm, sign, theta_name, phase_position, offset, transpose in (
        itertools.product(
            ["symmetric", "rotation", "hadamard", "single_bs"], ["top", "bottom"],
            [1, -1], theta_maps, ["top", "bottom"], [0.0, math.pi / 4],
            [False, True],
        )
    ):  # fmt: skip
        build = convention_unitary_builder(
            splitter, arm, sign, theta_maps[theta_name], phase_position
        )
        try:
            kernel_a = convention_kernel("A", build, offset, transpose)
            kernel_b1 = convention_kernel("B1", build, offset, transpose)
        except ValueError:  # a zero-probability post-selection under this convention
            continue
        entries = np.sort([kernel_a[0, 1], kernel_a[0, 2], kernel_a[1, 2]])
        rows.append(
            {
                "splitter": splitter, "phase_arm": arm, "phase_sign": sign,
                "theta_meaning": theta_name, "phase_shifter_position": phase_position,
                "adaptive_offset": offset, "transposed": transpose,
                "platform_a_max_error_sorted": float(
                    np.abs(entries - np.sort(PLATFORM_A_PRINTED)).max()
                ),
                "b1_pearson_r_identity_order": compare(kernel_b1, target)["pearson_r"],
            }
        )  # fmt: skip
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digitized = np.load(
        paper_data_dir("adaptive_boson_sampling") / "digitized" / "figure_kernels.npz"
    )
    target = digitized["fig3e_B1_theoretical"]
    experimental = digitized["fig3e_B1_experimental"]

    source = source_model_scan(target, experimental, [1.0, 0.95, 0.911, 0.83, 0.7])
    conventions = convention_scan(target)
    matching_a = [
        row for row in conventions if row["platform_a_max_error_sorted"] < 0.01
    ]
    report = {
        "validity": "simulation of the published B1 circuit vs digitised Fig. 3e",
        "digitised_theoretical": {
            "min": float(target[UPPER].min()), "mean": float(target[UPPER].mean())
        },
        "source_model": source,
        "conventions_tested": len(conventions),
        "conventions_matching_platform_a": matching_a,
        "best_b1_r_any_convention": max(
            conventions, key=lambda row: row["b1_pearson_r_identity_order"]
        ),
    }  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "source_model"}, indent=1)
    )
    for row in source:
        print(row["gram_overlap"], row["vs_theoretical"])


if __name__ == "__main__":
    main()
