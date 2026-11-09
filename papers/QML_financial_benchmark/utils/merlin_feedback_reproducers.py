"""Minimal MerLin reproducers cited in ``ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md``.

Each check is independent of this paper's data and code:

1. ``CircuitBuilder.add_angle_encoding(scale=...)`` has no effect on the phases.
2. ``FidelityKernel`` disagrees with the exact fidelity ``|<psi(x)|psi(x')>|^2``
   computed from FOCK amplitudes: it returns all ones when the angle encoding is
   the last circuit element, and it changes when a fixed final mesh is added
   (a fidelity kernel is invariant to any fixed final unitary).
3. ``FidelityKernel`` wall-clock against "FOCK states once, then Gram matrix".
4. ``merlin.__version__`` against the installed ``merlinquantum`` distribution.

Writes the printed report to ``results/merlin_feedback_reproducers.txt``.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/merlin_feedback_reproducers.py
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import time
from pathlib import Path

import merlin as ml
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
FOUR_MODE_STATE = [1, 0, 1, 0]
TEN_MODE_STATE = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]


def make_builder(
    n_modes: int, final_mesh: bool, scale: float = 1.0
) -> ml.CircuitBuilder:
    """Fixed mesh -> angle encoding ``x0`` (-> fixed mesh)."""
    builder = ml.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=False, name="U0", seed=1234)
    builder.add_angle_encoding(modes=list(range(n_modes)), name="x0", scale=scale)
    if final_mesh:
        builder.add_entangling_layer(trainable=False, name="U1", seed=1235)
    return builder


def quantum_layer(
    builder: ml.CircuitBuilder, state: list[int], amplitudes: bool
) -> ml.QuantumLayer:
    strategy = (
        ml.MeasurementStrategy.amplitudes(computation_space=ml.ComputationSpace.FOCK)
        if amplitudes
        else ml.MeasurementStrategy.probs(computation_space=ml.ComputationSpace.FOCK)
    )
    return ml.QuantumLayer(
        input_size=len(state),
        builder=builder,
        input_state=state,
        n_photons=sum(state),
        measurement_strategy=strategy,
        dtype=torch.float64,
    )


def exact_gram(
    builder: ml.CircuitBuilder, state: list[int], points: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        amplitudes = quantum_layer(builder, state, amplitudes=True)(points)
    return (amplitudes.conj() @ amplitudes.T).abs() ** 2


def fidelity_kernel(
    builder: ml.CircuitBuilder, state: list[int], space=None
) -> ml.FidelityKernel:
    feature_map = ml.FeatureMap(
        builder=builder,
        input_size=len(state),
        input_parameters="x0",
        dtype=torch.float64,
    )
    return ml.FidelityKernel(
        feature_map, input_state=state, computation_space=space, dtype=torch.float64
    )


def report() -> None:
    torch.manual_seed(0)
    print(
        f"merlin.__version__ = {ml.__version__}; installed merlinquantum = "
        f"{importlib.metadata.version('merlinquantum')}; torch = {torch.__version__}"
    )

    print("\n[1] add_angle_encoding(scale=pi), 2 photons, 4 modes, FOCK probabilities")
    inputs = torch.rand(3, 4, dtype=torch.float64)
    with torch.no_grad():
        scaled = quantum_layer(
            make_builder(4, True, scale=torch.pi), FOUR_MODE_STATE, False
        )(inputs)
        unscaled = quantum_layer(make_builder(4, True), FOUR_MODE_STATE, False)(inputs)
        manual = quantum_layer(make_builder(4, True), FOUR_MODE_STATE, False)(
            inputs * torch.pi
        )
    print(
        f"  max |P(scale=pi, x) - P(scale=1, x)|    = {float((scaled - unscaled).abs().max()):.3e}"
    )
    print(
        f"  max |P(scale=pi, x) - P(scale=1, pi*x)| = {float((scaled - manual).abs().max()):.3e}"
    )

    print("\n[2] FidelityKernel vs exact FOCK fidelity, 6 random points, 4 modes")
    points = torch.rand(6, 4, dtype=torch.float64)
    off_diagonal = ~torch.eye(6, dtype=torch.bool)
    for final_mesh in (False, True):
        exact = exact_gram(make_builder(4, final_mesh), FOUR_MODE_STATE, points)
        print(
            f"  final mesh={final_mesh}: exact off-diagonal mean {float(exact[off_diagonal].mean()):.4f}"
        )
        for space in (
            None,
            ml.ComputationSpace.FOCK,
            ml.ComputationSpace.UNBUNCHED,
        ):
            gram = fidelity_kernel(make_builder(4, final_mesh), FOUR_MODE_STATE, space)(
                points
            )
            print(
                f"    computation_space={space!s:28} off-diagonal mean {float(gram[off_diagonal].mean()):.4f}, "
                f"min {float(gram[off_diagonal].min()):.4f}, max |K - exact| {float((gram - exact).abs().max()):.4f}"
            )

    print(
        "\n[3] wall-clock, 3 photons, 10 modes, FOCK (mesh -> encoding -> mesh); one Gram matrix"
    )
    builder = make_builder(10, True)
    kernel = fidelity_kernel(
        make_builder(10, True), TEN_MODE_STATE, ml.ComputationSpace.FOCK
    )
    for n_points in (180, 360, 720):
        points = torch.rand(n_points, 10, dtype=torch.float64)
        start = time.perf_counter()
        kernel(points)
        kernel_seconds = time.perf_counter() - start
        start = time.perf_counter()
        exact_gram(builder, TEN_MODE_STATE, points)
        state_seconds = time.perf_counter() - start
        print(
            f"  N={n_points:4d}: FidelityKernel {kernel_seconds:7.2f} s; "
            f"FOCK states + Gram {state_seconds:6.3f} s; ratio {kernel_seconds / state_seconds:5.0f}x"
        )


def main() -> int:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        report()
    text = buffer.getvalue()
    print(text, end="")
    (PROJECT_DIR / "results" / "merlin_feedback_reproducers.txt").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
