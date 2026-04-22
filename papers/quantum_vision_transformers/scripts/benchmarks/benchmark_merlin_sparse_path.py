#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from typing import Iterable

import perceval as pcvl
import torch

import merlin as ML
from merlin.core.computation_space import ComputationSpace
from merlin.core.probability_distribution import ProbabilityDistribution
from merlin.core.state_vector import StateVector
from merlin.measurement import MeasurementStrategy
from merlin.utils.combinadics import Combinadics


def _sample_fock_states(
    *, n_modes: int, n_photons: int, n_terms: int, seed: int
) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    states: list[tuple[int, ...]] = []

    while len(states) < n_terms:
        counts = [0] * n_modes
        for _ in range(n_photons):
            counts[rng.randrange(n_modes)] += 1
        state = tuple(counts)
        if state in seen:
            continue
        seen.add(state)
        states.append(state)
    return states


def _make_sparse_statevector(
    *,
    n_modes: int,
    n_photons: int,
    n_terms: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> StateVector:
    combinadics = Combinadics("fock", n_photons, n_modes)
    basis_size = combinadics.compute_space_size()
    states = _sample_fock_states(
        n_modes=n_modes, n_photons=n_photons, n_terms=n_terms, seed=seed
    )

    indices_list: list[int] = []
    values_list: list[complex] = []
    for idx, state in enumerate(states):
        basis_idx = combinadics.fock_to_index(state)
        phase = math.pi * (idx + 1) / max(1, n_terms)
        amp = complex(math.cos(phase), math.sin(phase))
        indices_list.append(basis_idx)
        values_list.append(amp)

    values = torch.tensor(values_list, dtype=dtype, device=device)
    values = values / values.norm().clamp_min(1e-12)
    indices = torch.tensor([indices_list], dtype=torch.long, device=device)
    tensor = torch.sparse_coo_tensor(
        indices, values, (basis_size,), dtype=dtype, device=device
    ).coalesce()
    return StateVector.from_tensor(
        tensor, n_modes=n_modes, n_photons=n_photons, dtype=dtype, device=device
    )


def _to_dense_tensor(output) -> torch.Tensor:
    if isinstance(output, StateVector):
        return output.to_dense()
    if isinstance(output, ProbabilityDistribution):
        return output.to_dense()
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unsupported output type: {type(output)!r}")


def _memory_bytes(obj) -> int | None:
    if hasattr(obj, "memory_bytes"):
        return int(obj.memory_bytes())
    if isinstance(obj, torch.Tensor):
        if obj.is_sparse:
            obj = obj.coalesce()
            return int(
                obj.indices().numel() * obj.indices().element_size()
                + obj.values().numel() * obj.values().element_size()
            )
        return int(obj.numel() * obj.element_size())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MerLin sparse StateVector amplitude path."
    )
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--photons", type=int, default=3)
    parser.add_argument("--terms", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument(
        "--measurement", choices=("amplitudes", "probabilities"), default="amplitudes"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    n_modes = args.modes
    n_photons = args.photons

    sv = _make_sparse_statevector(
        n_modes=n_modes,
        n_photons=n_photons,
        n_terms=args.terms,
        seed=args.seed,
        dtype=dtype,
        device=device,
    )

    measurement_strategy = (
        MeasurementStrategy.amplitudes(ComputationSpace.FOCK)
        if args.measurement == "amplitudes"
        else MeasurementStrategy.probs(ComputationSpace.FOCK)
    )

    layer = ML.QuantumLayer(
        circuit=pcvl.Circuit(n_modes),
        n_photons=n_photons,
        measurement_strategy=measurement_strategy,
        return_object=True,
        device=device,
        dtype=torch.float32 if dtype == torch.complex64 else torch.float64,
    )

    start = time.perf_counter()
    output = layer(sv)
    elapsed_s = time.perf_counter() - start
    dense_output = _to_dense_tensor(output)
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    payload = {
        "modes": n_modes,
        "photons": n_photons,
        "terms": args.terms,
        "measurement": args.measurement,
        "dtype": args.dtype,
        "input_basis_size": sv.basis_size,
        "input_nnz": int(sv.tensor.coalesce()._nnz()) if sv.is_sparse else int(torch.count_nonzero(sv.tensor).item()),
        "input_memory_bytes": _memory_bytes(sv),
        "output_type": type(output).__name__,
        "output_shape": list(dense_output.shape),
        "output_memory_bytes": _memory_bytes(output),
        "output_abs_sum": float(dense_output.abs().sum().item()),
        "output_l2_norm": float(dense_output.norm().item()),
        "elapsed_seconds": elapsed_s,
        "peak_rss_kb": int(peak_rss_kb),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
