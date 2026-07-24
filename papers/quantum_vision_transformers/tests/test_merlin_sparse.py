from __future__ import annotations

from types import MethodType

import merlin as ML
import perceval as pcvl
import torch
from merlin.core.computation_space import ComputationSpace
from merlin.core.probability_distribution import ProbabilityDistribution
from merlin.core.state_vector import StateVector
from merlin.measurement import MeasurementStrategy
from merlin.utils.combinadics import Combinadics


def _make_sparse_statevector(
    *, n_modes: int, n_photons: int, states: list[tuple[int, ...]]
) -> StateVector:
    combinadics = Combinadics("fock", n_photons, n_modes)
    indices = [combinadics.fock_to_index(state) for state in states]
    values = torch.ones(len(indices), dtype=torch.complex64)
    values = values / values.norm().clamp_min(1e-12)
    tensor = torch.sparse_coo_tensor(
        torch.tensor([indices], dtype=torch.long),
        values,
        (combinadics.compute_space_size(),),
        dtype=torch.complex64,
    ).coalesce()
    return StateVector.from_tensor(tensor, n_modes=n_modes, n_photons=n_photons)


def _make_batched_sparse_statevector(
    *, n_modes: int, n_photons: int, batches: list[list[tuple[int, ...]]]
) -> StateVector:
    combinadics = Combinadics("fock", n_photons, n_modes)
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[complex] = []

    for row, states in enumerate(batches):
        amps = torch.ones(len(states), dtype=torch.complex64)
        amps = amps / amps.norm().clamp_min(1e-12)
        for amp, state in zip(amps.tolist(), states, strict=False):
            row_indices.append(row)
            col_indices.append(combinadics.fock_to_index(state))
            values.append(amp)

    tensor = torch.sparse_coo_tensor(
        torch.tensor([row_indices, col_indices], dtype=torch.long),
        torch.tensor(values, dtype=torch.complex64),
        (len(batches), combinadics.compute_space_size()),
        dtype=torch.complex64,
    ).coalesce()
    return StateVector.from_tensor(tensor, n_modes=n_modes, n_photons=n_photons)


def _make_layer(
    *, n_modes: int, n_photons: int, measurement: str = "amplitudes"
) -> ML.QuantumLayer:
    strategy = (
        MeasurementStrategy.amplitudes(ComputationSpace.FOCK)
        if measurement == "amplitudes"
        else MeasurementStrategy.probs(ComputationSpace.FOCK)
    )
    return ML.QuantumLayer(
        circuit=pcvl.Circuit(n_modes),
        n_photons=n_photons,
        measurement_strategy=strategy,
        return_object=True,
        device="cpu",
        dtype=torch.float32,
    )


def _to_dense(output) -> torch.Tensor:
    if isinstance(output, StateVector):
        return output.to_dense()
    if isinstance(output, ProbabilityDistribution):
        return output.to_dense()
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unsupported output type: {type(output)!r}")


def test_sparse_statevector_matches_dense_for_identity_amplitudes() -> None:
    states = [(2, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0)]
    sparse_sv = _make_sparse_statevector(n_modes=6, n_photons=2, states=states)
    dense_sv = StateVector.from_tensor(
        sparse_sv.to_dense(), n_modes=6, n_photons=2, dtype=torch.complex64
    )
    layer = _make_layer(n_modes=6, n_photons=2, measurement="amplitudes")

    sparse_out = _to_dense(layer(sparse_sv))
    dense_out = _to_dense(layer(dense_sv))

    assert torch.allclose(sparse_out, dense_out, atol=1e-6, rtol=1e-6)


def test_sparse_statevector_matches_dense_for_identity_probabilities() -> None:
    states = [(2, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0)]
    sparse_sv = _make_sparse_statevector(n_modes=6, n_photons=2, states=states)
    dense_sv = StateVector.from_tensor(
        sparse_sv.to_dense(), n_modes=6, n_photons=2, dtype=torch.complex64
    )
    layer = _make_layer(n_modes=6, n_photons=2, measurement="probabilities")

    sparse_out = _to_dense(layer(sparse_sv))
    dense_out = _to_dense(layer(dense_sv))

    assert torch.allclose(sparse_out, dense_out, atol=1e-6, rtol=1e-6)


def test_batched_sparse_statevector_matches_dense_identity_amplitudes() -> None:
    sparse_sv = _make_batched_sparse_statevector(
        n_modes=6,
        n_photons=2,
        batches=[
            [(2, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)],
            [(0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 1, 1)],
        ],
    )
    dense_sv = StateVector.from_tensor(
        sparse_sv.to_dense(), n_modes=6, n_photons=2, dtype=torch.complex64
    )
    layer = _make_layer(n_modes=6, n_photons=2, measurement="amplitudes")

    sparse_out = _to_dense(layer(sparse_sv))
    dense_out = _to_dense(layer(dense_sv))

    assert torch.allclose(sparse_out, dense_out, atol=1e-6, rtol=1e-6)


def test_statevector_forward_uses_single_vectorized_dispatch() -> None:
    """A StateVector forward must hit the amplitude path exactly once.

    merlin 0.4 routes StateVector inputs through compute_superposition_state
    (compute_ebs_simultaneously is reserved for the deprecated
    amplitude_encoding flag and parameter-batched inputs); the guarantee we
    care about is a single vectorized dispatch, not a per-state Python loop.
    """
    sparse_sv = _make_sparse_statevector(
        n_modes=6,
        n_photons=2,
        states=[(2, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0)],
    )
    layer = _make_layer(n_modes=6, n_photons=2, measurement="amplitudes")
    process = layer.computation_process
    tracker = {"ebs": 0, "super": 0}

    original_ebs = process.compute_ebs_simultaneously
    original_super = process.compute_superposition_state

    def tracked_ebs(self, parameters, *args, **kwargs):
        tracker["ebs"] += 1
        return original_ebs(parameters, *args, **kwargs)

    def tracked_super(self, parameters, *args, **kwargs):
        tracker["super"] += 1
        return original_super(parameters, *args, **kwargs)

    process.compute_ebs_simultaneously = MethodType(tracked_ebs, process)
    process.compute_superposition_state = MethodType(tracked_super, process)

    _ = layer(sparse_sv)

    assert tracker["ebs"] + tracker["super"] == 1
