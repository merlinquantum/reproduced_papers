from __future__ import annotations

import math

import perceval as pcvl
import torch

import merlin as ML
from merlin.core.computation_space import ComputationSpace
from merlin.measurement import MeasurementStrategy

from lib.models import (
    CompoundTransformerLayer,
    HierarchicalCompoundLayer,
)


def _block_signature(state: tuple[int, ...], blocks: tuple[int, ...]) -> tuple[int, ...]:
    signature = []
    start = 0
    for size in blocks:
        end = start + size
        signature.append(sum(state[start:end]))
        start = end
    return tuple(signature)


def test_partition_filtering_prunes_output_keys_for_probabilities() -> None:
    layer = ML.QuantumLayer(
        circuit=pcvl.Circuit(4),
        input_state=[1, 0, 0, 1],
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(
            ComputationSpace.FOCK,
            partition_blocks=[2, 2],
            allowed_counts=[(1, 1)],
        ),
        device="cpu",
        dtype=torch.float32,
    )

    assert len(layer.output_keys) == 4
    assert all(_block_signature(key, (2, 2)) == (1, 1) for key in layer.output_keys)

    probs = layer()
    assert probs.shape[-1] == 4
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))


def test_compound_transformer_cross_only_uses_reduced_basis() -> None:
    layer = CompoundTransformerLayer(
        n_patches=4,
        d=4,
        compound_readout="cross_only",
        circuit_family="generic",
        device="cpu",
    )

    assert len(layer.layer.output_keys) == 16
    assert all(
        _block_signature(key, (4, 4)) == (1, 1) for key in layer.layer.output_keys
    )

    full_basis_size = math.comb(4 + 4 + 2 - 1, 2)
    assert len(layer.layer.output_keys) < full_basis_size

    y, sector_mass = layer(torch.rand(2, 4, 4))
    assert y.shape == (2, 4, 4)
    assert sector_mass.shape == (2,)


def test_hierarchical_layer_prunes_to_requested_partition_sectors() -> None:
    layer = HierarchicalCompoundLayer(
        n_regions=2,
        n_patches_per_region=2,
        d=4,
        use_rpp_attention=True,
        circuit_family="generic",
        device="cpu",
    )

    allowed = {(1, 1, 1), (1, 2, 0)}
    assert layer.layer.output_keys
    assert all(
        _block_signature(key, (2, 2, 4)) in allowed for key in layer.layer.output_keys
    )
