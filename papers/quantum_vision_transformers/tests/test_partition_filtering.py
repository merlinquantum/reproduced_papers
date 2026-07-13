"""Sector selection under merlin 0.4.

merlin 0.4 removed ``partition_blocks``/``allowed_counts`` from
``MeasurementStrategy.probs``: layers now emit the full Fock distribution and
the QVT readout modules perform partition filtering classically. These tests
pin down that post-filtering behaviour.
"""

from __future__ import annotations

import math

import torch

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


def test_compound_transformer_emits_full_basis_and_readout_filters() -> None:
    layer = CompoundTransformerLayer(
        n_patches=4,
        d=4,
        compound_readout="cross_only",
        circuit_family="generic",
        device="cpu",
    )

    full_basis_size = math.comb(4 + 4 + 2 - 1, 2)
    output_keys = list(layer.layer.output_keys)
    assert len(output_keys) == full_basis_size

    # The readout selects exactly the cross-partition (1, 1) sector.
    assert layer.readout.vi.numel() == 4 * 4
    for idx in layer.readout.vi.tolist():
        assert _block_signature(output_keys[idx], (4, 4)) == (1, 1)

    y, sector_mass = layer(torch.rand(2, 4, 4))
    assert y.shape == (2, 4, 4)
    assert sector_mass.shape == (2,)
    assert torch.all(sector_mass >= 0) and torch.all(sector_mass <= 1 + 1e-6)
    # Cross-sector readout is renormalized over the selected sector.
    assert torch.allclose(y.sum(dim=(-2, -1)), torch.ones(2), atol=1e-5)


def test_readout_places_probability_mass_on_matching_entry() -> None:
    layer = CompoundTransformerLayer(
        n_patches=2,
        d=2,
        compound_readout="cross_only",
        circuit_family="generic",
        device="cpu",
    )
    output_keys = list(layer.layer.output_keys)

    # Synthetic distribution: all mass on one cross-partition key.
    probs = torch.zeros(1, len(output_keys))
    idx = layer.readout.vi[0].item()
    pi = layer.readout.pi[0].item()
    fi = layer.readout.fi[0].item()
    probs[0, idx] = 1.0

    Y, sector_mass = layer.readout(probs)
    assert torch.isclose(sector_mass[0], torch.tensor(1.0))
    assert torch.isclose(Y[0, pi, fi], torch.tensor(1.0))
    assert torch.isclose(Y.sum(), torch.tensor(1.0))


def test_hierarchical_layer_readout_selects_requested_sectors() -> None:
    layer = HierarchicalCompoundLayer(
        n_regions=2,
        n_patches_per_region=2,
        d=4,
        use_rpp_attention=True,
        circuit_family="generic",
        device="cpu",
    )

    full_basis_size = math.comb(2 + 2 + 4 + 3 - 1, 3)
    output_keys = list(layer.layer.output_keys)
    assert len(output_keys) == full_basis_size

    # Triple-cross sector: exactly r*p*d entries with signature (1, 1, 1).
    assert layer.readout.tc_idx.numel() == 2 * 2 * 4
    for idx in layer.readout.tc_idx.tolist():
        assert _block_signature(output_keys[idx], (2, 2, 4)) == (1, 1, 1)

    # Region-patch-patch sector: signature (1, 2, 0) only.
    assert layer.readout.rpp_idx.numel() > 0
    for idx in layer.readout.rpp_idx.tolist():
        assert _block_signature(output_keys[idx], (2, 2, 4)) == (1, 2, 0)

    y, info = layer(torch.rand(2, 4, 4))
    assert y.shape == (2, 4, 4)
    assert "sector_masses" in info
