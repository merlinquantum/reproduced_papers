from __future__ import annotations

import json
import os
from pathlib import Path

import merlin as ml
import numpy as np
import pytest
import torch
from common import PROJECT_DIR
from lib.models import (
    build_torch_model,
    count_trainable_parameters,
    sample_reference_inputs,
)
from lib.qk_lstm_merlin import PhotonicFidelityKernel

INPUT_SIZE = 11


def _photonic_cfg(gate_mode="shared", input_state=(1, 0, 1, 0)):
    return {
        "type": "qk_lstm_merlin",
        "params": {
            "gate_mode": gate_mode,
            "hidden_size": 16,
            "n_qubits": 4,
            "n_references": 4,
            "photonic": {
                "input_state": list(input_state),
                "mesh_seeds": [7, 8],
                "encoding_scale": 1.0,
            },
        },
    }


def _references():
    return sample_reference_inputs(
        np.random.default_rng(0).random((50, INPUT_SIZE)), 4, seed=0
    )


def test_kernel_shapes_and_fidelity_properties():
    kernel = PhotonicFidelityKernel(4, [1, 0, 1, 0], [7, 8])
    z = torch.randn(5, 4)
    gram = kernel(z, z)
    assert gram.shape == (5, 5)
    assert torch.allclose(torch.diagonal(gram), torch.ones(5), atol=1e-5)
    assert torch.allclose(gram, gram.T, atol=1e-5)
    assert gram.min() >= -1e-6 and gram.max() <= 1 + 1e-5
    assert kernel(torch.randn(3, 4), torch.randn(2, 4)).shape == (3, 2)


def test_configured_space_photons_and_no_trainable_circuit_parameters():
    kernel = PhotonicFidelityKernel(4, [1, 0, 1, 0], [7, 8])
    info = kernel.describe()
    assert kernel.computation_space == ml.ComputationSpace.FOCK
    assert info["computation_space"] == "FOCK"
    assert info["detector_model"].startswith("threshold")
    assert info["postselection"].startswith("none")
    assert (
        info["n_photons"] == 2
        and info["n_modes"] == 4
        and info["input_state"] == [1, 0, 1, 0]
    )
    assert info["trainable_circuit_parameters"] == 0


def test_mesh_is_fixed_by_its_seeds_not_the_global_rng():
    z, z_ref = torch.randn(2, 4), torch.randn(3, 4)
    torch.manual_seed(0)
    first = PhotonicFidelityKernel(4, [1, 0, 1, 0], [7, 8])(z, z_ref)
    torch.manual_seed(123)
    second = PhotonicFidelityKernel(4, [1, 0, 1, 0], [7, 8])(z, z_ref)
    other = PhotonicFidelityKernel(4, [1, 0, 1, 0], [9, 10])(z, z_ref)
    assert torch.allclose(first, second)
    assert not torch.allclose(first, other)


def test_unbunched_space_is_not_a_fidelity_kernel():
    """Documents the merlin 0.4.1 behaviour behind the FOCK choice (LOG.md amendment)."""
    torch.manual_seed(0)
    z = torch.randn(4, 4)
    unbunched = PhotonicFidelityKernel(
        4, [1, 0, 1, 0], [7, 8], computation_space="unbunched"
    )(z, z)
    fock = PhotonicFidelityKernel(4, [1, 0, 1, 0], [7, 8])(z, z)
    assert (unbunched - unbunched.T).abs().max() > 1e-3
    assert (fock - fock.T).abs().max() < 1e-5
    # UNBUNCHED renormalises by the collision-free mass, so it is never smaller.
    assert torch.all(unbunched >= fock - 1e-6)


def test_unbunched_gram_path_is_symmetric_but_renormalised():
    """Single-argument (Gram) path: symmetric, yet not the FOCK fidelity (LOG.md, MerLin API Notes)."""
    torch.manual_seed(0)
    z = torch.randn(4, 4)
    unbunched = PhotonicFidelityKernel(
        4, [1, 0, 1, 0], [7, 8], computation_space="unbunched"
    ).kernel(z)
    fock = PhotonicFidelityKernel(4, [1, 0, 1, 0], [7, 8]).kernel(z)
    assert (unbunched - unbunched.T).abs().max() < 1e-5
    assert (unbunched - fock).abs().max() > 1e-2


def test_invalid_input_state_is_rejected():
    with pytest.raises(ValueError):
        PhotonicFidelityKernel(4, [1, 0, 1], [7, 8])
    with pytest.raises(ValueError):
        PhotonicFidelityKernel(4, [0, 0, 0, 0], [7, 8])
    with pytest.raises(ValueError):
        PhotonicFidelityKernel(4, [2, 0, 0, 0], [7, 8])


@pytest.mark.parametrize(
    ("gate_mode", "expected"), [("shared", 209), ("per_gate", 449)]
)
def test_parameter_counts_match_qubit_variants(gate_mode, expected):
    model = build_torch_model(_photonic_cfg(gate_mode), INPUT_SIZE, _references())
    assert count_trainable_parameters(model) == expected


def test_config_values_reach_the_model():
    model = build_torch_model(
        _photonic_cfg(input_state=(1, 0, 0, 0)), INPUT_SIZE, _references()
    )
    kernel = model.cell.photonic_kernel
    assert model.cell.kernel == "photonic"
    assert (
        kernel.input_state == [1, 0, 0, 0]
        and kernel.n_photons == 1
        and kernel.mesh_seeds == [7, 8]
    )


def test_seeded_initialisation_matches_qubit_model():
    # Photonic runs are paired by seed with the Phase 3 qubit runs.
    qubit_cfg = _photonic_cfg()
    qubit_cfg = {
        "type": "qk_lstm",
        "params": {k: v for k, v in qubit_cfg["params"].items() if k != "photonic"},
    }
    torch.manual_seed(5)
    qubit = build_torch_model(qubit_cfg, INPUT_SIZE, _references())
    torch.manual_seed(5)
    photonic = build_torch_model(_photonic_cfg(), INPUT_SIZE, _references())
    qubit_state, photonic_state = qubit.state_dict(), photonic.state_dict()
    shared_keys = [k for k in qubit_state if k in photonic_state]
    assert set(shared_keys) == set(qubit_state)
    for key in shared_keys:
        assert torch.equal(qubit_state[key], photonic_state[key]), key


def test_forward_backward_and_gradient_reaches_compression():
    torch.manual_seed(0)
    model = build_torch_model(_photonic_cfg("per_gate"), INPUT_SIZE, _references())
    out = model(torch.rand(3, 3, INPUT_SIZE))
    assert out.shape == (3, 1)
    out.sum().backward()
    assert model.cell.compression.weight.grad.abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_reduced_merlin_run_through_shared_runtime(tmp_path):
    from runtime_lib import run_from_project

    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                "configs/aqi_qklstm_count209_merlin.json",
                "--epochs",
                "1",
                "--max-train-windows",
                "8",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["trainable_parameters"] == 209
    assert (
        metrics["photonic"]["n_photons"] == 2
        and metrics["photonic"]["computation_space"] == "FOCK"
    )
    assert "PHOTONIC_CONFIG" in (run_dir / "run.log").read_text()
