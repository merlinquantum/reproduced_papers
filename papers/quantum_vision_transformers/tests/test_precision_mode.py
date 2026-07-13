from __future__ import annotations

from contextlib import contextmanager

import torch
from lib.config import validate_run_config
from lib.models import CompoundTransformerLayer, ModelA
from lib.photonic_primitives import complex_dtype_for

from implementation import resolve_runtime_dtype


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_complex_dtype_for_matches_real_precision() -> None:
    assert complex_dtype_for(torch.float32) is torch.complex64
    assert complex_dtype_for(torch.bfloat16) is torch.complex64
    assert complex_dtype_for(torch.float64) is torch.complex128


def test_validate_run_config_accepts_gpu_friendly_precision_mode() -> None:
    cfg = validate_run_config({"model_type": "A", "precision_mode": "gpu_friendly"})
    assert cfg["precision_mode"] == "gpu_friendly"


def test_resolve_runtime_dtype_prefers_gpu_friendly_mode() -> None:
    cfg = {"dtype": "float64", "precision_mode": "gpu_friendly"}
    assert resolve_runtime_dtype(cfg, None) is torch.float32
    assert resolve_runtime_dtype(cfg, "float64") is torch.float64


def test_model_a_float32_matches_float64_reasonably() -> None:
    torch.manual_seed(0)
    with _default_dtype(torch.float64):
        model64 = ModelA(4, circuit_family="generic", device="cpu")
    torch.manual_seed(0)
    with _default_dtype(torch.float32):
        model32 = ModelA(4, circuit_family="generic", device="cpu")
    model32.load_state_dict(model64.state_dict())

    x64 = torch.rand(2, 3, 4, dtype=torch.float64)
    with torch.no_grad():
        y64 = model64(x64)
        y32 = model32(x64.float()).to(torch.float64)

    assert y64.shape == y32.shape
    assert torch.allclose(y64, y32, atol=1e-3, rtol=2e-3)


def test_compound_layer_float32_matches_float64_reasonably() -> None:
    torch.manual_seed(0)
    with _default_dtype(torch.float64):
        layer64 = CompoundTransformerLayer(
            n_patches=2,
            d=4,
            compound_readout="cross_only",
            circuit_family="generic",
            device="cpu",
        )
    torch.manual_seed(0)
    with _default_dtype(torch.float32):
        layer32 = CompoundTransformerLayer(
            n_patches=2,
            d=4,
            compound_readout="cross_only",
            circuit_family="generic",
            device="cpu",
        )
    layer32.load_state_dict(layer64.state_dict())

    x64 = torch.rand(2, 2, 4, dtype=torch.float64)
    with torch.no_grad():
        y64, sm64 = layer64(x64)
        y32, sm32 = layer32(x64.float())

    assert y64.shape == y32.shape
    assert torch.allclose(y64, y32.to(torch.float64), atol=2e-3, rtol=3e-3)
    assert torch.allclose(sm64, sm32.to(torch.float64), atol=2e-3, rtol=3e-3)
