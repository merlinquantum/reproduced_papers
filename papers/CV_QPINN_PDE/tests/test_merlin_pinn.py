from __future__ import annotations

import math
import sys

import pytest
import torch
from common import PROJECT_DIR

# The paper package lives under papers/, which pytest does not reliably put on
# sys.path at collection time (test_cli.py sidesteps this with in-function
# imports); insert it explicitly so module-level imports work.
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

pytest.importorskip("merlin", reason="merlinquantum not installed")

from CV_QPINN_PDE.lib.merlin_pinn import MerLinPINN  # noqa: E402


def small_model(seed: int = 42) -> MerLinPINN:
    """4 modes / 2 photons keeps the UNBUNCHED subspace tiny for fast tests."""
    return MerLinPINN(n_modes=4, input_modes=(0, 1), entangling_layers=1,
                      n_photons=2, scale=math.pi / 2, seed=seed)


def test_output_size_matches_unbunched_subspace():
    model = small_model()
    # UNBUNCHED with n photons in m modes has C(m, n) outcomes.
    assert model.qlayer.output_size == math.comb(4, 2)


def test_forward_shapes_dtype_and_trace():
    model = small_model()
    x = torch.linspace(0.0, math.pi / 2, 5, dtype=torch.float64)
    u, ux, trace = model(x)
    assert u.shape == (5,) and ux.shape == (5,)
    assert u.dtype == torch.float64 and ux.dtype == torch.float64
    assert torch.isfinite(u).all() and torch.isfinite(ux).all()
    # Probabilities are intrinsically normalised: trace is constant 1.
    assert torch.allclose(trace, torch.ones_like(trace))


def test_gradients_reach_quantum_parameters():
    model = small_model()
    x = torch.linspace(0.0, math.pi / 2, 4, dtype=torch.float64)
    u, ux, _ = model(x)
    loss = (u**2).sum() + (ux**2).sum()
    loss.backward()
    qlayer_grads = [p.grad for p in model.qlayer.parameters() if p.requires_grad]
    assert qlayer_grads, "quantum layer exposes no trainable parameters"
    assert all(g is not None for g in qlayer_grads)
    assert any(g.abs().sum() > 0 for g in qlayer_grads)


def test_same_seed_same_initial_outputs():
    # Parameter init draws from the global RNG (run_merlin_poisson calls
    # torch.manual_seed before construction), so seeding it reproduces init.
    x = torch.linspace(0.0, math.pi / 2, 3, dtype=torch.float64)
    torch.manual_seed(7)
    u_a, _, _ = small_model(seed=7)(x)
    torch.manual_seed(7)
    u_b, _, _ = small_model(seed=7)(x)
    assert torch.allclose(u_a, u_b)


def test_short_training_reduces_objective():
    model = small_model()
    x = torch.linspace(0.0, math.pi / 2, 8, dtype=torch.float64)
    target = torch.sin(4.0 * x) / 16.0
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    def objective() -> torch.Tensor:
        u, _, _ = model(x)
        return ((u - target) ** 2).mean()

    initial = objective().item()
    for _ in range(30):
        opt.zero_grad()
        loss = objective()
        loss.backward()
        opt.step()
    assert objective().item() < initial


def test_run_merlin_poisson_writes_artifacts(tmp_path):
    from CV_QPINN_PDE.lib.merlin_pinn import run_merlin_poisson

    cfg = {
        "experiment": "poisson_merlin",
        "seed": 42,
        "model": {
            "n_modes": 4,
            "input_modes": [0, 1],
            "entangling_layers": 1,
            "n_photons": 2,
            "encoding_scale": math.pi / 2,
        },
        "training": {
            "epochs": 2,
            "lr": 0.02,
            "collocation_points": 8,
            "lambdas": {"pde": 0.34, "bc": 0.34, "consistency": 0.32, "trace": 0.0},
            "log_every": 1,
        },
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = run_merlin_poisson(cfg, run_dir)
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "predictions.json").exists()
    assert "metrics" in summary
    assert summary["n_params"] > 0
