# utils.py

import os
from typing import Callable, Optional

import torch
import torch.nn as nn

from .config import (
    DTYPE,
    DEVICE,
    DHO_N_SAMPLES,
    SEE_X_MIN,
    SEE_X_MAX,
    SEE_T_MIN,
    SEE_T_MAX,
    SEE_N_IC,
    SEE_N_BC,
    SEE_N_F,
)


def make_time_grid():
    """Return the time grid t ∈ [0,1] as a torch tensor."""
    return torch.linspace(0.0, 1.0, DHO_N_SAMPLES, dtype=DTYPE, device=DEVICE).reshape(
        -1, 1
    )


def make_optimizer(model, lr):
    """Create an Adam optimizer for the given model."""
    return torch.optim.Adam(model.parameters(), lr=lr)


# ==========================
#  SEE – Smooth Euler Equation (Sec. 3.1)
#  1D Euler, solution lisse:
#  x ∈ (-1, 1), t ∈ (0, 2)
# ==========================


def sample_ic_points():
    """
    Generate Initial Condition (IC) points for the Smooth Euler Equation.

    In a PDE setting, the unknown is a function of TWO variables: U(x, t).
    Initial conditions are therefore not a single value like f(0) (ODE case),
    but a condition defined along the ENTIRE line t = 0:
        U(x, 0) = U0(x), for x in (SEE_X_MIN, SEE_X_MAX).

    We approximate this continuous constraint by sampling SEE_N_IC random x-points
    in the spatial domain, and pairing them with t = 0.
    """
    # Sample x uniformly in (0,1) then map to (SEE_X_MIN, SEE_X_MAX)
    x_ic = torch.rand(SEE_N_IC, 1, dtype=DTYPE, device=DEVICE)
    x_ic = SEE_X_MIN + (SEE_X_MAX - SEE_X_MIN) * x_ic

    # Initial condition line: all points are at t = 0
    t_ic = torch.zeros_like(x_ic)

    return x_ic, t_ic


def sample_bc_points():
    """
    Generate Boundary Condition (BC) points for periodic boundaries in x.

    For the Smooth Euler case, the paper uses periodic boundary conditions in x.
    "Periodic in x" means the left and right boundaries are IDENTIFIED:
        U(SEE_X_MIN, t) = U(SEE_X_MAX, t) for all t.

    Numerically, we cannot enforce "for all t" exactly, so we enforce it on
    a set of sampled times {t_bc_i}. For each sampled time t_bc_i, we create
    a PAIR of boundary points:
        (x_left = SEE_X_MIN,  t_bc_i)  and  (x_right = SEE_X_MAX, t_bc_i)

    During training, the BC loss typically penalizes the mismatch:
        ||U(x_left, t_bc) - U(x_right, t_bc)||^2
    which encourages periodicity across the domain boundaries.
    """
    # Sample times uniformly in (0,1) then map to (SEE_T_MIN, SEE_T_MAX)
    t_bc = torch.rand(SEE_N_BC, 1, dtype=DTYPE, device=DEVICE)
    t_bc = SEE_T_MIN + (SEE_T_MAX - SEE_T_MIN) * t_bc

    # Create the matching boundary x-locations at the same times t_bc
    x_left = torch.full_like(t_bc, SEE_X_MIN)
    x_right = torch.full_like(t_bc, SEE_X_MAX)

    return x_left, x_right, t_bc


def sample_collocation_points():
    """
    Generate interior collocation points (x_f, t_f) for the PDE residual term.

    This is where the PINN uses the governing equation (the "physics"):
    we sample points throughout the space-time domain and evaluate the PDE
    residual F(x, t) computed via automatic differentiation.

    The HQPINN/PINN framework forms a physics loss (often MSE) by enforcing:
        F(x_f, t_f) ≈ 0
    at many interior points (collocation points).
    """
    # Spatial samples: map uniform (0,1) to (SEE_X_MIN, SEE_X_MAX)
    x_f = torch.rand(SEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    x_f = SEE_X_MIN + (SEE_X_MAX - SEE_X_MIN) * x_f

    # Temporal samples: map uniform (0,1) to (SEE_T_MIN, SEE_T_MAX)
    t_f = torch.rand(SEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    t_f = SEE_T_MIN + (SEE_T_MAX - SEE_T_MIN) * t_f

    return x_f, t_f


def count_trainable_params(model: nn.Module) -> int:
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_training_info(n_epochs, elapsed, final_loss, loss_ic, loss_bc, loss_f, rows):
    """Log training information for debugging and monitoring."""
    print(
        f"Epoch {n_epochs:5d} | elapsed={elapsed:.2f}s  "
        f"L={final_loss:.3e} | "
        f"IC={loss_ic.item():.3e} | "
        f"BC={loss_bc.item():.3e} | "
        f"F={loss_f.item():.3e}"
    )

    rows.append(
        [
            n_epochs,
            f"{elapsed:.2f}",
            f"{final_loss:.3e}",
            f"{loss_ic.item():.3e}",
            f"{loss_bc.item():.3e}",
            f"{loss_f.item():.3e}",
        ]
    )

    return


def load_model(
    ckpt_path: str, model_ctor: Callable[..., nn.Module], processor=None
) -> nn.Module:
    model = model_ctor(processor=processor)  # use_remote implicite = False (local SLOS)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from: {ckpt_path} remote={processor is not None}")
    return model


def get_latest_checkpoint(ckpt_dir: str, case_prefix: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        print(f"No checkpoint directory found at {ckpt_dir}")
        return None

    files = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith(f"{case_prefix}_") and f.endswith(".pt")
    ]
    if not files:
        print(f"No checkpoints matching {case_prefix}_*.pt in {ckpt_dir}")
        return None

    files.sort()  # lexicographique => avec timestamp YYYYMMDD-HHMMSS c'est chronologique
    latest = files[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f"Latest checkpoint found: {ckpt_path}")
    return ckpt_path


def load_latest_model_local(
    ckpt_dir: str,
    case_prefix: str,
    model_ctor: Callable[[], nn.Module],
) -> Optional[nn.Module]:
    ckpt_path = get_latest_checkpoint(ckpt_dir, case_prefix)
    if ckpt_path is None:
        return None
    return load_model(ckpt_path, model_ctor)
