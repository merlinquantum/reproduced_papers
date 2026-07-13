import csv
import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .config import (
    DEVICE,
    DHO_N_SAMPLES,
    DTYPE,
    SEE_N_BC,
    SEE_N_F,
    SEE_N_IC,
    SEE_T_MAX,
    SEE_T_MIN,
    SEE_X_MAX,
    SEE_X_MIN,
)


def make_time_grid():
    """Return the time grid t ∈ [0,1] as a torch tensor."""
    return torch.linspace(0.0, 1.0, DHO_N_SAMPLES, dtype=DTYPE, device=DEVICE)[
        1:
    ].reshape(-1, 1)


def make_optimizer(model, lr):
    """Create an Adam optimizer for the given model."""
    return torch.optim.Adam(model.parameters(), lr=lr)


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
    x_ic = torch.rand(SEE_N_IC, 1, dtype=DTYPE, device=DEVICE)
    x_ic = SEE_X_MIN + (SEE_X_MAX - SEE_X_MIN) * x_ic
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
    t_bc = torch.rand(SEE_N_BC, 1, dtype=DTYPE, device=DEVICE)
    t_bc = SEE_T_MIN + (SEE_T_MAX - SEE_T_MIN) * t_bc
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
    x_f = torch.rand(SEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    x_f = SEE_X_MIN + (SEE_X_MAX - SEE_X_MIN) * x_f
    t_f = torch.rand(SEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    t_f = SEE_T_MIN + (SEE_T_MAX - SEE_T_MIN) * t_f
    return x_f, t_f


def count_trainable_params(model: nn.Module) -> int:
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def append_or_replace_training_row(rows: list[list[object]], row: list[object]) -> None:
    """Append one metrics row, or replace the last row when it targets the same step."""
    if rows and str(rows[-1][0]) == str(row[0]):
        rows[-1] = row
        return
    rows.append(row)


def write_metrics_csv(
    csv_path: str,
    header: list[str],
    rows: list[list[object]],
) -> None:
    """Persist the full in-memory metrics table to disk."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def log_training_info(n_epochs, elapsed, final_loss, loss_ic, loss_bc, loss_f, rows):
    """Log training information for debugging and monitoring."""
    print(
        f"Epoch {n_epochs:5d} | elapsed={elapsed:.2f}s  "
        f"L={final_loss:.3e} | "
        f"IC={loss_ic.item():.3e} | "
        f"BC={loss_bc.item():.3e} | "
        f"F={loss_f.item():.3e}"
    )

    append_or_replace_training_row(
        rows,
        [
            n_epochs,
            f"{elapsed:.2f}",
            f"{final_loss:.3e}",
            f"{loss_ic.item():.3e}",
            f"{loss_bc.item():.3e}",
            f"{loss_f.item():.3e}",
        ],
    )


def load_model(
    ckpt_path: str, model_ctor: Callable[..., nn.Module], processor=None
) -> nn.Module:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = model_ctor(processor=processor)
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

    files.sort()
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


def get_resume_checkpoint_path(ckpt_dir: str, case_prefix: str) -> str:
    """Return the hidden path used for interrupted-training state."""
    return os.path.join(ckpt_dir, f".{case_prefix}_resume.pt")


def save_training_checkpoint(
    checkpoint_path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_id: str,
    epoch: int,
    elapsed_s: float,
    rows: list[list[object]],
    extra_state: Optional[dict[str, Any]] = None,
) -> None:
    """Save enough state to resume training from the last completed step."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_id": run_id,
        "epoch": int(epoch),
        "elapsed_s": float(elapsed_s),
        "rows": [list(row) for row in rows],
        "rng_state": torch.get_rng_state(),
    }
    if extra_state:
        state["extra_state"] = extra_state
    torch.save(state, checkpoint_path)
    print(
        f"Training checkpoint saved to: {checkpoint_path} "
        f"(run_id={run_id}, epoch={epoch})"
    )


def load_training_checkpoint(
    checkpoint_path: str,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    """Load a resumable training checkpoint into model and optimizer."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(f"{checkpoint_path} is not a resumable training checkpoint")

    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])

    print(
        f"Resumed training from: {checkpoint_path} "
        f"(run_id={state.get('run_id', '')}, epoch={state.get('epoch', '')})"
    )
    return state


def remove_resume_checkpoint(checkpoint_path: str) -> None:
    """Delete an interrupted-training checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def prepare_training_session(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    case_prefix: str,
    default_run_id: str,
    force_retrain: bool = False,
) -> tuple[str, Optional[dict[str, Any]], str]:
    """Load a resumable checkpoint when available and return session metadata."""
    resume_checkpoint_path = get_resume_checkpoint_path(ckpt_dir, case_prefix)

    if force_retrain:
        if os.path.exists(resume_checkpoint_path):
            remove_resume_checkpoint(resume_checkpoint_path)
            print(f"Removed interrupted-training checkpoint: {resume_checkpoint_path}")
        return default_run_id, None, resume_checkpoint_path

    if not os.path.isfile(resume_checkpoint_path):
        return default_run_id, None, resume_checkpoint_path

    try:
        state = load_training_checkpoint(
            resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
        )
    except Exception as exc:
        print(
            f"Interrupted-training checkpoint at {resume_checkpoint_path} "
            f"could not be loaded: {exc}; starting from scratch."
        )
        remove_resume_checkpoint(resume_checkpoint_path)
        return default_run_id, None, resume_checkpoint_path

    resumed_run_id = str(state.get("run_id") or default_run_id)
    return resumed_run_id, state, resume_checkpoint_path


def finalize_training_session(
    *,
    model: nn.Module,
    ckpt_dir: str,
    case_prefix: str,
    run_id: str,
    resume_checkpoint_path: Optional[str] = None,
) -> str:
    """Write the final inference checkpoint and clear the resumable one."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{case_prefix}_{run_id}.pt")
    torch.save(model.state_dict(), ckpt_path)
    if resume_checkpoint_path is not None:
        remove_resume_checkpoint(resume_checkpoint_path)
    print(f"Model saved to: {ckpt_path}")
    return ckpt_path
