# core_dee.py

import os
import csv
from datetime import datetime
import sys
from typing import Tuple, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

import matplotlib

# Keep batch exports headless, but do not disable inline notebook rendering.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...config import (
    DTYPE,
    DEVICE,
    GAMMA,
    DEE_X_MIN,
    DEE_X_MAX,
    DEE_T_MIN,
    DEE_T_MAX,
    DEE_NX_SAMPLES,
    DEE_NT_SAMPLES,
    DEE_N_IC,
    DEE_N_BC,
    DEE_N_F,
    DEE_U,
    DEE_P,
    DEE_RHO_L,
    DEE_RHO_R,
    DEE_X0,
)
from ...paths import results_case_dir_for_model_dir
from ...utils import (
    count_trainable_params,
    log_training_info,
)

DEE_SUMMARY_COLUMNS = [
    "run_id",
    "Model",
    "Size",
    "epoch",
    "elapsed (s)",
    "Trainable parameters",
    "Loss",
    "IC",
    "BC",
    "F",
    "Density error",
    "Pressure error",
]

DEE_PAPER_CMAP = "jet"
DEE_PAPER_RHO_MIN = 1.0
DEE_PAPER_RHO_MAX = 1.4
DEE_PAPER_RHO_TICKS = (1.0, 1.1, 1.2, 1.3, 1.4)
DEE_PAPER_ABS_ERR_MAX = 0.18
DEE_PAPER_ABS_ERR_TICKS = (0.05, 0.10, 0.15)
DEE_PAPER_X_TICKS = (0.0, 0.25, 0.50, 0.75, 1.0)
DEE_PAPER_T_TICKS = (0.0, 0.5, 1.0, 1.5, 2.0)
DEE_PAPER_BOX_ASPECT = 0.95
DEE_PAPER_FIELD_ALPHA_WITH_POINTS = 0.42
DEE_PAPER_RESIDUAL_MARKER_SIZE = 28
DEE_PAPER_BOUNDARY_MARKER_SIZE = 30
DEE_PAPER_REFERENCE_FIGSIZE = (11.6, 4.9)
DEE_PAPER_SINGLE_FIGSIZE = (6.2, 4.8)


def _build_dee_plot_grid(
    nx: int, nt: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(DEE_X_MIN, DEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
    t = torch.linspace(DEE_T_MIN, DEE_T_MAX, nt, dtype=DTYPE, device=DEVICE)
    X, T = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
    return X, T, xt


def _evaluate_dee_density_fields(
    model: nn.Module,
    nx: int,
    nt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        X, T, xt = _build_dee_plot_grid(nx, nt)
        U_pred = model(xt)
        rho_pred = U_pred[:, 0].reshape(nx, nt)
        rho_exact = exact_rho(X, T)
        rho_abs_err = torch.abs(rho_pred - rho_exact)

    return (
        X.cpu().numpy(),
        T.cpu().numpy(),
        rho_pred.cpu().numpy(),
        rho_exact.cpu().numpy(),
        rho_abs_err.cpu().numpy(),
    )


def _stack_dee_training_points(
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    t_bc: torch.Tensor,
    x_f: torch.Tensor,
    t_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_xt = torch.cat([x_f, t_f], dim=1)
    boundary_xt = torch.cat(
        [
            torch.cat([x_ic, x_left, x_right], dim=0),
            torch.cat([t_ic, t_bc, t_bc], dim=0),
        ],
        dim=1,
    )
    return residual_xt, boundary_xt


def _sample_dee_training_points() -> tuple[torch.Tensor, torch.Tensor]:
    x_ic, t_ic = sample_ic_points()
    x_left, x_right, t_bc = sample_bc_points()
    x_f, t_f = sample_collocation_points()
    return _stack_dee_training_points(x_ic, t_ic, x_left, x_right, t_bc, x_f, t_f)


def _predict_dee_rho_at_points(model: nn.Module, xt: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        rho = model(xt)[:, 0]
    return np.clip(rho.detach().cpu().numpy(), DEE_PAPER_RHO_MIN, DEE_PAPER_RHO_MAX)


def _overlay_dee_training_points(
    ax: plt.Axes,
    model: nn.Module,
    residual_xt: torch.Tensor | None = None,
    boundary_xt: torch.Tensor | None = None,
) -> None:
    if residual_xt is None or boundary_xt is None:
        residual_xt, boundary_xt = _sample_dee_training_points()

    residual_rho = _predict_dee_rho_at_points(model, residual_xt)
    boundary_rho = _predict_dee_rho_at_points(model, boundary_xt)

    residual_np = residual_xt.detach().cpu().numpy()
    boundary_np = boundary_xt.detach().cpu().numpy()

    ax.scatter(
        residual_np[:, 0],
        residual_np[:, 1],
        c=residual_rho,
        cmap=DEE_PAPER_CMAP,
        vmin=DEE_PAPER_RHO_MIN,
        vmax=DEE_PAPER_RHO_MAX,
        s=DEE_PAPER_RESIDUAL_MARKER_SIZE,
        edgecolors=(0.0, 0.0, 0.0, 0.18),
        linewidths=0.2,
        alpha=1.0,
        zorder=3,
    )
    ax.scatter(
        boundary_np[:, 0],
        boundary_np[:, 1],
        c=boundary_rho,
        cmap=DEE_PAPER_CMAP,
        vmin=DEE_PAPER_RHO_MIN,
        vmax=DEE_PAPER_RHO_MAX,
        s=DEE_PAPER_BOUNDARY_MARKER_SIZE,
        edgecolors="k",
        linewidths=0.85,
        zorder=4,
    )


def _style_dee_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_xlim(DEE_X_MIN, DEE_X_MAX)
    ax.set_ylim(DEE_T_MIN, DEE_T_MAX)
    ax.set_xticks(DEE_PAPER_X_TICKS)
    ax.set_yticks(DEE_PAPER_T_TICKS)
    ax.set_box_aspect(DEE_PAPER_BOX_ASPECT)


def _plot_dee_density_panel(
    ax: plt.Axes,
    X_np: np.ndarray,
    T_np: np.ndarray,
    rho_np: np.ndarray,
    title: str,
    model_for_training_points: nn.Module | None = None,
    residual_xt: torch.Tensor | None = None,
    boundary_xt: torch.Tensor | None = None,
):
    rho_for_plot = np.clip(rho_np, DEE_PAPER_RHO_MIN, DEE_PAPER_RHO_MAX)
    field_alpha = (
        DEE_PAPER_FIELD_ALPHA_WITH_POINTS
        if model_for_training_points is not None
        else 1.0
    )
    contour = ax.contourf(
        X_np,
        T_np,
        rho_for_plot,
        levels=np.linspace(DEE_PAPER_RHO_MIN, DEE_PAPER_RHO_MAX, 200),
        cmap=DEE_PAPER_CMAP,
        alpha=field_alpha,
    )
    _style_dee_axis(ax, title)
    if model_for_training_points is not None:
        _overlay_dee_training_points(
            ax,
            model_for_training_points,
            residual_xt=residual_xt,
            boundary_xt=boundary_xt,
        )
    return contour


def _plot_dee_abs_error_panel(
    ax: plt.Axes,
    X_np: np.ndarray,
    T_np: np.ndarray,
    rho_abs_err_np: np.ndarray,
    title: str,
):
    rho_abs_err_for_plot = np.clip(rho_abs_err_np, 0.0, DEE_PAPER_ABS_ERR_MAX)
    contour = ax.contourf(
        X_np,
        T_np,
        rho_abs_err_for_plot,
        levels=np.linspace(0.0, DEE_PAPER_ABS_ERR_MAX, 200),
        cmap=DEE_PAPER_CMAP,
    )
    _style_dee_axis(ax, title)
    return contour


def _save_dee_single_panel_figure(
    png_path: str,
    contour,
    fig: plt.Figure,
    ax: plt.Axes,
    ticks: tuple[float, ...],
    cbar_format: str | None = None,
) -> None:
    fig.colorbar(
        contour,
        ax=ax,
        orientation="horizontal",
        pad=0.12,
        ticks=ticks,
        format=cbar_format,
    )
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _save_dee_reference_figure(
    png_path: str,
    X_np: np.ndarray,
    T_np: np.ndarray,
    rho_pred_np: np.ndarray,
    rho_abs_err_np: np.ndarray,
    plot_label: str | None,
    model_for_training_points: nn.Module | None = None,
    residual_xt: torch.Tensor | None = None,
    boundary_xt: torch.Tensor | None = None,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=DEE_PAPER_REFERENCE_FIGSIZE,
        constrained_layout=True,
    )

    rho_title = r"$\rho$"
    if plot_label:
        rho_title += f", {plot_label}"
    rho_contour = _plot_dee_density_panel(
        axes[0],
        X_np,
        T_np,
        rho_pred_np,
        rho_title,
        model_for_training_points=model_for_training_points,
        residual_xt=residual_xt,
        boundary_xt=boundary_xt,
    )
    err_contour = _plot_dee_abs_error_panel(
        axes[1],
        X_np,
        T_np,
        rho_abs_err_np,
        r"$|\Delta \rho|$",
    )

    fig.colorbar(
        rho_contour,
        ax=axes[0],
        orientation="horizontal",
        pad=0.12,
        ticks=DEE_PAPER_RHO_TICKS,
    )
    fig.colorbar(
        err_contour,
        ax=axes[1],
        orientation="horizontal",
        pad=0.12,
        ticks=DEE_PAPER_ABS_ERR_TICKS,
        format="%.2f",
    )
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


# ============================================================
#  Discontinue Euler Equation (DEE)
# ============================================================


def partial_derivative(y: torch.Tensor, x: torch.Tensor, index: int) -> torch.Tensor:
    """Compute dy/dx_index where x has shape [N, 2] = [x, t]."""
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grads[:, index : index + 1]  # index=0 -> d/dx, index=1 -> d/dt


def sample_ic_points():
    """Sample DEE initial-condition points on t=DEE_T_MIN."""
    x_ic = torch.rand(DEE_N_IC, 1, dtype=DTYPE, device=DEVICE)
    x_ic = DEE_X_MIN + (DEE_X_MAX - DEE_X_MIN) * x_ic
    t_ic = torch.full_like(x_ic, DEE_T_MIN)
    return x_ic, t_ic


def sample_bc_points():
    """Sample DEE boundary points at x=DEE_X_MIN and x=DEE_X_MAX."""
    t_bc = torch.rand(DEE_N_BC, 1, dtype=DTYPE, device=DEVICE)
    t_bc = DEE_T_MIN + (DEE_T_MAX - DEE_T_MIN) * t_bc
    x_left = torch.full_like(t_bc, DEE_X_MIN)
    x_right = torch.full_like(t_bc, DEE_X_MAX)
    return x_left, x_right, t_bc


def sample_collocation_points():
    """Sample DEE collocation points in (x,t) domain."""
    x_f = torch.rand(DEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    x_f = DEE_X_MIN + (DEE_X_MAX - DEE_X_MIN) * x_f
    t_f = torch.rand(DEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    t_f = DEE_T_MIN + (DEE_T_MAX - DEE_T_MIN) * t_f
    return x_f, t_f


def exact_rho(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Discontinuous density:
      rho(x, t) = 1.4 if x < 0.5 + 0.1*t
                  1.0 if x > 0.5 + 0.1*t
                  undefined if x == 0.5 + 0.1*t
    """
    front = DEE_X0 + DEE_U * t
    rho = torch.where(x <= front, DEE_RHO_L, DEE_RHO_R)
    return rho


def exact_u(x: torch.Tensor):
    return torch.full_like(x, DEE_U)


def exact_p(x: torch.Tensor):
    return torch.full_like(x, DEE_P)


# def exact_solution(x: torch.Tensor, t: torch.Tensor):
#     """Exact smooth Euler solution (rho,u,p) at (x,t)."""
#     rho = exact_rho(x, t)
#     return rho


def euler_loss_batched(
    model: nn.Module,
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    t_bc: torch.Tensor,
    x_f: torch.Tensor,
    t_f: torch.Tensor,
    n_f_batch: Optional[int] = 256,
    n_ic_batch: Optional[int] = None,
    n_bc_batch: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mini-batching

    Optionally supports mini-batching. By default, all sampled points
    are used each epoch (paper setting: 1000 F, 60 IC, 60 BC).
    If batch sizes are provided, a random subset is selected, e.g.:

        n_f_batch = 128     # F points
        n_ic_batch = 25     # initial condition points
        n_bc_batch = 25     # boundary points
    """

    # ==========================
    # 1. Initial condition loss
    # ==========================
    N_ic = x_ic.size(0)
    if n_ic_batch is not None and n_ic_batch < N_ic:
        idx_ic = torch.randperm(N_ic)[:n_ic_batch]
        x_ic = x_ic[idx_ic]
        t_ic = t_ic[idx_ic]
    X_ic = torch.cat([x_ic, t_ic], dim=1)  # [N_ic_batch, 2]

    U_ic = model(X_ic)  # [N_ic_batch, 3]
    rho_ic, u_ic, p_ic = U_ic.split(1, dim=1)

    rho_ic_exact = exact_rho(x_ic, t_ic)
    u_ic_exact = exact_u(x_ic)
    p_ic_exact = exact_p(x_ic)

    loss_ic = torch.mean(
        (rho_ic - rho_ic_exact) ** 2
        + (u_ic - u_ic_exact) ** 2
        + (p_ic - p_ic_exact) ** 2
    )

    # ==========================
    # 2. Dirichlet boundary loss
    # ==========================
    N_bc = x_left.size(0)
    if n_bc_batch is not None and n_bc_batch < N_bc:
        idx_bc = torch.randperm(N_bc)[:n_bc_batch]
        x_left = x_left[idx_bc]
        x_right = x_right[idx_bc]
        t_bc = t_bc[idx_bc]

    X_left = torch.cat([x_left, t_bc], dim=1)
    X_right = torch.cat([x_right, t_bc], dim=1)

    U_left = model(X_left)
    rho_left, u_left, p_left = U_left.split(1, dim=1)
    U_right = model(X_right)
    rho_right, u_right, p_right = U_right.split(1, dim=1)

    rho_left_exact = torch.full_like(x_left, DEE_RHO_R)  # Eq. (12): = 1.0
    rho_right_exact = torch.full_like(x_right, DEE_RHO_R)  # = 1.0
    u_left_exact = torch.full_like(x_left, DEE_U)
    u_right_exact = torch.full_like(x_right, DEE_U)
    p_left_exact = torch.full_like(x_left, DEE_P)
    p_right_exact = torch.full_like(x_right, DEE_P)

    loss_bc = torch.mean(
        # (rho_left - rho_left_exact) ** 2
        # + (rho_right - rho_right_exact) ** 2
        +((u_left - u_left_exact) ** 2)
        + (u_right - u_right_exact) ** 2
        + (p_left - p_left_exact) ** 2
        + (p_right - p_right_exact) ** 2
    )

    # ==========================
    # 3. PDE residual loss (batched)
    # ==========================
    N_f = x_f.size(0)

    if n_f_batch is not None and n_f_batch < N_f:
        idx_f = torch.randperm(N_f)[:n_f_batch]
        x_f = x_f[idx_f]
        t_f = t_f[idx_f]

    X_f = torch.cat([x_f, t_f], dim=1)  # [n_f_batch, 2]
    X_f = X_f.clone().detach().to(DTYPE).to(DEVICE)
    X_f.requires_grad_(True)

    U_f = model(X_f)  # [n_f_batch, 3]
    rho, u, p = U_f.split(1, dim=1)

    e = p / ((GAMMA - 1.0) * rho)
    E = e + 0.5 * u**2

    U1 = rho
    U2 = rho * u
    U3 = rho * E

    F1 = rho * u
    F2 = rho * u**2 + p
    F3 = u * (rho * E + p)

    U1_t = partial_derivative(U1, X_f, index=1)
    U2_t = partial_derivative(U2, X_f, index=1)
    U3_t = partial_derivative(U3, X_f, index=1)

    F1_x = partial_derivative(F1, X_f, index=0)
    F2_x = partial_derivative(F2, X_f, index=0)
    F3_x = partial_derivative(F3, X_f, index=0)

    r1 = U1_t + F1_x
    r2 = U2_t + F2_x
    r3 = U3_t + F3_x

    loss_f = torch.mean(r1**2 + r2**2 + r3**2)

    return loss_ic, loss_bc, loss_f


def evaluate_dee_errors(model, nx: int = 1000):
    """Relative L2 errors computed at t = T."""
    t_final = DEE_T_MAX
    x = torch.linspace(DEE_X_MIN, DEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
    t = torch.full_like(x, t_final)

    xt = torch.stack([x, t], dim=1)

    with torch.no_grad():
        U_pred = model(xt)
        rho_pred, u_pred, p_pred = U_pred.split(1, dim=1)

    rho_exact = exact_rho(x[:, None], t[:, None])
    p_exact = exact_p(x[:, None])

    # PINN-style L2 error
    def rel_l2(pred, exact):
        valid = torch.isfinite(pred) & torch.isfinite(exact)
        if not torch.any(valid):
            return float("nan")
        pred_v = pred[valid]
        exact_v = exact[valid]
        num = torch.sqrt(torch.mean((pred_v - exact_v) ** 2))
        den = torch.sqrt(torch.mean(exact_v**2))
        return (num / den).item()

    return rel_l2(rho_pred, rho_exact), rel_l2(p_pred, p_exact)


def load_training_loss_for_checkpoint(
    out_dir: str, model_label: str, ckpt_path: str, case_prefix: str
) -> Optional[float]:
    """Return the final Loss from the CSV that matches the checkpoint run_id."""
    ckpt_name = os.path.basename(ckpt_path)
    ckpt_prefix = f"{case_prefix}_"
    ckpt_suffix = ".pt"
    if not (ckpt_name.startswith(ckpt_prefix) and ckpt_name.endswith(ckpt_suffix)):
        return None

    run_id = ckpt_name[len(ckpt_prefix) : -len(ckpt_suffix)]
    if not run_id:
        return None

    csv_path = os.path.join(out_dir, f"dee-{model_label}_{run_id}.csv")
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    return float(rows[-1]["Loss"])


def get_run_id_from_checkpoint(ckpt_path: str, case_prefix: str) -> Optional[str]:
    """Extract the run_id encoded in a checkpoint filename."""
    ckpt_name = os.path.basename(ckpt_path)
    ckpt_prefix = f"{case_prefix}_"
    ckpt_suffix = ".pt"
    if not (ckpt_name.startswith(ckpt_prefix) and ckpt_name.endswith(ckpt_suffix)):
        return None

    run_id = ckpt_name[len(ckpt_prefix) : -len(ckpt_suffix)]
    return run_id or None


def load_training_row_for_run_id(
    out_dir: str,
    model_label: str,
    run_id: str,
) -> Optional[dict[str, str]]:
    """Return the last row from the detailed CSV for a given model/run_id pair."""
    csv_path = os.path.join(out_dir, f"dee-{model_label}_{run_id}.csv")
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    return rows[-1]


def append_summary_row(summary_path: str, row: dict[str, object]) -> bool:
    """
    Append one normalized DEE summary row, writing the header on first use.

    Returns True when the same `(run_id, Model, Size)` triplet was already
    present before the append, and False otherwise.
    """
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = (
        not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    )

    is_duplicate = False
    if not write_header:
        with open(summary_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        row_key = (
            str(row.get("run_id", "")),
            str(row.get("Model", "")),
            str(row.get("Size", "")),
        )
        for existing in existing_rows:
            existing_key = (
                existing.get("run_id", ""),
                existing.get("Model", ""),
                existing.get("Size", ""),
            )
            if existing_key == row_key:
                is_duplicate = True
                break

    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEE_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in DEE_SUMMARY_COLUMNS})
    return is_duplicate


def train_dee(
    model: nn.Module,
    t_train,  # unused, kept for API consistency
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    plot_every: int,
    out_dir: str,
    model_label: str,
    run_id: str,
    loss_fn: Callable[
        ..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = euler_loss_batched,
    # ] = euler_loss,
) -> tuple[float, float, float, int]:
    """Training loop for the Discontinuous Euler Equation PINN (classical-classical)."""

    os.makedirs(out_dir, exist_ok=True)

    rho_pred_png_path = os.path.join(
        out_dir, f"dee-{model_label}_{run_id}_rho_pred.png"
    )
    rho_exact_png_path = os.path.join(
        out_dir, f"dee-{model_label}_{run_id}_rho_exact.png"
    )
    rho_error_png_path = os.path.join(
        out_dir, f"dee-{model_label}_{run_id}_rho_error.png"
    )
    csv_path = os.path.join(out_dir, f"dee-{model_label}_{run_id}.csv")

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=500
    # )

    rows = []
    start = datetime.now()

    # Fixed training samples: draw once and reuse for all epochs.
    x_ic_all, t_ic_all = sample_ic_points()
    x_left_all, x_right_all, t_bc_all = sample_bc_points()
    x_f_all, t_f_all = sample_collocation_points()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Compute the three loss components
        loss_ic, loss_bc, loss_f = loss_fn(
            model,
            x_ic_all,
            t_ic_all,
            x_left_all,
            x_right_all,
            t_bc_all,
            x_f_all,
            t_f_all,
        )
        loss = loss_ic + loss_bc + loss_f

        loss.backward()
        optimizer.step()

        # Logging every plot_every epochs
        if epoch % plot_every == 0:
            elapsed = (datetime.now() - start).total_seconds()

            log_training_info(
                n_epochs=epoch,
                elapsed=elapsed,
                final_loss=loss,
                loss_ic=loss_ic,
                loss_bc=loss_bc,
                loss_f=loss_f,
                rows=rows,
            )

        # scheduler.step(loss.item())  # Adjust learning rate based on total loss

    # -------------------------
    # L-BFGS refinement
    # -------------------------
    # print("Starting L-BFGS refinement...")

    # optimizer_lbfgs = torch.optim.LBFGS(
    #     model.parameters(),
    #     lr=1.0,
    #     max_iter=500,
    #     max_eval=500,
    #     history_size=50,
    #     line_search_fn="strong_wolfe",
    # )

    # def closure():
    #     optimizer_lbfgs.zero_grad()  # resets gradient buffer
    #     lic, lbc, lf = loss_fn(model)
    #     l = lic + lbc + lf
    #     l.backward()  #  computes ∇loss
    #     return l

    # # Run L-BFGS optimization
    # optimizer_lbfgs.step(closure)  #  updates model weights

    # print("L-BFGS refinement done.")

    # -------------------------
    # Final loss after L-BFGS
    # -------------------------
    loss_ic, loss_bc, loss_f = loss_fn(
        model,
        x_ic_all,
        t_ic_all,
        x_left_all,
        x_right_all,
        t_bc_all,
        x_f_all,
        t_f_all,
    )
    final_loss = (loss_ic + loss_bc + loss_f).item()
    elapsed = (datetime.now() - start).total_seconds()

    log_training_info(
        n_epochs=n_epochs,
        elapsed=elapsed,
        final_loss=final_loss,
        loss_ic=loss_ic,
        loss_bc=loss_bc,
        loss_f=loss_f,
        rows=rows,
    )

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "elapsed (s)", "Loss", "IC", "BC", "F"])
        writer.writerows(rows)

    residual_xt, boundary_xt = _stack_dee_training_points(
        x_ic_all,
        t_ic_all,
        x_left_all,
        x_right_all,
        t_bc_all,
        x_f_all,
        t_f_all,
    )
    X_np, T_np, rho_pred_np, rho_exact_np, rho_abs_err_np = _evaluate_dee_density_fields(
        model=model,
        nx=DEE_NX_SAMPLES,
        nt=DEE_NT_SAMPLES,
    )

    fig, ax = plt.subplots(
        figsize=DEE_PAPER_SINGLE_FIGSIZE,
        constrained_layout=True,
    )
    rho_pred_contour = _plot_dee_density_panel(
        ax,
        X_np,
        T_np,
        rho_pred_np,
        rf"$\rho$, epoch={n_epochs}",
        model_for_training_points=model,
        residual_xt=residual_xt,
        boundary_xt=boundary_xt,
    )
    _save_dee_single_panel_figure(
        rho_pred_png_path,
        rho_pred_contour,
        fig,
        ax,
        DEE_PAPER_RHO_TICKS,
    )

    fig, ax = plt.subplots(
        figsize=DEE_PAPER_SINGLE_FIGSIZE,
        constrained_layout=True,
    )
    rho_exact_contour = _plot_dee_density_panel(
        ax,
        X_np,
        T_np,
        rho_exact_np,
        r"Exact $\rho$",
    )
    _save_dee_single_panel_figure(
        rho_exact_png_path,
        rho_exact_contour,
        fig,
        ax,
        DEE_PAPER_RHO_TICKS,
    )

    fig, ax = plt.subplots(
        figsize=DEE_PAPER_SINGLE_FIGSIZE,
        constrained_layout=True,
    )
    rho_abs_err_contour = _plot_dee_abs_error_panel(
        ax,
        X_np,
        T_np,
        rho_abs_err_np,
        rf"$|\Delta \rho|$, epoch={n_epochs}",
    )
    _save_dee_single_panel_figure(
        rho_error_png_path,
        rho_abs_err_contour,
        fig,
        ax,
        DEE_PAPER_ABS_ERR_TICKS,
        cbar_format="%.2f",
    )

    rho_slice_png_path = _save_rho_slice_plot_to_dir(
        model=model,
        output_dir=out_dir,
        file_prefix=f"dee-{model_label}_{run_id}",
        t_slice=2.0,
        n_points=400,
    )

    # Evaluate density and pressure errors on (x,t) grid
    err_rho, err_p = evaluate_dee_errors(model)

    # Count trainable parameters
    n_params = count_trainable_params(model)

    # print(f"Metrics CSV saved to: {metrics_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"PNG saved to: {rho_pred_png_path}")
    print(f"PNG saved to: {rho_exact_png_path}")
    print(f"PNG saved to: {rho_error_png_path}")
    print(f"PNG saved to: {rho_slice_png_path}")

    return final_loss, err_rho, err_p, n_params


def _save_rho_slice_plot_to_dir(
    *,
    model: nn.Module,
    output_dir: str,
    file_prefix: str,
    t_slice: float,
    n_points: int,
) -> str:
    """Save one DEE density slice into an explicit output directory."""
    model.eval()
    t_val = min(max(float(t_slice), DEE_T_MIN), DEE_T_MAX)

    with torch.no_grad():
        x = torch.linspace(DEE_X_MIN, DEE_X_MAX, n_points, dtype=DTYPE, device=DEVICE)
        t = torch.full_like(x, t_val)
        xt = torch.stack([x, t], dim=1)

        rho_pred = model(xt)[:, 0]
        rho_exact = exact_rho(x[:, None], t[:, None]).squeeze(1)

    x_np = x.detach().cpu().numpy()
    rho_pred_np = rho_pred.detach().cpu().numpy()
    rho_exact_np = rho_exact.detach().cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    t_tag = str(t_val).replace(".", "p")
    png_path = os.path.join(output_dir, f"{file_prefix}_rho_x_t_{t_tag}.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_np, rho_pred_np, lw=2, label="NN")
    ax.plot(x_np, rho_exact_np, "--", lw=2, label="Exact")
    ax.set_xlabel("x")
    ax.set_ylabel("rho")
    ax.set_title(f"rho(x, t={t_val:.2f})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return png_path


# Displaying of the result
def save_density_plot(
    model: nn.Module,
    ckpt_dir: str,
    case_prefix: str,
    plot_label: str | None,
    run_id: str,
    backend: str,
) -> str:
    """
    Save the DEE density contour used in `run` and `remote` modes.

    A 1D density slice is saved as well because the moving shock is easier to
    inspect on a line plot than on a full contour.
    """

    model.eval()

    nx, nt = DEE_NX_SAMPLES, DEE_NT_SAMPLES
    if backend.lower() != "local":
        # Remote backends create many cloud jobs; downsample for robustness.
        orig_nx, orig_nt = nx, nt
        nx = min(nx, 30)
        nt = min(nt, 30)
        if (nx, nt) != (orig_nx, orig_nt):
            print(
                f"Remote backend detected: reduced grid for plotting "
                f"from {orig_nx}x{orig_nt} to {nx}x{nt}."
            )

    X_np, T_np, rho_pred_np, _, rho_abs_err_np = _evaluate_dee_density_fields(
        model=model,
        nx=nx,
        nt=nt,
    )

    results_dir = results_case_dir_for_model_dir(ckpt_dir, case_prefix)
    os.makedirs(results_dir, exist_ok=True)

    png_path = os.path.join(results_dir, f"{case_prefix}_{backend}_{run_id}.png")
    _save_dee_reference_figure(
        png_path=png_path,
        X_np=X_np,
        T_np=T_np,
        rho_pred_np=rho_pred_np,
        rho_abs_err_np=rho_abs_err_np,
        plot_label=plot_label,
        model_for_training_points=model,
    )

    rho_slice_path = save_rho_slice_plot(
        model=model,
        ckpt_dir=ckpt_dir,
        case_prefix=case_prefix,
        run_id=run_id,
        backend=backend,
        t_slice=2.0,
    )
    print(f"Rho slice plot saved to: {rho_slice_path}")

    return png_path


def save_rho_slice_plot(
    model: nn.Module,
    ckpt_dir: str,
    case_prefix: str,
    run_id: str,
    backend: str,
    t_slice: float = 2.0,
) -> str:
    """Save rho(x, t_slice) together with the exact moving-front profile."""
    n_points = 400 if backend.lower() == "local" else 120
    results_dir = results_case_dir_for_model_dir(ckpt_dir, case_prefix)
    return _save_rho_slice_plot_to_dir(
        model=model,
        output_dir=results_dir,
        file_prefix=f"{case_prefix}_{backend}_{run_id}",
        t_slice=t_slice,
        n_points=n_points,
    )
