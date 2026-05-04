# core_see.py

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

from ..config import *
from ..paths import results_case_dir_for_model_dir
from ..utils import (
    append_or_replace_training_row,
    sample_ic_points,
    sample_bc_points,
    sample_collocation_points,
    count_trainable_params,
    log_training_info,
    save_training_checkpoint,
    write_metrics_csv,
)

SEE_SUMMARY_COLUMNS = [
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

SEE_PAPER_CMAP = "jet"
SEE_PAPER_RHO_MIN = 0.8
SEE_PAPER_RHO_MAX = 1.2
SEE_PAPER_RHO_TICKS = (0.8, 0.9, 1.0, 1.1, 1.2)
SEE_PAPER_ABS_ERR_MAX = 0.004
SEE_PAPER_ABS_ERR_TICKS = (0.000, 0.001, 0.002, 0.003, 0.004)
SEE_PAPER_X_TICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)
SEE_PAPER_T_TICKS = (0.0, 0.5, 1.0, 1.5, 2.0)


def _build_see_plot_grid(nx: int, nt: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(SEE_X_MIN, SEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
    t = torch.linspace(SEE_T_MIN, SEE_T_MAX, nt, dtype=DTYPE, device=DEVICE)
    X, T = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
    return X, T, xt


def _evaluate_see_density_fields(
    model: nn.Module,
    nx: int,
    nt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        X, T, xt = _build_see_plot_grid(nx, nt)
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


def _style_see_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_xlim(SEE_X_MIN, SEE_X_MAX)
    ax.set_ylim(SEE_T_MIN, SEE_T_MAX)
    ax.set_xticks(SEE_PAPER_X_TICKS)
    ax.set_yticks(SEE_PAPER_T_TICKS)
    ax.set_aspect("equal", adjustable="box")


def _plot_see_density_panel(
    ax: plt.Axes,
    X_np: np.ndarray,
    T_np: np.ndarray,
    rho_np: np.ndarray,
    title: str,
):
    rho_for_plot = np.clip(rho_np, SEE_PAPER_RHO_MIN, SEE_PAPER_RHO_MAX)
    contour = ax.contourf(
        X_np,
        T_np,
        rho_for_plot,
        levels=np.linspace(SEE_PAPER_RHO_MIN, SEE_PAPER_RHO_MAX, 200),
        cmap=SEE_PAPER_CMAP,
    )
    _style_see_axis(ax, title)
    return contour


def _plot_see_abs_error_panel(
    ax: plt.Axes,
    X_np: np.ndarray,
    T_np: np.ndarray,
    rho_abs_err_np: np.ndarray,
    title: str,
):
    rho_abs_err_for_plot = np.clip(rho_abs_err_np, 0.0, SEE_PAPER_ABS_ERR_MAX)
    contour = ax.contourf(
        X_np,
        T_np,
        rho_abs_err_for_plot,
        levels=np.linspace(0.0, SEE_PAPER_ABS_ERR_MAX, 200),
        cmap=SEE_PAPER_CMAP,
    )
    _style_see_axis(ax, title)
    return contour


def _save_see_single_panel_figure(
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


def _save_see_reference_figure(
    png_path: str,
    X_np: np.ndarray,
    T_np: np.ndarray,
    rho_pred_np: np.ndarray,
    rho_abs_err_np: np.ndarray,
    plot_label: str | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.8), constrained_layout=True)

    rho_title = r"$\rho$"
    if plot_label:
        rho_title += f", {plot_label}"
    rho_contour = _plot_see_density_panel(
        axes[0],
        X_np,
        T_np,
        rho_pred_np,
        rho_title,
    )
    err_contour = _plot_see_abs_error_panel(
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
        ticks=SEE_PAPER_RHO_TICKS,
    )
    fig.colorbar(
        err_contour,
        ax=axes[1],
        orientation="horizontal",
        pad=0.12,
        ticks=SEE_PAPER_ABS_ERR_TICKS,
        format="%.3f",
    )
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


# ============================================================
#  Smooth Euler Equation (SEE)
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


def euler_loss_batched(
    model: nn.Module,
    n_f_batch: Optional[int] = 256,
    n_ic_batch: Optional[int] = None,
    n_bc_batch: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mini-batching

    Instead of evaluating all PDE, IC, and BC points at every epoch
    (e.g., 2000 PDE points, 50 IC, 50 BC), we randomly sample a small
    subset such as:

        n_f_batch = 128     # PDE points
        n_ic_batch = 25     # initial condition points
        n_bc_batch = 25     # boundary points
    """

    # ==========================
    # 1. Initial condition loss
    # ==========================
    x_ic, t_ic = sample_ic_points()  # [N_ic, 1]
    N_ic = x_ic.size(0)
    if n_ic_batch is not None and n_ic_batch < N_ic:
        idx_ic = torch.randperm(N_ic)[:n_ic_batch]
        x_ic = x_ic[idx_ic]
        t_ic = t_ic[idx_ic]
    X_ic = torch.cat([x_ic, t_ic], dim=1)  # [N_ic_batch, 2]

    U_ic = model(X_ic)  # [N_ic_batch, 3]
    rho_ic, u_ic, p_ic = U_ic.split(1, dim=1)

    rho_ic_exact = 1.0 + 0.2 * torch.sin(torch.pi * x_ic)
    u_ic_exact = torch.ones_like(x_ic)
    p_ic_exact = torch.ones_like(x_ic)

    loss_ic = torch.mean(
        (rho_ic - rho_ic_exact) ** 2
        + (u_ic - u_ic_exact) ** 2
        + (p_ic - p_ic_exact) ** 2
    )

    # ==========================
    # 2. Periodic boundary loss
    # ==========================
    x_left, x_right, t_bc = sample_bc_points()  # [N_bc, 1]
    N_bc = x_left.size(0)
    if n_bc_batch is not None and n_bc_batch < N_bc:
        idx_bc = torch.randperm(N_bc)[:n_bc_batch]
        x_left = x_left[idx_bc]
        x_right = x_right[idx_bc]
        t_bc = t_bc[idx_bc]

    X_left = torch.cat([x_left, t_bc], dim=1)
    X_right = torch.cat([x_right, t_bc], dim=1)

    U_left = model(X_left)
    U_right = model(X_right)

    loss_bc = torch.mean((U_left - U_right) ** 2)

    # ==========================
    # 3. PDE residual loss (batched)
    # ==========================
    x_f, t_f = sample_collocation_points()  # [N_f, 1]
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

    gamma = GAMMA

    e = p / ((gamma - 1.0) * rho)
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


def exact_rho(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Exact density rho(x,t) = 1 + 0.2 * sin(pi * (x - t))."""
    return 1.0 + 0.2 * torch.sin(torch.pi * (x - t))


def exact_solution(x: torch.Tensor, t: torch.Tensor):
    """Exact smooth Euler solution (rho,u,p) at (x,t)."""
    rho = exact_rho(x, t)
    u = torch.ones_like(x)
    p = torch.ones_like(x)
    return rho, u, p


def evaluate_see_errors(model, nx: int = 1000):
    """Relative L2 errors computed at t = T."""
    t_final = SEE_T_MAX
    x = torch.linspace(SEE_X_MIN, SEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
    t = torch.full_like(x, t_final)

    xt = torch.stack([x, t], dim=1)

    with torch.no_grad():
        U_pred = model(xt)
        rho_pred, u_pred, p_pred = U_pred.split(1, dim=1)

    rho_exact, u_exact, p_exact = exact_solution(x[:, None], t[:, None])

    # PINN-style L2 error
    def rel_l2(pred, exact):
        num = torch.sqrt(torch.mean((pred - exact) ** 2))
        den = torch.sqrt(torch.mean(exact**2))
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

    csv_path = os.path.join(out_dir, f"see-{model_label}_{run_id}.csv")
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
    csv_path = os.path.join(out_dir, f"see-{model_label}_{run_id}.csv")
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    return rows[-1]


def append_summary_row(summary_path: str, row: dict[str, object]) -> bool:
    """
    Append one normalized SEE summary row, writing the header on first use.

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
        writer = csv.DictWriter(f, fieldnames=SEE_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SEE_SUMMARY_COLUMNS})
    return is_duplicate


def train_see(
    model: nn.Module,
    t_train,  # unused, kept for API consistency
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    plot_every: int,
    out_dir: str,
    model_label: str,
    run_id: str,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = None,
    resume_state: dict | None = None,
    loss_fn: Callable[
        [nn.Module], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = euler_loss_batched,
    # ] = euler_loss,
):
    # -> tuple[float, float, float, int]
    # :
    """Training loop for the Smooth Euler Equation PINN (classical-classical)."""

    os.makedirs(out_dir, exist_ok=True)

    rho_pred_png_path = os.path.join(
        out_dir, f"see-{model_label}_{run_id}_rho_pred.png"
    )
    rho_exact_png_path = os.path.join(
        out_dir, f"see-{model_label}_{run_id}_rho_exact.png"
    )
    rho_error_png_path = os.path.join(
        out_dir, f"see-{model_label}_{run_id}_rho_error.png"
    )
    csv_path = os.path.join(out_dir, f"see-{model_label}_{run_id}.csv")
    csv_header = ["epoch", "elapsed (s)", "Loss", "IC", "BC", "F"]

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=500
    # )

    rows = [list(row) for row in (resume_state or {}).get("rows", [])]
    start = datetime.now()
    start_epoch = int((resume_state or {}).get("epoch", -1)) + 1
    elapsed_offset = float((resume_state or {}).get("elapsed_s", 0.0))
    checkpoint_every = checkpoint_every or plot_every
    last_completed_epoch = start_epoch - 1
    last_elapsed = elapsed_offset

    try:
        for epoch in range(start_epoch, n_epochs):
            optimizer.zero_grad()

            # Compute the three loss components
            loss_ic, loss_bc, loss_f = loss_fn(model)
            loss = loss_ic + loss_bc + loss_f

            loss.backward()
            optimizer.step()
            elapsed = elapsed_offset + (datetime.now() - start).total_seconds()
            last_completed_epoch = epoch
            last_elapsed = elapsed

            # Logging every plot_every epochs
            if epoch % plot_every == 0:
                log_training_info(
                    n_epochs=epoch,
                    elapsed=elapsed,
                    final_loss=loss,
                    loss_ic=loss_ic,
                    loss_bc=loss_bc,
                    loss_f=loss_f,
                    rows=rows,
                )

            if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
                save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    run_id=run_id,
                    epoch=epoch,
                    elapsed_s=elapsed,
                    rows=rows,
                )
                write_metrics_csv(csv_path, csv_header, rows)
    except KeyboardInterrupt:
        if checkpoint_path is not None:
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                run_id=run_id,
                epoch=last_completed_epoch,
                elapsed_s=last_elapsed,
                rows=rows,
            )
            write_metrics_csv(csv_path, csv_header, rows)
        raise

    loss_ic, loss_bc, loss_f = loss_fn(model)
    final_loss = (loss_ic + loss_bc + loss_f).item()
    elapsed = elapsed_offset + (datetime.now() - start).total_seconds()

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
    write_metrics_csv(csv_path, csv_header, rows)
    if checkpoint_path is not None:
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            run_id=run_id,
            epoch=n_epochs - 1,
            elapsed_s=elapsed,
            rows=rows,
        )

    # Final evaluation grid for PNG outputs.
    X_np, T_np, rho_pred_np, rho_exact_np, rho_abs_err_np = _evaluate_see_density_fields(
        model=model,
        nx=SEE_NX_SAMPLES,
        nt=SEE_NT_SAMPLES,
    )

    fig, ax = plt.subplots(figsize=(5.2, 5.6), constrained_layout=True)
    rho_pred_contour = _plot_see_density_panel(
        ax,
        X_np,
        T_np,
        rho_pred_np,
        rf"$\rho$, epoch={n_epochs}",
    )
    _save_see_single_panel_figure(
        rho_pred_png_path,
        rho_pred_contour,
        fig,
        ax,
        SEE_PAPER_RHO_TICKS,
    )

    fig, ax = plt.subplots(figsize=(5.2, 5.6), constrained_layout=True)
    rho_exact_contour = _plot_see_density_panel(
        ax,
        X_np,
        T_np,
        rho_exact_np,
        r"Exact $\rho$",
    )
    _save_see_single_panel_figure(
        rho_exact_png_path,
        rho_exact_contour,
        fig,
        ax,
        SEE_PAPER_RHO_TICKS,
    )

    fig, ax = plt.subplots(figsize=(5.2, 5.6), constrained_layout=True)
    rho_abs_err_contour = _plot_see_abs_error_panel(
        ax,
        X_np,
        T_np,
        rho_abs_err_np,
        rf"$|\Delta \rho|$, epoch={n_epochs}",
    )
    _save_see_single_panel_figure(
        rho_error_png_path,
        rho_abs_err_contour,
        fig,
        ax,
        SEE_PAPER_ABS_ERR_TICKS,
        cbar_format="%.3f",
    )

    # Evaluate density and pressure errors on (x,t) grid
    err_rho, err_p = evaluate_see_errors(model)

    # Count trainable parameters
    n_params = count_trainable_params(model)

    # print(f"Metrics CSV saved to: {metrics_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"PNG saved to: {rho_pred_png_path}")
    print(f"PNG saved to: {rho_exact_png_path}")
    print(f"PNG saved to: {rho_error_png_path}")

    return final_loss, err_rho, err_p, n_params


# Displaying of the result
def save_density_plot(
    model: nn.Module,
    ckpt_dir: str,
    case_prefix: str,
    plot_label: str | None,
    run_id: str,
    backend: str,
) -> str:

    model.eval()

    nx, nt = SEE_NX_SAMPLES, SEE_NT_SAMPLES
    if backend.lower() != "local":
        # Remote backends create many cloud jobs; downsample for robustness.
        nx = min(nx, 30)
        nt = min(nt, 30)
        print(f"Remote backend detected: using reduced grid {nx}x{nt} for plotting.")

    X_np, T_np, rho_pred_np, _, rho_abs_err_np = _evaluate_see_density_fields(
        model=model,
        nx=nx,
        nt=nt,
    )

    results_dir = results_case_dir_for_model_dir(ckpt_dir, case_prefix)
    os.makedirs(results_dir, exist_ok=True)

    png_path = os.path.join(results_dir, f"{case_prefix}_{backend}_{run_id}.png")
    _save_see_reference_figure(
        png_path=png_path,
        X_np=X_np,
        T_np=T_np,
        rho_pred_np=rho_pred_np,
        rho_abs_err_np=rho_abs_err_np,
        plot_label=plot_label,
    )

    return png_path
