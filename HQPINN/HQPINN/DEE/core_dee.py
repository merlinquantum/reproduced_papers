# core_dee.py

import os
import csv
from datetime import datetime
from typing import Tuple, Callable, Optional

import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..config import (
    DTYPE,
    DEVICE,
    EE_GAMMA,
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
from ..utils import (
    count_trainable_params,
    log_training_info,
)

from typing import Optional

# Use non-interactive backend for batch PDF export
matplotlib.use("Agg")


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
    n_f_batch: Optional[int] = None,
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
        (rho_left - rho_left_exact) ** 2
        + (rho_right - rho_right_exact) ** 2
        + (u_left - u_left_exact) ** 2
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

    e = p / ((EE_GAMMA - 1.0) * rho)
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


def train_dee(
    model: nn.Module,
    t_train,  # unused, kept for API consistency
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    plot_every: int,
    out_dir: str,
    model_label: str,
    loss_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = euler_loss_batched,
    # ] = euler_loss,
) -> tuple[float, float, float, int]:
    """Training loop for the Discontinuous Euler Equation PINN (classical-classical)."""

    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf_path = os.path.join(out_dir, f"dee-{model_label}_{timestamp}.pdf")
    csv_path = os.path.join(out_dir, f"dee-{model_label}_{timestamp}.csv")

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

    # Final evaluation grid for PDF
    with torch.no_grad():
        nx, nt = DEE_NX_SAMPLES, DEE_NT_SAMPLES
        x = torch.linspace(DEE_X_MIN, DEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
        t = torch.linspace(DEE_T_MIN, DEE_T_MAX, nt, dtype=DTYPE, device=DEVICE)
        X, T = torch.meshgrid(x, t, indexing="ij")
        xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

        U_pred = model(xt)
        rho_pred = U_pred[:, 0].reshape(nx, nt)

        # Exact density
        rho_exact = exact_rho(X, T)  # [nx, nt]

        # Error (still in Tensor)
        rho_error = rho_pred - rho_exact  # Tensor [nx, nt]

        # ---- Only now: convert everything to numpy for plotting ----
        X_np = X.cpu().numpy()
        T_np = T.cpu().numpy()
        rho_pred_np = rho_pred.cpu().numpy()
        rho_exact_np = rho_exact.cpu().numpy()
        rho_err_np = rho_error.cpu().numpy()

        with PdfPages(pdf_path) as pdf:

            # 1) Predicted density
            fig, ax = plt.subplots(figsize=(8, 5))  # 8x5 inches
            cs = ax.contourf(
                X_np, T_np, rho_pred_np, levels=50
            )  # 50 contour levels for smooth color gradation
            fig.colorbar(cs, ax=ax)
            ax.set_title("Predicted density $\\rho_\\text{pred}(x,t)$")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # 2) Exact density
            fig, ax = plt.subplots(figsize=(8, 5))
            cs = ax.contourf(X_np, T_np, rho_exact_np, levels=50)
            fig.colorbar(cs, ax=ax)
            ax.set_title("Exact density $\\rho_\\text{exact}(x,t)$")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # 3) Error (pred - exact)
            fig, ax = plt.subplots(figsize=(8, 5))
            cs = ax.contourf(X_np, T_np, rho_err_np, levels=50, cmap="bwr")
            fig.colorbar(cs, ax=ax)
            ax.set_title(
                "Density error $\\rho_\\text{pred}(x,t)-\\rho_\\text{exact}(x,t)$"
            )
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # Evaluate density and pressure errors on (x,t) grid
    err_rho, err_p = evaluate_dee_errors(model)

    # Count trainable parameters
    n_params = count_trainable_params(model)

    # print(f"Metrics CSV saved to: {metrics_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"PDF saved to: {pdf_path}")

    return final_loss, err_rho, err_p, n_params


# Displaying of the result
def save_density_plot(
    model: nn.Module,
    ckpt_dir: str,
    case_prefix: str,
    n_photons: int,
    timestamp: str,
    backend: str,
) -> str:

    model.eval()

    with torch.no_grad():
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
        x = torch.linspace(DEE_X_MIN, DEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
        t = torch.linspace(DEE_T_MIN, DEE_T_MAX, nt, dtype=DTYPE, device=DEVICE)
        X, T = torch.meshgrid(x, t, indexing="ij")
        xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

        U_pred = model(xt)
        rho_pred = U_pred[:, 0].reshape(nx, nt)

        X_np = X.cpu().numpy()
        T_np = T.cpu().numpy()
        rho_pred_np = rho_pred.cpu().numpy()

        results_dir = os.path.join(ckpt_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        png_path = os.path.join(results_dir, f"{case_prefix}_{backend}_{timestamp}.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        cs = ax.contourf(
            X_np,
            T_np,
            rho_pred_np,
            levels=50,
        )
        fig.colorbar(cs, ax=ax)
        ax.set_title(
            f"Predicted density $\\rho_\\text{{pred}}(x,t)$, {n_photons} photons, backend: {backend}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.tight_layout()

        fig.savefig(png_path, dpi=300)
        plt.close(fig)

    return png_path
