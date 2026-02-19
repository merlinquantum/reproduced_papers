# core.py

import os
import csv
from datetime import datetime
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import (
    M,
    MU,
    K,
    LAMBDA1,
    LAMBDA2,
    DTYPE,
    DEVICE,
    SEE_GAMMA,
    SEE_X_MIN,
    SEE_X_MAX,
    SEE_T_MIN,
    SEE_T_MAX,
)
from utils import (
    sample_ic_points,
    sample_bc_points,
    sample_collocation_points,
    count_trainable_params,
)

# Use non-interactive backend for batch PDF export
matplotlib.use("Agg")


# ============================================================
#  Damped oscillator (dho)
# ============================================================


def omega(mu: float = MU, k: float = K) -> float:
    return np.sqrt(k - (mu / 2.0) ** 2)


def u_exact(t_array: np.ndarray, mu: float = MU, k: float = K) -> np.ndarray:
    w = omega(mu, k)
    return np.exp(-mu * t_array / 2.0) * (
        np.cos(w * t_array) + (mu / (2.0 * w)) * np.sin(w * t_array)
    )


def derivative(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute du/dt using PyTorch autograd."""
    return grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]


def second_derivative(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute d²u/dt² using PyTorch autograd."""
    du_dt = derivative(u, t)
    return derivative(du_dt, t)


def oscillator_loss(
    model: nn.Module,
    t: torch.Tensor,
    m: float = M,
    mu: float = MU,
    k: float = K,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three components of the PINN loss:

      1. Initial condition on u(0):  (u(0) - 1)^2
      2. Initial condition on u'(0): (u'(0))^2
      3. PDE residual: mean( (m u'' + μ u' + k u)^2 )

    Returns
    -------
    loss_ic_u : torch.Tensor
        Initial condition loss on u(0).
    loss_ic_du : torch.Tensor
        Initial condition loss on u'(0).
    loss_f : torch.Tensor
        PDE residual loss.
    """
    # Fresh differentiable copy of t
    t = t.clone().detach().requires_grad_(True)

    # Forward pass
    u = model(t)
    du = derivative(u, t)
    d2u = second_derivative(u, t)

    # PDE residual: m u'' + μ u' + k u
    f = m * d2u + mu * du + k * u

    # Initial conditions at t = 0
    t0 = torch.zeros((1, 1), dtype=DTYPE, device=DEVICE).requires_grad_(True)
    u0 = model(t0)
    du0 = derivative(u0, t0)

    loss_ic_u = (u0 - 1.0) ** 2  # u(0) = 1
    loss_ic_du = du0**2  # u'(0) = 0
    loss_f = torch.mean(f**2)

    return (
        loss_ic_u.squeeze(),
        loss_ic_du.squeeze(),
        loss_f.squeeze(),
    )


def train_oscillator_pinn(
    model: nn.Module,
    t_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    plot_every: int,
    out_dir: str,
    model_label: str,
    lambda1: float = LAMBDA1,
    lambda2: float = LAMBDA2,
    loss_fn: Callable[
        [nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = oscillator_loss,
) -> None:

    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf_path = os.path.join(out_dir, f"a2-dho-{model_label}_{timestamp}.pdf")
    csv_path = os.path.join(out_dir, f"a2-dho-{model_label}_{timestamp}.csv")

    start = datetime.now()
    rows = []

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    for epoch in range(n_epochs):

        optimizer.zero_grad()
        lic_u, lic_du, lf = loss_fn(model, t_train)
        loss = lic_u + lambda1 * lic_du + lambda2 * lf

        loss.backward()
        optimizer.step()

        if epoch % plot_every == 0:

            stop = datetime.now()
            elapsed = (stop - start).total_seconds()

            print(f"Epoch {epoch:4d} | Elapsed: {elapsed:.2f}seconds")
            print(
                f"  Loss={loss.item():.4e} | "
                f"IC_u={lic_u:.4e} | IC_du={lic_du:.4e} | PDE={lf:.4e}"
            )

            # -------------------------------------------------------
            # Append this epoch to CSV
            # -------------------------------------------------------

            # for ti, ui, dui, d2ui in zip(t_np, u_np, du_np, d2u_np):
            rows.append(
                [
                    epoch,
                    f"{elapsed:.2f}",
                    f"{loss.item():.4e}",
                    f"{lic_u:.4e}",
                    f"{lic_du:.4e}",
                    f"{lf:.4e}",
                ]
            )

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "elapsed time (s)",
                "Loss",
                "IC_u",
                "IC_du",
                "PDE",
            ]
        )
        writer.writerows(rows)

    stop = datetime.now()
    elapsed = (stop - start).total_seconds()

    # -------------------------------------------------------
    # Final PDF (only the prediction vs exact plot)
    # -------------------------------------------------------
    with torch.no_grad():
        t_np = t_train.squeeze().cpu().numpy()
        u_pred = model(t_train).cpu().numpy().flatten()
        u_ex = u_exact(t_np)

    with PdfPages(pdf_path) as pdf:

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_np, u_pred, label=f"PINN ({model_label})")
        ax.plot(t_np, u_ex, "--", label="Exact")
        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.set_title(f"{elapsed:.2f}s - {n_epochs - 1} epochs")
        ax.grid(True)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nCSV saved to: {csv_path}")
    print(f"PDF saved to: {pdf_path}")


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


def euler_loss(
    model: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PINN loss terms for the 1D smooth Euler equation:
      1) Initial condition loss on (rho, u, p) at t = 0
      2) Periodic boundary loss in x
      3) PDE residual loss for 1D Euler in conservative form
    """

    # ==========================
    # 1. Initial condition loss
    # ==========================
    x_ic, t_ic = sample_ic_points()  # shapes [N_ic, 1]
    X_ic = torch.cat([x_ic, t_ic], dim=1)  # [N_ic, 2]

    # Network prediction at t=0
    U_ic = model(X_ic)  # [N_ic, 3]
    rho_ic, u_ic, p_ic = U_ic.split(1, dim=1)

    # Exact IC from the paper:
    # rho_0(x) = 1.0 + 0.2 * sin(pi * x)
    # u_0(x)   = 1.0
    # p_0(x)   = 1.0
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
    x_left, x_right, t_bc = sample_bc_points()  # each [N_bc, 1]

    X_left = torch.cat([x_left, t_bc], dim=1)  # [N_bc, 2]
    X_right = torch.cat([x_right, t_bc], dim=1)

    U_left = model(X_left)  # [N_bc, 3]
    U_right = model(X_right)

    # Periodicity: U(x_min, t) = U(x_max, t)
    loss_bc = torch.mean((U_left - U_right) ** 2)

    # ==========================
    # 3. PDE residual loss
    # ==========================
    x_f, t_f = sample_collocation_points()  # [N_f, 1]
    X_f = torch.cat([x_f, t_f], dim=1)  # [N_f, 2]
    X_f = X_f.clone().detach().to(DTYPE).to(DEVICE)
    X_f.requires_grad_(True)

    # Network prediction in the interior
    U_f = model(X_f)  # [N_f, 3]
    rho, u, p = U_f.split(1, dim=1)

    gamma = SEE_GAMMA

    # Total energy per unit mass: E = e + 0.5 u^2
    # with p = (gamma - 1) * rho * e
    e = p / ((gamma - 1.0) * rho)
    E = e + 0.5 * u**2

    # Conservative variables:
    # U1 = rho
    # U2 = rho * u
    # U3 = rho * E
    U1 = rho
    U2 = rho * u
    U3 = rho * E

    # Fluxes in x:
    # F1 = rho * u
    # F2 = rho * u^2 + p
    # F3 = u * (rho * E + p)
    F1 = rho * u
    F2 = rho * u**2 + p
    F3 = u * (rho * E + p)

    # Time derivatives ∂_t U
    U1_t = partial_derivative(U1, X_f, index=1)
    U2_t = partial_derivative(U2, X_f, index=1)
    U3_t = partial_derivative(U3, X_f, index=1)

    # Spatial derivatives ∂_x F
    F1_x = partial_derivative(F1, X_f, index=0)
    F2_x = partial_derivative(F2, X_f, index=0)
    F3_x = partial_derivative(F3, X_f, index=0)

    # Residuals: ∂_t U + ∂_x F = 0
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


# def evaluate_see_errors(model, nx: int = 200, nt: int = 200):
#     """Compute relative L2 errors for rho and p on an (x,t) grid."""
#     x = torch.linspace(SEE_X_MIN, SEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
#     t = torch.linspace(SEE_T_MIN, SEE_T_MAX, nt, dtype=DTYPE, device=DEVICE)
#     X, T = torch.meshgrid(x, t, indexing="ij")  # [nx, nt]

#     # Build (x,t) pairs
#     xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)  # [nx*nt, 2]

#     with torch.no_grad():
#         U_pred = model(xt)  # [nx*nt, 3]
#         rho_pred, u_pred, p_pred = U_pred.split(1, dim=1)

#     rho_exact, u_exact, p_exact = exact_solution(xt[:, 0:1], xt[:, 1:2])

#     def rel_l2(pred: torch.Tensor, exact: torch.Tensor) -> float:
#         """Compute relative L2 norm."""
#         num = torch.norm(pred - exact)
#         den = torch.norm(exact)
#         return (num / den).item()

#     err_rho = rel_l2(rho_pred, rho_exact)
#     err_p = rel_l2(p_pred, p_exact)

#     return err_rho, err_p


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


def train_see_cc(
    model: nn.Module,
    t_train,  # unused, kept for API consistency
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    plot_every: int,
    out_dir: str,
    model_label: str,
    loss_fn: Callable[
        [nn.Module], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = euler_loss,
) -> tuple[float, float, float, int]:
    """Training loop for the Smooth Euler Equation PINN (classical-classical)."""

    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf_path = os.path.join(out_dir, f"see-cc-{model_label}_{timestamp}.pdf")
    csv_path = os.path.join(out_dir, f"see-cc-{model_label}_{timestamp}.csv")

    rows = []
    start = datetime.now()

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        # Compute the three loss components
        loss_ic, loss_bc, loss_f = loss_fn(model)
        loss = loss_ic + loss_bc + loss_f

        loss.backward()
        optimizer.step()

        # Logging every plot_every epochs
        if epoch % plot_every == 0:
            elapsed = (datetime.now() - start).total_seconds()

            print(
                f"Epoch {epoch:5d} | elapsed={elapsed:.2f}s  "
                f"L={loss.item():.3e} | "
                f"IC={loss_ic.item():.3e} | "
                f"BC={loss_bc.item():.3e} | "
                f"F={loss_f.item():.3e}"
            )

            rows.append(
                [
                    epoch,
                    f"{elapsed:.2f}",
                    f"{loss.item():.4e}",
                    f"{loss_ic.item():.4e}",
                    f"{loss_bc.item():.4e}",
                    f"{loss_f.item():.4e}",
                ]
            )

    # -------------------------
    # L-BFGS refinement
    # -------------------------
    print("Starting L-BFGS refinement...")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=500,
        max_eval=500,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()  # resets gradient buffer
        lic, lbc, lf = euler_loss(model)
        l = lic + lbc + lf
        l.backward()  #  computes ∇loss
        return l

    # Run L-BFGS optimization
    optimizer_lbfgs.step(closure)  #  updates model weights

    print("L-BFGS refinement done.")

    # -------------------------
    # Final loss after L-BFGS
    # -------------------------
    loss_ic, loss_bc, loss_f = euler_loss(model)
    final_loss = (loss_ic + loss_bc + loss_f).item()

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "elapsed (s)", "Loss", "IC", "BC", "F"])
        writer.writerows(rows)

    # Final evaluation grid for PDF
    with torch.no_grad():
        nx, nt = 200, 200
        x = torch.linspace(SEE_X_MIN, SEE_X_MAX, nx, dtype=DTYPE, device=DEVICE)
        t = torch.linspace(SEE_T_MIN, SEE_T_MAX, nt, dtype=DTYPE, device=DEVICE)
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

        # Exact density
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
    err_rho, err_p = evaluate_see_errors(model)

    # Count trainable parameters
    n_params = count_trainable_params(model)

    # print(f"Metrics CSV saved to: {metrics_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"PDF saved to: {pdf_path}")

    return final_loss, err_rho, err_p, n_params
