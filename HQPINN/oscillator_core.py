# oscillator_core.py

import os
from datetime import datetime
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Use non-interactive backend for batch PDF export
matplotlib.use("Agg")


# ============================================================
#  Physical parameters and exact solution
# ============================================================

# Damped oscillator constants
M = 1.0
MU = 4.0
K = 400.0

# Loss weights
LAMBDA1 = 1e-1
LAMBDA2 = 1e-4


def omega(mu: float = MU, k: float = K) -> float:
    """Angular frequency of the damped oscillator."""
    return np.sqrt(k - (mu / 2.0) ** 2)


def u_exact(t_array: np.ndarray, mu: float = MU, k: float = K) -> np.ndarray:
    """Closed-form solution u(t) for the damped oscillator with u(0)=1, u'(0)=0."""
    w = omega(mu, k)
    return np.exp(-mu * t_array / 2.0) * (
        np.cos(w * t_array) + (mu / (2.0 * w)) * np.sin(w * t_array)
    )


# ============================================================
#  Autograd utilities: first and second derivatives w.r.t. t
# ============================================================


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


# ============================================================
#  Physics-informed loss for the damped oscillator
# ============================================================


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
    t0 = torch.zeros((1, 1), dtype=t.dtype, device=t.device).requires_grad_(True)
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


# ============================================================
#  Generic training loop with PDF logging
# ============================================================


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
    """
    Generic training loop for the damped oscillator PINN.

    Parameters
    ----------
    model : nn.Module
        Model u(t) to train. Must accept t of shape (N, 1) and return (N, 1).
    t_train : torch.Tensor
        Time grid for training, shape (N, 1).
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    n_epochs : int
        Number of training epochs.
    plot_every : int
        Plot and log diagnostics every 'plot_every' epochs.
    out_dir : str
        Output directory for PDF logs.
    model_label : str
        Label used in plots (e.g. "classical–classical", "quantum–quantum").
    lambda1, lambda2 : float
        Loss weights for derivative IC and PDE residual.
    loss_fn : callable
        Function computing (loss_ic_u, loss_ic_du, loss_f).
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf_path = os.path.join(out_dir, f"HQPINN-A2-{model_label}_{timestamp}.pdf")

    with PdfPages(pdf_path) as pdf:
        for epoch in range(n_epochs):

            optimizer.zero_grad()

            lic_u, lic_du, lf = loss_fn(model, t_train)

            # Total weighted loss
            loss = lic_u + lambda1 * lic_du + lambda2 * lf

            # Backpropagation
            loss.backward()
            optimizer.step()

            if epoch % plot_every != 0:
                continue

            # --- Console diagnostics ---
            print(
                f"Epoch {epoch:4d} | "
                f"Loss = {loss.item():.4e} | "
                f"IC_u = {lic_u:.4e} | "
                f"IC_du = {lic_du:.4e} | "
                f"PDE = {lf:.4e}"
            )

            # Diagnostic grid
            t_diag = t_train.clone().detach().requires_grad_(True)
            u_diag = model(t_diag)
            du_diag = derivative(u_diag, t_diag)
            d2u_diag = second_derivative(u_diag, t_diag)

            print(
                "||u||:",
                u_diag.abs().mean().item(),
                "\n||u'||:",
                du_diag.abs().mean().item(),
                "\n||u''||:",
                d2u_diag.abs().mean().item(),
            )

            # --- Numpy conversion ---
            with torch.no_grad():
                t_np = t_diag.squeeze().cpu().numpy()
                u_np = u_diag.squeeze().cpu().numpy()
                du_np = du_diag.squeeze().cpu().numpy()
                d2u_np = d2u_diag.squeeze().cpu().numpy()

                u_pred = model(t_train).cpu().numpy().flatten()
                u_ex = u_exact(t_np)

            # --------------------------------------------------------
            # Page 1: u(t), u'(t), u''(t)
            # --------------------------------------------------------
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(t_np, u_np, label="u(t)")
            ax1.plot(t_np, du_np, label="u'(t)")
            ax1.plot(t_np, d2u_np, label="u''(t)")
            ax1.legend()
            ax1.set_xlabel("t")
            ax1.set_title(f"Diagnostics: u, u', u'' — epoch {epoch}")
            ax1.grid(True)
            fig1.tight_layout()
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)

            # --------------------------------------------------------
            # Page 2: prediction vs exact solution
            # --------------------------------------------------------
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(t_np, u_pred, label=f"PINN ({model_label})")
            ax2.plot(t_np, u_ex, "--", label="Exact")
            ax2.legend()
            ax2.set_xlabel("t")
            ax2.set_ylabel("u(t)")
            ax2.set_title(f"Prediction vs Exact — epoch {epoch}")
            ax2.grid(True)
            fig2.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)
