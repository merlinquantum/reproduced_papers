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

from config import M, MU, K, LAMBDA1, LAMBDA2, DTYPE, DEVICE

# Use non-interactive backend for batch PDF export
matplotlib.use("Agg")


def omega(mu: float = MU, k: float = K) -> float:
    return np.sqrt(k - (mu / 2.0) ** 2)


def u_exact(t_array: np.ndarray, mu: float = MU, k: float = K) -> np.ndarray:
    w = omega(mu, k)
    return np.exp(-mu * t_array / 2.0) * (
        np.cos(w * t_array) + (mu / (2.0 * w)) * np.sin(w * t_array)
    )


# ============================================================
#  Autograd utilities: first and second derivatives
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

            print(f"Epoch {epoch:4d} | Elapsed: {elapsed:.2f}s ")
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
        ax.set_title("Final Prediction vs Exact")
        ax.grid(True)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nCSV saved to: {csv_path}")
    print(f"PDF saved to: {pdf_path}")
