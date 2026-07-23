"""Composite loss functions for the QPINN.

The paper combines a physics-residual term, a boundary-condition term, a
trace-normalisation term, and a consistency term that ties the second
output of the network to the spatial derivative of the first output (which
removes the need for nested automatic differentiation).
"""

from __future__ import annotations

import torch


def poisson_total_loss(model, x_collocation: torch.Tensor, x_bc_left: torch.Tensor,
                       x_bc_right: torch.Tensor,
                       lambdas: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """1D Poisson: u'' + sin(4x) = 0, u(0)=u(pi/2)=0.

    The "physics" residual uses the chain ux -> d(ux)/dx (one autograd hop,
    no nested gradients) instead of d^2 u / dx^2 directly.
    """
    x_collocation = x_collocation.detach().clone().requires_grad_(True)
    u, ux, trace = model(x_collocation)
    ux_x = torch.autograd.grad(ux.sum(), x_collocation, create_graph=True)[0]
    pde_residual = ux_x + torch.sin(4 * x_collocation)
    loss_pde = (pde_residual ** 2).mean()

    u_left, _, _ = model(x_bc_left)
    u_right, _, _ = model(x_bc_right)
    loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()

    u_x_auto = torch.autograd.grad(u.sum(), x_collocation, create_graph=True)[0]
    loss_consistency = ((u_x_auto - ux) ** 2).mean()

    loss_trace = ((trace - 1.0) ** 2).mean()

    total = (lambdas["pde"] * loss_pde
             + lambdas["bc"] * loss_bc
             + lambdas["consistency"] * loss_consistency
             + lambdas["trace"] * loss_trace)
    return total, {
        "pde": loss_pde.detach(),
        "bc": loss_bc.detach(),
        "consistency": loss_consistency.detach(),
        "trace": loss_trace.detach(),
        "total": total.detach(),
    }


def heat_total_loss(model, xt_collocation: torch.Tensor, xt_ic: torch.Tensor,
                    T_ic: torch.Tensor, xt_bc_left: torch.Tensor,
                    xt_bc_right: torch.Tensor, alpha: float,
                    lambdas: dict[str, float],
                    pre_train_only_ic: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """1D heat: T_t = alpha T_xx.

    `xt_collocation`: (B, 2) interior points (x, t).
    `xt_ic`:          (B_ic, 2) initial-condition points, t = 0.
    `T_ic`:           (B_ic,) target temperatures at t = 0.
    `xt_bc_left/right`: (B_bc, 2) boundary collocations at x = ±pi/2.
    """
    if pre_train_only_ic:
        u_ic, _, _ = model(xt_ic)
        loss_ic = ((u_ic - T_ic) ** 2).mean()
        return loss_ic, {"ic": loss_ic.detach(), "total": loss_ic.detach()}

    xt_collocation = xt_collocation.detach().clone().requires_grad_(True)
    u, ux, trace = model(xt_collocation)
    grads = torch.autograd.grad(u.sum(), xt_collocation, create_graph=True)[0]
    u_t = grads[:, 1]
    ux_grads = torch.autograd.grad(ux.sum(), xt_collocation, create_graph=True)[0]
    ux_x = ux_grads[:, 0]
    pde_residual = u_t - alpha * ux_x
    loss_pde = (pde_residual ** 2).mean()

    u_ic, _, _ = model(xt_ic)
    loss_ic = ((u_ic - T_ic) ** 2).mean()

    u_left, _, _ = model(xt_bc_left)
    u_right, _, _ = model(xt_bc_right)
    loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()

    u_x_auto = grads[:, 0]
    loss_consistency = ((u_x_auto - ux) ** 2).mean()

    loss_trace = ((trace - 1.0) ** 2).mean()

    total = (lambdas["pde"] * loss_pde
             + lambdas["ic"] * loss_ic
             + lambdas["bc"] * loss_bc
             + lambdas["consistency"] * loss_consistency
             + lambdas["trace"] * loss_trace)
    return total, {
        "pde": loss_pde.detach(),
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "consistency": loss_consistency.detach(),
        "trace": loss_trace.detach(),
        "total": total.detach(),
    }


def poisson_nested_loss(model, x_collocation: torch.Tensor, x_bc_left: torch.Tensor,
                        x_bc_right: torch.Tensor,
                        lambdas: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ablation: no consistency loss, no second-output dependence.

    Computes the second derivative `u_xx` directly via nested autograd. The
    paper claims this is impractical in CV simulators because nested
    gradients blow up memory; we use this loss to test that claim
    head-on.
    """
    x_collocation = x_collocation.detach().clone().requires_grad_(True)
    u, _, trace = model(x_collocation)
    u_x = torch.autograd.grad(u.sum(), x_collocation, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_collocation, create_graph=True)[0]
    pde_residual = u_xx + torch.sin(4 * x_collocation)
    loss_pde = (pde_residual ** 2).mean()

    u_left, _, _ = model(x_bc_left)
    u_right, _, _ = model(x_bc_right)
    loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()

    loss_trace = ((trace - 1.0) ** 2).mean()

    total = (lambdas["pde"] * loss_pde
             + lambdas["bc"] * loss_bc
             + lambdas["trace"] * loss_trace)
    return total, {
        "pde": loss_pde.detach(),
        "bc": loss_bc.detach(),
        "trace": loss_trace.detach(),
        "total": total.detach(),
        # Zeroed entries so the training loop's logger does not crash.
        "consistency": torch.tensor(0.0),
    }


def heat_nested_loss(model, xt_collocation: torch.Tensor, xt_ic: torch.Tensor,
                     T_ic: torch.Tensor, xt_bc_left: torch.Tensor,
                     xt_bc_right: torch.Tensor, alpha: float,
                     lambdas: dict[str, float],
                     pre_train_only_ic: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ablation: no consistency loss for the heat equation.

    `u_xx = d^2 u / dx^2` and `u_t = du/dt` are computed by nested autograd.
    """
    if pre_train_only_ic:
        u_ic, _, _ = model(xt_ic)
        loss_ic = ((u_ic - T_ic) ** 2).mean()
        return loss_ic, {"ic": loss_ic.detach(), "total": loss_ic.detach()}

    xt_collocation = xt_collocation.detach().clone().requires_grad_(True)
    u, _, trace = model(xt_collocation)
    grads = torch.autograd.grad(u.sum(), xt_collocation, create_graph=True)[0]
    u_x = grads[:, 0]
    u_t = grads[:, 1]
    u_xx_grads = torch.autograd.grad(u_x.sum(), xt_collocation, create_graph=True)[0]
    u_xx = u_xx_grads[:, 0]
    pde_residual = u_t - alpha * u_xx
    loss_pde = (pde_residual ** 2).mean()

    u_ic, _, _ = model(xt_ic)
    loss_ic = ((u_ic - T_ic) ** 2).mean()

    u_left, _, _ = model(xt_bc_left)
    u_right, _, _ = model(xt_bc_right)
    loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()

    loss_trace = ((trace - 1.0) ** 2).mean()

    total = (lambdas["pde"] * loss_pde
             + lambdas["ic"] * loss_ic
             + lambdas["bc"] * loss_bc
             + lambdas["trace"] * loss_trace)
    return total, {
        "pde": loss_pde.detach(),
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "trace": loss_trace.detach(),
        "total": total.detach(),
        "consistency": torch.tensor(0.0),
    }


__all__ = ["poisson_total_loss", "heat_total_loss",
           "poisson_nested_loss", "heat_nested_loss"]
