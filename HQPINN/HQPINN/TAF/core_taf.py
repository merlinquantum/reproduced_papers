"""
Core training utilities for TAF (2D transonic aerofoil flow, Sec. 3.3).
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import (
    DEVICE,
    DTYPE,
    GAMMA,
    TAF_EPSILON_LAMBDA,
    TAF_LBFGS_STEPS,
    TAF_P_OUT,
    TAF_R_GAS,
    TAF_X_BOT_FILE,
    TAF_X_DATA_INT_FILE,
    TAF_X_F_FILE,
    TAF_X_IN_FILE,
    TAF_X_OUT_FILE,
    TAF_X_TOP_FILE,
    TAF_X_WALL_FILE,
    TAF_X_WALL_NORMALS_FILE,
)
from ..utils import count_trainable_params


DATA_DIR = Path(__file__).resolve().parent / "NACA0012"


def load_points(path: str) -> torch.Tensor:
    """Load .npy points on configured dtype/device."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = DATA_DIR / candidate
    arr = np.load(candidate)
    return torch.tensor(arr, dtype=DTYPE, device=DEVICE)


def load_training_sets() -> dict[str, torch.Tensor]:
    """Load TAF geometry/training point sets."""
    return {
        "X_in": load_points(TAF_X_IN_FILE),
        "X_out": load_points(TAF_X_OUT_FILE),
        "X_top": load_points(TAF_X_TOP_FILE),
        "X_bot": load_points(TAF_X_BOT_FILE),
        "X_wall": load_points(TAF_X_WALL_FILE),
        "X_data_int": load_points(TAF_X_DATA_INT_FILE),
        "X_f": load_points(TAF_X_F_FILE),
        "X_wall_normals": load_points(TAF_X_WALL_NORMALS_FILE),
    }


def unpack_primitives(
    raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map network output to (rho, u, v, T)."""
    # TAF networks always output 4 channels in this order:
    # density, x-velocity, y-velocity, temperature.
    # Keeping this helper centralizes the convention in one place.
    rho = raw[:, 0:1]
    u = raw[:, 1:2]
    v = raw[:, 2:3]
    temp = raw[:, 3:4]
    return rho, u, v, temp


def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return gradient of scalar field y(N,1) wrt x(N,2) -> (N,2)."""
    assert y.ndim == 2 and y.shape[1] == 1
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grads


def divergence_uv(u: torch.Tensor, v: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Compute div(u,v) = u_x + v_y."""
    # Velocity divergence is a local compression/expansion indicator.
    # In compressive zones (negative divergence), shocks are likely.
    du = grad_scalar(u, xy)
    dv = grad_scalar(v, xy)
    return du[:, 0:1] + dv[:, 1:2]


def euler_residual(
    model: nn.Module, xy_in: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Steady 2D Euler residual for the TAF case (paper Sec. 3.3),
    written in conservative form on (x, y):
        dG1/dx + dG2/dy = 0.
    Here G1 is the x-flux vector and G2 is the y-flux vector.
    """
    xy = xy_in.clone().detach().to(DTYPE).to(DEVICE)
    xy.requires_grad_(True)

    raw = model(xy)
    rho, u, v, temp = unpack_primitives(raw)

    # Primitive -> thermodynamic quantities used in Euler fluxes.
    # p = rho * R * T (ideal gas), E = e + kinetic energy.
    p = rho * TAF_R_GAS * temp
    E = p / ((GAMMA - 1.0) * rho) + 0.5 * (u**2 + v**2)

    rho_u = rho * u
    rho_v = rho * v

    # Conservative flux in x-direction:
    # G1 = [rho*u, rho*u^2 + p, rho*u*v, u*(rho*E + p)].
    G1 = torch.cat(
        [
            rho_u,
            p + rho * u**2,
            rho * u * v,
            u * (rho * E + p),
        ],
        dim=1,
    )
    # Conservative flux in y-direction:
    # G2 = [rho*v, rho*u*v, rho*v^2 + p, v*(rho*E + p)].
    G2 = torch.cat(
        [
            rho_v,
            rho * u * v,
            p + rho * v**2,
            v * (rho * E + p),
        ],
        dim=1,
    )

    residuals = []
    for i in range(4):
        # For each conservation law i, compute:
        # Ri = dG1_i/dx + dG2_i/dy.
        dG1 = grad_scalar(G1[:, i : i + 1], xy)
        dG2 = grad_scalar(G2[:, i : i + 1], xy)
        residuals.append(dG1[:, 0:1] + dG2[:, 1:2])

    # R has 4 columns: mass, x-momentum, y-momentum, total-energy residuals.
    R = torch.cat(residuals, dim=1)
    return R, rho, u, v, temp


def compute_lambda(
    model: nn.Module,
    xy_in: torch.Tensor,
    eps: float = TAF_EPSILON_LAMBDA,
) -> torch.Tensor:
    """
    Shock-adaptive weight used in the TAF PDE loss (paper Sec. 3.3):
        lambda = 1 / (1 + eps * (|div(u)| - div(u))).

    Here div(u) = du/dx + dv/dy.
    Equivalent piecewise form:
      - if div(u) >= 0:  |div|-div = 0      -> lambda = 1
      - if div(u) < 0:   |div|-div = -2*div -> lambda < 1

    So the weight is unchanged in expansion/smooth zones and reduced in
    compression zones, which is the shock-sensitive behavior used by the
    weighted residual formulation.
    """
    xy = xy_in.clone().detach().to(DTYPE).to(DEVICE)
    xy.requires_grad_(True)

    raw = model(xy)
    _, u, v, _ = unpack_primitives(raw)

    # div(u) is computed from model-predicted velocity components (u, v).
    div = divergence_uv(u, v, xy)
    # Direct implementation of lambda from the paper-style weighted loss.
    lam = 1.0 / (eps * (torch.abs(div) - div) + 1.0)
    # Numerical safeguard: keep lambda in (0, 1] and avoid zero weights.
    return torch.clamp(lam, min=1e-4, max=1.0)


mse = nn.MSELoss()


def loss_boundary(
    model: nn.Module, data: dict[str, torch.Tensor], U_in: torch.Tensor
) -> torch.Tensor:
    """
    Boundary terms for the TAF setup of paper Sec. 3.3, using the
    boundary point sets generated in `generate_aerofoil_training_sets.py`:
    - inlet Dirichlet on (rho,u,v,T)
    - outlet pressure on p = Pout
    - wall no-penetration u.n = 0
    - top/bottom periodicity

    Total boundary loss:
        L_bc = L_in + L_out + L_wall + L_per
    """
    X_in = data["X_in"]
    X_out = data["X_out"]
    X_top = data["X_top"]
    X_bot = data["X_bot"]
    X_wall = data["X_wall"]
    X_wall_normals = data["X_wall_normals"]

    # Sec. 3.3 inlet BC, applied on `X_in` (left boundary of the box domain):
    # enforce the prescribed primitive vector U_in = [rho, u, v, T].
    pred_in = model(X_in)
    rho_i, u_i, v_i, T_i = unpack_primitives(pred_in)
    U_pred_in = torch.cat([rho_i, u_i, v_i, T_i], dim=1)
    L_in = mse(U_pred_in, U_in.view(1, 4).expand_as(U_pred_in))

    # Sec. 3.3 outlet BC, applied on `X_out` (right boundary):
    # enforce static pressure p = rho * R * T = Pout.
    pred_out = model(X_out)
    rho_o, _, _, T_o = unpack_primitives(pred_out)
    p_out_pred = rho_o * TAF_R_GAS * T_o
    p_out_target = torch.full_like(p_out_pred, TAF_P_OUT)
    L_out = mse(p_out_pred, p_out_target)

    # Sec. 3.3 wall BC on NACA0012 surface points `X_wall`:
    # impermeability (slip wall), i.e. normal velocity is zero: (u, v) dot n = 0.
    # Wall normals come from `X_wall_normals` generated with the geometry set.
    pred_wall = model(X_wall)
    _, u_w, v_w, _ = unpack_primitives(pred_wall)
    normals = X_wall_normals[:, 2:4]
    u_dot_n = u_w * normals[:, 0:1] + v_w * normals[:, 1:2]
    L_wall = mse(u_dot_n, torch.zeros_like(u_dot_n))

    # Sec. 3.3 far-field closure in this repo: periodic pairing of
    # top and bottom boundaries (`X_top`, `X_bot`) for all primitives.
    pred_top = model(X_top)
    pred_bot = model(X_bot)
    rho_t, u_t, v_t, T_t = unpack_primitives(pred_top)
    rho_b, u_b, v_b, T_b = unpack_primitives(pred_bot)
    U_top = torch.cat([rho_t, u_t, v_t, T_t], dim=1)
    U_bot = torch.cat([rho_b, u_b, v_b, T_b], dim=1)
    L_per = mse(U_top, U_bot)

    return L_in + L_out + L_wall + L_per


def loss_pde(
    model: nn.Module,
    data: dict[str, torch.Tensor],
    eps_lambda: float = TAF_EPSILON_LAMBDA,
) -> torch.Tensor:
    """Weighted PDE residual term."""
    X_f = data["X_f"]
    R, _, _, _, _ = euler_residual(model, X_f)
    # Same collocation points, different role:
    # - R: Euler residuals to minimize
    # - lam: shock-aware per-point weight
    lam = compute_lambda(model, X_f, eps=eps_lambda)
    R2 = torch.sum(R**2, dim=1, keepdim=True)
    return torch.mean(lam * R2)


def log_training_info(
    epoch: int,
    elapsed: float,
    loss: torch.Tensor,
    loss_bc: torch.Tensor,
    loss_f: torch.Tensor,
    rows: list[list[str]],
) -> None:
    """Console log + in-memory CSV row append."""
    print(
        f"Step {epoch:6d} | elapsed={elapsed:.2f}s | "
        f"L={loss.item():.3e} | "
        f"BC={loss_bc.item():.3e} | "
        f"F={loss_f.item():.3e}"
    )

    rows.append(
        [
            str(epoch),
            f"{elapsed:.2f}",
            f"{loss.item():.3e}",
            f"{loss_bc.item():.3e}",
            f"{loss_f.item():.3e}",
        ]
    )


def train_taf(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    plot_every: int,
    out_dir: str,
    model_label: str,
    data: dict[str, torch.Tensor],
    U_in: torch.Tensor,
    lbfgs_steps: int = TAF_LBFGS_STEPS,
    eps_lambda: float = TAF_EPSILON_LAMBDA,
) -> Tuple[float, float, float, int]:
    """Train TAF model and return summary metrics."""
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(out_dir, f"taf-{model_label}_{timestamp}.csv")

    rows: list[list[str]] = []
    start = datetime.now()

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        # Paper objective:
        #   L = L_bc + L_f
        L_bc = loss_boundary(model, data, U_in)
        L_f = loss_pde(model, data, eps_lambda=eps_lambda)
        L_total = L_bc + L_f
        L_total.backward()
        optimizer.step()

        if epoch % plot_every == 0:
            elapsed = (datetime.now() - start).total_seconds()
            log_training_info(epoch, elapsed, L_total, L_bc, L_f, rows)

    if lbfgs_steps > 0:
        print("Switching to L-BFGS...")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_steps,
            tolerance_grad=1e-7,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            L_bc_c = loss_boundary(model, data, U_in)
            L_f_c = loss_pde(model, data, eps_lambda=eps_lambda)
            L_total_c = L_bc_c + L_f_c
            L_total_c.backward()
            return L_total_c

        optimizer_lbfgs.step(closure)

    # Do not use torch.no_grad() here: PDE loss relies on autograd
    # to evaluate spatial derivatives in euler_residual().
    L_bc = loss_boundary(model, data, U_in)
    L_f = loss_pde(model, data, eps_lambda=eps_lambda)
    L_total = L_bc + L_f

    elapsed = (datetime.now() - start).total_seconds()
    log_training_info(n_epochs, elapsed, L_total, L_bc, L_f, rows)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "elapsed (s)", "Loss", "BC", "F"])
        writer.writerows(rows)

    n_params = count_trainable_params(model)
    print(f"CSV saved to: {csv_path}")

    return (
        float(L_total.item()),
        float(L_bc.item()),
        float(L_f.item()),
        n_params,
    )
