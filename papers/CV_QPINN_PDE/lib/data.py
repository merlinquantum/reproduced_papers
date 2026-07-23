"""Collocation-point generation and reference solutions for the two PDEs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.stats import qmc


@dataclass
class PoissonProblem:
    """1D Poisson: u''(x) + sin(4 x) = 0, u(0) = u(pi/2) = 0."""

    x_min: float = 0.0
    x_max: float = math.pi / 2

    def analytic(self, x: torch.Tensor) -> torch.Tensor:
        """Exact solution u(x) = sin(4 x) / 16.

        The Poisson equation u'' = -sin(4 x) integrates to
        u' = cos(4 x) / 4 + C1 and u = sin(4 x) / 16 + C1 x + C2.
        The boundary conditions u(0) = u(pi/2) = 0 force C1 = C2 = 0
        because sin(2 pi) = 0.
        """
        return torch.sin(4 * x) / 16.0

    def analytic_grad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(4 * x) / 4.0

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(4 * x)


@dataclass
class HeatProblem:
    """1D heat equation, T_t = alpha T_xx on [-pi/2, pi/2]."""

    x_min: float = -math.pi / 2
    x_max: float = math.pi / 2
    t_min: float = 0.0
    t_max: float = 0.5
    alpha: float = 0.30
    sigma_sq: float = 0.2
    mu: float = -math.pi / 8

    def initial(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma_sq))

    def reference_solution(self, nx: int = 41, nt: int = 11) -> dict:
        """Compute the RK45 reference solution of the heat equation.

        Uses a centred-difference spatial discretisation with Dirichlet BCs and
        SciPy's ``solve_ivp`` (RK45) for the time integration. Returns a dict
        with arrays ``x``, ``t``, and ``T`` of shapes ``(nx,)``, ``(nt,)`` and
        ``(nt, nx)``.
        """
        x = np.linspace(self.x_min, self.x_max, nx)
        dx = x[1] - x[0]
        T0 = 0.5 * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma_sq))
        T0[0] = 0.0
        T0[-1] = 0.0

        def rhs(_t: float, T: np.ndarray) -> np.ndarray:
            d2 = np.zeros_like(T)
            d2[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx ** 2
            return self.alpha * d2

        t_eval = np.linspace(self.t_min, self.t_max, nt)
        sol = solve_ivp(rhs, (self.t_min, self.t_max), T0, method="RK45",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
        return {"x": x, "t": t_eval, "T": sol.y.T}


def sobol_1d(n: int, x_min: float, x_max: float, *, seed: int = 0) -> torch.Tensor:
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    pts = sampler.random(n=n).squeeze(-1)
    return torch.tensor(x_min + (x_max - x_min) * pts, dtype=torch.float64)


def sobol_2d(n: int, x_min: float, x_max: float, t_min: float, t_max: float,
             *, seed: int = 0) -> torch.Tensor:
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    pts = sampler.random(n=n)
    x = x_min + (x_max - x_min) * pts[:, 0]
    t = t_min + (t_max - t_min) * pts[:, 1]
    return torch.tensor(np.stack([x, t], axis=1), dtype=torch.float64)


def regular_grid_2d(nx: int, nt: int, x_min: float, x_max: float,
                    t_min: float, t_max: float) -> torch.Tensor:
    x = torch.linspace(x_min, x_max, nx, dtype=torch.float64)
    t = torch.linspace(t_min, t_max, nt, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="xy")
    return torch.stack([xx.flatten(), tt.flatten()], dim=1)


__all__ = [
    "PoissonProblem",
    "HeatProblem",
    "sobol_1d",
    "sobol_2d",
    "regular_grid_2d",
]
