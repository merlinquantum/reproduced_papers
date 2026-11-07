"""Multi-layer continuous-variable QPINN used by the paper.

The architecture follows Killoran et al. (Phys. Rev. Research 1, 033063):

    Layer = K . D . U2 . S . U1

For two qumodes a *multi-qumode* layer instantiates U1 and U2 as
two-mode interferometers (beam splitter + phase shifts), while a
*single-qumode* layer reduces U1 and U2 to per-mode phase shifters
(no mode mixing). The paper uses 4 multi + 4 single layers for the
Poisson experiment and 2 + 2 for the heat experiment.

Inputs are encoded as real displacements on the relevant qumodes:

    |psi_in> = D(x_1) D(x_2) |0, 0>   (heat case)
    |psi_in> = D(x)        |0, 0>     (Poisson case, x on mode 0)

The network returns one homodyne expectation per qumode, which the
training loop interprets as (u, du/dx) via the consistency loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .cv_simulator import (
    CVOperators,
    apply_single_mode,
    apply_two_mode,
    beamsplitter,
    displacement,
    expectation_x,
    kerr,
    rotation,
    squeezing,
    state_norm_sq,
    vacuum_state,
)


def _param(
    shape: tuple[int, ...] | int, sd: float, generator: torch.Generator
) -> nn.Parameter:
    shape_t = (shape,) if isinstance(shape, int) else shape
    t = sd * torch.randn(shape_t, dtype=torch.float64, generator=generator)
    return nn.Parameter(t)


@dataclass
class QPINNConfig:
    n_qumodes: int = 2
    n_multi_layers: int = 4
    n_single_layers: int = 4
    cutoff: int = 10
    active_sd: float = 0.001
    passive_sd: float = 0.1
    seed: int = 42


class CVMultiQumodeLayer(nn.Module):
    """One Killoran block for ``n_qumodes`` (>=2) with mode-mixing interferometers."""

    def __init__(
        self,
        n_qumodes: int,
        active_sd: float,
        passive_sd: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        assert n_qumodes == 2, "Reference implementation supports 2 qumodes"
        self.n_qumodes = n_qumodes
        # interferometer U_k for two modes: 1 BS (theta, phi) + 2 final phases (alpha, beta).
        self.theta1 = _param((), passive_sd, generator)
        self.phi1 = _param((), passive_sd, generator)
        self.alpha1 = _param((), passive_sd, generator)
        self.beta1 = _param((), passive_sd, generator)
        self.r = _param(n_qumodes, active_sd, generator)
        self.theta2 = _param((), passive_sd, generator)
        self.phi2 = _param((), passive_sd, generator)
        self.alpha2 = _param((), passive_sd, generator)
        self.beta2 = _param((), passive_sd, generator)
        self.d_alpha = _param(n_qumodes, active_sd, generator)
        self.kappa = _param(n_qumodes, active_sd, generator)

    def forward(self, state: torch.Tensor, ops: CVOperators) -> torch.Tensor:
        BS1 = beamsplitter(self.theta1, self.phi1, ops)
        state = apply_two_mode(state, BS1, (0, 1), self.n_qumodes)
        R1a = rotation(self.alpha1, ops)
        R1b = rotation(self.beta1, ops)
        state = apply_single_mode(state, R1a, 0, self.n_qumodes)
        state = apply_single_mode(state, R1b, 1, self.n_qumodes)
        S0 = squeezing(self.r[0], ops)
        S1 = squeezing(self.r[1], ops)
        state = apply_single_mode(state, S0, 0, self.n_qumodes)
        state = apply_single_mode(state, S1, 1, self.n_qumodes)
        BS2 = beamsplitter(self.theta2, self.phi2, ops)
        state = apply_two_mode(state, BS2, (0, 1), self.n_qumodes)
        R2a = rotation(self.alpha2, ops)
        R2b = rotation(self.beta2, ops)
        state = apply_single_mode(state, R2a, 0, self.n_qumodes)
        state = apply_single_mode(state, R2b, 1, self.n_qumodes)
        D0 = displacement(self.d_alpha[0], ops)
        D1 = displacement(self.d_alpha[1], ops)
        state = apply_single_mode(state, D0, 0, self.n_qumodes)
        state = apply_single_mode(state, D1, 1, self.n_qumodes)
        K0 = kerr(self.kappa[0], ops)
        K1 = kerr(self.kappa[1], ops)
        state = apply_single_mode(state, K0, 0, self.n_qumodes)
        state = apply_single_mode(state, K1, 1, self.n_qumodes)
        return state


class CVSingleQumodeLayer(nn.Module):
    """One Killoran block per qumode, with no inter-mode mixing."""

    def __init__(
        self,
        n_qumodes: int,
        active_sd: float,
        passive_sd: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.n_qumodes = n_qumodes
        self.phi_a = _param(n_qumodes, passive_sd, generator)
        self.r = _param(n_qumodes, active_sd, generator)
        self.phi_b = _param(n_qumodes, passive_sd, generator)
        self.d_alpha = _param(n_qumodes, active_sd, generator)
        self.kappa = _param(n_qumodes, active_sd, generator)

    def forward(self, state: torch.Tensor, ops: CVOperators) -> torch.Tensor:
        for m in range(self.n_qumodes):
            R1 = rotation(self.phi_a[m], ops)
            state = apply_single_mode(state, R1, m, self.n_qumodes)
            S = squeezing(self.r[m], ops)
            state = apply_single_mode(state, S, m, self.n_qumodes)
            R2 = rotation(self.phi_b[m], ops)
            state = apply_single_mode(state, R2, m, self.n_qumodes)
            D = displacement(self.d_alpha[m], ops)
            state = apply_single_mode(state, D, m, self.n_qumodes)
            K = kerr(self.kappa[m], ops)
            state = apply_single_mode(state, K, m, self.n_qumodes)
        return state


class QPINN(nn.Module):
    """Two-output CVQNN for the consistency-loss PINN scheme.

    For Poisson the input is a single scalar `x` encoded by D(x) on
    mode 0 only. For the heat equation `x` is encoded on mode 0 and
    `t` on mode 1.
    """

    def __init__(
        self,
        cfg: QPINNConfig,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.ops = CVOperators(
            d=cfg.cutoff, dtype=dtype, device=device or torch.device("cpu")
        )
        gen = torch.Generator().manual_seed(cfg.seed)
        self.multi_layers = nn.ModuleList(
            [
                CVMultiQumodeLayer(cfg.n_qumodes, cfg.active_sd, cfg.passive_sd, gen)
                for _ in range(cfg.n_multi_layers)
            ]
        )
        self.single_layers = nn.ModuleList(
            [
                CVSingleQumodeLayer(cfg.n_qumodes, cfg.active_sd, cfg.passive_sd, gen)
                for _ in range(cfg.n_single_layers)
            ]
        )
        vac = vacuum_state(cfg.n_qumodes, cfg.cutoff, dtype, self.ops.device)
        self.register_buffer("vac", vac)

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Build the input state for a batch of `inputs`.

        inputs has shape (B,) for 1-variable problems and (B, 2) for 2-variable
        problems. Each variable is encoded as a displacement on a different
        qumode.
        """
        d = self.cfg.cutoff
        if inputs.dim() == 1:
            B = inputs.shape[0]
            # Encode on mode 0 only; mode 1 stays in vacuum.
            state = self.vac.unsqueeze(0).expand(B, d, d).contiguous()
            D0 = displacement(inputs, self.ops)
            state = apply_single_mode(state, D0, 0, self.cfg.n_qumodes)
            return state
        elif inputs.dim() == 2 and inputs.shape[1] == 2:
            B = inputs.shape[0]
            state = self.vac.unsqueeze(0).expand(B, d, d).contiguous()
            D0 = displacement(inputs[:, 0], self.ops)
            D1 = displacement(inputs[:, 1], self.ops)
            state = apply_single_mode(state, D0, 0, self.cfg.n_qumodes)
            state = apply_single_mode(state, D1, 1, self.cfg.n_qumodes)
            return state
        raise ValueError(f"Unsupported input shape {inputs.shape}")

    def _run_circuit(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.multi_layers:
            state = layer(state, self.ops)
        for layer in self.single_layers:
            state = layer(state, self.ops)
        return state

    def forward_state(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._run_circuit(self._encode(inputs))

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (u, ux, trace) for a batch of inputs.

        u  = <psi|X_0|psi>
        ux = <psi|X_1|psi>
        trace = <psi|psi> (used by the trace loss for normalisation).
        """
        state = self.forward_state(inputs)
        u = expectation_x(state, 0, self.cfg.n_qumodes, self.ops)
        ux = expectation_x(state, 1, self.cfg.n_qumodes, self.ops)
        trace = state_norm_sq(state, self.cfg.n_qumodes)
        return u, ux, trace

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["QPINN", "QPINNConfig", "CVMultiQumodeLayer", "CVSingleQumodeLayer"]
