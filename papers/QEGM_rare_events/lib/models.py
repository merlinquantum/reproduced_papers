"""QEGM model variants used in this reproduction.

Three nn.Modules with the same encoder/decoder skeleton but different
quantum-randomness modulators:

* ``VAEBaseline`` — standard Gaussian-prior VAE (no quantum component);
  the classical fair baseline for the synthetic GMM task.
* ``QEGMGate`` — VAE with a gate-based hardware-efficient VQC modulating
  the latent noise variance through ``z̃ = z + ε, ε ~ N(0, σ² r)``
  (paper Eq. 7) and a learnable σ.
* ``QEGMMerlin`` — same skeleton, with the photonic ``MerlinPhotonicLayer``
  in place of the gate-based circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .merlin_layer import MerlinPhotonicLayer
from .vqc import HardwareEfficientVQC


class _MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.mu(h), self.log_var(h)


class _MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAEBaseline(nn.Module):
    """Fair-baseline VAE with the same encoder/decoder as QEGM."""

    variant = "vae"

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = _MLPEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = _MLPDecoder(latent_dim, hidden_dim, in_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> dict:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return {"x_hat": x_hat, "mu": mu, "log_var": log_var, "z": z, "r": None}

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


class _QEGMBase(VAEBaseline):
    """VAE-style backbone with a quantum-randomness modulated noise term."""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__(in_dim, hidden_dim, latent_dim)
        # σ scale for the quantum-noise channel (paper Eq. 7, σ²r perturbation).
        self.log_sigma_q = nn.Parameter(torch.tensor(-1.0))

    def _quantum_randomness(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict:
        mu, log_var = self.encoder(x)
        z_base = self.reparameterize(mu, log_var)
        r = self._quantum_randomness(z_base)
        sigma = torch.exp(self.log_sigma_q)
        eps = torch.randn_like(z_base)
        # Eq. 7: z̃ = z + ε, ε ~ N(0, σ²r); apply per-latent r.
        z_q = z_base + sigma * torch.sqrt(r + 1e-6) * eps
        x_hat = self.decoder(z_q)
        return {"x_hat": x_hat, "mu": mu, "log_var": log_var, "z": z_q, "r": r}

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        r = self._quantum_randomness(z)
        sigma = torch.exp(self.log_sigma_q)
        eps = torch.randn_like(z)
        z_q = z + sigma * torch.sqrt(r + 1e-6) * eps
        return self.decoder(z_q)


class QEGMGate(_QEGMBase):
    """QEGM with a gate-based hardware-efficient VQC."""

    variant = "qegm"

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_qubits: int,
        n_layers: int,
    ):
        if latent_dim != n_qubits:
            raise ValueError("Gate QEGM requires latent_dim == n_qubits")
        super().__init__(in_dim, hidden_dim, latent_dim)
        self.vqc = HardwareEfficientVQC(n_qubits=n_qubits, n_layers=n_layers)

    def _quantum_randomness(self, z: torch.Tensor) -> torch.Tensor:
        return self.vqc(z)


class QEGMMerlin(_QEGMBase):
    """QEGM with a MerLin photonic layer as the quantum-randomness source."""

    variant = "qegm_merlin"

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_modes: int,
        n_photons: int,
        encoding_scale: float,
    ):
        super().__init__(in_dim, hidden_dim, latent_dim)
        self.merlin = MerlinPhotonicLayer(
            n_qubits=latent_dim,
            n_modes=n_modes,
            n_photons=n_photons,
            encoding_scale=encoding_scale,
        )

    def hardware_settings(self) -> dict:
        return self.merlin.hardware_settings()

    def _quantum_randomness(self, z: torch.Tensor) -> torch.Tensor:
        return self.merlin(z)


class QEGMConst(_QEGMBase):
    """Ablation: same QEGM skeleton but with ``r`` pinned to a constant.

    If Eq. 7 carries any information through the VQC, ``QEGMGate`` should
    beat this variant. If they match within seed variance, the
    quantum-randomness modulation is provably inert: the layer reduces
    to a VAE with a re-scaled σ_q.
    """

    variant = "qegm_const"

    def __init__(
        self, in_dim: int, hidden_dim: int, latent_dim: int, r_value: float = 0.5
    ):
        super().__init__(in_dim, hidden_dim, latent_dim)
        self.register_buffer("r_const", torch.tensor(float(r_value)))

    def _quantum_randomness(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        return self.r_const.expand(batch, self.latent_dim).clone()


def build_model(variant: str, cfg: dict, in_dim: int) -> nn.Module:
    """Instantiate one of the three model variants from a config dict."""

    model_cfg = cfg["model"]
    hidden_dim = int(model_cfg["hidden_dim"])
    latent_dim = int(model_cfg["latent_dim"])
    if variant == "vae":
        return VAEBaseline(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    if variant == "qegm":
        vqc_cfg = model_cfg["vqc"]
        return QEGMGate(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_qubits=int(vqc_cfg["n_qubits"]),
            n_layers=int(vqc_cfg["n_layers"]),
        )
    if variant == "qegm_merlin":
        merlin_cfg = model_cfg["merlin"]
        return QEGMMerlin(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_modes=int(merlin_cfg["n_modes"]),
            n_photons=int(merlin_cfg["n_photons"]),
            encoding_scale=float(merlin_cfg["encoding_scale"]),
        )
    if variant == "qegm_const":
        r_value = float(model_cfg.get("const_r_value", 0.5))
        return QEGMConst(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            r_value=r_value,
        )
    raise ValueError(f"Unknown model variant: {variant!r}")
