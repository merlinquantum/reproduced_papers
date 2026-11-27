"""Minimal WGAN-GP used for every vector-valued experiment.

The generator follows the paper's design constraint: hidden widths are
*non-decreasing*, which is what makes the network invertible-in-principle and
therefore places it inside the class of generators covered by Theorem 1.

Two details differ per experiment and are therefore configurable rather than
hard-coded:

* **Latent re-injection.** Appendix E describes *"feeding affine transforms of
  the latent code z to all layers"* for the QM9 generator only. The toy
  experiments do not use it, and neither does the authors' released code, so
  ``reinject`` defaults to False.
* **Optimizer and activation.** Appendix D (synthetic datasets) specifies
  *"a RMSProp optimizer with a learning rate of 5e-4"* and *"a ReLU activation
  function in all their hidden layers"*, while Appendix E (QM9) specifies Adam
  at 1e-4 with LeakyReLU. The authors' released 2D code uses Adam with
  ``betas=(0.0, 0.9)`` at 5e-4 and LeakyReLU.

Weight initialisation follows the released code: Xavier uniform with zero
biases.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["Critic", "Generator", "WGANGPConfig", "train_wgan_gp"]

_ACTIVATIONS = {
    "relu": lambda slope: nn.ReLU(),
    "leaky_relu": lambda slope: nn.LeakyReLU(slope),
}


def _xavier_init(module: nn.Module) -> None:
    """Xavier-uniform weights with zero biases, as in the authors' release."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Generator(nn.Module):
    """Feed-forward generator with non-decreasing widths and latent re-injection.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent code.
    out_dim : int
        Dimension of a generated sample.
    hidden : tuple[int, ...]
        Hidden layer widths; must be non-decreasing for Theorem 1 to apply.
    reinject : bool
        If True, an affine transform of ``z`` is added to every hidden
        activation - the paper's *"affine transformations of the latent code z
        fed to all layers"*, which Appendix E specifies for QM9 only. Default
        value is False.
    activation : {"relu", "leaky_relu"}
        Hidden activation. Appendix D uses ReLU, Appendix E LeakyReLU. Default
        value is "leaky_relu".
    negative_slope : float
        Slope of the LeakyReLU, ignored for ReLU. Default value is 0.2.

    Raises
    ------
    ValueError
        If ``hidden`` is not non-decreasing, or ``activation`` is unknown.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden: tuple[int, ...] = (64, 176, 288, 400, 512),
        reinject: bool = False,
        activation: str = "leaky_relu",
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if any(b < a for a, b in zip(hidden, hidden[1:])):
            raise ValueError("hidden widths must be non-decreasing (Theorem 1)")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation!r}")
        self.latent_dim = latent_dim
        self.reinject = reinject
        dims = (latent_dim, *hidden)
        self.blocks = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))
        )
        self.skips = nn.ModuleList(
            nn.Linear(latent_dim, h, bias=False) if reinject else nn.Identity()
            for h in hidden
        )
        self.act = _ACTIVATIONS[activation](negative_slope)
        # No output activation: the released generator "Outputs *unnormalized*
        # coordinates (no final activation)".
        self.head = nn.Linear(hidden[-1], out_dim)
        self.apply(_xavier_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for block, skip in zip(self.blocks, self.skips):
            h = block(h)
            if self.reinject:
                h = h + skip(z)
            h = self.act(h)
        return self.head(h)


class Critic(nn.Module):
    """Feed-forward critic (no normalisation layers -- required by WGAN-GP).

    Parameters
    ----------
    in_dim : int
        Dimension of a data sample.
    hidden : tuple[int, ...]
        Hidden layer widths. Default value is (256, 256).
    activation : {"relu", "leaky_relu"}
        Hidden activation. Default value is "leaky_relu".
    negative_slope : float
        Slope of the LeakyReLU, ignored for ReLU. Default value is 0.2.

    Raises
    ------
    ValueError
        If ``activation`` is unknown.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        activation: str = "leaky_relu",
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation!r}")
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), _ACTIVATIONS[activation](negative_slope)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.apply(_xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class WGANGPConfig:
    """Training hyper-parameters (paper values are the defaults)."""

    iterations: int = 20_000
    batch_size: int = 256
    lr: float = 1e-4
    optimizer: str = "adam"
    betas: tuple[float, float] = (0.0, 0.9)
    n_critic: int = 5
    # The paper never states the gradient-penalty coefficient; the authors'
    # released code uses penalty=10, the WGAN-GP default.
    gp_weight: float = 10.0
    device: str = "cpu"
    log_every: int = 1_000


def _make_optimizer(parameters, cfg: WGANGPConfig):
    """Build the optimizer named by ``cfg.optimizer``.

    Parameters
    ----------
    parameters : Iterable[torch.nn.Parameter]
        Parameters to optimize.
    cfg : WGANGPConfig
        Training configuration.

    Returns
    -------
    torch.optim.Optimizer
        Adam (QM9 and the released 2D code) or RMSProp (Appendix D).

    Raises
    ------
    ValueError
        If ``cfg.optimizer`` is neither "adam" nor "rmsprop".
    """
    if cfg.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=cfg.lr, betas=tuple(cfg.betas))
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=cfg.lr)
    raise ValueError(f"unknown optimizer: {cfg.optimizer!r}")


def _gradient_penalty(critic, real, fake, device):
    eps = torch.rand(real.size(0), 1, device=device)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(mix)
    grad = torch.autograd.grad(
        outputs=score,
        inputs=mix,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(
    data: torch.Tensor,
    latent,
    generator: Generator,
    critic: Critic,
    cfg: WGANGPConfig = WGANGPConfig(),
    callback=None,
):
    """Train ``generator``/``critic`` on ``data`` with latent codes from ``latent``.

    Parameters
    ----------
    data : torch.Tensor
        Real samples, shape ``(n, out_dim)``.
    latent : merlin.LatentDistribution
        Any of the samplers in :mod:`lib.latents`.
    callback : callable | None
        Called as ``callback(step, generator)`` every ``cfg.log_every`` steps.

    Returns
    -------
    dict
        Training history (critic loss, generator loss).
    """
    dev = torch.device(cfg.device)
    generator, critic, data = generator.to(dev), critic.to(dev), data.to(dev)
    opt_g = _make_optimizer(generator.parameters(), cfg)
    opt_d = _make_optimizer(critic.parameters(), cfg)
    hist = {"step": [], "d_loss": [], "g_loss": []}

    def real_batch(n):
        idx = torch.randint(0, data.shape[0], (n,), device=dev)
        return data[idx]

    for step in range(1, cfg.iterations + 1):
        for _ in range(cfg.n_critic):
            real = real_batch(cfg.batch_size)
            with torch.no_grad():
                fake = generator(latent.sample(cfg.batch_size, device=dev))
            d_loss = (
                critic(fake).mean()
                - critic(real).mean()
                + cfg.gp_weight * _gradient_penalty(critic, real, fake, dev)
            )
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        fake = generator(latent.sample(cfg.batch_size, device=dev))
        g_loss = -critic(fake).mean()
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        if step % cfg.log_every == 0 or step == 1:
            hist["step"].append(step)
            hist["d_loss"].append(d_loss.detach().item())
            hist["g_loss"].append(g_loss.detach().item())
            if callback is not None:
                callback(step, generator)
    return hist
