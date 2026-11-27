"""Classical priors adapted from Pep-Q-GAN's samples/get_samples.py.

Gaussian: independent N(0, 1) coordinates.
Coordinates have theoretical zero mean and unit variance; do not standardize batches.
Original algorithm: Engdal et al., DOI 10.64898/2026.07.09.736951 (Apache-2.0).
Default sampling uses a private CPU RNG. The global_device option uses upstream
device-native draws and requires one isolated training process per experiment.
"""

import torch


class _ClassicalPrior:
    def __init__(self, z_dim: int = 32, seed: int = 46, rng_mode: str = "private_cpu"):
        if type(z_dim) is not int or z_dim < 1:
            raise ValueError("z_dim must be a positive integer")
        self.z_dim = z_dim
        self.seed = seed
        if rng_mode not in ("private_cpu", "global_device"):
            raise ValueError("Unknown RNG mode")
        self.rng_mode = rng_mode
        self._rng = torch.Generator(device="cpu").manual_seed(seed)

    def sample(
        self, batch_size: int, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        """Return float32 latent vectors, shape [batch_size, z_dim], without gradients.

        private_cpu draws on CPU, then transfers to the requested device.
        global_device consumes the device's global RNG, without reseeding here.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.rng_mode == "global_device":
            shape = (batch_size, self.z_dim)
            return torch.randn(shape, device=device)
        return self._draw((batch_size, self.z_dim)).to(device)

    def __call__(
        self, batch_size: int, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        return self.sample(batch_size, device=device)

    def _draw(self, shape: tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError


class GaussianPrior(_ClassicalPrior):
    """Factorized standard normal prior, matching upstream torch.randn."""

    def _draw(self, shape: tuple[int, int]) -> torch.Tensor:
        return torch.randn(shape, generator=self._rng, dtype=torch.float32)
