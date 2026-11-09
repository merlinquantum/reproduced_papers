"""Conditioning vectors ``z`` for the dictionary-convolution student.

PQKD conditions every compressed layer on one global vector
``z(theta) = gamma * (Phi(p_hat_theta) - mu) / (sigma + eps)`` (Supp. Eq. for
standardisation). The reference statistics ``(mu, sigma)`` are fixed once
before training.

The paper computes ``(mu, sigma)`` "from an initial set of feature evaluations
at the initial theta". With the paper's initial ``theta = 0`` the beam
splitters are the identity, the output pattern is deterministic and every
``sigma`` would be zero. The reproduction therefore fits ``(mu, sigma)`` on
``reference_evaluations`` shot-limited features at parameters drawn uniformly
in ``[-theta_max, theta_max]`` (LOG.md, Reproducibility Assessment).

Sources (baseline IDs from LOG.md, Fair Baseline Plan):

- ``photonic``: PQKD, ``theta`` learned by SPSA.
- ``photonic_fixed`` (B4): same sampler, ``theta`` frozen.
- ``surrogate`` (B5): classical generator ``cos(G phi + b)`` with
  ``dim(phi) = dim(theta)``, standardised the same way and learned by the same
  SPSA schedule.
- ``random_fixed`` / ``random_resampled`` (B3): ``z ~ N(0, I)`` fixed per run or
  redrawn per epoch.
- ``classical_histogram`` (B6): independent Bernoulli click patterns with the
  photonic per-mode click rates, passed through the same Phi with the same
  shot budget.
- ``zero`` (B0): ``z = 0``. This makes every compressed kernel zero; it is the
  paper's Fig. 4 control, kept for metric agreement only.
- ``photonic_native`` / ``surrogate_native`` (Phase 4 MerLin-native
  adaptation, arms N1-N3): exact, differentiable features whose parameters
  are trained by autograd jointly with the student.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from .photonic import TimeBinFeatureMap

logger = logging.getLogger(__name__)

SOURCES = (
    "photonic",
    "photonic_fixed",
    "surrogate",
    "random_fixed",
    "random_resampled",
    "classical_histogram",
    "zero",
    "photonic_native",
    "surrogate_native",
)
THETA_INIT_MODES = ("constant", "uniform")


class FeatureStandardiser:
    """Elementwise ``gamma * (x - mu) / (sigma + eps)`` with fixed statistics."""

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, gamma: float, eps: float
    ) -> None:
        self.mean = mean
        self.std = std
        self.gamma = gamma
        self.eps = eps

    @classmethod
    def fit(
        cls, samples: torch.Tensor, gamma: float, eps: float
    ) -> FeatureStandardiser:
        """Fit ``mu`` and ``sigma`` on reference features of shape ``(n, d)``."""
        return cls(samples.mean(dim=0), samples.std(dim=0, unbiased=False), gamma, eps)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return self.gamma * (raw - self.mean) / (self.std + self.eps)


class ConditioningSource:
    """Base class: produces one standardised conditioning vector per call.

    Attributes
    ----------
    name : str
        Source key from :data:`SOURCES`.
    feature_dim : int
        Length of ``z``.
    learnable : bool
        Whether SPSA updates :meth:`vector`.
    """

    name = "base"
    learnable = False
    differentiable = False

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        """Mutable state needed to resume training (fixed parts are rebuilt from the seed)."""
        return {}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            raise ValueError(
                f"Source {self.name!r} has no checkpoint state, got {sorted(state)}"
            )

    def vector(self) -> torch.Tensor | None:
        """Current learnable parameter vector, or ``None``."""
        return None

    def set_vector(self, vector: torch.Tensor) -> None:
        raise RuntimeError(f"Source {self.name!r} has no learnable parameters")

    @property
    def parameter_count(self) -> int:
        vector = self.vector()
        return 0 if vector is None or not self.learnable else int(vector.numel())

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        """Draw one conditioning vector (float32, shape ``(feature_dim,)``)."""
        raise NotImplementedError

    def zero_feature(self) -> torch.Tensor:
        return torch.zeros(self.feature_dim)


class PhotonicSource(ConditioningSource):
    """Shot-limited photonic feature ``z(theta)`` (PQKD, or B4 when frozen)."""

    def __init__(
        self,
        feature_map: TimeBinFeatureMap,
        theta: torch.Tensor,
        shots: int,
        standardiser: FeatureStandardiser,
        learnable: bool,
    ) -> None:
        super().__init__(feature_map.feature_dim)
        self.feature_map = feature_map
        self.theta = theta.to(torch.float64).clone()
        self.shots = shots
        self.standardiser = standardiser
        self.learnable = learnable
        self.name = "photonic" if learnable else "photonic_fixed"

    def vector(self) -> torch.Tensor:
        return self.theta.clone()

    def set_vector(self, vector: torch.Tensor) -> None:
        if not self.learnable:
            raise RuntimeError("photonic_fixed keeps theta frozen")
        self.theta = vector.to(torch.float64).clone()

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        raw = self.feature_map.raw_feature(self.theta, self.shots, generator)
        return self.standardiser(raw).to(torch.float32)

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {"theta": self.theta.clone()}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        self.theta = state["theta"].to(torch.float64).clone()


class SurrogateSource(ConditioningSource):
    """Classical iso-parameter generator ``z = std(cos(G phi + b))`` (B5).

    A random-Fourier-feature map keeps the generator bounded and periodic in
    its parameters like interferometer angles, and non-degenerate at
    ``phi = 0``. It has no shot noise.
    """

    name = "surrogate"
    learnable = True

    def __init__(
        self,
        projection: torch.Tensor,
        offset: torch.Tensor,
        phi: torch.Tensor,
        standardiser: FeatureStandardiser,
    ) -> None:
        super().__init__(projection.shape[0])
        self.projection = projection
        self.offset = offset
        self.phi = phi.to(torch.float64).clone()
        self.standardiser = standardiser

    @staticmethod
    def raw(
        projection: torch.Tensor, offset: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        return torch.cos(projection @ phi + offset)

    def vector(self) -> torch.Tensor:
        return self.phi.clone()

    def set_vector(self, vector: torch.Tensor) -> None:
        self.phi = vector.to(torch.float64).clone()

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        raw = self.raw(self.projection, self.offset, self.phi)
        return self.standardiser(raw).to(torch.float32)

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {"phi": self.phi.clone()}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        self.phi = state["phi"].to(torch.float64).clone()


class RandomSource(ConditioningSource):
    """Gaussian ``z ~ N(0, gamma^2 I)``, fixed per run or redrawn per call (B3)."""

    def __init__(
        self, feature_dim: int, gamma: float, resample: bool, generator: torch.Generator
    ) -> None:
        super().__init__(feature_dim)
        self.gamma = gamma
        self.resample = resample
        self.name = "random_resampled" if resample else "random_fixed"
        self._fixed = gamma * torch.randn(feature_dim, generator=generator)

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        if self.resample:
            return self.gamma * torch.randn(self.feature_dim, generator=generator)
        return self._fixed.clone()

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {"fixed": self._fixed.clone()}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        self._fixed = state["fixed"].clone()


class ClassicalHistogramSource(ConditioningSource):
    """Independent-mode Bernoulli click patterns through the photonic Phi (B6)."""

    name = "classical_histogram"

    def __init__(
        self,
        feature_map: TimeBinFeatureMap,
        click_probabilities: torch.Tensor,
        shots: int,
        standardiser: FeatureStandardiser,
    ) -> None:
        super().__init__(feature_map.feature_dim)
        self.feature_map = feature_map
        self.click_probabilities = click_probabilities.to(torch.float64)
        self.shots = shots
        self.standardiser = standardiser

    @staticmethod
    def raw(
        feature_map: TimeBinFeatureMap,
        click_probabilities: torch.Tensor,
        shots: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        uniforms = torch.rand(
            shots, click_probabilities.numel(), generator=generator, dtype=torch.float64
        )
        return feature_map.raw_feature_from_bits(uniforms < click_probabilities)

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        raw = self.raw(
            self.feature_map, self.click_probabilities, self.shots, generator
        )
        return self.standardiser(raw).to(torch.float32)


class ZeroSource(ConditioningSource):
    """``z = 0`` (B0, the paper's Fig. 4 control)."""

    name = "zero"

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        return self.zero_feature()


class NativePhotonicSource(torch.nn.Module, ConditioningSource):
    """MerLin-native adaptation: exact photonic feature, angles trained by autograd.

    The MerLin layer's beam-splitter angles are an ``nn.Parameter`` optimised
    jointly with the student (Phase 4, arms N1/N2). The feature is the
    infinite-shot ``[h1; h2]`` of the same circuit and threshold detection as
    PQKD, standardised with fixed statistics. This is a simulation-only
    capability: hardware would need parameter-shift gradients from samples.
    """

    name = "photonic_native"
    differentiable = True

    def __init__(
        self,
        feature_map: TimeBinFeatureMap,
        initial_angles: torch.Tensor,
        standardiser: FeatureStandardiser,
    ) -> None:
        torch.nn.Module.__init__(self)
        ConditioningSource.__init__(self, feature_map.feature_dim)
        self.feature_map = feature_map
        self.layer = feature_map.layer
        self.standardiser = standardiser
        self.layer.requires_grad_(False)
        with torch.no_grad():
            feature_map.angles.copy_(initial_angles.to(feature_map.angles))
        feature_map.angles.requires_grad_(True)

    @property
    def parameter_count(self) -> int:
        return int(self.feature_map.angles.numel())

    def vector(self) -> torch.Tensor:
        return self.feature_map.angles.detach().clone()

    def differentiable_feature(self) -> torch.Tensor:
        raw = self.feature_map.differentiable_exact_feature()
        return self.standardiser(raw).to(torch.float32)

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        with torch.no_grad():
            return self.differentiable_feature()

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {"angles": self.vector()}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.feature_map.angles.copy_(state["angles"].to(self.feature_map.angles))


class NativeSurrogateSource(torch.nn.Module, ConditioningSource):
    """Classical counterpart of :class:`NativePhotonicSource` (Phase 4, arm N3).

    ``z = standardise(cos(G phi + b))`` with ``dim(phi) = dim(theta)``, trained
    by the same optimiser and learning rate; no shots, no photonics.
    """

    name = "surrogate_native"
    differentiable = True

    def __init__(
        self,
        projection: torch.Tensor,
        offset: torch.Tensor,
        initial_phi: torch.Tensor,
        standardiser: FeatureStandardiser,
    ) -> None:
        torch.nn.Module.__init__(self)
        ConditioningSource.__init__(self, projection.shape[0])
        self.register_buffer("projection", projection)
        self.register_buffer("offset", offset)
        self.phi = torch.nn.Parameter(initial_phi.to(torch.float64).clone())
        self.standardiser = standardiser

    @property
    def parameter_count(self) -> int:
        return int(self.phi.numel())

    def vector(self) -> torch.Tensor:
        return self.phi.detach().clone()

    def differentiable_feature(self) -> torch.Tensor:
        raw = SurrogateSource.raw(self.projection, self.offset, self.phi)
        return self.standardiser(raw).to(torch.float32)

    def feature(self, generator: torch.Generator) -> torch.Tensor:
        with torch.no_grad():
            return self.differentiable_feature()

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {"phi": self.vector()}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.phi.copy_(state["phi"].to(self.phi))


def _required(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required config key {where}.{key}")
    return section[key]


def _uniform(
    shape: tuple[int, ...], bound: float, generator: torch.Generator
) -> torch.Tensor:
    return (torch.rand(shape, generator=generator, dtype=torch.float64) * 2 - 1) * bound


def _initial_vector(
    conditioning: dict[str, Any], size: int, bound: float, generator: torch.Generator
) -> torch.Tensor:
    """Initial theta/phi: the paper's constant (``theta_init``) or ``U[-bound, bound]``."""
    mode = _required(conditioning, "theta_init_mode", "conditioning")
    if mode not in THETA_INIT_MODES:
        raise ValueError(
            f"theta_init_mode must be one of {THETA_INIT_MODES}, got {mode!r}"
        )
    if mode == "uniform":
        return _uniform((size,), bound, generator)
    return torch.full(
        (size,),
        float(_required(conditioning, "theta_init", "conditioning")),
        dtype=torch.float64,
    )


def build_conditioning_source(
    cfg: dict[str, Any], generator: torch.Generator
) -> ConditioningSource:
    """Construct the conditioning source named by ``cfg["conditioning"]["source"]``.

    Parameters
    ----------
    cfg : dict
        Resolved run configuration (``conditioning``, ``photonic``, ``spsa``).
    generator : torch.Generator
        Seeds reference statistics, fixed projections and random features.

    Returns
    -------
    ConditioningSource
        Ready-to-use source with fixed standardisation.

    Raises
    ------
    ValueError
        If the source name is unknown.
    """
    conditioning = _required(cfg, "conditioning", "cfg")
    source = _required(conditioning, "source", "conditioning")
    if source not in SOURCES:
        raise ValueError(
            f"Unknown conditioning source {source!r}; expected one of {SOURCES}"
        )
    feature_dim = int(_required(conditioning, "feature_dim", "conditioning"))
    gamma = float(_required(conditioning, "gamma", "conditioning"))
    eps = float(_required(conditioning, "std_eps", "conditioning"))
    shots = int(_required(conditioning, "shots", "conditioning"))
    theta_dim = int(_required(conditioning, "theta_dim", "conditioning"))
    n_reference = int(_required(conditioning, "reference_evaluations", "conditioning"))
    theta_max = float(_required(cfg["spsa"], "theta_max", "spsa"))

    if source == "zero":
        return ZeroSource(feature_dim)
    if source in ("random_fixed", "random_resampled"):
        return RandomSource(feature_dim, gamma, source == "random_resampled", generator)
    if source in ("surrogate", "surrogate_native"):
        projection = torch.randn(
            feature_dim, theta_dim, generator=generator, dtype=torch.float64
        )
        offset = _uniform((feature_dim,), math.pi, generator) + math.pi
        reference = torch.stack(
            [
                SurrogateSource.raw(
                    projection, offset, _uniform((theta_dim,), theta_max, generator)
                )
                for _ in range(n_reference)
            ]
        )
        standardiser = FeatureStandardiser.fit(reference, gamma, eps)
        phi = _initial_vector(conditioning, theta_dim, theta_max, generator)
        if source == "surrogate_native":
            return NativeSurrogateSource(projection, offset, phi, standardiser)
        return SurrogateSource(projection, offset, phi, standardiser)

    photonic = _required(cfg, "photonic", "cfg")
    feature_map = TimeBinFeatureMap(
        int(_required(photonic, "n_modes", "photonic")),
        theta_dim,
        _required(photonic, "input_state", "photonic"),
    )
    if feature_map.feature_dim != feature_dim:
        raise ValueError(
            f"conditioning.feature_dim={feature_dim} but the photonic feature map gives "
            f"{feature_map.feature_dim} for {feature_map.n_modes} modes"
        )

    if source == "classical_histogram":
        reference_theta = _uniform((theta_dim,), theta_max, generator)
        click_probabilities = feature_map.mode_click_probabilities(reference_theta)
        reference = torch.stack(
            [
                ClassicalHistogramSource.raw(
                    feature_map, click_probabilities, shots, generator
                )
                for _ in range(n_reference)
            ]
        )
        standardiser = FeatureStandardiser.fit(reference, gamma, eps)
        return ClassicalHistogramSource(
            feature_map, click_probabilities, shots, standardiser
        )

    if source == "photonic_native":
        if theta_dim != feature_map.n_angles:
            raise ValueError(
                f"photonic_native trains every angle; theta_dim={theta_dim} must equal "
                f"{feature_map.n_angles} (a multiple of n_modes - 1)"
            )
        # Exact (shots=None) reference features: the native feature has no shot noise.
        reference = torch.stack(
            [
                feature_map.raw_feature(
                    _uniform((theta_dim,), theta_max, generator), None, generator
                )
                for _ in range(n_reference)
            ]
        )
        standardiser = FeatureStandardiser.fit(reference, gamma, eps)
        angles = _initial_vector(conditioning, theta_dim, theta_max, generator)
        return NativePhotonicSource(feature_map, angles, standardiser)

    reference = torch.stack(
        [
            feature_map.raw_feature(
                _uniform((theta_dim,), theta_max, generator), shots, generator
            )
            for _ in range(n_reference)
        ]
    )
    standardiser = FeatureStandardiser.fit(reference, gamma, eps)
    theta = _initial_vector(conditioning, theta_dim, theta_max, generator)
    return PhotonicSource(
        feature_map, theta, shots, standardiser, learnable=source == "photonic"
    )
