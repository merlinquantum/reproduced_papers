"""Teacher, dense student and photonic-conditioned dictionary-convolution student.

Architecture (paper Methods "Teacher CNN"; Supp. Note 5):
conv1 5x5 pad 2 -> ReLU -> 2x2 max-pool, conv2 3x3 pad 1 -> ReLU -> 2x2 max-pool,
conv3 3x3 pad 1 -> ReLU -> dropout(0.25), global average pool, linear -> 10.

A compressed layer replaces the dense kernel by the dictionary model of
Eqs. (6)-(7):  W[o, i] = g[o] * sum_r M[o, i, r] B[r],  vec(M) = A z,
with trainable basis B (R x k x k), fixed Gaussian A (variance 1/d) and the
conditioning vector z supplied from outside the module.

The per-output-channel gain ``g`` is an inferred detail: the paper's reported
compression factors (Table 2, Fig. 2 axes) are reproduced only when each
compressed layer carries R*k^2 + 2*C_out trainable parameters, i.e. one
parameter per output channel beyond the bias (LOG.md, Reproducibility
Assessment).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional

KERNEL_SIZES = (5, 3, 3)
PADDINGS = (2, 1, 1)
NUM_CLASSES = 10


class DictionaryConv2d(nn.Module):
    """Convolution whose kernel is a channel-mixed dictionary of spatial filters.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel counts ``C_in`` and ``C_out``.
    kernel_size : int
        Spatial size ``k``.
    padding : int
        Zero padding.
    rank : int
        Number of basis filters ``R``.
    feature_dim : int
        Dimension ``d`` of the conditioning vector ``z``.
    mixing : str
        ``"feature"`` generates ``M = reshape(A z)`` from the conditioning
        vector (PQKD); ``"trainable"`` learns ``M`` directly (Supp. Note 5,
        dictionary-only baseline).
    generator : torch.Generator
        Seeds the fixed projection ``A`` and the basis initialisation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        rank: int,
        feature_dim: int,
        mixing: str,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if mixing not in ("feature", "trainable"):
            raise ValueError(f"mixing must be 'feature' or 'trainable', got {mixing!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.rank = rank
        self.mixing_mode = mixing

        # The paper does not give the basis initialisation. M has unit-variance
        # entries when z is standardised (A has variance 1/d), so this He-style
        # scale gives the reconstructed kernel the variance of a dense He init.
        fan_in = in_channels * kernel_size * kernel_size
        basis_std = math.sqrt(2.0 / (fan_in * rank))
        self.basis = nn.Parameter(
            torch.randn(rank, kernel_size, kernel_size, generator=generator) * basis_std
        )
        bias_bound = 1.0 / math.sqrt(fan_in)
        self.bias = nn.Parameter(
            (torch.rand(out_channels, generator=generator) * 2 - 1) * bias_bound
        )
        self.gain = nn.Parameter(torch.ones(out_channels))

        mixing_shape = (out_channels, in_channels, rank)
        if mixing == "feature":
            projection = torch.randn(
                out_channels * in_channels * rank, feature_dim, generator=generator
            ) / math.sqrt(feature_dim)
            self.register_buffer("projection", projection)
            self.register_buffer("mixing", torch.zeros(mixing_shape))
        else:
            self.mixing = nn.Parameter(torch.randn(mixing_shape, generator=generator))

    @torch.no_grad()
    def set_feature(self, feature: torch.Tensor) -> None:
        """Set ``M = reshape(A z)`` from a conditioning vector ``z`` of shape ``(d,)``."""
        if self.mixing_mode != "feature":
            raise RuntimeError("set_feature is only defined for mixing='feature'")
        mixing = self.projection @ feature.to(self.projection)
        self.mixing.copy_(mixing.view_as(self.mixing))

    def kernel(self, feature: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct the dense kernel ``W`` of shape ``(C_out, C_in, k, k)``.

        With ``feature`` given, ``M = reshape(A z)`` is built inside the graph
        so gradients reach ``z`` (MerLin-native adaptation); otherwise the
        stored mixing from :meth:`set_feature` is used.
        """
        mixing = self.mixing
        if feature is not None:
            if self.mixing_mode != "feature":
                raise RuntimeError("A feature can only drive mixing='feature' layers")
            mixing = (self.projection @ feature.to(self.projection)).view_as(
                self.mixing
            )
        kernel = torch.einsum("oir,rxy->oixy", mixing, self.basis)
        return kernel * self.gain.view(-1, 1, 1, 1)

    def forward(
        self, inputs: torch.Tensor, feature: torch.Tensor | None = None
    ) -> torch.Tensor:
        return functional.conv2d(
            inputs, self.kernel(feature), self.bias, padding=self.padding
        )


class ThreeBlockCNN(nn.Module):
    """Teacher / student CNN with optional dictionary-compressed conv layers.

    Parameters
    ----------
    in_channels : int
        Image channels (1 for MNIST-like, 3 for CIFAR-10).
    widths : sequence of int
        ``(c1, c2, c3)``.
    dropout : float
        Dropout after conv3 (paper: 0.25).
    compressed : sequence of bool, optional
        Which of conv1..conv3 use :class:`DictionaryConv2d`. Default value is
        None (all dense, i.e. the teacher).
    ranks : sequence of int, optional
        Basis rank ``R_l`` per layer, required when any layer is compressed.
        Default value is None.
    feature_dim : int, optional
        Conditioning dimension ``d``. Default value is None.
    mixing : str
        Mixing mode for compressed layers. Default value is ``"feature"``.
    generator : torch.Generator, optional
        Seeds dictionary layers. Required when any layer is compressed.
        Default value is None.
    """

    def __init__(
        self,
        in_channels: int,
        widths: Sequence[int],
        dropout: float,
        compressed: Sequence[bool] | None = None,
        ranks: Sequence[int] | None = None,
        feature_dim: int | None = None,
        mixing: str = "feature",
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if len(widths) != 3:
            raise ValueError(f"widths must have three entries, got {widths}")
        compressed = (
            tuple(compressed) if compressed is not None else (False, False, False)
        )
        if len(compressed) != 3:
            raise ValueError(f"compressed must have three entries, got {compressed}")
        if any(compressed) and (
            ranks is None or feature_dim is None or generator is None
        ):
            raise ValueError("compressed layers need ranks, feature_dim and generator")
        self.widths = tuple(int(width) for width in widths)
        self.compressed = compressed
        channels = (in_channels, *self.widths)
        convs: list[nn.Module] = []
        for index in range(3):
            if compressed[index]:
                convs.append(
                    DictionaryConv2d(
                        channels[index],
                        channels[index + 1],
                        KERNEL_SIZES[index],
                        PADDINGS[index],
                        int(ranks[index]),
                        int(feature_dim),
                        mixing,
                        generator,
                    )
                )
            else:
                convs.append(
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        KERNEL_SIZES[index],
                        padding=PADDINGS[index],
                    )
                )
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.widths[2], NUM_CLASSES)

    @property
    def uses_feature(self) -> bool:
        """Whether any layer takes its channel mixing from a conditioning vector."""
        return any(
            isinstance(conv, DictionaryConv2d) and conv.mixing_mode == "feature"
            for conv in self.convs
        )

    def set_feature(self, feature: torch.Tensor) -> None:
        """Broadcast one conditioning vector ``z`` to every feature-mixed layer."""
        for conv in self.convs:
            if isinstance(conv, DictionaryConv2d) and conv.mixing_mode == "feature":
                conv.set_feature(feature)

    def _conv(
        self, index: int, inputs: torch.Tensor, feature: torch.Tensor | None
    ) -> torch.Tensor:
        conv = self.convs[index]
        if (
            feature is not None
            and isinstance(conv, DictionaryConv2d)
            and conv.mixing_mode == "feature"
        ):
            return conv(inputs, feature)
        return conv(inputs)

    def forward(
        self, images: torch.Tensor, feature: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Logits; ``feature`` (optional) conditions feature-mixed layers in-graph."""
        hidden = functional.max_pool2d(
            functional.relu(self._conv(0, images, feature)), 2
        )
        hidden = functional.max_pool2d(
            functional.relu(self._conv(1, hidden, feature)), 2
        )
        hidden = self.dropout(functional.relu(self._conv(2, hidden, feature)))
        hidden = functional.adaptive_avg_pool2d(hidden, 1).flatten(1)
        return self.classifier(hidden)


def trainable_parameter_count(module: nn.Module) -> int:
    """Number of trainable parameters (fixed projections ``A`` are buffers)."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def conv_parameter_count(model: ThreeBlockCNN) -> int:
    """Trainable parameters of the convolutional stack only (paper "conv-only")."""
    return sum(trainable_parameter_count(conv) for conv in model.convs)


def teacher_parameter_formula(widths: Sequence[int], in_channels: int = 1) -> int:
    """Closed-form dense parameter count (paper Methods, for ``in_channels = 1``)."""
    c1, c2, c3 = widths
    return (
        25 * in_channels * c1 + c1 + 9 * c1 * c2 + c2 + 9 * c2 * c3 + c3 + 10 * c3 + 10
    )


def iso_parameter_widths(
    target: int, in_channels: int, c3: int
) -> tuple[int, int, int]:
    """Dense widths ``(c1, c2, c3')`` whose parameter count is closest to ``target``.

    Used for the KD-only dense baseline (B1). Candidates keep the teacher's
    proportions: ``(c2/2, c2, c3)`` with ``c3`` fixed, or ``(s, 2s, 2s)`` when
    the target is too small for ``c3`` to stay fixed.
    """
    candidates = [(width2 // 2, width2, c3) for width2 in range(2, 4 * c3 + 1, 2)]
    candidates += [(scale, 2 * scale, 2 * scale) for scale in range(1, (c3 + 1) // 2)]
    return min(
        candidates,
        key=lambda widths: abs(teacher_parameter_formula(widths, in_channels) - target),
    )
