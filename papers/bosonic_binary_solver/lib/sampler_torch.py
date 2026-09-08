"""Batched click sources on a torch device.

Same four sources as :mod:`lib.sampler`, same meanings, but drawing for many
circuits at once: the contract is ``U`` of shape ``(B, n, n)`` and ``shots`` per
circuit, returning ``(B, shots, m)`` threshold patterns. One update step of the
paper's algorithm needs ``1 + 2K`` circuits per instance, so batching them is
what turns a size-30 row from tens of hours into minutes.

The boson source needs the GPU Clifford & Clifford sampler; the interference-free
sources are plain torch and run anywhere, which is what makes them testable
against :mod:`lib.sampler` on CPU.
"""

from __future__ import annotations

import torch

SOURCES = ("boson", "shuffled_boson", "distinguishable", "bernoulli")


def _threshold(occupations):
    return (occupations > 0).to(torch.int8)


def draw_clicks(unitaries, input_modes, shots, generator, source, m, rate_scale=1.0,
                sampler_backend="triton", sampler_algo="auto"):
    """Draw threshold patterns for a batch of circuits.

    Parameters
    ----------
    unitaries : torch.Tensor
        Shape ``(B, n, n)``. ``n`` may exceed ``m`` when the loop topology adds
        rail modes; only the first ``m`` modes are detected.
    input_modes : sequence of int
        Indices of the occupied input modes.
    shots : int
        Samples per circuit.
    generator : torch.Generator
        Seeded generator on the same device.
    source : str
        One of :data:`SOURCES`.
    m : int
        Number of detected time bins.
    rate_scale : float
        Click-rate multiplier for the ``bernoulli`` source. Default value is 1.0.

    Returns
    -------
    torch.Tensor
        Shape ``(B, shots, m)``, entries in {0, 1}.
    """
    batch = unitaries.shape[0]

    if source in ("boson", "shuffled_boson"):
        try:
            from cliffordgpu.sampler import clifford_sample
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "the boson source needs the GPU Clifford & Clifford sampler (cliffordgpu). "
                "Install it on the GPU host, or use lib.sampler (Perceval, CPU) for small sizes, "
                "or pick an interference-free source, which is pure torch and runs anywhere."
            ) from error

        occupations = clifford_sample(
            unitaries, list(input_modes), shots, generator=generator,
            algo=sampler_algo, backend=sampler_backend,
        )
        clicks = _threshold(occupations).reshape(batch, shots, -1)[:, :, :m]
        if source == "shuffled_boson":
            # permute each mode's column within its own circuit's block of shots:
            # every per-mode marginal is preserved exactly, the joint is destroyed
            order = torch.argsort(
                torch.rand(clicks.shape, generator=generator, device=clicks.device), dim=1
            )
            clicks = torch.gather(clicks, 1, order)
        return clicks

    probabilities = (unitaries[:, :m, list(input_modes)] ** 2).to(torch.float64)   # (B, m, n_photons)

    if source == "distinguishable":
        # each photon lands independently in a mode drawn from its own column of
        # |U|^2: photon number is conserved, multi-photon interference is not
        normalised = probabilities / probabilities.sum(dim=1, keepdim=True)
        cumulative = torch.cumsum(normalised, dim=1)                              # (B, m, n)
        draw = torch.rand(batch, shots, probabilities.shape[2], generator=generator,
                          device=unitaries.device, dtype=torch.float64)
        landed = (draw[:, :, None, :] > cumulative[:, None, :, :]).sum(dim=2).clamp(max=m - 1)
        clicks = torch.zeros(batch, shots, m, dtype=torch.int8, device=unitaries.device)
        return clicks.scatter_(2, landed, 1)

    if source == "bernoulli":
        # keep only the per-mode click rate and drop every correlation; rate_scale
        # moves the mean click density without touching anything else
        rate = (1.0 - torch.prod(1.0 - probabilities, dim=2)).mul(rate_scale).clamp(0.0, 1.0)
        draw = torch.rand(batch, shots, m, generator=generator, device=unitaries.device, dtype=torch.float64)
        return (draw < rate[:, None, :]).to(torch.int8)

    raise ValueError(f"unknown click source {source!r}; available: {SOURCES}")
