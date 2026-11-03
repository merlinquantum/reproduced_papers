"""Synthetic Fourier-series targets of paper Figs. 3-4 and their data loaders.

The paper has no external dataset: its "data" are closed-form truncated Fourier
series sampled on a fixed grid (LOG.md, "Phase 2 generator specification").
A target is

    g(x) = c_0 + sum_{n=1}^{D} (c_n exp(i sign s n x) + conj(c_n) exp(-i sign s n x)),

which is real by construction. ``sign`` selects the caption's exponent
convention (Fig. 3 uses ``exp(-inx)``, Fig. 4 uses ``exp(+2inx)``) and ``s`` is
the target's input scaling. Nothing is read from or written to the data root.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# The complex sum is real up to rounding; a larger imaginary part means the
# coefficients were not paired with their conjugates.
_MAX_IMAGINARY_PART = 1e-12


def build_grid(
    grid_cfg: Mapping[str, Any], dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Return the evaluation grid ``linspace(start, stop, num)``.

    Parameters
    ----------
    grid_cfg : Mapping
        Keys ``start``, ``stop`` and ``num``. The paper grid is
        ``linspace(-6, 6, 70)`` (reference notebook, cell 20).
    dtype : torch.dtype
        Real dtype of the grid. Default value is ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Grid of shape ``(num,)``.

    Raises
    ------
    KeyError
        If a grid key is missing.
    """
    return torch.linspace(
        float(grid_cfg["start"]),
        float(grid_cfg["stop"]),
        int(grid_cfg["num"]),
        dtype=dtype,
    )


def target_coefficients(target_cfg: Mapping[str, Any]) -> tuple[float, list[complex]]:
    """Parse ``c_0`` and the complex coefficients ``c_1 ... c_D`` of a target.

    Parameters
    ----------
    target_cfg : Mapping
        Keys ``c0`` and ``coefficients`` (a list of ``{"re": ..., "im": ...}``).

    Returns
    -------
    tuple of (float, list of complex)
        ``c_0`` and the positive-frequency coefficients in order ``n = 1..D``.

    Raises
    ------
    KeyError
        If a key is missing.
    """
    constant = float(target_cfg["c0"])
    coefficients = [
        complex(float(entry["re"]), float(entry["im"]))
        for entry in target_cfg["coefficients"]
    ]
    return constant, coefficients


def evaluate_target(
    inputs: torch.Tensor, target_cfg: Mapping[str, Any]
) -> torch.Tensor:
    """Evaluate a target Fourier series on ``inputs``.

    Parameters
    ----------
    inputs : torch.Tensor
        Real inputs of shape ``(N,)``.
    target_cfg : Mapping
        Keys ``c0``, ``coefficients``, ``sign`` (``+1`` or ``-1``) and
        ``scaling``.

    Returns
    -------
    torch.Tensor
        Real target values of shape ``(N,)`` with the dtype of ``inputs``.

    Raises
    ------
    KeyError
        If a key is missing.
    ValueError
        If ``sign`` is not ``+1`` or ``-1``, or the sum is not real.
    """
    sign = int(target_cfg["sign"])
    if sign not in (1, -1):
        raise ValueError(f"target.sign must be +1 or -1, got {sign}")
    scaling = float(target_cfg["scaling"])
    constant, coefficients = target_coefficients(target_cfg)
    complex_inputs = inputs.to(torch.complex128)
    values = torch.full_like(complex_inputs, constant)
    for frequency, coefficient in enumerate(coefficients, start=1):
        phase = torch.exp(1j * sign * scaling * frequency * complex_inputs)
        values = values + coefficient * phase + coefficient.conjugate() * phase.conj()
    largest_imaginary = values.imag.abs().max().item()
    if largest_imaginary > _MAX_IMAGINARY_PART:
        raise ValueError(f"Target is not real: max |Im| = {largest_imaginary:.3e}")
    return values.real.to(inputs.dtype)


def build_dataset(
    case_cfg: Mapping[str, Any], dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the full grid and its target values for one fitting case.

    Parameters
    ----------
    case_cfg : Mapping
        Keys ``grid`` and ``target``.
    dtype : torch.dtype
        Real dtype. Default value is ``torch.float64``.

    Returns
    -------
    tuple of torch.Tensor
        Inputs and targets, both of shape ``(num,)``.
    """
    inputs = build_grid(case_cfg["grid"], dtype)
    return inputs, evaluate_target(inputs, case_cfg["target"])


def build_dataloader(
    case_cfg: Mapping[str, Any], split: str, seed: int | None = None
) -> DataLoader:
    """Return the loader for one fitting case.

    The paper (Fig. 4 caption) and the reference notebook train on the full
    grid, drawing each mini-batch of 25 points with replacement
    (``np.random.randint(0, len(x), (batch_size,))``). There is no held-out
    split: ``"eval"`` returns the same full grid in one batch.

    Parameters
    ----------
    case_cfg : Mapping
        Keys ``grid``, ``target`` and ``training`` (``steps``, ``full_batch``,
        and ``batch_size`` when ``full_batch`` is false).
    split : str
        ``"train"`` for one batch per optimizer step: seeded mini-batches with
        replacement (paper protocol), or the whole grid when
        ``training.full_batch`` is true (Phase 4 amendment). ``"eval"`` for the
        whole grid.
    seed : int | None
        Seed of the batch sampler. Required for ``"train"``. Default value is
        None.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(inputs, targets)`` batches.

    Raises
    ------
    ValueError
        If ``split`` is unknown or ``"train"`` is requested without a seed.
    """
    inputs, targets = build_dataset(case_cfg)
    dataset = TensorDataset(inputs, targets)
    if split == "eval":
        return DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    if split != "train":
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'eval'")
    if seed is None:
        raise ValueError("The train loader needs a seed for its batch sampler")
    training_cfg = case_cfg["training"]
    steps = int(training_cfg["steps"])
    if bool(training_cfg["full_batch"]):
        # Phase 4 amendment protocols P1-P3: every step sees the whole grid in order.
        return DataLoader(
            dataset, batch_size=len(dataset), sampler=list(range(len(dataset))) * steps
        )
    batch_size = int(training_cfg["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=batch_size * steps, generator=generator
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
