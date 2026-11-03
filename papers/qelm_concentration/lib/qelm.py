"""Glue between the config, the reservoir simulator and the feature extractors."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from runtime_lib.dtypes import DtypeSpec

from .observables import (
    nearest_neighbour_features,
    shadow_gaussian_noise,
    tuning_features,
)
from .reservoir import (
    KickedIsingReservoir,
    ReservoirArchitecture,
    ReservoirHyperparameters,
    sample_block_parameters,
    simulate_pair_rdms,
)


def real_dtype_from_config(cfg: dict) -> torch.dtype:
    """Real torch dtype of the run (``float32`` -> complex64 states, ``float64`` -> complex128).

    Raises
    ------
    ValueError
        If ``cfg["dtype"]`` is unset.
    """
    spec = cfg["dtype"]
    if isinstance(spec, DtypeSpec):
        if spec.torch is None:
            raise ValueError("cfg['dtype'] must name float32 or float64 for this paper")
        return spec.torch
    if spec in ("float32", "float64"):
        return getattr(torch, spec)
    raise ValueError(f"cfg['dtype'] must be float32 or float64, got {spec!r}")


def architecture_from_config(reservoir_cfg: dict, **overrides) -> ReservoirArchitecture:
    """Architecture from ``cfg["reservoir"]`` with optional field overrides (e.g. ``n_qubits``)."""
    arch = ReservoirArchitecture.from_config(reservoir_cfg)
    return dataclasses.replace(arch, **overrides) if overrides else arch


def build_reservoir(
    arch: ReservoirArchitecture,
    hyper: ReservoirHyperparameters,
    seed: int | list[int],
    device: str,
    real_dtype: torch.dtype,
) -> KickedIsingReservoir:
    """Sample one frozen realisation with ``numpy.random.default_rng(seed)``."""
    params = sample_block_parameters(hyper, arch, np.random.default_rng(seed))
    return KickedIsingReservoir(
        arch, hyper, params, device=device, real_dtype=real_dtype
    )


def ring_features(
    reservoir: KickedIsingReservoir,
    inputs: np.ndarray,
    feature_set: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Pauli features on ring neighbours and the Pauli weight of each feature.

    Parameters
    ----------
    feature_set : str
        ``"tuning"`` (``X, Y, Z`` and ``XX, YY, ZZ``; 6 N features, paper Sec. II) or
        ``"nearest_neighbour"`` (all weight-1 and nearest-neighbour weight-2 strings;
        12 N features, paper Sec. III / Table A1).
    """
    rdms = simulate_pair_rdms(
        reservoir, inputs, reservoir.arch.ring_neighbour_pairs(), chunk_size
    )
    if feature_set == "tuning":
        n = reservoir.arch.n_qubits
        weights = np.concatenate([np.ones(3 * n), 2 * np.ones(3 * n)])
        return tuning_features(rdms), weights
    if feature_set == "nearest_neighbour":
        return nearest_neighbour_features(rdms)
    raise ValueError(f"Unknown feature_set {feature_set!r}")


def apply_feature_noise(
    features: np.ndarray, weights: np.ndarray, noise_cfg: dict, rng: np.random.Generator
) -> np.ndarray:
    """Finite-precision model of the expectation values.

    ``noise_cfg["kind"]``:

    - ``"none"``: exact expectations;
    - ``"gaussian"``: additive ``N(0, std)`` (paper's ``N(0, 10^{-3/2})`` proxy; the
      config states whether ``10^{-3/2}`` is read as the std or the variance);
    - ``"shadow"``: classical-shadow estimator variance at ``shots`` randomized shots.
    """
    kind = noise_cfg["kind"]
    if kind == "none":
        return features
    if kind == "gaussian":
        return features + rng.normal(0.0, float(noise_cfg["std"]), size=features.shape)
    if kind == "shadow":
        return shadow_gaussian_noise(features, weights, int(noise_cfg["shots"]), rng)
    raise ValueError(f"Unknown noise kind {kind!r}")
