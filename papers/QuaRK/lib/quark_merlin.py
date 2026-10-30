"""Photonic adaptation of the QuaRK reservoir with MerLin (an adaptation, not the paper).

Each sub-reservoir is a linear-optical circuit on ``m`` modes with ``n`` photons:

    W_r, then for each time step t:  phases phi_t on all modes, W_r,

with ``W_r`` a fixed random MZI mesh (MerLin non-trainable entangling layer) and
``phi_t = pi (1 - 1e-6) tanh(x_t @ Pi)``, ``Pi`` a Gaussian projection with entries
``N(0, 1/m)``. A phase acts trivially on a Fock state and just before photon
counting, so every encoding sits between two meshes; the first mesh plays the role
of QuaRK's ``|+>`` preparation.

QuaRK's reset channel ``rho -> lambda U rho U^dag + (1 - lambda) sigma`` resets to its
initial state, so after ``w`` steps the state is a classical mixture over memory
lengths: only the last ``j`` steps acted, with probability ``(1 - lambda) lambda^j``
(``j < w``) or ``lambda^w`` (``j = w``). Probabilities are linear in the state, so the
photonic reservoir mixes the output distributions of its depth-``j`` circuits with the
same weights, using QuaRK's own ``lambda_r`` at the same seed. On hardware, a classical
coin per shot picks ``j``.

Detection: threshold detectors, simulated in the Fock space with MerLin's occupancy
readout (bunched photons give fewer clicks; no postselection). Features: the 1- and
2-mode click marginals, ``m + m(m-1)/2`` per sub-reservoir (105 for 14 modes), either
exact or estimated from ``shots`` click patterns shared by all marginals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import merlin as ml
import numpy as np
import perceval as pcvl
import torch

from .reservoir import ANGLE_SCALE, draw_reservoir_parameters

ESTIMATORS = ("exact", "click_sampling")


@dataclass
class PhotonicReservoirParameters:
    """Random draws of the photonic reservoir.

    Attributes
    ----------
    projection : numpy.ndarray
        Gaussian projection of shape ``(d, m)``, entries ``N(0, 1/m)``.
    mesh_seeds : list[int]
        Seed of the fixed MZI mesh of each sub-reservoir.
    contractions : numpy.ndarray
        ``lambda_r`` of shape ``(R,)``, QuaRK's values at the same seed.
    input_state : list[int]
        Fock input state (one photon per listed mode).
    """

    projection: np.ndarray
    mesh_seeds: list[int]
    contractions: np.ndarray
    input_state: list[int]


def spread_input_state(num_modes: int, num_photons: int) -> list[int]:
    """One photon in each of ``num_photons`` modes spread evenly over the circuit."""
    if not 2 <= num_photons <= num_modes:
        raise ValueError("need 2 <= num_photons <= num_modes")
    occupied = {round(k * num_modes / num_photons) for k in range(num_photons)}
    if len(occupied) != num_photons:
        raise ValueError("photon positions collide")
    return [1 if mode in occupied else 0 for mode in range(num_modes)]


def draw_photonic_parameters(
    *, input_dim, num_modes, num_photons, num_reservoirs, quark_cfg, seed
) -> PhotonicReservoirParameters:
    """Draw the projection and mesh seeds; take ``lambda_r`` from QuaRK's own draw."""
    rng = np.random.default_rng([seed, 2])
    projection = rng.normal(0.0, 1.0 / np.sqrt(num_modes), size=(input_dim, num_modes))
    mesh_seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=num_reservoirs)]
    contractions = draw_reservoir_parameters(
        input_dim=input_dim,
        num_qubits=int(quark_cfg["num_qubits"]),
        num_reservoirs=num_reservoirs,
        first_contraction=float(quark_cfg["first_contraction"]),
        seed=seed,
    ).contractions
    return PhotonicReservoirParameters(
        projection=projection,
        mesh_seeds=mesh_seeds,
        contractions=contractions,
        input_state=spread_input_state(num_modes, num_photons),
    )


def truncated_layer(
    depth: int,
    *,
    num_modes: int,
    mesh_seed: int,
    input_state: list[int],
    distinguishable: bool,
) -> ml.QuantumLayer:
    """MerLin layer of the circuit that applies only the last ``depth`` time steps.

    Input columns are the phases of step 1 (modes 0..m-1), then step 2, and so on.
    """
    builder = ml.CircuitBuilder(n_modes=num_modes)
    builder.add_entangling_layer(trainable=False, seed=mesh_seed)
    for step in range(depth):
        # MerLin names each phase prefix + running input index (t00m1..t00m14,
        # t01m15, ...); fixed-width prefixes keep the names unambiguous.
        builder.add_angle_encoding(name=f"t{step:02d}m", scale=1.0)
        builder.add_entangling_layer(trainable=False, seed=mesh_seed)
    noise = pcvl.NoiseModel(indistinguishability=0.0) if distinguishable else None
    return ml.QuantumLayer(
        input_size=depth * num_modes if depth else None,
        builder=builder,
        input_state=input_state,
        n_photons=sum(input_state),
        noise=noise,
        dtype=torch.float64,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK, occupancy_readout=True
        ),
    )


def marginal_matrix(output_keys, num_modes: int) -> np.ndarray:
    """0/1 matrix mapping click-pattern probabilities to 1- and 2-mode click marginals.

    Columns: single modes 0..m-1, then mode pairs (i < j) in lexicographic order.
    """
    pairs = list(combinations(range(num_modes), 2))
    keys = np.array([[1 if c else 0 for c in key] for key in output_keys])
    singles = keys
    doubles = np.stack([keys[:, i] * keys[:, j] for i, j in pairs], axis=1)
    return np.concatenate([singles, doubles], axis=1).astype(float)


def mixture_weights(contraction: float, window: int) -> np.ndarray:
    """Weights of the memory lengths ``j = 0..w`` of QuaRK's reset channel."""
    weights = (1.0 - contraction) * contraction ** np.arange(window + 1)
    weights[window] = contraction**window
    return weights


def click_distributions(
    windows: np.ndarray,
    params: PhotonicReservoirParameters,
    *,
    distinguishable: bool,
    chunk_size: int = 2000,
) -> tuple[np.ndarray, list]:
    """Mixture click-pattern distributions of every window and sub-reservoir.

    Returns
    -------
    tuple[numpy.ndarray, list]
        Array ``(N, R, K)`` over the ``K`` click patterns, and the pattern keys.
    """
    num_windows, window, _ = windows.shape
    num_modes = params.projection.shape[1]
    phases = ANGLE_SCALE * np.tanh(windows @ params.projection)  # (N, w, m)
    out, keys = None, None
    for r, mesh_seed in enumerate(params.mesh_seeds):
        weights = mixture_weights(float(params.contractions[r]), window)
        mixture = None
        for depth in range(window + 1):
            layer = truncated_layer(
                depth,
                num_modes=num_modes,
                mesh_seed=mesh_seed,
                input_state=params.input_state,
                distinguishable=distinguishable,
            )
            if keys is None:
                keys = list(layer.output_keys)
                out = np.empty((num_windows, len(params.mesh_seeds), len(keys)))
            elif list(layer.output_keys) != keys:
                raise RuntimeError("output keys differ between depths")
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if depth == 0:
                    probs = layer().numpy()[0][None, :]
                else:
                    last = phases[:, window - depth :, :].reshape(num_windows, -1)
                    probs = np.concatenate(
                        [
                            layer(torch.from_numpy(last[s : s + chunk_size])).numpy()
                            for s in range(0, num_windows, chunk_size)
                        ]
                    )
            term = weights[depth] * probs
            mixture = term if mixture is None else mixture + term
        out[:, r] = np.broadcast_to(mixture, (num_windows, len(keys)))
    return out, keys


def photonic_features(
    windows: np.ndarray, merlin_cfg: dict, quark_cfg: dict, seed: int
) -> tuple[np.ndarray, dict]:
    """Click-marginal features ``(N, R * (m + m(m-1)/2))`` and model information."""
    estimator = merlin_cfg["estimator"]
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown photonic estimator {estimator!r}")
    if int(merlin_cfg["locality"]) != 2:
        raise ValueError("only 1- and 2-mode click marginals are implemented")
    num_modes = int(merlin_cfg["num_modes"])
    params = draw_photonic_parameters(
        input_dim=windows.shape[-1],
        num_modes=num_modes,
        num_photons=int(merlin_cfg["num_photons"]),
        num_reservoirs=int(merlin_cfg["num_reservoirs"]),
        quark_cfg=quark_cfg,
        seed=seed,
    )
    distributions, keys = click_distributions(
        windows, params, distinguishable=bool(merlin_cfg["distinguishable"])
    )
    matrix = marginal_matrix(keys, num_modes)
    if estimator == "click_sampling":
        shots = int(merlin_cfg["shots"])
        estimated = np.empty_like(distributions)
        for r in range(distributions.shape[1]):
            rng = np.random.default_rng([seed, 3, r])
            pvals = np.clip(distributions[:, r], 0.0, None)
            pvals /= pvals.sum(axis=1, keepdims=True)
            estimated[:, r] = rng.multinomial(shots, pvals) / shots
        distributions = estimated
    features = np.concatenate(
        [distributions[:, r] @ matrix for r in range(distributions.shape[1])], axis=1
    )
    info = {
        "contractions": params.contractions.tolist(),
        "mesh_seeds": params.mesh_seeds,
        "input_state": params.input_state,
        "num_click_patterns": len(keys),
        "num_observables": matrix.shape[1],
    }
    return features, info
