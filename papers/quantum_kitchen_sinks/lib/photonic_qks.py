"""Photonic Quantum Kitchen Sinks via MerLin.

Per-episode random photonic circuit (entangling layer → angle encoding →
entangling layer); parameters are frozen (QKS is open-loop).  One shot is
sampled per episode from the output occupation distribution; the resulting
``n_modes``-bit pattern is the QKS feature contribution.

See README and INSIGHTS for the design rationale.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

import merlin as ml

from .encoding import EpisodeEncoding, make_episodes


def _validate_input_modes(n_modes: int, input_modes: Sequence[int]) -> List[int]:
    ordered_modes = []
    seen = set()
    for mode in input_modes:
        mode_int = int(mode)
        if mode_int < 0 or mode_int >= n_modes:
            raise ValueError("input_modes entries must lie in [0, n_modes)")
        if mode_int not in seen:
            ordered_modes.append(mode_int)
            seen.add(mode_int)
    return ordered_modes


def _default_input_state(
    n_modes: int,
    n_photons: int,
    input_modes: Sequence[int],
    computation_space: ml.ComputationSpace,
) -> List[int]:
    if n_photons > n_modes:
        raise ValueError("n_photons must be <= n_modes")
    ordered_modes = _validate_input_modes(n_modes, input_modes)
    if computation_space is ml.ComputationSpace.DUAL_RAIL:
        if n_modes % 2 != 0:
            raise ValueError("dual_rail requires an even number of modes")
        if n_photons * 2 != n_modes:
            raise ValueError("dual_rail requires n_photons = n_modes // 2")
        if len(ordered_modes) != n_photons:
            raise ValueError(
                "dual_rail requires exactly one encoded mode per logical qubit pair"
            )
        pair_to_mode = {}
        for mode in ordered_modes:
            pair_idx = mode // 2
            if pair_idx in pair_to_mode:
                raise ValueError(
                    "dual_rail input_modes must select at most one mode in each pair"
                )
            pair_to_mode[pair_idx] = mode
        if len(pair_to_mode) != n_photons:
            raise ValueError(
                "dual_rail input_modes must cover each logical qubit pair exactly once"
            )
        state = [0] * n_modes
        for pair_idx in range(n_photons):
            state[pair_to_mode[pair_idx]] = 1
        return state
    for mode in range(n_modes):
        if mode not in ordered_modes:
            ordered_modes.append(mode)
    state = [0] * n_modes
    for mode in ordered_modes[:n_photons]:
        state[mode] = 1
    return state


class PhotonicQKSFeaturizer:
    def __init__(
        self,
        n_modes: int,
        n_photons: int,
        n_episodes: int,
        sigma: float,
        encoding: str,
        n_layers: int = 1,
        shots_per_episode: int = 1,
        input_modes: Sequence[int] | None = None,
        angle_scale: float = 1.0,
        computation_space: ml.ComputationSpace | str = ml.ComputationSpace.UNBUNCHED,
    ) -> None:
        self.n_modes = int(n_modes)
        self.n_photons = int(n_photons)
        self.n_episodes = int(n_episodes)
        self.sigma = float(sigma)
        self.encoding = encoding
        self.n_layers = int(n_layers)
        self.shots_per_episode = int(shots_per_episode)
        self.computation_space = ml.ComputationSpace.coerce(computation_space)
        self.input_modes = (
            list(range(0, self.n_modes, 2))
            if input_modes is None
            and self.computation_space is ml.ComputationSpace.DUAL_RAIL
            else list(range(self.n_modes)) if input_modes is None else list(input_modes)
        )
        self.input_modes = _validate_input_modes(self.n_modes, self.input_modes)
        self.angle_scale = float(angle_scale)
        self.input_state = _default_input_state(
            self.n_modes,
            self.n_photons,
            self.input_modes,
            self.computation_space,
        )
        self.episodes: List[EpisodeEncoding] = []
        self._layer_seeds: List[int] = []
        self.input_dim = 0

    def _build_layer(self, seed: int) -> ml.QuantumLayer:
        builder = ml.CircuitBuilder(n_modes=self.n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=self.input_modes, scale=self.angle_scale)
        builder.add_entangling_layer()
        layer = ml.QuantumLayer(
            input_size=len(self.input_modes),
            builder=builder,
            input_state=self.input_state,
            n_photons=self.n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=self.computation_space
            ),
        )
        for p in layer.parameters():
            p.requires_grad = False
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        for p in layer.parameters():
            with torch.no_grad():
                p.copy_(
                    torch.empty_like(p).uniform_(0.0, 2.0 * np.pi, generator=gen)
                )
        return layer

    def fit_episodes(self, input_dim: int, seed: int = 0) -> "PhotonicQKSFeaturizer":
        self.input_dim = int(input_dim)
        total_episodes = self.n_episodes * self.n_layers
        self.episodes = make_episodes(
            n_episodes=total_episodes,
            input_dim=input_dim,
            n_gate_params=len(self.input_modes),
            sigma=self.sigma,
            encoding=self.encoding,
            seed=seed,
        )
        self._layer_seeds = [seed + 1000 * (e + 1) for e in range(self.n_episodes)]
        return self

    def _sample_outcomes(self, probs: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        probs_np = np.clip(probs_np, 0.0, None)
        probs_np /= probs_np.sum(axis=1, keepdims=True)
        n = probs_np.shape[0]
        cum = np.cumsum(probs_np, axis=1)
        u = rng.uniform(size=(n, 1))
        outcome_indices = (u > cum).sum(axis=1)
        if not hasattr(self, "_outcome_table"):
            from itertools import combinations

            outcomes = list(combinations(range(self.n_modes), self.n_photons))
            self._outcome_table = np.zeros((len(outcomes), self.n_modes), dtype=np.int8)
            for i, combo in enumerate(outcomes):
                for m in combo:
                    self._outcome_table[i, m] = 1
        return self._outcome_table[outcome_indices]

    def transform(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        if not self._layer_seeds:
            raise RuntimeError("Call fit_episodes(...) before transform(...).")
        rng = np.random.default_rng(seed)
        feature_chunks = []
        for e in range(self.n_episodes):
            layer = self._build_layer(self._layer_seeds[e])
            episode_idx = e * self.n_layers + (self.n_layers - 1)
            episode = self.episodes[episode_idx]
            theta_np = X @ episode.omega.T + episode.beta
            theta = torch.from_numpy(theta_np.astype(np.float32, copy=False))
            probs = layer(theta)
            if self.shots_per_episode == 1:
                bits = self._sample_outcomes(probs, rng)
            else:
                acc = np.zeros((n_samples, self.n_modes), dtype=np.float32)
                for _ in range(self.shots_per_episode):
                    acc += self._sample_outcomes(probs, rng).astype(np.float32)
                bits = acc / self.shots_per_episode
            feature_chunks.append(np.asarray(bits, dtype=np.float32))
        return np.concatenate(feature_chunks, axis=1)


__all__ = ["PhotonicQKSFeaturizer"]
