"""Photonic Quantum Kitchen Sinks via MerLin.

This is the photonic adaptation we add on top of the original (gate-model) QKS
algorithm.  The structure mirrors the paper:

  * Each "episode" is an independent random draw of an encoding ``Omega, beta``.
  * The encoded angles ``theta = Omega @ u + beta`` drive a *fixed* photonic
    circuit per episode.
  * The encoding parameters are also fixed once per episode and reused for
    every sample.
  * A single shot is drawn from the output detection distribution and the
    occupation pattern (one binary value per mode) becomes the QKS feature.

The fixed photonic circuit is:

  - ``add_entangling_layer``                   # universal MZI mesh, fixed phases
  - ``add_angle_encoding(modes=input_modes,
                         scale=angle_scale)``  # data-dependent phases
  - ``add_entangling_layer``                   # fixed phases

The mesh phases are re-initialised per episode (their random initialisation
*is* the per-episode randomness).  They are not trained — QKS is open-loop.

This intentionally departs from a verbatim translation of the gate-model
CNOT/CZ ansätze: MerLin does not expose qubit-level CNOTs, and a faithful
photonic counterpart is a random interferometer rather than a specific gate
network.  The scientific question is whether the same QKS *recipe* (random
non-linear quantum features + linear classifier) works in the photonic
regime; the gate structure itself is paper-specific.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

import merlin as ml

from .encoding import EpisodeEncoding, encode_batch, make_episodes


def _default_input_state(n_modes: int, n_photons: int) -> List[int]:
    """Pack ``n_photons`` photons across ``n_modes``.

    Spread photons across the active modes so the trainable mesh actually
    acts on the photon light cone, as recommended by MERLIN_COOKBOOK section 0.5.
    """
    if n_photons > n_modes:
        raise ValueError("n_photons must be <= n_modes for UNBUNCHED dual-rail style")
    state = [0] * n_modes
    if n_photons * 2 <= n_modes:
        for i in range(n_photons):
            state[2 * i + 1] = 1
    else:
        for i in range(n_photons):
            state[i] = 1
    return state


class PhotonicQKSFeaturizer:
    """QKS-style feature extractor whose per-episode circuit lives in MerLin.

    Notes
    -----
    For each episode we build one ``ml.QuantumLayer`` whose entangling-layer
    phases are sampled at construction (MerLin initialises them randomly when
    no value is provided).  We *freeze* the parameters so no training happens
    on them — the only "training" is the downstream logistic regression on the
    concatenated features.
    """

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
    ) -> None:
        self.n_modes = int(n_modes)
        self.n_photons = int(n_photons)
        self.n_episodes = int(n_episodes)
        self.sigma = float(sigma)
        self.encoding = encoding
        self.n_layers = int(n_layers)
        self.shots_per_episode = int(shots_per_episode)
        self.input_modes = (
            list(range(self.n_modes)) if input_modes is None else list(input_modes)
        )
        self.angle_scale = float(angle_scale)
        self.input_state = _default_input_state(self.n_modes, self.n_photons)
        self.episodes: List[EpisodeEncoding] = []
        self._layers: List[ml.QuantumLayer] = []
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
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            computation_space=ml.ComputationSpace.UNBUNCHED,
        )
        # Freeze the entangling-layer parameters: QKS is open-loop.
        for p in layer.parameters():
            p.requires_grad = False
        # Re-seed the entangling phases so different episodes look different.
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
        # Build one frozen QuantumLayer per *physical* episode (not per layer).
        # When n_layers > 1, the same QuantumLayer is reused with successive
        # angle batches; this matches the "stacked encoding" trick the paper
        # mentions for keeping CNOT-network depth meaningful.
        self._layers = [
            self._build_layer(seed=seed + 1000 * (e + 1))
            for e in range(self.n_episodes)
        ]
        return self

    def _sample_outcomes(self, probs: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
        """Sample one outcome per row from a (n, output_size) probability tensor.

        Returns an (n_samples, n_modes) int8 array where each row is the
        occupation pattern of the sampled outcome (in UNBUNCHED that is a
        binary vector of length ``n_modes`` with exactly ``n_photons`` ones).
        """
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        probs_np = np.clip(probs_np, 0.0, None)
        probs_np /= probs_np.sum(axis=1, keepdims=True)
        n = probs_np.shape[0]
        cum = np.cumsum(probs_np, axis=1)
        u = rng.uniform(size=(n, 1))
        outcome_indices = (u > cum).sum(axis=1)
        # Compute the occupation pattern of each index.  In UNBUNCHED with k
        # photons in m modes, the outcome ordering is the lexicographic
        # enumeration of all size-k subsets of {0, ..., m-1}.  Build the table
        # once.
        if not hasattr(self, "_outcome_table"):
            from itertools import combinations

            outcomes = list(combinations(range(self.n_modes), self.n_photons))
            self._outcome_table = np.zeros((len(outcomes), self.n_modes), dtype=np.int8)
            for i, combo in enumerate(outcomes):
                for m in combo:
                    self._outcome_table[i, m] = 1
        return self._outcome_table[outcome_indices]

    def transform(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        if not self._layers:
            raise RuntimeError("Call fit_episodes(...) before transform(...).")
        rng = np.random.default_rng(seed)
        angles_all = encode_batch(X, self.episodes)
        E_total, n_samples, q = angles_all.shape
        if self.n_layers == 1:
            angles_per_layer = angles_all[:, None, :, :]
        else:
            angles_per_layer = angles_all.reshape(
                self.n_episodes, self.n_layers, n_samples, q
            )
        feature_chunks = []
        for e in range(self.n_episodes):
            layer = self._layers[e]
            # Take only the last-layer encoding for sampling (single-circuit per
            # episode; the n_layers > 1 case is reserved for future use).
            theta = torch.from_numpy(
                angles_per_layer[e, -1].astype(np.float32)
            )
            probs = layer(theta)  # (n_samples, output_size)
            if self.shots_per_episode == 1:
                bits = self._sample_outcomes(probs, rng)  # (n_samples, n_modes)
            else:
                acc = np.zeros((n_samples, self.n_modes), dtype=np.float32)
                for _ in range(self.shots_per_episode):
                    acc += self._sample_outcomes(probs, rng).astype(np.float32)
                bits = acc / self.shots_per_episode
            feature_chunks.append(np.asarray(bits, dtype=np.float32))
        return np.concatenate(feature_chunks, axis=1)


__all__ = ["PhotonicQKSFeaturizer"]
