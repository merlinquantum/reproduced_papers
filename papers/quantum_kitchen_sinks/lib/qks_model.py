"""End-to-end Quantum Kitchen Sinks feature pipeline.

Given an ansatz, an encoding, and the number of episodes, build:

    QKSFeaturizer.fit_episodes(input_dim, seed)
    QKSFeaturizer.transform(X) -> features of shape (n_samples, n_episodes * n_qubits)

Where each episode contributes `n_qubits` measured bits (single-shot sampling)
per sample.  The episodes are fixed once at fit time; only the measurement
randomness changes between fits, controlled by the `seed` passed to `transform`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .circuits import make_ansatz, number_of_gate_params
from .encoding import EpisodeEncoding, encode_batch, make_episodes


class QKSFeaturizer:
    def __init__(
        self,
        circuit: str,
        n_qubits: int,
        n_episodes: int,
        sigma: float,
        encoding: str,
        n_layers: int = 1,
        shots_per_episode: int = 1,
    ) -> None:
        self.circuit = circuit
        self.n_qubits = int(n_qubits)
        self.n_episodes = int(n_episodes)
        self.sigma = float(sigma)
        self.encoding = encoding
        self.n_layers = int(n_layers)
        self.shots_per_episode = int(shots_per_episode)
        self.episodes: list[EpisodeEncoding] = []
        self.input_dim = 0
        self._ansatz = None

    def fit_episodes(self, input_dim: int, seed: int = 0) -> "QKSFeaturizer":
        self.input_dim = int(input_dim)
        q = number_of_gate_params(self.circuit, self.n_qubits)
        total_episodes = self.n_episodes * self.n_layers
        self.episodes = make_episodes(
            n_episodes=total_episodes,
            input_dim=input_dim,
            n_gate_params=q,
            sigma=self.sigma,
            encoding=self.encoding,
            seed=seed,
        )
        self._ansatz = make_ansatz(self.circuit, self.n_qubits)
        return self

    def _layered_angles(self, X: np.ndarray) -> np.ndarray:
        """Encode X using all episodes and reshape into (E, n_layers, n_samples, q)."""
        all_angles = encode_batch(X, self.episodes)  # (n_episodes_total, n_samples, q)
        E_total, n_samples, q = all_angles.shape
        if self.n_layers == 1:
            return all_angles[:, None, :, :]  # (E, 1, n_samples, q)
        E = self.n_episodes
        return all_angles.reshape(E, self.n_layers, n_samples, q)

    def transform(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        if self._ansatz is None:
            raise RuntimeError("Call fit_episodes(...) before transform(...).")
        rng = np.random.default_rng(seed)
        angles = self._layered_angles(X)
        E, n_layers, n_samples, q = angles.shape
        n_qubits = self.n_qubits
        if self.shots_per_episode == 1:
            features = np.empty((n_samples, E * n_qubits), dtype=np.float32)
            for e in range(E):
                theta_batch = angles[e].transpose(1, 0, 2)  # (n_samples, n_layers, q)
                bits = self._ansatz(theta_batch, n_layers, rng)  # (n_samples, n_qubits)
                features[:, e * n_qubits:(e + 1) * n_qubits] = bits
            return features
        # shots > 1: average the bit value over `shots_per_episode` independent shots.
        features = np.zeros((n_samples, E * n_qubits), dtype=np.float32)
        S = self.shots_per_episode
        for e in range(E):
            theta_batch = angles[e].transpose(1, 0, 2)
            acc = np.zeros((n_samples, n_qubits), dtype=np.float32)
            for _ in range(S):
                acc += self._ansatz(theta_batch, n_layers, rng).astype(np.float32)
            features[:, e * n_qubits:(e + 1) * n_qubits] = acc / S
        return features


__all__ = ["QKSFeaturizer"]
