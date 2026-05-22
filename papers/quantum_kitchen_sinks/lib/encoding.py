"""QKS linear encoding `theta = Omega @ u + beta` as described in arXiv:1806.08321.

`Omega` is a (q, p) matrix with exactly `r` non-zero entries per row, drawn from
N(0, sigma^2). `beta` is a (q,) bias vector drawn from U(0, 2*pi).

We expose two encoding schemes from the paper:
- "split": q = p, r = 1 — each input dimension feeds exactly one gate parameter.
- "tile": q = chosen, r = p / q — the input is partitioned into `q` contiguous
  tiles (each of length p/q); each row of Omega has its non-zero entries on a
  distinct tile.  This matches the "tile" partitioning used for MNIST (Fig. 4).

Each row of Omega is also independently constructed across "episodes": one
episode = one fresh `(Omega, beta)` draw, used identically on every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class EpisodeEncoding:
    """Static encoding for one episode: Omega (q, p), beta (q,)."""
    omega: np.ndarray
    beta: np.ndarray

    @property
    def n_params(self) -> int:
        return self.beta.shape[0]


def _split_episode(p: int, sigma: float, rng: np.random.Generator) -> EpisodeEncoding:
    # r = 1, q = p: each dim of u goes to exactly one gate parameter.
    omega = np.zeros((p, p), dtype=np.float64)
    cols = rng.permutation(p)
    for k in range(p):
        omega[k, cols[k]] = rng.normal(loc=0.0, scale=sigma)
    beta = rng.uniform(0.0, 2.0 * np.pi, size=p)
    return EpisodeEncoding(omega=omega, beta=beta)


def _tile_episode(
    p: int, q: int, sigma: float, rng: np.random.Generator
) -> EpisodeEncoding:
    """Tile encoding: q rows, each row has r = p // q nonzeros covering a tile.

    Rounds up the tile size when q does not divide p (the last tile carries the
    remainder).  Tile assignment to rows is random per episode (different
    episodes look at tiles in different orders).
    """
    if q <= 0 or q > p:
        raise ValueError(f"Invalid q={q} for p={p}")
    base = p // q
    rem = p - base * q
    tile_sizes = [base + (1 if i < rem else 0) for i in range(q)]
    boundaries = np.cumsum([0] + tile_sizes)
    omega = np.zeros((q, p), dtype=np.float64)
    row_order = rng.permutation(q)
    for row_idx, tile_idx in enumerate(row_order):
        start = boundaries[tile_idx]
        end = boundaries[tile_idx + 1]
        omega[row_idx, start:end] = rng.normal(loc=0.0, scale=sigma, size=end - start)
    beta = rng.uniform(0.0, 2.0 * np.pi, size=q)
    return EpisodeEncoding(omega=omega, beta=beta)


def make_episodes(
    n_episodes: int,
    input_dim: int,
    n_gate_params: int,
    sigma: float,
    encoding: str = "split",
    seed: int = 0,
) -> list[EpisodeEncoding]:
    """Generate `n_episodes` independent (Omega, beta) draws.

    `n_gate_params` is `q` (the number of trainable gate parameters per circuit).
    For "split" encoding the function requires q == input_dim.
    """
    rng = np.random.default_rng(seed)
    episodes: list[EpisodeEncoding] = []
    for _ in range(n_episodes):
        if encoding == "split":
            if n_gate_params != input_dim:
                raise ValueError(
                    "Split encoding requires n_gate_params == input_dim "
                    f"(got q={n_gate_params}, p={input_dim})."
                )
            ep = _split_episode(input_dim, sigma, rng)
        elif encoding == "tile":
            ep = _tile_episode(input_dim, n_gate_params, sigma, rng)
        else:
            raise ValueError(f"Unknown encoding {encoding!r}")
        episodes.append(ep)
    return episodes


def encode_batch(
    X: np.ndarray, episodes: Sequence[EpisodeEncoding]
) -> np.ndarray:
    """Apply each episode encoding to a batch.

    Returns a (n_episodes, n_samples, q) array of gate angles.
    """
    out = np.empty((len(episodes), X.shape[0], episodes[0].n_params), dtype=np.float64)
    for e, ep in enumerate(episodes):
        out[e] = X @ ep.omega.T + ep.beta
    return out


__all__ = ["EpisodeEncoding", "make_episodes", "encode_batch"]
