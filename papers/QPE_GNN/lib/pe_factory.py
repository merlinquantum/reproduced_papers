"""Per-batch positional-encoding tensor construction.

Given a padded adjacency batch ``A: (B, Nmax, Nmax)``, mask, and an encoding
spec, return the (B, Nmax, Nmax, K) edge feature tensor consumed by full GRIT
or GRITLite.

The heavy QPE computations are cached per graph and complete encoding
specification so that training pays the cost once per graph and encoding, not
once per batch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .qpe import (
    cqrw_features,
    ground_state_correlation_eigvecs,
    qirw_features,
    rrwp,
)


def _pe_for_one_graph(
    A: np.ndarray,
    encoding: str,
    K: int,
    times: Sequence[float] | None = None,
    rrwp_dim: int | None = None,
    qpe_dim: int | None = None,
) -> np.ndarray:
    if K < 1:
        raise ValueError("positional-encoding dimension K must be positive")
    if encoding == "rrwp":
        return rrwp(A, K)
    if encoding == "cqrw1":
        times = times or list(np.linspace(0.1, 2.0, K))
        return cqrw_features(A, 1, times)
    if encoding == "qirw2":
        # K is the total channel count. Channel zero is M^0|psi_init>, followed
        # by K - 1 transition powers, matching the RRWP channel convention.
        return qirw_features(A, 2, num_features=K)
    if encoding == "ground_state_corr":
        feats = ground_state_correlation_eigvecs(A, K)
        N = feats.shape[0]
        P = np.zeros((K, N, N), dtype=np.float64)
        for k in range(K):
            P[k] = np.outer(feats[:, k], feats[:, k])
        return P
    if encoding in {"rrwp+cqrw1", "rrwp+qirw2"}:
        if rrwp_dim is None or qpe_dim is None:
            raise ValueError(
                f"{encoding} requires separate rrwp_dim and qpe_dim values"
            )
        if rrwp_dim + qpe_dim != K:
            raise ValueError("combined positional-encoding dimensions do not sum to K")
        rrwp_features = rrwp(A, rrwp_dim)
        if encoding == "rrwp+cqrw1":
            if times is None or len(times) != qpe_dim:
                raise ValueError("rrwp+cqrw1 requires one time per QPE channel")
            quantum_features = cqrw_features(A, 1, times)
        else:
            quantum_features = qirw_features(A, 2, num_features=qpe_dim)
        return np.concatenate([rrwp_features, quantum_features], axis=0)
    raise ValueError(f"unknown encoding: {encoding}")


# Module-level cache. Cleared at the start of each test by `clear_cache()`.
_CACHE: dict[tuple, np.ndarray] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _cached_pe(
    A: np.ndarray,
    encoding: str,
    K: int,
    times: Sequence[float] | None,
    rrwp_dim: int | None,
    qpe_dim: int | None,
) -> np.ndarray:
    times_key = tuple(times) if times is not None else None
    key = (A.tobytes(), A.shape, encoding, K, times_key, rrwp_dim, qpe_dim)
    if key not in _CACHE:
        _CACHE[key] = _pe_for_one_graph(
            A,
            encoding,
            K,
            times=times,
            rrwp_dim=rrwp_dim,
            qpe_dim=qpe_dim,
        )
    return _CACHE[key]


def pe_batch(
    A_batch: torch.Tensor,
    mask: torch.Tensor,
    encoding: str,
    K: int,
    times: Sequence[float] | None = None,
    rrwp_dim: int | None = None,
    qpe_dim: int | None = None,
) -> torch.Tensor:
    """Build a dense positional-encoding tensor for padded graphs.

    Parameters
    ----------
    A_batch : torch.Tensor
        Padded adjacency tensor with shape ``(B, N, N)``.
    mask : torch.Tensor
        Boolean valid-node mask with shape ``(B, N)``.
    encoding : str
        Positional-encoding method.
    K : int
        Total output channel count.
    times : Sequence[float] | None
        Evolution times for 1-CQRW channels. Default value is None.
    rrwp_dim : int | None
        RRWP channel count for a concatenated encoding. Default value is None.
    qpe_dim : int | None
        Quantum channel count for a concatenated encoding. Default value is
        None.

    Returns
    -------
    torch.Tensor
        Positional encodings with shape ``(B, N, N, K)``.
    """
    B, Nmax, _ = A_batch.shape
    out = np.zeros((B, Nmax, Nmax, K), dtype=np.float32)
    for b in range(B):
        n = int(mask[b].sum().item())
        if n == 0:
            continue
        A = A_batch[b, :n, :n].cpu().numpy().astype(np.float64)
        P = _cached_pe(A, encoding, K, times, rrwp_dim, qpe_dim)  # (K, n, n)
        out[b, :n, :n] = P.transpose(1, 2, 0)
    return torch.from_numpy(out)
