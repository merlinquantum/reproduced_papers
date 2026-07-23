"""End-to-end QRC training and generation pipeline (paper §II)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from .fnn import ReservoirHead, train_head

_LOG = logging.getLogger(__name__)


def teacher_forcing_features(
    reservoir,
    original: Sequence[int],
    leaking_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk through the original sequence with teacher forcing.

    Returns ``(features, targets)`` where ``features[t]`` is the reservoir
    probability vector after processing the pair ``(x_t, h_t)`` and
    ``targets[t]`` is ``x_{t+1}`` - the next feature in the original sequence.
    The leaking-rate update follows Eq. 2 of the paper.
    """
    n = len(original) - 1
    if n <= 0:
        raise ValueError("Need at least 2 features for teacher forcing")
    feats = np.zeros((n, reservoir.output_dim), dtype=np.float64)
    targets = np.asarray(original[1:], dtype=np.int64)

    h = reservoir.initial_hidden()
    for t in range(n):
        x_t = int(original[t])
        p = reservoir.step(x_t, h)
        feats[t] = p
        h = (1.0 - leaking_rate) * h + leaking_rate * p
    return feats, targets


def generate(
    reservoir,
    head: ReservoirHead,
    length: int,
    temperature: float,
    leaking_rate: float,
    seed_feature: int,
    rng: np.random.Generator,
) -> list[int]:
    """Run the autoregressive generation mode (Eqs. 1 and 2)."""
    sequence: list[int] = [int(seed_feature)]
    h = reservoir.initial_hidden()
    head.eval()

    for _ in range(length - 1):
        x_t = sequence[-1]
        p = reservoir.step(x_t, h)
        h = (1.0 - leaking_rate) * h + leaking_rate * p
        with torch.no_grad():
            logits = head(torch.tensor(p, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        probs = _softmax(logits / max(float(temperature), 1e-6))
        # ``rng.choice`` would normalise away tiny numerical drift, but it is
        # safer to clip first.
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
        sequence.append(int(rng.choice(probs.shape[0], p=probs)))
    return sequence


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def train_and_generate(
    reservoir,
    original: Sequence[int],
    num_features: int,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    leaking_rate: float,
    temperatures: Sequence[float],
    n_samples: int,
    gen_length: int,
    seed: int,
    log_progress: bool = True,
) -> dict:
    """Train the FNN with teacher forcing and sample new levels per temperature."""
    feats, targets = teacher_forcing_features(reservoir, original, leaking_rate)
    head = ReservoirHead(
        input_dim=reservoir.output_dim, num_features=num_features, hidden_dim=hidden_dim
    )
    history = train_head(
        head, feats, targets, epochs=epochs, lr=lr, weight_decay=weight_decay
    )
    if log_progress:
        _LOG.info(
            "Trained FNN: final loss=%.4f after %d epochs (input_dim=%d, num_features=%d)",
            history[-1] if history else float("nan"),
            epochs,
            reservoir.output_dim,
            num_features,
        )

    rng = np.random.default_rng(seed + 1)
    generated_per_T: dict[float, list[list[int]]] = {}
    seed_feature = int(original[0])
    for temperature in temperatures:
        seqs = []
        for _ in range(int(n_samples)):
            seqs.append(
                generate(
                    reservoir=reservoir,
                    head=head,
                    length=int(gen_length),
                    temperature=float(temperature),
                    leaking_rate=leaking_rate,
                    seed_feature=seed_feature,
                    rng=rng,
                )
            )
        generated_per_T[float(temperature)] = seqs
        if log_progress:
            _LOG.info(
                "Generated %d sequences at T=%.4g (length=%d)",
                n_samples,
                float(temperature),
                int(gen_length),
            )

    return {
        "training_history": history,
        "generated": generated_per_T,
        "teacher_features_shape": tuple(feats.shape),
    }
