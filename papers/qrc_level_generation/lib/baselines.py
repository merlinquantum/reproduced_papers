"""Markov-chain and uncorrelated reference generators (paper §III)."""

from __future__ import annotations

import random
from collections.abc import Sequence


def feature_probabilities(
    sequence: Sequence[int], num_features: int
) -> dict[int, float]:
    length = len(sequence)
    return {feat: sequence.count(feat) / length for feat in range(num_features)}


def markov_transitions(
    sequence: Sequence[int], num_features: int
) -> dict[int, dict[int, float]]:
    """Empirical transition probabilities P(x_{t+1} | x_t) from the training sequence."""
    n = len(sequence)
    successors: dict[int, list[int]] = {feat: [] for feat in range(num_features)}
    for index in range(n):
        successors[sequence[index]].append(sequence[(index + 1) % n])
    transitions: dict[int, dict[int, float]] = {}
    for feat, succ in successors.items():
        if not succ:
            transitions[feat] = {}
            continue
        bucket: dict[int, float] = {}
        for value in succ:
            bucket[value] = bucket.get(value, 0.0) + 1.0
        total = float(len(succ))
        transitions[feat] = {key: count / total for key, count in bucket.items()}
    return transitions


def generate_uncorrelated(
    probs: dict[int, float], length: int, rng: random.Random | None = None
) -> list[int]:
    rng = rng or random
    keys = list(probs.keys())
    weights = [probs[k] for k in keys]
    return rng.choices(keys, weights=weights, k=length)


def generate_markov(
    probs: dict[int, float],
    transitions: dict[int, dict[int, float]],
    length: int,
    rng: random.Random | None = None,
) -> list[int]:
    rng = rng or random
    start = rng.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
    sequence = [start]
    for _ in range(length - 1):
        row = transitions[sequence[-1]]
        if not row:  # fallback: dead-end state - resample uniformly from feature priors
            sequence.append(
                rng.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
            )
            continue
        next_feat = rng.choices(list(row.keys()), weights=list(row.values()), k=1)[0]
        sequence.append(next_feat)
    return sequence


def make_baseline_sequences(
    original: Sequence[int],
    num_features: int,
    length: int,
    n_samples: int,
    seed: int = 0,
) -> dict[str, list[list[int]]]:
    """Produce Markov and uncorrelated baseline corpora seeded deterministically."""
    rng = random.Random(seed)
    probs = feature_probabilities(original, num_features)
    transitions = markov_transitions(original, num_features)
    uncorrelated = [generate_uncorrelated(probs, length, rng) for _ in range(n_samples)]
    markov = [
        generate_markov(probs, transitions, length, rng) for _ in range(n_samples)
    ]
    return {"uncorrelated": uncorrelated, "markov": markov}
