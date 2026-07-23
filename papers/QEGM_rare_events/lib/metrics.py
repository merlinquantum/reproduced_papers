"""Evaluation metrics for QEGM rare-event reproduction.

Matches paper Section VI.B:

* ``tail_kl_divergence`` — KL(P_T || Q_T) restricted to the tail region.
* ``rare_event_recall`` — recall on tail-membership of generated samples.
* ``coverage_calibration`` — empirical coverage of α-level predictive
  intervals built from the model's samples.
"""

from __future__ import annotations

import numpy as np


def _tail_density(samples: np.ndarray, edges: np.ndarray, eps: float) -> np.ndarray:
    counts, _ = np.histogram(samples, bins=edges)
    counts = counts.astype(np.float64) + eps
    return counts / counts.sum()


def tail_kl_divergence(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    tail_threshold: float,
    n_bins: int = 20,
    eps: float = 1e-4,
) -> float:
    """KL(P_T || Q_T) on the union of the two distributions' tail mass."""

    real = real.flatten()
    generated = generated.flatten()
    real_tail = real[np.abs(real) > tail_threshold]
    gen_tail = generated[np.abs(generated) > tail_threshold]
    if real_tail.size == 0:
        return float("nan")
    lo = float(
        min(real_tail.min(), gen_tail.min() if gen_tail.size else real_tail.min())
    )
    hi = float(
        max(real_tail.max(), gen_tail.max() if gen_tail.size else real_tail.max())
    )
    edges = np.linspace(lo, hi, n_bins + 1)
    p = _tail_density(real_tail, edges, eps)
    if gen_tail.size == 0:
        gen_tail = np.zeros_like(real_tail)
    q = _tail_density(gen_tail, edges, eps)
    return float(np.sum(p * np.log(p / q)))


def rare_event_recall(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    tail_threshold: float,
) -> float:
    """Recall = fraction of distinct real-tail bins covered by generated samples."""

    real = real.flatten()
    generated = generated.flatten()
    real_tail = real[np.abs(real) > tail_threshold]
    if real_tail.size == 0:
        return float("nan")
    edges = np.linspace(real_tail.min() - 1e-6, real_tail.max() + 1e-6, 21)
    real_bins = np.unique(np.digitize(real_tail, edges))
    gen_tail = generated[np.abs(generated) > tail_threshold]
    if gen_tail.size == 0:
        return 0.0
    gen_bins = np.unique(np.digitize(gen_tail, edges))
    covered = np.intersect1d(real_bins, gen_bins).size
    return float(covered) / float(real_bins.size)


def coverage_calibration(
    real: np.ndarray,
    generated: np.ndarray,
    alphas: tuple = (0.5, 0.7, 0.9, 0.95),
) -> dict:
    """Empirical coverage of α-level predictive intervals derived from samples."""

    generated = generated.flatten()
    real = real.flatten()
    out = {}
    for alpha in alphas:
        lower_q = (1.0 - alpha) / 2.0
        upper_q = 1.0 - lower_q
        lo, hi = np.quantile(generated, [lower_q, upper_q])
        out[str(alpha)] = float(np.mean((real >= lo) & (real <= hi)))
    return out


def summarize(metrics_per_seed: list[dict]) -> dict:
    """Compute mean and std across seeds for a list of metric dicts."""

    if not metrics_per_seed:
        return {}
    keys = metrics_per_seed[0].keys()
    summary: dict = {}
    for key in keys:
        values = [m[key] for m in metrics_per_seed if key in m]
        if not values:
            continue
        if isinstance(values[0], dict):
            sub_keys = values[0].keys()
            summary[key] = {
                sk: {
                    "mean": float(np.mean([v[sk] for v in values])),
                    "std": float(np.std([v[sk] for v in values])),
                }
                for sk in sub_keys
            }
        else:
            summary[key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }
    return summary
