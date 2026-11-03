"""Temperature (Platt) scaling calibration for expert probabilities.

Given raw probabilities ``p_hat`` and labels ``y`` on a validation set:

1. Convert probabilities to logits: ``logit = ln(p_hat / (1 - p_hat))``,
   clipping ``p_hat`` to ``[eps, 1 - eps]`` first to avoid ``inf``.
2. Fit a single scalar temperature ``t`` (``nn.Parameter``, initialized to
   1.0) minimizing ``BCELoss(sigmoid(logits / t), y)`` with a few dozen Adam
   steps.

We use Adam rather than LBFGS here purely for simplicity/consistency with
the rest of the training code in this reproduction (LBFGS is the more common
choice for 1-D temperature scaling in the literature, e.g. Guo et al. 2017,
and would also work fine) — documented per the architecture spec.

KNOWN ISSUE (found 2026-09-01 in the n=100-fold statistically-powered runs,
NOT yet re-validated with the fix below -- see LOG.md "Calibration
Instability Finding"): with ``temperature.clamp_(min=1e-3)`` (the value used
for ALL results reported before this fix), Adam occasionally drives ``t``
toward that floor on folds where the secondary (GQC) expert's raw validation
predictions are separable-ish on a tiny (~28k-row) validation set --
minimizing BCE by making the sigmoid arbitrarily steep is a textbook
temperature-scaling failure mode on small/separable data (Platt scaling is
NOT immune to this; it is the reason more robust variants exist in the
calibration literature). Once ``t`` collapses to ~0.001, ``apply_temperature``
produces near-binary (saturated 0/1) probabilities for almost any nonzero
logit, and a handful of confidently-wrong saturated predictions among a
severely imbalanced holdout set (0.17% fraud) can catastrophically distort
threshold-independent ranking metrics like AUCPR -- observed swings as large
as -0.40 AUCPR on affected folds (~8-11% of folds), versus a typical/median
effect near zero. ``min=0.1`` (below) is a substantially less permissive
floor while still allowing meaningful calibration sharpening; it has NOT
been validated to eliminate 100% of collapse events, only to raise the floor
by 100x.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEFAULT_EPS = 1e-7


def probs_to_logits(p_hat: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    p_clipped = np.clip(p_hat, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    n_steps: int = 50,
    lr: float = 0.05,
) -> float:
    """Fit a scalar temperature minimizing calibrated BCE on (logits, y)."""
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([temperature], lr=lr)
    bce = nn.BCELoss()

    for _ in range(n_steps):
        optimizer.zero_grad()
        calibrated = torch.sigmoid(logits_t / temperature)
        loss = bce(calibrated, y_t)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # Temperature must stay positive; clamp defensively. min=0.1 (not
            # the original 1e-3) -- see module docstring "KNOWN ISSUE": a
            # too-permissive floor lets Adam collapse t on separable-ish
            # small-validation-set folds, producing near-binary saturated
            # probabilities that can catastrophically distort AUCPR.
            temperature.clamp_(min=0.1)

    return float(temperature.detach().item())


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    """Return calibrated probabilities ``sigmoid(logits / t)``.

    Clips ``logits / t`` to ``[-30, 30]`` before ``exp()`` -- mathematically
    a no-op (``sigmoid`` saturates well inside this range in float64) but
    avoids a spurious ``RuntimeWarning: overflow encountered in exp`` when a
    small fitted temperature combined with a large-magnitude raw logit would
    otherwise overflow ``exp()``.
    """
    logits = np.asarray(logits)
    scaled = np.clip(logits / t, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-scaled))


__all__ = ["probs_to_logits", "fit_temperature", "apply_temperature"]
