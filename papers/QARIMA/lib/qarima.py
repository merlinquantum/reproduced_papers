"""Core QARIMA pipeline: differencing, quantum ACF/PACF, AR/MA loss, fit, forecast.

The forecast function is a standard Box--Jenkins ARIMA(p,d,q) one-step predictor;
the AR and MA *coefficients* are produced by a pluggable refiner (classical / gate
VQC / photonic MerLin VQC) that minimises the paper's quantum-inspired loss
(Sec. 4.5 / 4.7).  See LOG.md (D1--D4) for the modelling decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lib.swaptest import binary_entropy, cosine_similarity, swap_test_cosine


# --------------------------------------------------------------------------- #
# Differencing
# --------------------------------------------------------------------------- #
def difference(y: np.ndarray, d: int) -> np.ndarray:
    w = np.asarray(y, dtype=np.float64).copy()
    for _ in range(d):
        w = np.diff(w)
    return w


def _delay_matrix(w: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows x_t = [w_{t-1},...,w_{t-p}], targets y_t = w_t, for t = p..len-1."""
    T = w.size
    X = np.stack([w[p - 1 - i : T - 1 - i] for i in range(p)], axis=1)
    y = w[p:]
    return X, y


# --------------------------------------------------------------------------- #
# Quantum-inspired ACF / PACF (swap-test cosine); analytic limit == classical
# --------------------------------------------------------------------------- #
def quantum_acf(
    w: np.ndarray,
    nlags: int,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    w = w - w.mean()
    out = np.empty(nlags + 1)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        a, b = w[k:], w[:-k]
        out[k] = swap_test_cosine(a, b, shots=shots, rng=rng)
    return out


def quantum_pacf(
    w: np.ndarray,
    nlags: int,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """PACF via Durbin--Levinson on the (quantum) autocovariances."""
    w0 = w - w.mean()
    n = w0.size
    acov = np.array(
        [
            np.dot(w0[k:], w0[:-k]) / n if k else np.dot(w0, w0) / n
            for k in range(nlags + 1)
        ]
    )
    if shots:  # inject swap-test-style noise on the normalised acf
        acf = quantum_acf(w, nlags, shots=shots, rng=rng)
        acov = acf * acov[0]
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = acov[1] / acov[0]
    pacf[1] = phi[1, 1]
    for k in range(2, nlags + 1):
        num = acov[k] - sum(phi[k - 1, j] * acov[k - j] for j in range(1, k))
        den = acov[0] - sum(phi[k - 1, j] * acov[j] for j in range(1, k))
        phi[k, k] = num / den if den != 0 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf[k] = phi[k, k]
    return pacf


# --------------------------------------------------------------------------- #
# Quantum-inspired AR / MA loss (paper Eq. 17 / 38)
# --------------------------------------------------------------------------- #
@dataclass
class LossWeights:
    lambda_cos: float = 0.1
    lambda_ent: float = 0.05
    lambda_l2: float = 0.0
    omega: float = 1.0  # phase-correction weight
    shots: int | None = None  # None => analytic swap test


def _binary_entropy_vec(p0: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p0 = np.clip(p0, eps, 1.0 - eps)
    return -p0 * np.log2(p0) - (1.0 - p0) * np.log2(1.0 - p0)


def ar_loss_fn(
    X: np.ndarray, y: np.ndarray, w: LossWeights, rng: np.random.Generator | None = None
):
    """Quantum-inspired AR/MA loss (paper Eq. 17/38).

    Fast analytic path (``shots is None``): the phase-corrected swap-test
    prediction reduces to ``X @ b`` and the cosine-alignment penalty vanishes, so
    the loss is ``||y - Xb||^2 + lambda_ent * sum H(1 - cos^2) + lambda_l2 ||b||^2``.
    A per-sample path handles finite shots (swap-test measurement noise).
    """
    norms_x = np.linalg.norm(X, axis=1)

    def loss(b: np.ndarray) -> float:
        b = np.asarray(b, dtype=np.float64)
        nb = float(np.linalg.norm(b)) + 1e-12
        if w.shots is None:  # analytic (infinite-shot) limit
            preds = X @ b
            err = float(np.sum((y - preds) ** 2))
            pen_ent = 0.0
            if w.lambda_ent:
                cos = (X @ b) / (norms_x * nb + 1e-12)
                pen_ent = float(
                    np.sum(_binary_entropy_vec(1.0 - np.clip(cos, -1, 1) ** 2))
                )
            return err + w.lambda_ent * pen_ent + w.lambda_l2 * float(b @ b)
        # finite-shot path (swap-test noise enters as in the paper)
        preds = np.empty(X.shape[0])
        pen_cos = pen_ent = 0.0
        for t in range(X.shape[0]):
            c_dot = cosine_similarity(X[t], b)
            c_swap = swap_test_cosine(X[t], b, shots=w.shots, rng=rng)
            th_corr = np.arccos(np.clip(c_swap, -1, 1)) + w.omega * (
                np.arccos(np.clip(c_dot, -1, 1)) - np.arccos(np.clip(c_swap, -1, 1))
            )
            preds[t] = norms_x[t] * nb * np.cos(th_corr)
            pen_cos += (c_dot - c_swap) ** 2
            pen_ent += binary_entropy(1.0 - c_swap**2)
        err = float(np.sum((y - preds) ** 2))
        return (
            err
            + w.lambda_cos * pen_cos
            + w.lambda_ent * pen_ent
            + w.lambda_l2 * float(b @ b)
        )

    return loss


# --------------------------------------------------------------------------- #
# Fit AR / MA
# --------------------------------------------------------------------------- #
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def fit_ar(w_train: np.ndarray, p: int, refiner, weights: LossWeights, seed: int):
    X, y = _delay_matrix(w_train, p)
    b_ols = _ols(X, y)
    if refiner is None or refiner.name == "ols_only":
        return b_ols, b_ols
    loss = ar_loss_fn(X, y, weights, np.random.default_rng(seed))
    b = refiner.refine(b_ols, loss, seed=seed)
    return b, b_ols


def ar_residuals(w: np.ndarray, p: int, b: np.ndarray) -> np.ndarray:
    X, y = _delay_matrix(w, p)
    return y - X @ b


def fit_ma(resid: np.ndarray, q: int, refiner, weights: LossWeights, seed: int):
    if q == 0:
        return np.zeros(0)
    E, e = _delay_matrix(resid, q)
    if E.shape[0] < q + 1:
        return np.zeros(q)
    t_ols = _ols(E, e)
    if refiner is None:
        return t_ols
    loss = ar_loss_fn(E, e, weights, np.random.default_rng(seed + 1))
    return refiner.refine(t_ols, loss, seed=seed + 1)


# --------------------------------------------------------------------------- #
# Rolling one-step forecast (coefficients fixed from training)
# --------------------------------------------------------------------------- #
def forecast_oos(
    y_full: np.ndarray,
    n_oos: int,
    p: int,
    d: int,
    q: int,
    b: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Dynamic multi-step OOS forecast (single origin at end of training).

    Coefficients ``b`` (AR) and ``theta`` (MA) are fixed on the training set and
    the whole OOS horizon is forecast recursively: the AR term consumes the
    model's *own* predictions and future MA innovations are set to their
    expectation (zero).  Differencing is integrated back to the original scale.
    This matches the error magnitudes reported in the paper (multi-step, not
    one-step rolling); see LOG.md 7.3.7 note.
    """
    y_full = np.asarray(y_full, dtype=np.float64)
    y_train = y_full[:-n_oos]

    # Difference levels D[0]=y_train, D[k]=diff(D[k-1]); seed = last value of each.
    levels = [y_train]
    for _ in range(d):
        levels.append(np.diff(levels[-1]))
    wf = list(levels[d])  # d-th differences (AR operates here)
    seed = [float(levels[k][-1]) for k in range(d + 1)]

    # training innovations from the fixed AR coefficients
    ef = [0.0] * len(wf)
    for t in range(p, len(wf)):
        ef[t] = wf[t] - float(np.dot(b, [wf[t - 1 - i] for i in range(p)]))

    preds = np.empty(n_oos)
    for h in range(n_oos):
        ar = float(np.dot(b, [wf[-1 - i] for i in range(p)])) if p else 0.0
        ma = sum(theta[j - 1] * ef[-j] for j in range(1, q + 1) if len(ef) >= j)
        pred_d = ar + ma
        wf.append(pred_d)
        ef.append(0.0)  # expected future innovation
        # integrate the d-th difference back up to the original scale
        cur = pred_d
        seed[d] = pred_d
        for k in range(d - 1, -1, -1):
            cur = seed[k] + cur
            seed[k] = cur
        preds[h] = seed[0] if d > 0 else pred_d
    return preds


# --------------------------------------------------------------------------- #
# End-to-end fit + forecast for one (p,d,q) with one refiner
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    p: int
    d: int
    q: int
    refiner: str
    b: np.ndarray
    b_ols: np.ndarray
    theta: np.ndarray
    y_pred: np.ndarray
    extra: dict = field(default_factory=dict)


def fit_and_forecast(y_full, n_oos, p, d, q, refiner, weights, seed):
    y_train = y_full[:-n_oos]
    w_train = difference(y_train, d)
    b, b_ols = fit_ar(w_train, p, refiner, weights, seed)
    resid = ar_residuals(w_train, p, b)
    theta = fit_ma(resid, q, refiner, weights, seed)
    y_pred = forecast_oos(y_full, n_oos, p, d, q, b, theta)
    return FitResult(
        p, d, q, getattr(refiner, "name", "classical"), b, b_ols, theta, y_pred
    )
