"""Coefficient refiners for QARIMA AR/MA estimation.

All three backends share one interface::

    b_refined = refiner.refine(b_ols, loss_fn, seed)

where ``loss_fn(b) -> float`` is the paper's quantum-inspired AR/MA loss and
``b_ols`` is the OLS warm start.  They differ only in how a candidate coefficient
vector is *parameterized*:

* ``OLSRefiner``   -- identity; returns the (loss-refined) OLS solution.
* ``GateVQCRefiner``   -- gate VQC (RY rotations + CNOT ladder) statevector; the
  coefficient is ``b(beta) = b_ols + step * (z(beta) - z(0))`` with
  ``z_j = <Z_j>`` the Pauli-Z expectation on qubit ``j``.  This is the paper's
  VQC-AR / VQC-MA refiner (Sec. 4.5, Fig. 2), with the underspecified
  angle->coefficient readout resolved as documented in LOG.md (D2).
* ``MerlinVQCRefiner`` -- photonic MerLin ``QuantumLayer`` counterpart: a no-input
  trainable interferometer whose output probabilities are read out linearly to a
  ``p``-vector ``z(beta)``; same warm-started refinement form.

Both VQC refiners are trained with COBYLA over a small, parameter-matched set of
angles/phases (paper: fixed optimizer + budget, reps configurable), so the only
difference between "gate" and "merlin" results is gate-unitary vs photonic-unitary
feature map.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

LossFn = Callable[[np.ndarray], float]


class OLSRefiner:
    """Classical baseline: minimise the loss directly over the coefficient vector.

    Uses COBYLA over ``b`` itself (same optimizer/budget family as the VQCs) so
    the comparison isolates the effect of the *quantum parameterization* rather
    than of the optimizer.  Warm-started at the OLS solution.
    """

    name = "classical"

    def __init__(self, max_iter: int = 200):
        self.max_iter = max_iter

    def refine(self, b_ols: np.ndarray, loss_fn: LossFn, seed: int = 0) -> np.ndarray:
        res = minimize(
            loss_fn,
            np.asarray(b_ols, dtype=np.float64),
            method="COBYLA",
            options={
                "maxiter": self.max_iter,
                "rhobeg": 0.1 * (np.linalg.norm(b_ols) + 1e-6),
            },
        )
        return np.asarray(res.x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Gate VQC (numpy statevector: RY rotations + CNOT ladder)
# --------------------------------------------------------------------------- #
def _apply_ry(state: np.ndarray, qubit: int, theta: float, n: int) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    st = state.reshape([2] * n)
    st = np.moveaxis(st, qubit, 0)
    a, b = st[0].copy(), st[1].copy()
    st[0] = c * a - s * b
    st[1] = s * a + c * b
    st = np.moveaxis(st, 0, qubit)
    return st.reshape(-1)


def _apply_cnot(state: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
    st = state.reshape([2] * n)
    st = np.moveaxis(st, ctrl, 0)
    # where control==1, flip target
    sub = st[1]
    sub = np.moveaxis(sub, tgt if tgt < ctrl else tgt - 1, 0)
    sub[[0, 1]] = sub[[1, 0]]
    sub = np.moveaxis(sub, 0, tgt if tgt < ctrl else tgt - 1)
    st[1] = sub
    st = np.moveaxis(st, 0, ctrl)
    return st.reshape(-1)


def _z_expectations(state: np.ndarray, n: int) -> np.ndarray:
    probs = np.abs(state) ** 2
    idx = np.arange(2**n)
    z = np.empty(n)
    for q in range(n):
        bit = (idx >> (n - 1 - q)) & 1  # qubit 0 = most significant
        z[q] = float(np.sum(probs[bit == 0]) - np.sum(probs[bit == 1]))
    return z


class GateVQCRefiner:
    name = "gate"

    def __init__(self, reps: int = 1, max_iter: int = 200, step_frac: float = 0.5):
        self.reps = reps
        self.max_iter = max_iter
        self.step_frac = step_frac

    def _z_of(self, angles: np.ndarray, p: int) -> np.ndarray:
        state = np.zeros(2**p, dtype=np.complex128)
        state[0] = 1.0
        a = angles.reshape(self.reps, p)
        for r in range(self.reps):
            for q in range(p):
                state = _apply_ry(state, q, float(a[r, q]), p)
            for q in range(p - 1):
                state = _apply_cnot(state, q, q + 1, p)
        return _z_expectations(state, p)

    def refine(self, b_ols: np.ndarray, loss_fn: LossFn, seed: int = 0) -> np.ndarray:
        b_ols = np.asarray(b_ols, dtype=np.float64)
        p = b_ols.size
        if p == 1:  # nothing to entangle; refine the scalar directly
            return OLSRefiner(self.max_iter).refine(b_ols, loss_fn, seed)
        step = self.step_frac * (np.linalg.norm(b_ols) / np.sqrt(p) + 1e-6)
        z0 = self._z_of(np.zeros(self.reps * p), p)

        def coef(beta: np.ndarray) -> np.ndarray:
            return b_ols + step * (self._z_of(beta, p) - z0)

        def obj(beta: np.ndarray) -> float:
            return loss_fn(coef(beta))

        rng = np.random.default_rng(seed)
        beta0 = rng.normal(0.0, 1e-2, size=self.reps * p)
        res = minimize(
            obj,
            beta0,
            method="COBYLA",
            options={"maxiter": self.max_iter, "rhobeg": 0.3},
        )
        return coef(np.asarray(res.x, dtype=np.float64))


# --------------------------------------------------------------------------- #
# MerLin photonic VQC
# --------------------------------------------------------------------------- #
class MerlinVQCRefiner:
    name = "merlin"

    def __init__(
        self,
        reps: int = 1,
        max_iter: int = 200,
        step_frac: float = 0.5,
        n_train: int | None = None,
        max_modes: int = 8,
    ):
        self.reps = reps
        self.max_iter = max_iter
        self.step_frac = step_frac
        self.n_train = n_train  # trainable phase count (parameter-matched to gate)
        self.max_modes = max_modes  # cap circuit size so output_size stays tractable
        self.hardware = {}  # filled in per refine() for reporting

    def _build_layer(self, p: int, seed: int):
        import merlin as ml
        import torch

        # Cap the interferometer size: a small fixed photonic feature map suffices
        # to reparameterise a p-vector, and keeps output_size = C(m, m/2) tractable.
        n_modes = min(max(4, 2 * p), self.max_modes)
        n_modes -= n_modes % 2
        n_photons = n_modes // 2
        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        builder.add_rotations(list(range(n_modes)), axis="y", trainable=True)
        builder.add_entangling_layer()
        input_state = [0] * n_modes
        for i in range(n_photons):
            input_state[2 * i] = 1
        layer = ml.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(),
            dtype=torch.float64,
        )
        self.hardware = {
            "computation_space": "UNBUNCHED",
            "detector_model": "threshold",
            "n_photons": n_photons,
            "n_modes": n_modes,
            "input_state": input_state,
            "encoding": "none (no-input trainable interferometer)",
            "measurement_strategy": "PROBABILITIES",
            "postselection": "none",
            "backend": "MerLin CPU simulator (analytic, shots=0)",
            "output_size": int(layer.output_size),
        }
        return layer

    def refine(self, b_ols: np.ndarray, loss_fn: LossFn, seed: int = 0) -> np.ndarray:
        import torch

        b_ols = np.asarray(b_ols, dtype=np.float64)
        p = b_ols.size
        if p == 1:
            return OLSRefiner(self.max_iter).refine(b_ols, loss_fn, seed)

        torch.manual_seed(seed)
        layer = self._build_layer(p, seed)
        params = list(layer.parameters())
        flat0 = torch.cat([q.detach().reshape(-1) for q in params]).clone()
        n_total = flat0.numel()
        n_train = min(self.n_train or (self.reps * p), n_total)
        self.hardware["n_train_params"] = int(n_train)

        rng = np.random.default_rng(seed)
        readout = rng.normal(0.0, 1.0, size=(p, self.hardware["output_size"]))
        readout /= np.linalg.norm(readout, axis=1, keepdims=True)
        step = self.step_frac * (np.linalg.norm(b_ols) / np.sqrt(p) + 1e-6)

        def probs_of(free: np.ndarray) -> np.ndarray:
            flat = flat0.clone()
            flat[:n_train] = torch.tensor(free, dtype=torch.float64)
            with torch.no_grad():
                offset = 0
                for q in params:
                    k = q.numel()
                    q.copy_(flat[offset : offset + k].reshape(q.shape))
                    offset += k
                out = layer().detach().reshape(-1).numpy()
            return out

        z0 = readout @ probs_of(np.zeros(n_train))

        def coef(free: np.ndarray) -> np.ndarray:
            return b_ols + step * (readout @ probs_of(free) - z0)

        def obj(free: np.ndarray) -> float:
            return loss_fn(coef(free))

        free0 = rng.normal(0.0, 1e-2, size=n_train)
        res = minimize(
            obj,
            free0,
            method="COBYLA",
            options={"maxiter": self.max_iter, "rhobeg": 0.3},
        )
        return coef(np.asarray(res.x, dtype=np.float64))


def make_refiner(name: str, **kw):
    name = name.lower()
    if name == "classical":
        return OLSRefiner(max_iter=kw.get("max_iter", 200))
    if name == "gate":
        return GateVQCRefiner(
            reps=kw.get("reps", 1),
            max_iter=kw.get("max_iter", 200),
            step_frac=kw.get("step_frac", 0.5),
        )
    if name == "merlin":
        return MerlinVQCRefiner(
            reps=kw.get("reps", 1),
            max_iter=kw.get("max_iter", 200),
            step_frac=kw.get("step_frac", 0.5),
            n_train=kw.get("n_train"),
        )
    raise ValueError(f"unknown refiner '{name}'")
