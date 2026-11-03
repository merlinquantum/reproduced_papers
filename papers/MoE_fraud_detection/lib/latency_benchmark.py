"""Table 1 reproduction: per-sample inference-latency benchmark for QMKL,
the Genetic Feature Map (GFM), and GQC (Section 2, "Motivation").

This module is self-contained: it reads the raw dataset via
``lib.data.load_raw_data`` and reuses the gate-model quantum block via
``lib.gqc_model.GQCModel`` / ``lib.vqc_gate.VQCLayer``, but does not import
from or modify ``lib.pipeline`` (the MoE CV pipeline) -- Table 1 is a purely
architectural latency comparison, not an accuracy benchmark, and has its own
much smaller/simpler protocol (Section 2):

1. Reduce the dataset to 10,000 rows, split 90/10 train/test, balance the
   TRAIN pool to 90 samples/class (~180 rows total).
2. QMKL and GFM are restricted to 2 qubits in the paper's benchmark, so they
   need a PCA(2) reduction fit on the (scaled) balanced train pool. GQC does
   its own dimensionality reduction internally (its autoencoder), so it
   consumes the full-dimensional scaled features directly, no PCA.
3. Benchmark protocol: 5 timing runs, each executing 10 internal repetitions
   over a fixed batch of 50 test samples; run 0 is a hardware warm-up and is
   excluded from the reported statistics. Report mean ms/sample and
   coefficient-of-variation (%) across the remaining 4 runs.

Ambiguity / assumptions (see also the reproduction's final report):

- **10,000-row reduction**: a literal uniform random subsample of the full
  ULB dataset (0.172% fraud) would contain only ~17 fraud rows in total --
  far too few to satisfy the paper's own downstream "balance train pool to
  90 samples/class" requirement. We instead build the 10,000-row subset from
  ALL available fraud rows (492) plus a random sample of non-fraud rows
  filling the remainder to 10,000. This is a deliberate, documented deviation
  from a literal i.i.d. subsample, made solely to make the paper's stated
  protocol executable; it does not change what is being measured (per-sample
  *inference latency*, which does not depend on the class distribution of
  the timed batch).
- **QMKL's "weighted combination" of three kernels**: the paper's text
  states QMKL evaluates three kernels per pair but does not specify the
  combination weights. We use an equal-weighted average (1/3 each) -- a
  simple, defensible default that preserves the "evaluate three kernels per
  pair" cost structure the paper cites as why QMKL is slower than GFM.
- **Exact circuit ansatze**: neither QMKL's three kernels nor GFM's
  (genetically-searched) single kernel are specified in enough detail in the
  paper text alone to reproduce literally. We implement three distinct,
  genuinely different small 2-qubit PennyLane feature-map circuits for QMKL
  (plain angle encoding; angle encoding + one ZZ-style interaction term;
  angle encoding + one entangling CNOT) and one fixed 2-qubit entangling
  feature map for GFM (angle encoding + one CNOT), each turned into a
  standard fidelity/overlap kernel: encode ``x``, apply the adjoint encoding
  of ``x'``, and read off the probability of the all-zeros outcome.
- **What "inference" costs**: Table 1 measures per-sample *inference*
  latency, which for a kernel method is dominated by evaluating the kernel
  between the new point and the (small, ~180-row) training pool -- this is
  what actually determines wall-clock cost at prediction time, whether or
  not one bothers to also fit/evaluate the downstream SVM. We therefore
  benchmark kernel-row evaluation against the training pool directly and do
  NOT fit an ``sklearn.svm.SVC(kernel="precomputed")`` for Table 1 -- fitting
  an SVM is a one-time training-time cost, not part of "per-sample inference
  latency", and would not change any timing numbers reported here. This is
  documented explicitly per the task's guidance to pick one approach and
  explain why.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pennylane as qml
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .data import load_raw_data
from .gqc_model import GQCModel

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _reduce_to_n_samples(
    X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Build a `n_samples`-row subset that keeps every fraud row.

    See module docstring "10,000-row reduction" for why a literal uniform
    subsample would not work with the paper's downstream 90/class balancing
    step.
    """
    fraud_idx = np.flatnonzero(y == 1)
    nonfraud_idx = np.flatnonzero(y == 0)
    n_nonfraud_needed = max(n_samples - len(fraud_idx), 0)
    n_nonfraud_needed = min(n_nonfraud_needed, len(nonfraud_idx))
    chosen_nonfraud = rng.choice(nonfraud_idx, size=n_nonfraud_needed, replace=False)
    combined_idx = np.concatenate([fraud_idx, chosen_nonfraud])
    rng.shuffle(combined_idx)
    combined_idx = combined_idx[: min(n_samples, len(combined_idx))]
    return X[combined_idx], y[combined_idx]


def _balance_to_n_per_class(
    X: np.ndarray, y: np.ndarray, n_per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample both classes to exactly ``n_per_class`` rows each."""
    fraud_idx = np.flatnonzero(y == 1)
    nonfraud_idx = np.flatnonzero(y == 0)
    if len(fraud_idx) < n_per_class or len(nonfraud_idx) < n_per_class:
        raise ValueError(
            f"Not enough rows to balance to {n_per_class}/class: "
            f"fraud={len(fraud_idx)} nonfraud={len(nonfraud_idx)}"
        )
    chosen_fraud = rng.choice(fraud_idx, size=n_per_class, replace=False)
    chosen_nonfraud = rng.choice(nonfraud_idx, size=n_per_class, replace=False)
    combined_idx = np.concatenate([chosen_fraud, chosen_nonfraud])
    rng.shuffle(combined_idx)
    return X[combined_idx], y[combined_idx]


def prepare_benchmark_pools(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the train/test pools used by the Table 1 benchmark.

    Returns a dict with (all scaled via a MinMaxScaler fit on the balanced
    train pool):
      - ``X_train_full``, ``y_train``: balanced ``n_per_class``/class train
        pool, full feature dimensionality (used directly by GQC).
      - ``X_test_full``, ``y_test``: the held-out 10% test pool, full
        feature dimensionality.
      - ``X_train_pca``, ``X_test_pca``: the same two pools PCA-reduced to
        ``pca.n_components`` (default 2) features, fit on the train pool
        only (used by QMKL/GFM).
      - ``pca``: the fitted ``sklearn.decomposition.PCA`` object.
    """
    seed = int(cfg.get("seed", 42))
    ds_cfg = cfg.get("dataset", {})
    n_samples = int(ds_cfg.get("n_samples", 10000))
    test_fraction = float(ds_cfg.get("test_fraction", 0.1))
    n_per_class = int(ds_cfg.get("train_balance_per_class", 90))
    n_pca = int(cfg.get("pca", {}).get("n_components", 2))

    X, y, _feature_names = load_raw_data(cfg)

    rng = np.random.default_rng(seed)
    X_sub, y_sub = _reduce_to_n_samples(X, y, n_samples, rng)

    X_train_pool, X_test_pool, y_train_pool, y_test_pool = train_test_split(
        X_sub, y_sub, test_size=test_fraction, stratify=y_sub, random_state=seed
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train_pool)
    X_train_pool_s = scaler.transform(X_train_pool)
    X_test_pool_s = scaler.transform(X_test_pool)

    X_train_bal, y_train_bal = _balance_to_n_per_class(
        X_train_pool_s, y_train_pool, n_per_class, rng
    )

    pca = PCA(n_components=n_pca, random_state=seed)
    pca.fit(X_train_bal)
    X_train_pca = pca.transform(X_train_bal)
    X_test_pca = pca.transform(X_test_pool_s)

    return {
        "X_train_full": X_train_bal,
        "y_train": y_train_bal,
        "X_test_full": X_test_pool_s,
        "y_test": y_test_pool,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "pca": pca,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Quantum kernel feature maps (QMKL: three distinct maps; GFM: one)
# ---------------------------------------------------------------------------


def _fm_angle_only(x) -> None:
    """Feature map 1: plain angle encoding, no entanglement."""
    qml.RY(x[..., 0], wires=0)
    qml.RY(x[..., 1], wires=1)


def _fm_zz_interaction(x) -> None:
    """Feature map 2: angle encoding + one ZZ-style interaction term."""
    qml.RY(x[..., 0], wires=0)
    qml.RY(x[..., 1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(x[..., 0] * x[..., 1], wires=1)
    qml.CNOT(wires=[0, 1])


def _fm_entangled(x) -> None:
    """Feature map 3 (also used, alone, as the GFM feature map): angle
    encoding + one entangling CNOT."""
    qml.RY(x[..., 0], wires=0)
    qml.RY(x[..., 1], wires=1)
    qml.CNOT(wires=[0, 1])


_QMKL_FEATURE_MAPS = [_fm_angle_only, _fm_zz_interaction, _fm_entangled]
_GFM_FEATURE_MAP = _fm_entangled


def _make_kernel_row_circuit(feature_map: Callable, n_qubits: int = 2):
    """Build a QNode computing ``k(x_test, x_train_i)`` for every row of a
    batched ``x_train`` array in one call (PennyLane parameter broadcasting).

    Returns probability of the all-zeros outcome, i.e.
    ``|<0|U(x_train)^dagger U(x_test)|0>|^2``, the standard fidelity kernel.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x_test, x_train_batch):
        feature_map(x_test)
        qml.adjoint(feature_map)(x_train_batch)
        return qml.probs(wires=range(n_qubits))

    return circuit


class QuantumKernelRow:
    """Callable computing a kernel row ``k(x_test, X_train)`` for one or more
    2-qubit feature maps, combined by simple averaging (QMKL: 3 maps; GFM: 1
    map -- see module docstring for the equal-weighting assumption)."""

    def __init__(self, feature_maps: list[Callable]) -> None:
        self.circuits = [_make_kernel_row_circuit(fm) for fm in feature_maps]

    def __call__(self, x_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        kernel_sum = None
        for circuit in self.circuits:
            probs = circuit(x_test, X_train)  # shape (n_train, 4)
            k = np.asarray(probs)[:, 0]  # prob of |00>
            kernel_sum = k if kernel_sum is None else kernel_sum + k
        return kernel_sum / len(self.circuits)


def qmkl_kernel_row(x_test: np.ndarray, X_train_pca: np.ndarray) -> np.ndarray:
    """QMKL: equal-weighted average of three distinct 2-qubit kernels."""
    return QuantumKernelRow(_QMKL_FEATURE_MAPS)(x_test, X_train_pca)


def gfm_kernel_row(x_test: np.ndarray, X_train_pca: np.ndarray) -> np.ndarray:
    """GFM: a single fixed 2-qubit entangling kernel."""
    return QuantumKernelRow([_GFM_FEATURE_MAP])(x_test, X_train_pca)


# ---------------------------------------------------------------------------
# Timing protocol
# ---------------------------------------------------------------------------


def time_batch_inference(
    batch_call: Callable[[], None], *, n_runs: int, n_reps: int, batch_size: int
) -> dict[str, float]:
    """Run the paper's warm-up + repeated-run timing protocol.

    ``batch_call`` must, in one call, perform inference over exactly
    ``batch_size`` samples (whatever "inference" means for the method being
    timed). It is called ``n_reps`` times per run, ``n_runs`` times total.
    Run 0 is a hardware warm-up and excluded from the reported statistics
    (matching the paper's stated protocol).

    Returns ``{"ms_per_sample": mean, "cv_pct": coefficient_of_variation}``
    computed over per-run ms/sample across runs ``1..n_runs-1``.
    """
    if n_runs < 2:
        raise ValueError("n_runs must be >= 2 (run 0 is warm-up and excluded)")
    samples_per_run = n_reps * batch_size
    run_times_s: list[float] = []
    for _run in range(n_runs):
        t0 = time.perf_counter()
        for _rep in range(n_reps):
            batch_call()
        t1 = time.perf_counter()
        run_times_s.append(t1 - t0)

    measured_runs = run_times_s[1:]  # exclude warm-up run 0
    ms_per_sample_per_run = np.array(
        [(t / samples_per_run) * 1000.0 for t in measured_runs]
    )
    mean_ms = float(np.mean(ms_per_sample_per_run))
    std_ms = (
        float(np.std(ms_per_sample_per_run, ddof=1)) if len(measured_runs) > 1 else 0.0
    )
    cv_pct = float(100.0 * std_ms / mean_ms) if mean_ms > 0 else 0.0
    return {"ms_per_sample": mean_ms, "cv_pct": cv_pct, "std_ms_per_sample": std_ms}


# ---------------------------------------------------------------------------
# Per-method batch-inference callables
# ---------------------------------------------------------------------------


def _make_kernel_batch_call(
    kernel_row_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X_test_batch: np.ndarray,
    X_train_pca: np.ndarray,
) -> Callable[[], None]:
    def _call() -> None:
        for i in range(X_test_batch.shape[0]):
            kernel_row_fn(X_test_batch[i], X_train_pca)

    return _call


def _build_gqc_model(cfg: dict[str, Any], input_dim: int) -> GQCModel:
    model_cfg = dict(cfg.get("gqc_model", {}))
    model_cfg.setdefault("backend", "gate")
    gqc = GQCModel(input_dim=input_dim, cfg={"model": model_cfg})
    gqc.eval()
    return gqc


def _make_gqc_batch_call(
    model: GQCModel, X_test_batch: np.ndarray
) -> Callable[[], None]:
    x_t = torch.tensor(X_test_batch, dtype=torch.float32)

    def _call() -> None:
        with torch.no_grad():
            model.predict_proba(x_t)

    return _call


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_latency_benchmark(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Reproduce Table 1: per-sample inference latency for QMKL, GFM, GQC.

    Returns ``{"QMKL": {...}, "GFM": {...}, "GQC": {...}}`` where each value
    is the dict returned by :func:`time_batch_inference`.
    """
    bench_cfg = cfg.get("benchmark", {})
    n_runs = int(bench_cfg.get("n_runs", 5))
    n_reps = int(bench_cfg.get("n_reps", 10))
    batch_size = int(bench_cfg.get("batch_size", 50))

    pools = prepare_benchmark_pools(cfg)

    n_test = pools["X_test_full"].shape[0]
    if n_test < batch_size:
        raise ValueError(
            f"Test pool ({n_test} rows) smaller than requested batch_size ({batch_size})"
        )
    # Fixed batch reused across every run/rep, per the paper's "10 internal
    # repetitions on a batch of 50 samples" protocol (timing the same batch
    # repeatedly, not resampling it each rep).
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 999)
    batch_idx = rng.choice(n_test, size=batch_size, replace=False)
    X_test_pca_batch = pools["X_test_pca"][batch_idx]
    X_test_full_batch = pools["X_test_full"][batch_idx]

    results: dict[str, dict[str, float]] = {}

    qmkl_call = _make_kernel_batch_call(
        qmkl_kernel_row, X_test_pca_batch, pools["X_train_pca"]
    )
    results["QMKL"] = time_batch_inference(
        qmkl_call, n_runs=n_runs, n_reps=n_reps, batch_size=batch_size
    )

    gfm_call = _make_kernel_batch_call(
        gfm_kernel_row, X_test_pca_batch, pools["X_train_pca"]
    )
    results["GFM"] = time_batch_inference(
        gfm_call, n_runs=n_runs, n_reps=n_reps, batch_size=batch_size
    )

    input_dim = pools["X_train_full"].shape[1]
    gqc_model = _build_gqc_model(cfg, input_dim)
    gqc_call = _make_gqc_batch_call(gqc_model, X_test_full_batch)
    results["GQC"] = time_batch_inference(
        gqc_call, n_runs=n_runs, n_reps=n_reps, batch_size=batch_size
    )

    return results


__all__ = [
    "prepare_benchmark_pools",
    "qmkl_kernel_row",
    "gfm_kernel_row",
    "time_batch_inference",
    "run_latency_benchmark",
]
