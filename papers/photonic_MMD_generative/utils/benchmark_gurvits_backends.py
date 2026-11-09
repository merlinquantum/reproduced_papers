"""Benchmark the Gurvits/Glynn permanent estimators: this reproduction's JAX code vs MerLin photonic_engines.

Quantity estimated (the inner loop of the paper's loss, Fig. 1 / Eq. (15)): for a Haar
interferometer U (m modes), input modes S (n photons, ``middle_alternating``) and K Walsh
operators drawn as in the paper's loss, k ~ P_sigma (Gaussian kernel, Prop. 1; default sigma = 3,
the Fig. 3a setting, so each bit is 1 with probability p_sigma ~ 0.014), the expectations

    <W_k> = Per(A_k),   A_k = U_S^dagger diag(1 - 2k) U_S   (n x n),

each from Z random sign vectors. Backends (same U, S, K operators, Z, dtype complex128, device):

- ``jax``            : lib/estimator.py, A_k from the n input columns, Glynn signs z in {+-1}^n,
                       vmapped over the K operators and jit-compiled (compile time excluded,
                       reported separately).
- ``pe_matched_rad`` : photonic_engines ``EmitterGurvitsEstimator`` (Fock input as a chi=1
                       EmitterMPS) on the n-mode problem (matrix A_k, output 1^n), Rademacher
                       sampling: the same estimator as ``jax``, a different implementation.
- ``pe_matched_gauss``: same, complex Gaussian sampling.
- ``pe_native_rad``  : the estimator as photonic_engines is meant to be used: amplitude
                       <s| V_k |s> of the full m-mode unitary V_k = U^dagger diag(1-2k) U, so
                       every shot also draws the m - n unused modes (O(m^2) per shot, larger variance).
- ``pe_native_gauss``: same, Gaussian sampling (the variant with the second-moment theorem).

photonic_engines builds one estimate per call (python loop over the K operators) and returns a
python number, so it has no autograd; only forward times are compared. Building A_k / V_k is
included in each backend's time.

Fairness: the estimators have different variances, so the headline comparison is at equal
precision: utils/plot_gurvits_benchmark.py reports (t_backend / t_jax) * (SE_backend / SE_jax)^2,
SE the median per-operator standard error over the K operators. ``seconds_per_operator_at_eps``
(time to an absolute SE eps) is recorded too but is not comparable across n, since permanents
shrink quickly with n. Raw time per operator at equal
shot count is reported too. Timing: warm-up call excluded, median of ``--repeats`` timed calls,
CUDA synchronized. Correctness: for n <= 10 every backend's estimates are compared with exact
Ryser permanents (max |z| over operators).

Usage (from the paper folder)::

    python utils/benchmark_gurvits_backends.py --device cpu --sweep photons --m 256 --n 2 4 8 12 16 24
    python utils/benchmark_gurvits_backends.py --device cuda --sweep modes --n 8 --m 16 32 64 128 256 512

Writes ``outdir/bench_gurvits_<device>_<sweep>_<stamp>/`` with run.log, run_status.json,
config_snapshot.json and metrics.json (one record per (backend, m, n)).
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

jax.config.update("jax_enable_x64", True)

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "papers"))

from photonic_MMD_generative.lib import circuits, data, estimator, exact, kernels  # noqa: E402
from photonic_MMD_generative.lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)

logger = logging.getLogger("bench_gurvits")
BACKENDS = (
    "jax",
    "pe_matched_rad",
    "pe_matched_gauss",
    "pe_native_rad",
    "pe_native_gauss",
)
EXACT_MAX_N = 10


def problem(m: int, n: int, n_ops: int, seed: int, sigma: float):
    """Haar U, input modes S and K Walsh operators k ~ P_sigma (rows of 0/1), as in the loss."""
    U = np.asarray(
        circuits.haar_unitary(
            jax.random.normal(jax.random.PRNGKey(seed), (2 * m * m,)), m
        )
    )
    modes = data.input_modes(m, n, "middle_alternating")
    sampler = kernels.make_operator_sampler({"type": "gaussian", "sigma": sigma}, m, n)
    K = np.asarray(sampler(jax.random.PRNGKey(seed + 1), n_ops), dtype=np.float64)
    return U, modes, K


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_jax(U, modes, K, n_z, device):
    """Return a callable producing (estimates [K], per-operator SE [K]) with the JAX estimator."""
    jdev = jax.devices("gpu" if device.startswith("cuda") else "cpu")[0]
    U_S = jax.device_put(jnp.asarray(U[:, modes]), jdev)
    Kj = jax.device_put(jnp.asarray(K), jdev)
    n = len(modes)

    @jax.jit
    def run(key):
        Z = jax.random.rademacher(key, (n_z, n), dtype=jnp.float64)

        def one(k):
            g = estimator.glynn_samples(estimator.walsh_operator_matrix(U_S, k), Z)
            return jnp.mean(g), jnp.sqrt(jnp.var(g) / n_z)

        return jax.vmap(one)(Kj)

    counter = [0]

    def call():
        counter[0] += 1
        mean, se = run(jax.random.PRNGKey(counter[0]))
        mean.block_until_ready()
        return np.asarray(mean), np.asarray(se)

    return call


def fock_state(modes, m):
    from merlin.photonic_engines.states import EmitterMPS

    n = len(modes)
    return EmitterMPS(
        ops=np.ones((n, 1, 1, 1)),
        modes=tuple((int(s),) for s in modes),
        s0=np.ones(1),
        final=np.ones(1),
        m=m,
    )


def make_pe(U, modes, K, n_z, device, variant):
    """photonic_engines EmitterGurvitsEstimator, one amplitude() call per operator."""
    from merlin.photonic_engines.slos.estimate import EmitterGurvitsEstimator

    m, n = U.shape[0], len(modes)
    sampling = "rademacher" if variant.endswith("rad") else "gaussian"
    native = "native" in variant
    tdev = torch.device(device)
    Ut = torch.as_tensor(U, dtype=torch.complex128, device=tdev)
    U_S = Ut[:, torch.as_tensor(modes, device=tdev)]
    signs = torch.as_tensor(1.0 - 2.0 * K, dtype=torch.complex128, device=tdev)
    if native:
        est = EmitterGurvitsEstimator(
            fock_state(modes, m), sampling=sampling, device=device, seed=1
        )
        output = [1 if j in set(modes) else 0 for j in range(m)]
    else:
        est = EmitterGurvitsEstimator(
            fock_state(list(range(n)), n), sampling=sampling, device=device, seed=1
        )
        output = [1] * n

    def call():
        means, ses = [], []
        for i in range(K.shape[0]):
            if native:
                matrix = Ut.conj().T @ (signs[i][:, None] * Ut)  # V_k, m x m
            else:
                matrix = U_S.conj().T @ (signs[i][:, None] * U_S)  # A_k, n x n
            e = est.amplitude(matrix, output, n_z)
            means.append(e.value)
            ses.append(e.stderr)
        sync(device)
        return np.asarray(means), np.asarray(ses)

    return call


def exact_permanents(U, modes, K):
    U_S = U[:, modes]
    return np.array(
        [exact.permanent(U_S.conj().T @ ((1 - 2 * k)[:, None] * U_S)) for k in K]
    )


def measure(call, repeats: int, device: str):
    t0 = time.perf_counter()
    mean, se = call()  # warm-up (JIT compile, CUDA context, allocator)
    sync(device)
    warmup = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mean, se = call()
        sync(device)
        times.append(time.perf_counter() - t0)
    return mean, se, warmup, times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sweep", choices=("photons", "modes"), required=True)
    parser.add_argument("--m", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--n-ops", type=int, default=50, help="Walsh operators K")
    parser.add_argument(
        "--n-z", type=int, default=2000, help="sign vectors / shots per operator"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--eps", type=float, default=1e-3, help="target SE for the equal-precision time"
    )
    parser.add_argument(
        "--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sigma", type=float, default=3.0, help="Gaussian kernel bandwidth of P_sigma"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=600.0,
        help="skip larger sizes for a backend once one point exceeds this many seconds",
    )
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (
        PAPER_DIR
        / "outdir"
        / f"bench_gurvits_{args.device.replace(':', '')}_{args.sweep}_{stamp}"
    )
    run_dir.mkdir(parents=True)
    configure_logging(run_dir / "run.log")
    write_json(run_dir / "config_snapshot.json", vars(args))
    import merlin

    status = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "sweep_id": None,
        "candidate": {
            "device": args.device,
            "sweep": args.sweep,
            "backends": args.backends,
        },
        "seed": args.seed,
        "repetition": 0,
        "dataset": {
            "name": "synthetic_haar_walsh",
            "splits": [],
            "preprocessing": {
                "unitary": "Haar (lib.circuits.haar_unitary)",
                "operators": f"P_sigma, sigma={args.sigma}",
                "input_state": "middle_alternating",
            },
            "subset": None,
        },
        "selection": {
            "name": "none (timing benchmark, no selection)",
            "direction": "minimize",
            "split": None,
            "checkpoint": None,
        },
        "config_sha256": sha256(run_dir / "config_snapshot.json"),
        "metrics_sha256": None,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "config_path": "config_snapshot.json",
        "metrics_path": "metrics.json",
        "log_path": "run.log",
        "environment": {
            "torch": torch.__version__,
            "jax": jax.__version__,
            "merlin": getattr(merlin, "__version__", "unknown"),
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else None,
            "torch_threads": torch.get_num_threads(),
        },
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info("RUN_STARTED | run_id=%s | args=%s", run_dir.name, vars(args))
    logger.info(
        "DATASET_READY | name=synthetic_haar_walsh | U = Haar(seed=%d) per (m, n) | operators k ~ P_sigma(sigma=%g), K=%d | input=middle_alternating",
        args.seed,
        args.sigma,
        args.n_ops,
    )
    records, skip = [], set()
    try:
        grid = [(m, n) for m in args.m for n in args.n if m >= 2 * n]
        for m, n in grid:
            U, modes, K = problem(m, n, args.n_ops, args.seed, args.sigma)
            perms = exact_permanents(U, modes, K) if n <= EXACT_MAX_N else None
            for backend in args.backends:
                if backend in skip:
                    logger.info(
                        "SKIPPED | backend=%s | m=%d | n=%d | over time budget",
                        backend,
                        m,
                        n,
                    )
                    continue
                call = (
                    make_jax(U, modes, K, args.n_z, args.device)
                    if backend == "jax"
                    else make_pe(U, modes, K, args.n_z, args.device, backend)
                )
                mean, se, warmup, times = measure(call, args.repeats, args.device)
                t_call = statistics.median(times)
                t_op = t_call / args.n_ops
                se_med = float(np.median(se))
                # Root-mean-square SE over the K operators: the total variance is what sets the noise
                # of the loss; the median can be 0 when most sparse operators are trivial (k = 0, A = I).
                se_rms = float(np.sqrt(np.mean(np.asarray(se) ** 2)))
                record = {
                    "backend": backend,
                    "m": m,
                    "n": n,
                    "n_ops": args.n_ops,
                    "n_z": args.n_z,
                    "warmup_seconds": warmup,
                    "seconds_per_call": t_call,
                    "seconds_all_repeats": times,
                    "seconds_per_operator": t_op,
                    "seconds_per_shot": t_op / args.n_z,
                    "median_se": se_med,
                    "rms_se": se_rms,
                    "trivial_operators": int((K.sum(axis=1) == 0).sum()),
                    "seconds_per_operator_at_eps": t_op * (se_med / args.eps) ** 2,
                    "max_abs_z_vs_exact": None
                    if perms is None
                    else float(np.max(np.abs(mean - perms) / np.maximum(se, 1e-10))),
                }
                records.append(record)
                logger.info(
                    "EVALUATION_COMPLETED | backend=%s | m=%d | n=%d | s/op=%.3e | SE_rms=%.2e | trivial_ops=%d | warmup=%.2fs | max|z|=%s",
                    backend,
                    m,
                    n,
                    t_op,
                    se_rms,
                    record["trivial_operators"],
                    warmup,
                    "n/a"
                    if record["max_abs_z_vs_exact"] is None
                    else f"{record['max_abs_z_vs_exact']:.2f}",
                )
                if t_call * (args.repeats + 1) > args.time_budget:
                    skip.add(backend)
                write_json(
                    run_dir / "metrics.json", {"records": records, "eps": args.eps}
                )
        write_json(run_dir / "metrics.json", {"records": records, "eps": args.eps})
        status["metrics_sha256"] = sha256(run_dir / "metrics.json")
        logger.info("METRICS_WRITTEN | path=metrics.json | records=%d", len(records))
        status.update(status="COMPLETED", completed_at=utc_now())
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "RUN_COMPLETED | run_id=%s | records=%d", run_dir.name, len(records)
        )
        return 0
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.error("RUN_FAILED | %s\n%s", exc, traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
