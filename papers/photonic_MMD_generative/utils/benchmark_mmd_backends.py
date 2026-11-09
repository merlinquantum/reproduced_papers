"""Benchmark backends on the paper's task: evaluating the MMD^2 training loss of a photonic QCBM.

For a Haar interferometer U (m modes, n photons at ``middle_alternating`` inputs) and boson-sampling
data X (the repository recipe, a different Haar unitary), every route estimates the same quantity,
the paper's loss (Prop. 1 with the Gaussian kernel, sigma = 3 by default):

    L(U) = sum_k P_sigma(k) [ <W_k>_U^2 - 2 <W_k>_U T_k ] + U-statistic of the data term
         = sum_{x,y} p(x) p(y) kappa(x, y) - 2 sum_x p(x) mean_j kappa(x, X_j) + data term,
    kappa(x, y) = c^{H(x mod 2, y mod 2)},  c = exp(-1 / (2 sigma^2)),

with p the boson-sampling distribution of U (collisions included; W_k only sees parities).

Routes (same U, X, sigma, device, complex128):

- ``jax_gurvits``    this reproduction (lib/estimator.make_mmd_loss): K operators k ~ P_sigma, Z
                     Glynn sign vectors per operator, jit-compiled; forward and value_and_grad.
- ``pe_gurvits``     the same loss with each <W_k> estimated by photonic_engines'
                     ``EmitterGurvitsEstimator`` (matched mapping: n x n matrix A_k, Rademacher;
                     the fastest of the mappings measured before); forward only (no autograd).
- ``pe_slos_exact``  photonic_engines ``FockSLOSKernel`` output distribution over all
                     C(m+n-1, n) states, then the exact L (the D x D kernel contraction is part of
                     the route); forward and backward (torch autograd). No statistical error.
- ``pe_sampler``     photonic_engines Clifford sampler (``clifford.fock_sample``, CUDA only): N
                     samples of U, unbiased sample MMD^2 against X with kappa; forward only.

Fair comparison: stochastic routes are repeated R times with fresh randomness; SE = std over the
repeats. Equal-precision time t* = t_eval * (SE / eps)^2 (averaging r independent evaluations
divides the variance by r and multiplies the time by r), eps = ``--rel-eps`` * |L_ref|, L_ref
the exact SLOS value when it is computed, else the mean of the JAX repeats. SLOS: t* = t_eval.
Correctness: each stochastic route's mean vs the exact value (z-score) where SLOS ran.
Warm-up (JIT, CUDA context, SLOS basis construction) excluded and reported.

Usage (from the paper folder)::

    python utils/benchmark_mmd_backends.py --device cpu --sweep photons --m 32 --n 2 3 4 6 8 \
        --data-root ../../data/photonic_MMD_generative_prepared

Writes ``outdir/bench_mmd_<device>_<sweep>_<stamp>/`` (run.log, run_status.json,
config_snapshot.json, metrics.json: one record per (route, m, n)).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import statistics
import sys
import time
import traceback
from math import comb
from pathlib import Path

import numpy as np
import torch

# JAX preallocates ~75% of GPU memory by default, which starves the torch routes running in the
# same process (CUDA OOM in the SLOS contraction at 52k states); allocate on demand instead.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "papers"))

from photonic_MMD_generative.lib import (  # noqa: E402
    circuits,
    data,
    estimator,
    kernels,
    qcbm_merlin,
)
from photonic_MMD_generative.lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)

logger = logging.getLogger("bench_mmd")
ROUTES = ("jax_gurvits", "pe_gurvits", "pe_slos_exact", "pe_sampler")


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def parity_kernel_sums(
    A: torch.Tensor, B: torch.Tensor, wa, wb, c: float, max_bytes: int = 256 << 20
):
    """sum_ij wa_i wb_j c^{H(A_i, B_j)} for 0/1 float matrices A [Na, m], B [Nb, m].

    Rows of A are processed in chunks whose [chunk, Nb] float64 temporaries stay near ``max_bytes``.
    """
    chunk = max(1, max_bytes // (8 * B.shape[0]))
    log_c = math.log(c)
    norms_b = B.sum(1)
    total = torch.zeros((), dtype=A.dtype, device=A.device)
    for start in range(0, A.shape[0], chunk):
        a = A[start : start + chunk]
        hamming = a.sum(1)[:, None] + norms_b[None, :] - 2.0 * (a @ B.T)
        total = total + (wa[start : start + chunk] @ torch.exp(log_c * hamming) @ wb)
    return total


@torch.no_grad()
def parity_kernel_matvec(
    A: torch.Tensor, B: torch.Tensor, w, c: float, max_bytes: int = 256 << 20
):
    """(K w)_i = sum_j c^{H(A_i, B_j)} w_j, chunked over the rows of A, no autograd graph."""
    chunk = max(1, max_bytes // (8 * B.shape[0]))
    log_c = math.log(c)
    norms_b = B.sum(1)
    out = torch.empty(A.shape[0], dtype=A.dtype, device=A.device)
    for start in range(0, A.shape[0], chunk):
        a = A[start : start + chunk]
        hamming = a.sum(1)[:, None] + norms_b[None, :] - 2.0 * (a @ B.T)
        out[start : start + chunk] = torch.exp(log_c * hamming) @ w
    return out


def data_term(Xb: torch.Tensor, c: float) -> torch.Tensor:
    """Unbiased U-statistic of kappa over distinct pairs of data samples."""
    N = Xb.shape[0]
    ones = torch.ones(N, dtype=Xb.dtype, device=Xb.device)
    return (parity_kernel_sums(Xb, Xb, ones, ones, c) - N) / (N * (N - 1))


def build_problem(m, n, args):
    """Model parameters (Haar QR params), input modes, data X (train split of the BS recipe)."""
    modes = data.input_modes(m, n, "middle_alternating")
    params = jax.random.normal(jax.random.PRNGKey(args.seed), (2 * m * m,))
    cfg = {
        "dataset": {
            "name": "boson_sampling",
            "m": m,
            "n": n,
            "n_train": args.n_data,
            "n_test": args.n_data,
            "input_state": "middle_alternating",
            "unitary_seed": 10_000 + 100 * m + n,
            "sample_seed": 20_000 + 100 * m + n,
        },
        "data_root": str(args.data_root),
    }
    ds = data.load_dataset(cfg)
    return params, modes, ds["train"], ds["manifest"]


def route_jax(m, n, params, modes, X, args):
    unitary_fn = circuits.make_unitary_fn(m, "haar")
    sampler = kernels.make_operator_sampler(
        {"type": "gaussian", "sigma": args.sigma}, m, n
    )
    loss = estimator.make_mmd_loss(
        unitary_fn, modes, sampler, args.n_ops, args.n_z, "indistinguishable"
    )
    Xj = jnp.asarray(X, dtype=jnp.float64)
    fwd = jax.jit(lambda p, k: loss(p, k, Xj))
    vg = jax.jit(jax.value_and_grad(lambda p, k: loss(p, k, Xj)))
    counter = [0]

    def forward():
        counter[0] += 1
        return float(fwd(params, jax.random.PRNGKey(counter[0])).block_until_ready())

    def grad():
        counter[0] += 1
        value, g = vg(params, jax.random.PRNGKey(counter[0]))
        g.block_until_ready()
        return float(value)

    return forward, grad


def route_pe_gurvits(m, n, params, modes, X, args):
    from merlin.photonic_engines.slos.estimate import EmitterGurvitsEstimator
    from merlin.photonic_engines.states import EmitterMPS

    dev = torch.device(args.device)
    U = torch.as_tensor(
        np.asarray(circuits.haar_unitary(params, m)), dtype=torch.complex128, device=dev
    )
    U_S = U[:, torch.as_tensor(modes, device=dev)]
    sampler = kernels.make_operator_sampler(
        {"type": "gaussian", "sigma": args.sigma}, m, n
    )
    state = EmitterMPS(
        ops=np.ones((n, 1, 1, 1)),
        modes=tuple((i,) for i in range(n)),
        s0=np.ones(1),
        final=np.ones(1),
        m=n,
    )
    est = EmitterGurvitsEstimator(
        state, sampling="rademacher", device=args.device, seed=1
    )
    N = X.shape[0]
    counter = [0]

    def forward():
        counter[0] += 1
        K = np.asarray(
            sampler(jax.random.PRNGKey(10_000 + counter[0]), args.n_ops),
            dtype=np.float64,
        )
        T = (1.0 - 2.0 * ((X @ K.T) % 2)).mean(axis=0)  # data Walsh expectations, [K]
        signs = torch.as_tensor(1.0 - 2.0 * K, dtype=torch.complex128, device=dev)
        total = 0.0
        for i in range(K.shape[0]):
            e = est.amplitude(
                U_S.conj().T @ (signs[i][:, None] * U_S), [1] * n, args.n_z
            )
            g = e.value.real
            # unbiased square of the real part up to O(stderr^2) (the API returns mean and stderr only)
            total += (
                (g * g - 0.5 * e.stderr**2)
                - 2.0 * g * T[i]
                + (N * T[i] ** 2 - 1.0) / (N - 1)
            )
        sync(args.device)
        return total / K.shape[0]

    return forward, None


def route_slos(m, n, params, modes, X, args):
    from merlin.photonic_engines.slos import FockSLOSKernel

    dev = torch.device(args.device)
    occ = [1 if j in set(modes) else 0 for j in range(m)]
    kernel = FockSLOSKernel(n, m, occ, device=args.device, dtype=torch.complex128)
    B = torch.as_tensor(
        kernel.layers.final_occupations() % 2, dtype=torch.float64, device=dev
    )
    Xb = torch.as_tensor(X % 2, dtype=torch.float64, device=dev)
    c = math.exp(-1.0 / (2.0 * args.sigma**2))
    const = data_term(Xb, c)
    ones_x = torch.full(
        (Xb.shape[0],), 1.0 / Xb.shape[0], dtype=torch.float64, device=dev
    )
    theta = torch.as_tensor(np.asarray(params), dtype=torch.float64, device=dev)

    # Data cross term sum_j kappa(x, X_j) / N for every basis state x: independent of U, so it is
    # computed once (like the data term), not per loss evaluation.
    cross = parity_kernel_matvec(B, Xb, ones_x, c)

    def loss(th):
        p = kernel.probs(qcbm_merlin.haar_unitary_torch(th, m))
        # p^T K p with K the D x D parity kernel. Exact gradient 2 K p without storing K: K p is
        # computed in chunks outside autograd and attached through a surrogate with the same
        # value and gradient (memory O(D) instead of O(D^2)).
        Kp = parity_kernel_matvec(B, B, p.detach(), c)
        p0 = p.detach()
        quadratic = p0 @ Kp + 2.0 * ((p - p0) @ Kp)
        return quadratic - 2.0 * (p @ cross) + const

    def forward():
        with torch.no_grad():
            value = loss(theta)
        sync(args.device)
        return float(value)

    def grad():
        th = theta.clone().requires_grad_()
        value = loss(th)
        value.backward()
        sync(args.device)
        return float(value)

    return forward, grad, kernel.output_size()


def route_sampler(m, n, params, modes, X, args):
    from merlin.photonic_engines.clifford import fock_sample

    dev = torch.device(args.device)
    U = torch.as_tensor(
        np.asarray(circuits.haar_unitary(params, m)), dtype=torch.complex128, device=dev
    )
    Xb = torch.as_tensor(X % 2, dtype=torch.float64, device=dev)
    c = math.exp(-1.0 / (2.0 * args.sigma**2))
    const = data_term(Xb, c)
    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    N, Nx = args.n_samples, Xb.shape[0]
    ones_s = torch.ones(N, dtype=torch.float64, device=dev)
    ones_x = torch.ones(Nx, dtype=torch.float64, device=dev)

    def forward():
        S = torch.as_tensor(fock_sample(U, modes, N, generator=gen), device=dev) % 2
        S = S.to(torch.float64)
        model = (parity_kernel_sums(S, S, ones_s, ones_s, c) - N) / (N * (N - 1))
        cross = parity_kernel_sums(S, Xb, ones_s, ones_x, c) / (N * Nx)
        value = model - 2.0 * cross + const
        sync(args.device)
        return float(value)

    return forward, None


def timed(fn, repeats, device):
    t0 = time.perf_counter()
    fn()
    sync(device)
    warmup = time.perf_counter() - t0
    values, times = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        values.append(fn())
        sync(device)
        times.append(time.perf_counter() - t0)
    return values, times, warmup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sweep", choices=("photons", "modes"), required=True)
    parser.add_argument("--m", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument(
        "--n-ops", type=int, default=500, help="Walsh operators K per loss estimate"
    )
    parser.add_argument(
        "--n-z", type=int, default=5000, help="Glynn sign vectors Z per operator"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="sampler route: samples per estimate",
    )
    parser.add_argument(
        "--n-data", type=int, default=2000, help="data samples (train split)"
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--rel-eps", type=float, default=0.01, help="target SE relative to |L|"
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=60_000,
        help="SLOS route only up to this C(m+n-1,n)",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=900.0,
        help="drop a route after a point slower than this",
    )
    parser.add_argument("--routes", nargs="+", default=list(ROUTES), choices=ROUTES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # Relative data roots are resolved from the paper folder (the shared runtime's convention), so
    # the command works from any working directory, including a remote launch's repository copy.
    if not args.data_root.is_absolute():
        args.data_root = PAPER_DIR / args.data_root
    args.data_root = args.data_root.resolve()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (
        PAPER_DIR
        / "outdir"
        / f"bench_mmd_{args.device.replace(':', '')}_{args.sweep}_{stamp}"
    )
    run_dir.mkdir(parents=True)
    configure_logging(run_dir / "run.log")
    write_json(
        run_dir / "config_snapshot.json",
        {**vars(args), "data_root": str(args.data_root)},
    )
    import merlin

    status = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "sweep_id": None,
        "candidate": {
            "device": args.device,
            "sweep": args.sweep,
            "routes": args.routes,
        },
        "seed": args.seed,
        "repetition": 0,
        "dataset": {
            "name": "boson_sampling (repository recipe, one prepared dataset per (m, n))",
            "splits": ["train"],
            "preprocessing": {
                "input_state": "middle_alternating",
                "kernel": f"gaussian sigma={args.sigma}",
            },
            "subset": [],
        },
        "selection": {
            "name": "none (timing benchmark)",
            "direction": "minimize",
            "split": None,
            "checkpoint": None,
        },
        "code": git_state(REPO_ROOT),
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(run_dir / "config_snapshot.json"),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "log_path": "run.log",
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "environment": {
            "torch": torch.__version__,
            "jax": jax.__version__,
            "merlin": getattr(merlin, "__version__", "unknown"),
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else None,
            "jax_devices": [str(d) for d in jax.devices()],
            "torch_threads": torch.get_num_threads(),
        },
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info("RUN_STARTED | run_id=%s | args=%s", run_dir.name, vars(args))
    records, dropped = [], set()
    routes = [
        r
        for r in args.routes
        if not (r == "pe_sampler" and not args.device.startswith("cuda"))
    ]
    try:
        grid = [(m, n) for m in args.m for n in args.n if m >= 2 * n]
        problems = {}
        for m, n in grid:
            params, modes, X, manifest = build_problem(m, n, args)
            problems[(m, n)] = (params, modes, X)
            status["dataset"]["subset"].append(
                {
                    "m": m,
                    "n": n,
                    "dataset_id": manifest["dataset_id"],
                    "train_sha256": manifest["train_sha256"],
                    "train_shape": list(X.shape),
                    "collision_fraction": manifest.get("collision_fraction"),
                }
            )
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "DATASET_READY | %d prepared boson-sampling datasets | root=%s",
            len(grid),
            args.data_root,
        )
        for m, n in grid:
            params, modes, X = problems[(m, n)]
            states = comb(m + n - 1, n)
            exact = None
            point = []
            for route in routes:
                if route in dropped or (
                    route == "pe_slos_exact" and states > args.max_states
                ):
                    logger.info(
                        "SKIPPED | route=%s | m=%d | n=%d | states=%d",
                        route,
                        m,
                        n,
                        states,
                    )
                    continue
                if route == "jax_gurvits":
                    fwd, grd = route_jax(m, n, params, modes, X, args)
                elif route == "pe_gurvits":
                    fwd, grd = route_pe_gurvits(m, n, params, modes, X, args)
                elif route == "pe_slos_exact":
                    fwd, grd, _ = route_slos(m, n, params, modes, X, args)
                else:
                    fwd, grd = route_sampler(m, n, params, modes, X, args)
                reps = 3 if route == "pe_slos_exact" else args.repeats
                values, times, warmup = timed(fwd, reps, args.device)
                rec = {
                    "route": route,
                    "m": m,
                    "n": n,
                    "states": states,
                    "repeats": reps,
                    "loss_mean": float(np.mean(values)),
                    "loss_se": 0.0
                    if route == "pe_slos_exact"
                    else float(np.std(values, ddof=1)),
                    "values": values,
                    "seconds_forward": statistics.median(times),
                    "seconds_forward_all": times,
                    "warmup_seconds": warmup,
                    "seconds_grad": None,
                }
                if grd is not None:
                    _, gtimes, gwarm = timed(grd, 3, args.device)
                    rec["seconds_grad"] = statistics.median(gtimes)
                    rec["grad_warmup_seconds"] = gwarm
                if route == "pe_slos_exact":
                    exact = rec["loss_mean"]
                point.append(rec)
                logger.info(
                    "EVALUATION_COMPLETED | route=%s | m=%d | n=%d | L=%.6e | SE=%.2e | t_fwd=%.3es | t_grad=%s | warmup=%.2fs",
                    route,
                    m,
                    n,
                    rec["loss_mean"],
                    rec["loss_se"],
                    rec["seconds_forward"],
                    "n/a"
                    if rec["seconds_grad"] is None
                    else f"{rec['seconds_grad']:.3e}s",
                    warmup,
                )
                if rec["seconds_forward"] * (reps + 1) > args.time_budget:
                    dropped.add(route)
            ref = (
                exact
                if exact is not None
                else next(r["loss_mean"] for r in point if r["route"] == "jax_gurvits")
            )
            eps = args.rel_eps * abs(ref)
            for rec in point:
                rec["loss_ref"] = ref
                rec["loss_ref_is_exact"] = exact is not None
                rec["eps"] = eps
                rec["seconds_at_eps"] = (
                    rec["seconds_forward"] * max(1.0, (rec["loss_se"] / eps) ** 2)
                    if rec["route"] != "pe_slos_exact"
                    else rec["seconds_forward"]
                )
                rec["z_vs_exact"] = (
                    None
                    if exact is None or rec["route"] == "pe_slos_exact"
                    else (rec["loss_mean"] - exact)
                    / (rec["loss_se"] / math.sqrt(rec["repeats"]))
                )
            records.extend(point)
            write_json(run_dir / "metrics.json", {"records": records})
        write_json(run_dir / "metrics.json", {"records": records})
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
