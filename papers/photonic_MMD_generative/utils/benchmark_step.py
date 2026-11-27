"""Controlled step-time benchmark of MMD training variants (one variant per process).

Usage (one variant per process, so memory does not carry over):

    python papers/photonic_MMD_generative/utils/benchmark_step.py <variant> <prepared_dataset_dir> <m> <n> <K> <Z> <steps>

Example (results of 2026-10-12 in results/benchmark_step_2026-10-12.jsonl; peak RSS
measured with /usr/bin/time -l):

    python papers/photonic_MMD_generative/utils/benchmark_step.py cols-step \
        data/photonic_MMD_generative/prepared/sushi_top10_m100_n10_a0424c0792 100 10 2000 5000 6

variant = <matrix>-<compile>[-chunk<C>]
  matrix : full   -> A_k = (U^H diag(1-2k) U)[S, S]   (original code)
           cols   -> A_k = U_S^H diag(1-2k) U_S       (this reproduction)
  compile: loss   -> only the loss is jitted; value_and_grad and the Adam update
                     run outside jit (original Trainer, jaxopt opt_jit=False)
           step   -> value_and_grad + Adam update compiled as one function
  chunk<C>: operators processed with lax.map in chunks of C instead of one vmap.

Prints one JSON line: variant, compile+first step seconds, median step seconds,
losses (to check all variants compute the same values).
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent))
sys.path.insert(0, str(PAPER))
from photonic_MMD_generative.lib import (  # noqa: E402
    circuits,
    data,
    estimator,
    initialization,
    kernels,
)

variant, dataset_dir = sys.argv[1], Path(sys.argv[2])
m, n, n_ops, n_z, steps = map(int, sys.argv[3:8])
parts = variant.split("-")
matrix_mode, compile_mode = parts[0], parts[1]
chunk = int(parts[2][5:]) if len(parts) > 2 else None

X = jnp.asarray(np.load(dataset_dir / "train.npy"), dtype=jnp.float64)
modes = jnp.asarray(data.input_modes(m, n, "middle_alternating"))
unitary_fn = circuits.make_unitary_fn(m, "haar")
sampler = kernels.make_operator_sampler({"type": "gaussian", "sigma": 3.0}, m, n)
params0, _ = initialization.initial_parameters(
    {"strategy": "identity", "perturbation": 0.01}, jax.random.PRNGKey(0), m, "haar"
)


def terms_for(U, K, Z):
    T = estimator.walsh_expectations(K, X)
    count = X.shape[0]
    U_S = U[:, modes]

    def one(args):
        k, t = args
        if matrix_mode == "full":
            A = (jnp.conj(U.T) @ ((1.0 - 2.0 * k)[:, None] * U))[modes[:, None], modes]
        else:
            A = estimator.walsh_operator_matrix(U_S, k)
        G = estimator.glynn_samples(A, Z)
        sq = (jnp.sum(G) ** 2 - jnp.sum(G**2)) / (G.shape[0] * (G.shape[0] - 1))
        return sq - 2.0 * jnp.mean(G) * t + (count * t * t - 1.0) / (count - 1)

    if chunk is None:
        return jax.vmap(one)((K, T))
    return jax.lax.map(one, (K, T), batch_size=chunk)


@jax.jit
def loss(params, key):
    key_k, key_z = jax.random.split(key)
    K = sampler(key_k, n_ops)
    Z = (2 * jax.random.randint(key_z, (n_z, n), 0, 2) - 1).astype(jnp.float64)
    return jnp.real(jnp.mean(terms_for(unitary_fn(params), K, Z)))


optimizer = optax.adam(0.01)
if compile_mode == "step":
    value_and_grad = jax.value_and_grad(loss)

    @jax.jit
    def update(params, state, key):
        value, grad = value_and_grad(params, key)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, value
else:
    value_and_grad = jax.value_and_grad(
        loss
    )  # gradient of a jitted loss, not itself jitted

    def update(params, state, key):
        value, grad = value_and_grad(params, key)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, value


params, state, key = params0, optimizer.init(params0), jax.random.PRNGKey(1)
times, losses = [], []
for _ in range(steps):
    key, sub = jax.random.split(key)
    start = time.perf_counter()
    params, state, value = update(params, state, sub)
    value = float(value)
    times.append(time.perf_counter() - start)
    losses.append(value)
print(
    json.dumps(
        {
            "variant": variant,
            "m": m,
            "n": n,
            "K": n_ops,
            "Z": n_z,
            "first_step_s": times[0],
            "median_step_s": float(np.median(times[1:])),
            "losses": losses,
        }
    )
)
