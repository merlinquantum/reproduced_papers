"""Lowest test MSE any model can reach on the one-step forecast (F1 as written).

The label is ``u . tanh(Z_{t+1})`` with ``Z_{t+1} = mu_{t+1} + eps_{t+1}``: ``mu_{t+1}`` is
fixed by the past of the VARMA and the innovation ``eps_{t+1} ~ N(0, sigma^2 I)`` is
independent of it. No predictor that sees only the past beats the conditional mean
``E[u . tanh(mu + eps) | mu]``. This script computes, on the test windows of each run:

* the expected floor ``mean_i sum_k u_k^2 Var(tanh(mu_ik + eps_k))`` (the coordinates of
  ``eps`` are independent), by Gauss-Hermite quadrature;
* the test MSE of that conditional-mean predictor on the actual test labels.

A window model sees only the last 25 points, less than the whole past, so both values
are lower bounds for it. The latent path is replayed with the generator's own functions
and checked against ``generate_dataset`` (windows and forecast labels must match exactly).

Usage (from the repository root)::

    python papers/QuaRK/utils/forecast_noise_floor.py \\
        papers/QuaRK/outdir/sweep_main_comparison --candidate quark

Writes ``results/forecast_noise_floor.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for _path in (PROJECT_DIR.parents[1], PROJECT_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from QuaRK.lib.data import (  # noqa: E402
    TASK_NAMES,
    draw_varma_parameters,
    generate_dataset,
    varma_step,
)

QUADRATURE_NODES = 80


def conditional_moments(mu: np.ndarray, sigma: float, nodes: int) -> tuple:
    """Mean and variance of ``tanh(mu + sigma * e)``, ``e ~ N(0, 1)``, elementwise."""
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    values = np.tanh(mu[..., None] + sigma * points)
    mean = values @ weights
    variance = (values**2) @ weights - mean**2
    return mean, variance


def replay_conditional_means(data_cfg: dict, seed: int) -> tuple:
    """Return ``(dataset, mu)``: ``mu[i]`` is the conditional mean of ``Z`` one step after window i."""
    dataset = generate_dataset(data_cfg, seed)
    varma = data_cfg["varma"]
    window, gap = int(data_cfg["window"]), int(data_cfg["gap"])
    input_dim, burn_in = int(data_cfg["input_dim"]), int(data_cfg["burn_in"])
    num_windows = int(data_cfg["num_windows"])
    scale = float(varma["innovation_scale"])

    # Same draws, in the same order, as generate_dataset.
    rng = np.random.default_rng(seed)
    ar, ma = draw_varma_parameters(
        rng,
        input_dim=input_dim,
        ar_order=int(varma["ar_order"]),
        ma_order=int(varma["ma_order"]),
        gamma=float(varma["gamma"]),
        eta=float(varma["eta"]),
        rho=float(varma["rho"]),
    )
    stride = window + gap
    length = burn_in + window + (num_windows - 1) * stride
    innovations = rng.normal(0.0, scale, size=(length, input_dim))
    extra = np.random.default_rng([seed, 1]).normal(0.0, scale, size=(1, input_dim))
    innovations = np.vstack([innovations, extra])
    latent = np.zeros((length + 1, input_dim))
    for t in range(length + 1):
        latent[t] = varma_step(ar, ma, latent, innovations, t)

    ends = burn_in + (window - 1) + stride * np.arange(num_windows)
    windows = np.stack([np.tanh(latent[e - window + 1 : e + 1]) for e in ends])
    forecast = np.tanh(latent[ends + 1]) @ np.asarray(dataset.meta["u_readout"])
    row = TASK_NAMES.index("one_step_forecast")
    if not (
        np.array_equal(windows, dataset.windows)
        and np.array_equal(forecast, dataset.labels[row])
    ):
        raise RuntimeError(f"replayed path differs from generate_dataset (seed {seed})")
    mu = latent[ends + 1] - innovations[ends + 1]
    return dataset, mu, scale


def floor_for_run(data_cfg: dict, seed: int) -> dict:
    dataset, mu, scale = replay_conditional_means(data_cfg, seed)
    u = np.asarray(dataset.meta["u_readout"])
    test = dataset.test_idx
    y_test = dataset.labels[TASK_NAMES.index("one_step_forecast"), test]
    label_variance = float(np.var(y_test))

    result = {
        "seed": seed,
        "num_test": int(test.size),
        "label_variance_test": label_variance,
    }
    for nodes in (QUADRATURE_NODES, 2 * QUADRATURE_NODES):
        mean, variance = conditional_moments(mu[test], scale, nodes)
        result[f"expected_floor_mse__{nodes}_nodes"] = float(np.mean(variance @ u**2))
        oracle = mean @ u
        result[f"oracle_test_mse__{nodes}_nodes"] = float(
            np.mean((y_test - oracle) ** 2)
        )
    floor = result[f"expected_floor_mse__{QUADRATURE_NODES}_nodes"]
    oracle_mse = result[f"oracle_test_mse__{QUADRATURE_NODES}_nodes"]
    result["expected_floor_mse"] = floor
    result["oracle_test_mse"] = oracle_mse
    result["expected_floor_normalized"] = floor / label_variance
    result["oracle_test_mse_normalized"] = oracle_mse / label_variance
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results" / "forecast_noise_floor.json",
    )
    args = parser.parse_args()

    status = json.loads((args.sweep_dir / "sweep_status.json").read_text())
    runs = [
        r
        for r in status["runs"]
        if r["candidate"]["name"] == args.candidate and r["status"] == "COMPLETED"
    ]
    if not runs:
        raise SystemExit(f"no completed run of candidate {args.candidate!r}")
    results = []
    for run in runs:
        run_dir = (args.sweep_dir / run["metrics_path"]).parent
        snapshot = json.loads((run_dir / "config_snapshot.json").read_text())
        results.append(floor_for_run(snapshot["data"], int(snapshot["seed"])))
        print(json.dumps(results[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "sweep_id": status["sweep_id"],
                "candidate": args.candidate,
                "quadrature_nodes": QUADRATURE_NODES,
                "runs": results,
            },
            indent=2,
        )
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
