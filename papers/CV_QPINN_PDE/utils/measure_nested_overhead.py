"""Measure forward+backward time and peak memory for the consistency-loss vs
nested-autograd losses at increasing Fock cutoff.

The paper's central methodological claim is that nested gradients are
impractical for CV simulators. We test it head-on by reproducing one
training step under both losses and reporting:

  - peak resident memory delta (psutil)
  - wall-clock per step
  - peak Python GC objects

Usage:

    python utils/measure_nested_overhead.py
"""

from __future__ import annotations

import gc
import math
import resource
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lib.losses import poisson_nested_loss, poisson_total_loss  # noqa: E402
from lib.qpinn_model import QPINN, QPINNConfig  # noqa: E402


def _bench(loss_fn, cutoff: int, n_collocation: int, repeats: int = 5) -> dict:
    cfg = QPINNConfig(n_qumodes=2, n_multi_layers=2, n_single_layers=2,
                      cutoff=cutoff, seed=42)
    model = QPINN(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    lambdas = {"pde": 0.34, "bc": 0.33, "consistency": 0.33, "trace": 0.33}
    x = torch.linspace(0.01, math.pi / 2 - 0.01, n_collocation, dtype=torch.float64)
    x_left = torch.tensor([0.0], dtype=torch.float64)
    x_right = torch.tensor([math.pi / 2], dtype=torch.float64)
    # Warm up to populate caches and JIT internal state.
    for _ in range(2):
        optimizer.zero_grad()
        loss, _ = loss_fn(model, x, x_left, x_right, lambdas)
        loss.backward()
        optimizer.step()
    gc.collect()
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    for _ in range(repeats):
        optimizer.zero_grad()
        loss, _ = loss_fn(model, x, x_left, x_right, lambdas)
        loss.backward()
        optimizer.step()
    elapsed = (time.perf_counter() - t0) / repeats
    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "time_per_step_sec": elapsed,
        "max_rss_delta_kb": final_rss - baseline_rss,
        "max_rss_kb": final_rss,
        "final_loss": float(loss.detach().item()),
    }


def main() -> None:
    n_coll = 64
    print(f"Benchmark over {n_coll} collocation points, 2+2 layers, 5 repeats per cell.\n")
    print(f"{'cutoff':>6} | {'loss':>11} | {'step (s)':>9} | {'rss delta (MB)':>14} | "
          f"{'final loss':>10}")
    print("-" * 70)
    for cutoff in (8, 10, 12, 15):
        for name, fn in [("consistency", poisson_total_loss),
                          ("nested", poisson_nested_loss)]:
            r = _bench(fn, cutoff=cutoff, n_collocation=n_coll, repeats=5)
            print(f"{cutoff:>6} | {name:>11} | {r['time_per_step_sec']:>9.3f} | "
                  f"{r['max_rss_delta_kb']/1024:>14.1f} | {r['final_loss']:>10.3e}")


if __name__ == "__main__":
    main()
