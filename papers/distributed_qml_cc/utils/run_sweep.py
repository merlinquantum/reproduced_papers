"""Run a grid over (scheme, L, seed) and collect metrics.

Usage:

    python utils/run_sweep.py [--schemes non,nc,cc,qc] [--layers 3,5,7,9]
                              [--seeds 0,1,2] [--iterations 1000]
                              [--outdir results/sweep]

The script invokes ``lib.runner._run_quantum`` in-process for each
configuration and aggregates everything into a single ``sweep.json``.
Each individual run also writes its own ``metrics.json`` under
``<outdir>/<scheme>_L<L>/run_<seed>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PAPER_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402
from lib.runner import _run_quantum  # noqa: E402


def parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", type=str, default="non,nc,cc,qc")
    parser.add_argument("--layers", type=str, default="3,5,7,9")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results/sweep")
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    schemes = parse_list(args.schemes, str)
    layers = parse_list(args.layers, int)
    seeds = parse_list(args.seeds, int)
    base_outdir = (PAPER_ROOT / args.outdir).resolve()
    base_outdir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    dtype = dtype_map.get(args.dtype, torch.float32)

    sweep = {
        "schemes": schemes,
        "layers": layers,
        "seeds": seeds,
        "iterations": args.iterations,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dataset_seed": args.dataset_seed,
        "dtype": args.dtype,
        "results": [],
    }
    started = time.time()
    for scheme in schemes:
        for L in layers:
            run_dir = base_outdir / f"{scheme}_L{L}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "pipeline": "quantum",
                "dtype": args.dtype,
                "seeds": seeds,
                "dataset": {"params": {"seed": args.dataset_seed}},
                "model": {"params": {"scheme": scheme, "n_layers": L, "qubits_per_qpu": 4}},
                "training": {
                    "n_iterations": args.iterations,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "eval_every": args.eval_every,
                },
            }
            print(f"\n[{scheme} L={L}] starting...")
            t0 = time.time()
            summary = _run_quantum(cfg, run_dir, dtype)
            print(
                f"[{scheme} L={L}] mean_val_acc={summary['mean_val_acc']:.4f} "
                f"± {summary['std_val_acc']:.4f} ({len(seeds)} seeds, "
                f"{time.time()-t0:.1f}s)"
            )
            sweep["results"].append({
                "scheme": scheme,
                "n_layers": L,
                "mean_val_acc": summary["mean_val_acc"],
                "std_val_acc": summary["std_val_acc"],
                "runs": summary["runs"],
            })
    sweep["total_seconds"] = time.time() - started
    out_path = base_outdir / "sweep.json"
    out_path.write_text(json.dumps(sweep, indent=2))
    print(f"\nWrote sweep summary to {out_path} (total {sweep['total_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
