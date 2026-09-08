"""Run a queue of configs on a GPU, batching instances into lockstep chunks.

    python utils/run_gpu.py configs/a.json configs/b.json --chunk 20 --out results/gpu.jsonl

Writes one JSON line per instance, in the same schema as the CPU runner's
``metrics.json`` rows, and resumes: a key already present in the output file is
skipped, so an interrupted queue continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Deliberately narrow imports: the GPU host needs torch and numpy, nothing else.
# Reaching lib.runner here would drag in Perceval (through lib.sampler) and MerLin
# (through lib.bbs_merlin), neither of which the GPU path uses.
from lib.bbs_gpu import solve_batch                        # noqa: E402
from lib.metrics import relative_error                     # noqa: E402
from lib.problems_torch import build_batch                 # noqa: E402
from lib.tbi import beamsplitter_layout, candidate_budget  # noqa: E402


def run_key(cfg, seed):
    """Identity of one run. Everything that defines an arm belongs in here.

    Omitting any of these silently merges two different arms under one key, and
    then resume skips all but the first -- the single most repeated mistake in
    the exploratory work that preceded this reproduction.
    """
    problem, training, method = cfg["problem"], cfg["training"], cfg["method"]
    return "|".join([
        problem["family"], f"m{problem['m']}", method["name"],
        method.get("source", "-"), f"rate{method.get('rate_scale', 1.0)}",
        method.get("topology", "chain"),
        f"N{training['updates']}", f"S{training['samples']}",
        f"lr{training['lr_theta']}:{training['lr_alpha']}",
        f"shift{training['shift']}:{training.get('shift_scale', 1.0)}",
        f"freeze{int(bool(method.get('freeze_theta', False)))}",
        f"i{seed}",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--chunk", type=int, default=20, help="instances solved in lockstep")
    parser.add_argument("--out", default=str(ROOT / "results" / "gpu_results.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instances", type=int, default=None, help="override problem.instances")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"device: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["key"] for line in out_path.open() if line.strip()}
        print(f"resuming: {len(done)} runs already recorded in {out_path}", flush=True)

    handle = out_path.open("a", buffering=1)
    for config_path in args.configs:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if args.instances is not None:
            cfg["problem"]["instances"] = args.instances
        problem, training, method = cfg["problem"], cfg["training"], cfg["method"]
        m, delays = problem["m"], tuple(cfg["circuit"]["delays"])
        topology = method.get("topology", "chain")
        n_angles = len(beamsplitter_layout(m, delays, topology))
        budget = candidate_budget(m, delays, training["updates"], training["samples"])

        todo = [problem.get("instance_seed_base", 0) + i
                for i in range(problem["instances"])
                if run_key(cfg, problem.get("instance_seed_base", 0) + i) not in done]
        print(f"\n=== {Path(config_path).name}: m={m} {topology} {n_angles} angles -> "
              f"{1 + 2 * n_angles} circuits/update, budget {budget:,} = "
              f"{budget / 2 ** m:.3g} of the space; {len(todo)} instances to run", flush=True)

        for start in range(0, len(todo), args.chunk):
            seeds = todo[start:start + args.chunk]
            cost_batch, optima = build_batch(problem["family"], m, seeds, device, **problem.get("kwargs", {}))
            began = time.time()
            result = solve_batch(
                cost_batch, m, seeds,
                delays=delays,
                updates=training["updates"], samples=training["samples"],
                lr_theta=training["lr_theta"], lr_alpha=training["lr_alpha"],
                shift=training["shift"], shift_scale=training.get("shift_scale", 1.0),
                source=method.get("source", "boson"), rate_scale=method.get("rate_scale", 1.0),
                topology=topology, device=device,
                sampler_backend=cfg.get("sampler_backend", "triton"),
                sampler_algo=cfg.get("sampler_algo", "auto"),
                freeze_theta=bool(method.get("freeze_theta", False)),
            )
            elapsed = time.time() - began
            optimal = 0
            for slot, seed in enumerate(seeds):
                best = float(result["best_cost"][slot])
                found = abs(best - optima[slot]) < 1e-9
                optimal += found
                handle.write(json.dumps({
                    "key": run_key(cfg, seed), "config": Path(config_path).name,
                    "family": problem["family"], "m": m, "method": method["name"],
                    "source": method.get("source"), "rate_scale": method.get("rate_scale", 1.0),
                    "topology": topology, "freeze_theta": bool(method.get("freeze_theta", False)),
                    "instance_seed": seed, "best_cost": best, "optimum": float(optima[slot]),
                    "found_optimum": bool(found),
                    "relative_error_percent": relative_error(best, optima[slot]),
                    "evaluations": int(result["evaluations"]), "budget": int(budget),
                    "space_coverage": budget / 2 ** m,
                    "mean_clicks": float(result["mean_clicks"][slot]),
                    "seconds": round(elapsed / len(seeds), 2),
                }) + "\n")
            print(f"  instances {seeds[0]}-{seeds[-1]}: {optimal}/{len(seeds)} optimal, "
                  f"{elapsed:.1f}s ({elapsed / len(seeds):.1f}s per instance), "
                  f"mean clicks {np.mean(result['mean_clicks']):.2f}/{m}", flush=True)
    handle.close()
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
