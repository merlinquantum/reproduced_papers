"""Run a sweep across {quantum, classical, merlin, random_decoder} x digits x seeds.

Usage::

    python utils/run_sweep.py --digits 0 5 9 --seeds 0 1 2 --base configs/mnist_reduced.json

Writes per-run output under ``outdir/`` and a summary CSV at
``results/sweep_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from lib.runner import train_and_evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, nargs="+", default=[0, 5, 9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--models",
        nargs="+",
        default=["quantum", "classical", "merlin", "random_decoder"],
    )
    ap.add_argument("--base", default="configs/mnist_reduced.json",
                    help="Base config (per-model overrides loaded from configs/mnist_<model>.json)")
    args = ap.parse_args()

    base_path = HERE / args.base
    base = json.loads(base_path.read_text())

    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "sweep_summary.csv"
    fieldnames = ["model", "digit", "seed", "gen_params", "disc_params",
                  "final_fd", "best_fd", "ae_time_s", "gan_time_s", "run_dir"]

    rows = []
    for model in args.models:
        per_model = HERE / f"configs/mnist_{model}.json"
        if per_model.exists():
            cfg_template = json.loads(per_model.read_text())
        else:
            cfg_template = copy.deepcopy(base)
            cfg_template["model"] = model
        # Inherit anything the per-model config doesn't override from the base.
        for k, v in base.items():
            cfg_template.setdefault(k, v)
        for digit in args.digits:
            for seed in args.seeds:
                cfg = copy.deepcopy(cfg_template)
                cfg["digit"] = digit
                cfg["seed"] = seed
                ts = time.strftime("%Y%m%d_%H%M%S")
                run_dir = HERE / "outdir" / f"run_{ts}_seed{seed}_{model}_d{digit}"
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n=== model={model} digit={digit} seed={seed} ===")
                out = train_and_evaluate(cfg, run_dir)
                m = out["test_metrics"]
                rows.append({
                    "model": model,
                    "digit": digit,
                    "seed": seed,
                    "gen_params": m.get("gen_params"),
                    "disc_params": m.get("disc_params"),
                    "final_fd": m.get("final_fd"),
                    "best_fd": m.get("best_fd"),
                    "ae_time_s": m.get("ae_train_time_s"),
                    "gan_time_s": m.get("gan_train_time_s"),
                    "run_dir": str(run_dir),
                })
                # Save incremental snapshot in case of interrupt.
                with open(summary_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
