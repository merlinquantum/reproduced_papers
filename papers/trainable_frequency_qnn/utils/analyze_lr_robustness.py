#!/usr/bin/env python3
"""Check whether the TF vs FF ranking of E2 and E5 holds at every learning rate.

Combines the lr = 1e-3 runs of the main sweeps (``e2_fig2b``, ``e5_fig3``; seeds
1-3 only, to match) with the lr = 3e-3 and 1e-2 runs of ``lr_robustness``.
Reads only ``sweep_status.json`` and ``metrics.json``. Writes
``results/lr_robustness_results.{json,md}``.

Example (from the paper folder)::

    python utils/analyze_lr_robustness.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
SWEEPS = PAPER_DIR / "outdir" / "sweeps"
SEEDS = (1, 2, 3)
LEARNING_RATES = ("1e-3", "3e-3", "1e-2")
E2_MODELS = ("ff", "tf", "ffs")
E5_MODELS = ("simple", "tower", "exponential", "tf")


def train_mse_by_candidate(sweep: str) -> dict[str, dict[int, float]]:
    """Return ``{candidate: {seed: train_mse}}`` for completed runs of a sweep."""
    sweep_dir = SWEEPS / sweep
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    values: dict[str, dict[int, float]] = {}
    for run in status["runs"]:
        if run["status"] == "COMPLETED" and run["seed"] in SEEDS:
            metrics = json.loads(
                (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
            )
            values.setdefault(run["candidate"], {})[run["seed"]] = metrics["train_mse"]
    return values


def main() -> int:
    main_e2 = train_mse_by_candidate("e2_fig2b")
    main_e5 = train_mse_by_candidate("e5_fig3")
    robustness = train_mse_by_candidate("lr_robustness")

    def e2_values(model: str, lr: str) -> dict[int, float]:
        return main_e2[model] if lr == "1e-3" else robustness[f"e2_{model}_lr{lr}"]

    def e5_values(model: str, k: int, lr: str) -> dict[int, float]:
        return (
            main_e5[f"{model}_k{k}"]
            if lr == "1e-3"
            else robustness[f"e5_{model}_k{k}_lr{lr}"]
        )

    result: dict[str, dict] = {"e2": {}, "e5": {}}
    for lr in LEARNING_RATES:
        result["e2"][lr] = {}
        for model in E2_MODELS:
            values = e2_values(model, lr)
            if sorted(values) != list(SEEDS):
                raise ValueError(
                    f"E2 {model} lr {lr}: seeds {sorted(values)} != {SEEDS}"
                )
            result["e2"][lr][model] = {
                "mean": statistics.fmean(values.values()),
                "values": values,
            }
        tf_seeds = result["e2"][lr]["tf"]["values"]
        ff_seeds = result["e2"][lr]["ff"]["values"]
        result["e2"][lr]["tf_lower_than_ff_seeds"] = sum(
            tf_seeds[s] < ff_seeds[s] for s in SEEDS
        )

        result["e5"][lr] = {}
        for model in E5_MODELS:
            dataset_means = {}
            for k in range(1, 8):
                values = e5_values(model, k, lr)
                if sorted(values) != list(SEEDS):
                    raise ValueError(
                        f"E5 {model} k{k} lr {lr}: seeds {sorted(values)} != {SEEDS}"
                    )
                dataset_means[k] = statistics.fmean(values.values())
            result["e5"][lr][model] = {
                "mean_over_datasets": statistics.fmean(dataset_means.values()),
                "worst_dataset": max(dataset_means.values()),
                "best_dataset": min(dataset_means.values()),
                "per_dataset": dataset_means,
            }
        fixed = [result["e5"][lr][m] for m in ("simple", "tower", "exponential")]
        tf_row = result["e5"][lr]["tf"]
        result["e5"][lr]["tf_best_mean"] = all(
            tf_row["mean_over_datasets"] < f["mean_over_datasets"] for f in fixed
        )
        result["e5"][lr]["tf_best_worst_case"] = all(
            tf_row["worst_dataset"] < f["worst_dataset"] for f in fixed
        )

    out = PAPER_DIR / "results"
    out.mkdir(exist_ok=True)
    (out / "lr_robustness_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# lr robustness (seeds 1-3 at every lr; lr 1e-3 from the main sweeps)",
        "",
        "| lr | E2 FF | E2 FF+S | E2 TF | TF < FF (seeds) | E5 mean simple / tower / exp / TF | E5 worst simple / tower / exp / TF | TF best mean | TF best worst |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for lr in LEARNING_RATES:
        e2, e5 = result["e2"][lr], result["e5"][lr]
        means = " / ".join(f"{e5[m]['mean_over_datasets']:.3g}" for m in E5_MODELS)
        worsts = " / ".join(f"{e5[m]['worst_dataset']:.3g}" for m in E5_MODELS)
        lines.append(
            f"| {lr} | {e2['ff']['mean']:.3g} | {e2['ffs']['mean']:.3g} | {e2['tf']['mean']:.3g} | "
            f"{e2['tf_lower_than_ff_seeds']}/3 | {means} | {worsts} | {e5['tf_best_mean']} | {e5['tf_best_worst_case']} |"
        )
    (out / "lr_robustness_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
