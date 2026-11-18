"""Apply the Phase 5 selection rule to the tuning sweep and write the final sweeps.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/select_hyperparameters.py \\
        --tuning-dir papers/contextuality_inductive_bias/outdir/sweeps/fair_tuning \\
        --tuning-dir papers/contextuality_inductive_bias/outdir/sweeps/fair_tuning_ext \\
        --out-dir papers/contextuality_inductive_bias/results \\
        --spec-dir papers/contextuality_inductive_bias/configs/sweeps

Rule (LOG.md, "Phase 5 Comparison Plan" and its amendments):

1. A (model, lr, L2) candidate is eligible only if every tuning seed (0-2)
   COMPLETED with a finite final validation NLL.
2. Per model, select the eligible candidate with the lowest mean final
   validation NLL. Candidates within 1e-4 of the minimum are tied; ties go to
   the smallest lr, then the smallest L2.
3. Test metrics are read only to be reported, never to select.

Writes ``fair_tuning_selection.json`` / ``.md`` (every candidate, with its
eligibility and mean validation NLL) and the sweep specs ``fair_final.json``
(seeds 10-19) and ``fair_robustness.json`` (init seed 10 on data seeds 101-103).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TIE_TOLERANCE = 1e-4
TUNING_SEEDS = (0, 1, 2)
FINAL_SEEDS = tuple(range(10, 20))
ROBUSTNESS_DATA_SEEDS = (101, 102, 103)
ROBUSTNESS_INIT_SEED = 10
HYPERPARAMETER_FLAGS = ("--lr", "--l2")


def _model_flags(grid_point: dict) -> dict:
    return {k: v for k, v in grid_point.items() if k not in HYPERPARAMETER_FLAGS}


def collect(tuning_dirs: list[Path]) -> dict:
    """Group tuning runs of one or more sweeps by model, then by (lr, L2)."""
    models: dict[str, dict] = defaultdict(
        lambda: defaultdict(lambda: {"values": {}, "runs": []})
    )
    for tuning_dir in tuning_dirs:
        _collect_one(tuning_dir, models)
    return models


def _collect_one(tuning_dir: Path, models: dict) -> None:
    status = json.loads((tuning_dir / "sweep_status.json").read_text(encoding="utf-8"))
    for run in status["runs"]:
        model = json.dumps(_model_flags(run["grid_point"]), sort_keys=True)
        hyper = (float(run["grid_point"]["--lr"]), float(run["grid_point"]["--l2"]))
        entry = models[model][hyper]
        record = {"run_id": run["run_id"], "seed": run["seed"], "status": run["status"]}
        if run["status"] == "COMPLETED":
            metrics = json.loads(
                (tuning_dir / run["metrics_path"]).read_text(encoding="utf-8")
            )
            record["validation_nll"] = float(metrics["final"]["validation_nll"])
            record["test_kl"] = float(metrics["final"]["test_kl"])
        if any(r["seed"] == run["seed"] for r in entry["runs"]):
            raise ValueError(
                f"Duplicate tuning run for {model} {hyper} seed {run['seed']}"
            )
        entry["runs"].append(record)


def select(models: dict) -> list[dict]:
    """Apply the eligibility, minimum and tie rules for every model."""
    selections = []
    for model, candidates in sorted(models.items()):
        rows = []
        for (lr, l2), entry in sorted(candidates.items()):
            seeds = sorted(r["seed"] for r in entry["runs"])
            valid = [
                r
                for r in entry["runs"]
                if r["status"] == "COMPLETED"
                and math.isfinite(r.get("validation_nll", math.inf))
            ]
            eligible = seeds == list(TUNING_SEEDS) and len(valid) == len(TUNING_SEEDS)
            mean = (
                float(np.mean([r["validation_nll"] for r in valid]))
                if valid
                else math.nan
            )
            rows.append(
                {
                    "lr": lr,
                    "l2": l2,
                    "eligible": eligible,
                    "valid_runs": len(valid),
                    "expected_runs": len(TUNING_SEEDS),
                    "mean_validation_nll": mean,
                    "statuses": {r["run_id"]: r["status"] for r in entry["runs"]},
                }
            )
        eligible_rows = [r for r in rows if r["eligible"]]
        if not eligible_rows:
            raise ValueError(f"No eligible candidate for model {model}")
        best = min(r["mean_validation_nll"] for r in eligible_rows)
        tied = sorted(
            (
                r
                for r in eligible_rows
                if r["mean_validation_nll"] - best <= TIE_TOLERANCE
            ),
            key=lambda r: (r["lr"], r["l2"]),
        )
        chosen = tied[0]
        selections.append(
            {
                "model": json.loads(model),
                "selected": {
                    "lr": chosen["lr"],
                    "l2": chosen["l2"],
                    "mean_validation_nll": chosen["mean_validation_nll"],
                },
                "tied": [
                    {
                        "lr": r["lr"],
                        "l2": r["l2"],
                        "mean_validation_nll": r["mean_validation_nll"],
                    }
                    for r in tied
                ],
                "candidates": rows,
            }
        )
    return selections


def _spec(
    sweep_id: str,
    description: str,
    grid: list[dict],
    repetitions: list[dict],
    note: str,
) -> dict:
    return {
        "sweep_id": sweep_id,
        "description": description,
        "base_config": "configs/fair_quantum_tuning.json",
        "grid": grid,
        "repetitions": repetitions,
        "progress_total": 2000,
        "plan": {
            "metrics": [
                "final.test_kl",
                "min_validation_nll.test_kl",
                "min_validation_nll.step",
                "final.validation_nll",
                "final.train_nll",
                "final.test_zero_sum_violation",
                "train_seconds",
            ],
            "direction": "minimize",
            "split": "test (fixed seeded set, seed 2302, n = 10000); validation used only for the early-stopping view",
            "checkpoint": "final step 2000 (primary); step of minimum validation NLL (secondary)",
            "aggregation": "mean and sample std (ddof = 1) over repetitions; Welch and Mann-Whitney tests",
            "tie_tolerance": "difference resolved only if Welch and Mann-Whitney p < 0.05",
            "selection_rule": "none (hyperparameters fixed by fair_tuning_selection.json)",
            "note": note,
        },
        "fixed_args": {
            "--protocol": "paper",
            "--validation": "true",
            "--optimizer": "adam",
        },
        "max_workers": 6,
    }


def write(selections: list[dict], out_dir: Path, spec_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fair_tuning_selection.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "### Phase 5 tuning: selection by mean final validation NLL (seeds 0-2)",
        "",
        "| Model | lr | L2 | Eligible | Valid / expected | Mean validation NLL | Selected |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for item in selections:
        name = json.dumps(item["model"], sort_keys=True)
        for row in item["candidates"]:
            mark = (
                "**yes**"
                if (row["lr"], row["l2"])
                == (item["selected"]["lr"], item["selected"]["l2"])
                else ""
            )
            nll = (
                f"{row['mean_validation_nll']:.6f}"
                if math.isfinite(row["mean_validation_nll"])
                else "n/a"
            )
            lines.append(
                f"| `{name}` | {row['lr']:g} | {row['l2']:g} | {'yes' if row['eligible'] else 'no'} | "
                f"{row['valid_runs']}/{row['expected_runs']} | {nll} | {mark} |"
            )
    (out_dir / "fair_tuning_selection.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    grid = [
        dict(
            item["model"],
            **{"--lr": item["selected"]["lr"], "--l2": item["selected"]["l2"]},
        )
        for item in selections
    ]
    final = _spec(
        "fair_final",
        "Phase 5 final fair comparison: every model at its validation-selected (lr, L2), seeds 10-19, protocol paper, author training data, 2000 Adam steps.",
        grid,
        [{"--seed": s} for s in FINAL_SEEDS],
        "Generated by utils/select_hyperparameters.py from fair_tuning (LOG.md Phase 5 Comparison Plan).",
    )
    final["fixed_args"]["--experiment-id"] = "P5_fair_final"
    robustness_grid = [
        dict(point, **{"--train-source": "generated", "--train-seed": s})
        for s in ROBUSTNESS_DATA_SEEDS
        for point in grid
    ]
    robustness = _spec(
        "fair_robustness",
        "Phase 5 data robustness: every model at its selected (lr, L2) on generated training sets (data seeds 101-103), init seed 10, protocol paper.",
        robustness_grid,
        [{"--seed": ROBUSTNESS_INIT_SEED}],
        "Generated by utils/select_hyperparameters.py; paired over data seeds.",
    )
    robustness["fixed_args"]["--experiment-id"] = "P5_fair_robustness"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "fair_final.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    (spec_dir / "fair_robustness.json").write_text(
        json.dumps(robustness, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tuning-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spec-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    selections = select(collect(args.tuning_dir))
    write(selections, args.out_dir, args.spec_dir)
    for item in selections:
        tied = "" if len(item["tied"]) == 1 else f" (tied: {len(item['tied'])})"
        print(
            f"{json.dumps(item['model'], sort_keys=True)} -> lr={item['selected']['lr']:g}, l2={item['selected']['l2']:g}, val NLL={item['selected']['mean_validation_nll']:.6f}{tied}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
