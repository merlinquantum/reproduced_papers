"""Phase 5 stage A -> B: select each model's learning rate and write the stage-B configs.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/select_p5.py \
        papers/fermi_dirac_neurons/outdir/sweeps/p5_select_v1 \
        papers/fermi_dirac_neurons/outdir/sweeps/p5_photonic_select_v1

Rule frozen in LOG.md ("Phase 5 Comparison Plan — 2026-11-10T13:44:36Z"),
per task and model: selection metric = log10 MSE on the selection split
(regression) or the model's own loss on the selection split
(classification); median over the stage-A seeds; the minimum wins; a
learning rate at which the model diverged (or gave a non-finite metric) for
any seed is ineligible; exact ties go to the smaller rate; a model with no
eligible rate is FAILED for the task.

Writes ``results/p5_selection.json``, one ``configs/p5_final_<task>.json`` per
qubit task, one ``configs/p5_phfinal_m<m>_n<N>.json`` per photonic size, each
with the selected per-model ``learning_rate``, and the stage-B sweep specs
``configs/sweeps/p5_final_v1.json`` and ``p5_photonic_final_v1.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"
QUBIT_PATTERN = re.compile(r"p5_sel_(\w+?)_eta([\d.]+)")
PHOTONIC_PATTERN = re.compile(r"p5_phsel_(m\d+_n\d+)_lr([\d.]+)")
FINAL_SEEDS = list(range(10))


def _selection_value(metrics: dict, model: str) -> float:
    record = metrics["models"][model]
    if record["diverged_at_epoch"] is not None:
        return math.nan
    selection = record["final"]["selection"]
    if metrics["task_kind"] == "regression":
        return math.log10(selection["mse"]) if selection["mse"] > 0 else math.nan
    return selection["loss"]


def _read(sweep_dir: Path, pattern: re.Pattern) -> tuple[dict, list[int]]:
    """Return ``values[(task, model)][rate] = [per-seed metric]`` and the stage-A seeds."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    # Coverage, not the coordinator's final status, decides inclusion: the
    # p5_select_v1 coordinator ended FAILED in its summary step after every
    # planned run had COMPLETED (LOG.md, Phase 5 ledger).
    incomplete = [
        run["run_id"] for run in status["runs"] if run["status"] != "COMPLETED"
    ]
    if incomplete or len(status["runs"]) != status["expected_runs"]:
        raise ValueError(
            f"{sweep_dir}: {len(incomplete)} planned runs not COMPLETED: {incomplete[:5]}"
        )
    seeds = sorted(set(status["spec"]["seeds"]))
    values: dict = defaultdict(lambda: defaultdict(dict))
    for run in status["runs"]:
        task, rate_text = pattern.fullmatch(run["config"]).groups()
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        for model in metrics["models"]:
            per_seed = values[(task, model)][float(rate_text)]
            if run["seed"] in per_seed:
                raise ValueError(f"Duplicate seed {run['seed']} for {run['config']}")
            per_seed[run["seed"]] = _selection_value(metrics, model)
    return values, seeds


def select(values: dict, seeds: list[int]) -> list[dict]:
    """Apply the frozen selection rule to every (task, model)."""
    rows = []
    for (task, model), by_rate in sorted(values.items()):
        candidates = []
        for rate, per_seed in sorted(by_rate.items()):
            if sorted(per_seed) != seeds:
                raise ValueError(
                    f"{task}/{model} rate {rate}: seeds {sorted(per_seed)}"
                )
            metrics = np.array([per_seed[seed] for seed in seeds])
            eligible = bool(np.all(np.isfinite(metrics)))
            candidates.append(
                {
                    "learning_rate": rate,
                    "eligible": eligible,
                    "median_selection_metric": float(np.median(metrics))
                    if eligible
                    else None,
                    "per_seed": metrics.tolist(),
                }
            )
        eligible = [row for row in candidates if row["eligible"]]
        chosen = (
            min(
                eligible,
                key=lambda row: (row["median_selection_metric"], row["learning_rate"]),
            )
            if eligible
            else None
        )
        rows.append(
            {
                "task": task,
                "model": model,
                "status": "SELECTED" if chosen else "FAILED",
                "selected_learning_rate": chosen["learning_rate"] if chosen else None,
                "candidates": candidates,
            }
        )
    return rows


def _write_final(rows: list[dict], kind: str) -> list[str]:
    names = []
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row["task"]][row["model"]] = row["selected_learning_rate"]
    for task, rates in sorted(by_task.items()):
        if kind == "qubit":
            source = f"p5_sel_{task}_eta1"
            stem = f"p5_final_{task}"
        else:
            source = f"p5_phsel_{task}_lr0.01"
            stem = f"p5_phfinal_{task}"
        config = json.loads((CONFIG_DIR / f"{source}.json").read_text())
        config = copy.deepcopy(config)
        failed = [model for model, rate in rates.items() if rate is None]
        config["models"] = [
            {**model, "learning_rate": rates[model["name"]]}
            for model in config["models"]
            if model["name"] not in failed
        ]
        config["experiment"]["id"] = stem
        config["description"] = (
            f"Phase 5 stage B: {task}, each model at its selected learning rate "
            f"({', '.join(f'{name} {rate:g}' for name, rate in rates.items() if rate is not None)})"
            + (f"; FAILED in stage A: {', '.join(failed)}" if failed else "")
        )
        (CONFIG_DIR / f"{stem}.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        names.append(stem)
    return names


def main() -> None:
    """Select learning rates and write the stage-B configs and sweep specs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("qubit_sweep_dir", type=Path)
    parser.add_argument("photonic_sweep_dir", type=Path)
    args = parser.parse_args()
    qubit_values, qubit_seeds = _read(args.qubit_sweep_dir, QUBIT_PATTERN)
    photonic_values, photonic_seeds = _read(args.photonic_sweep_dir, PHOTONIC_PATTERN)
    qubit_rows = select(qubit_values, qubit_seeds)
    photonic_rows = select(photonic_values, photonic_seeds)
    qubit_configs = _write_final(qubit_rows, "qubit")
    photonic_configs = _write_final(photonic_rows, "photonic")
    for sweep_id, configs, workers, budget, description in (
        (
            "p5_final_v1",
            qubit_configs,
            3,
            2.5,
            "Phase 5 stage B: 14 paper tasks, every model at its selected learning rate, seeds 0-9.",
        ),
        (
            "p5_photonic_final_v1",
            photonic_configs,
            1,
            0.8,
            "Phase 5 stage B (MerLin adaptation): photonic models with Adam at their selected learning rates, seeds 0-9.",
        ),
    ):
        spec = {
            "sweep_id": sweep_id,
            "description": description,
            "configs": configs,
            "seeds": FINAL_SEEDS,
            "max_workers": workers,
            "plan": {
                "decision_rule": "see LOG.md 'Phase 5 Comparison Plan — 2026-11-10T13:44:36Z' (paired comparisons and claim outcome rules)",
                "selection": "none (learning rates fixed by utils/select_p5.py from stage A)",
                "expected_runs": len(configs) * len(FINAL_SEEDS),
                "memory_budget_gib": budget,
            },
        }
        (CONFIG_DIR / "sweeps" / f"{sweep_id}.json").write_text(
            json.dumps(spec, indent=2) + "\n", encoding="utf-8"
        )
    output = PROJECT_DIR / "results" / "p5_selection.json"
    output.write_text(
        json.dumps(
            {
                "qubit_sweep_dir": str(args.qubit_sweep_dir),
                "photonic_sweep_dir": str(args.photonic_sweep_dir),
                "stage_a_seeds": {"qubit": qubit_seeds, "photonic": photonic_seeds},
                "coordinator_status": {
                    "qubit": json.loads(
                        (args.qubit_sweep_dir / "sweep_status.json").read_text()
                    )["status"],
                    "photonic": json.loads(
                        (args.photonic_sweep_dir / "sweep_status.json").read_text()
                    )["status"],
                },
                "qubit": qubit_rows,
                "photonic": photonic_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
