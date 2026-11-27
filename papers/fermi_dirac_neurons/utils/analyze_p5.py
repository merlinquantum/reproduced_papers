"""Phase 5 comparison table: tuned quantum neurons vs B1, B2, B3, B4, and the MerLin adaptation.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_p5.py \
        papers/fermi_dirac_neurons/outdir/sweeps/p5_final_v1 \
        papers/fermi_dirac_neurons/outdir/sweeps/p5_photonic_final_v1 \
        papers/fermi_dirac_neurons/outdir/sweeps/p5_reference_v1

Applies the protocol frozen in LOG.md ("Phase 5 Comparison Plan —
2026-11-10T13:44:36Z"). The quantum neuron of each task is compared with every
other model of the task, paired by seed (seeds 0-9), with
``utils/sweep_statistics.paired_comparison`` (two-sided Wilcoxon, tolerance
0.1 decade or 1 accuracy point). Primary metric: test MSE (the 500 held-out
states, split ``validation``) for regression, test accuracy with the linear
rule for classification. B4 chooses its regularization with the stage-A rule
on seeds 0-2 (selection split) and is reported on seeds 0-9. A model that
diverged at any seed makes its comparisons ``EVIDENCE_MISSING``. Claim
outcomes follow the frozen rules. Writes ``results/p5_comparison.json`` and
``results/p5_comparison.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.utils.sweep_statistics import paired_comparison  # noqa: E402

STAGE_A_SEEDS = [0, 1, 2]
QUANTUM = {
    "fig7": "quantum_tfim",
    "fig8": "quantum_heisenberg",
    "fig9_tfim": "quantum_tfim_tanh",
    "fig9_heis": "quantum_heis_tanh",
    "photonic": "photonic_tanh",
}
ROLES = {
    "quantum_tfim": "paper method (quantum neuron)",
    "quantum_heisenberg": "paper method (quantum neuron)",
    "quantum_tfim_tanh": "paper method (quantum neuron)",
    "quantum_heis_tanh": "paper method (quantum neuron)",
    "classical_im": "B1 paper baseline (commuting)",
    "classical_fcim": "B1 paper baseline (commuting)",
    "classical_ising": "B1 paper baseline (commuting)",
    "b2_pauli_features": "B2 fair classical baseline",
    "b3_quantum_linear": "B3 quantum linear",
    "quantum_tfim_linear": "B3 quantum linear (paper's Fig. 9 model)",
    "quantum_heis_linear": "B3 quantum linear (paper's Fig. 9 model)",
    "b4_w2": "B4 reference (Pauli weight <= 2)",
    "b4_w3": "B4 reference (full tomography)",
    "photonic_tanh": "MerLin adaptation (photonic neuron)",
    "photon_count_tanh": "MerLin adaptation: commuting photon-count neuron",
    "correlation_tanh": "MerLin adaptation: B2 counterpart (correlation neuron)",
    "photonic_linear": "MerLin adaptation: photonic linear",
}
CLAIMS = {
    "C4": [f"fig7_{act}_{n}q" for act in ("tanh", "softplus") for n in (2, 7)],
    "C5": [f"fig8_logistic_{n}q" for n in range(2, 8)],
    "C6": [f"fig9_{family}_{n}q" for family in ("tfim", "heis") for n in (2, 3)],
    "MerLin": ["m4_n2", "m6_n3", "m8_n4"],
}


def _family(task: str) -> str:
    if task.startswith("fig9"):
        return "_".join(task.split("_")[:2])
    if task.startswith("fig"):
        return task.split("_")[0]
    return "photonic"


def _metrics_for(task: str) -> tuple[str, list[str]]:
    if task.startswith("fig8"):
        return "validation_accuracy_linear", [
            "validation_accuracy_activation",
            "train_loss",
        ]
    return "validation_mse", ["train_loss"]


def _load_final(sweep_dir: Path, prefix: str) -> tuple[dict, dict, dict, dict]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    incomplete = [
        run["run_id"] for run in status["runs"] if run["status"] != "COMPLETED"
    ]
    if incomplete or len(status["runs"]) != status["expected_runs"]:
        raise ValueError(
            f"{sweep_dir}: not every planned run COMPLETED: {incomplete[:5]}"
        )
    values: dict = defaultdict(dict)
    meta: dict = {}
    diverged: dict = defaultdict(list)
    elapsed: dict = defaultdict(list)
    for run in status["runs"]:
        task = run["config"].removeprefix(prefix)
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        elapsed[task].append(run["attempts"][-1]["elapsed_s"])
        for model, record in metrics["models"].items():
            meta[(task, model)] = {
                "n_parameters": record["n_parameters"],
                "learning_rate": record["learning_rate"],
            }
            values[(task, model, "training_seconds")][run["seed"]] = record[
                "training_seconds"
            ]
            if record["diverged_at_epoch"] is not None:
                diverged[(task, model)].append(run["seed"])
                continue
            entries = {"train_loss": record["last_logged_train_loss"]}
            for name, value in record["final"]["validation"].items():
                entries[f"validation_{name}"] = value
            for metric, value in entries.items():
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {metric} for {run['run_id']}/{model}")
                if run["seed"] in values[(task, model, metric)]:
                    raise ValueError(f"Duplicate seed {run['seed']} for {task}/{model}")
                values[(task, model, metric)][run["seed"]] = value
    return values, meta, diverged, elapsed


def _b4_selection_metric(record: dict, classification: bool) -> float:
    if classification:
        return record["selection_logistic_loss"]
    return math.log10(record["selection_mse"])


def _b4(reference_dir: Path) -> tuple[dict, dict, list]:
    """Select B4's regularization on seeds 0-2 and return seed values for every task."""
    status = json.loads((reference_dir / "sweep_status.json").read_text())
    if any(run["status"] != "COMPLETED" for run in status["runs"]):
        raise ValueError(f"{reference_dir}: not every planned run COMPLETED")
    records = defaultdict(dict)
    for run in status["runs"]:
        task = run["config"].removeprefix("p5_ref_")
        metrics = json.loads((reference_dir / run["metrics_path"]).read_text())
        records[task][run["seed"]] = metrics
    values: dict = defaultdict(dict)
    meta, selections = {}, []
    for task, by_seed in records.items():
        classification = task.startswith("fig8")
        for variant in by_seed[0]["variants"]:
            strengths = [
                record["strength"] for record in by_seed[0]["variants"][variant]
            ]

            medians = [
                float(
                    np.median(
                        [
                            _b4_selection_metric(
                                by_seed[seed]["variants"][variant][index],
                                classification,
                            )
                            for seed in STAGE_A_SEEDS
                        ]
                    )
                )
                for index in range(len(strengths))
            ]
            best = min(range(len(strengths)), key=lambda i: (medians[i], strengths[i]))
            selections.append(
                {
                    "task": task,
                    "model": variant,
                    "selected_strength": strengths[best],
                    "median_selection_metric_by_strength": dict(
                        zip(strengths, medians)
                    ),
                }
            )
            meta[(task, variant)] = {
                "n_parameters": by_seed[0]["variants"][variant][best]["n_features"] + 1,
                "learning_rate": None,
                "regularization": strengths[best],
            }
            for seed, metrics in by_seed.items():
                record = metrics["variants"][variant][best]
                if classification:
                    values[(task, variant, "validation_accuracy_linear")][seed] = (
                        record["validation_accuracy"]
                    )
                else:
                    values[(task, variant, "validation_mse")][seed] = record[
                        "validation_mse"
                    ]
                    values[(task, variant, "train_loss")][seed] = record["train_mse"]
    return values, meta, selections


def _summary(values: list[float], metric: str) -> dict:
    transformed = (
        [100 * value for value in values]
        if "accuracy" in metric
        else [math.log10(value) for value in values]
    )
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median_transformed": float(np.median(transformed)),
    }


def build(final_dir: Path, photonic_dir: Path, reference_dir: Path) -> dict:
    """Return the full comparison: system rows, pairwise comparisons, claim outcomes."""
    qubit_values, qubit_meta, qubit_diverged, qubit_elapsed = _load_final(
        final_dir, "p5_final_"
    )
    photonic_values, photonic_meta, photonic_diverged, photonic_elapsed = _load_final(
        photonic_dir, "p5_phfinal_"
    )
    b4_values, b4_meta, b4_selection = _b4(reference_dir)
    values = {**qubit_values, **photonic_values, **b4_values}
    meta = {**qubit_meta, **photonic_meta, **b4_meta}
    diverged = {**qubit_diverged, **photonic_diverged}
    elapsed = {**qubit_elapsed, **photonic_elapsed}
    seeds = list(range(10))
    systems, comparisons = [], []
    for claim, tasks in CLAIMS.items():
        for task in tasks:
            quantum = QUANTUM[_family(task)]
            primary, secondary = _metrics_for(task)
            models = sorted({model for (t, model) in meta if t == task})
            for model in models:
                observed = sorted(values.get((task, model, primary), {}))
                timing = values.get((task, model, "training_seconds"), {})
                row = {
                    "claim": claim,
                    "task": task,
                    "system": model,
                    "role": ROLES[model],
                    "provenance": "locally executed (Phase 5)",
                    "scope": (
                        "MerLin adaptation, simulation, tuned (Adam), 10 seeds"
                        if claim == "MerLin"
                        else "paper task, tuned learning rate, 10 seeds, 80 % of the paper's training states"
                    ),
                    "successful_runs": len(observed),
                    "expected_runs": len(seeds),
                    "diverged_seeds": diverged.get((task, model), []),
                    "metric": primary,
                    "n_parameters": meta[(task, model)]["n_parameters"],
                    "learning_rate": meta[(task, model)].get("learning_rate"),
                    "regularization": meta[(task, model)].get("regularization"),
                    "median_training_seconds": float(np.median(list(timing.values())))
                    if timing
                    else None,
                    "median_run_seconds": float(np.median(elapsed[task]))
                    if task in elapsed and not model.startswith("b4")
                    else None,
                }
                if observed:
                    row.update(
                        _summary(
                            [values[(task, model, primary)][s] for s in observed],
                            primary,
                        )
                    )
                row["status"] = "COMPLETE" if observed == seeds else "PARTIAL"
                systems.append(row)
                if model == quantum:
                    continue
                for metric in [primary, *secondary]:
                    if (task, model, metric) not in values:
                        continue
                    if diverged.get((task, model)) or diverged.get((task, quantum)):
                        comparisons.append(
                            {
                                "claim": claim,
                                "task": task,
                                "metric": metric,
                                "better_model": quantum,
                                "other_model": model,
                                "commensurability": "EVIDENCE_MISSING",
                                "reason": "a model diverged for at least one seed",
                            }
                        )
                        continue
                    result = paired_comparison(
                        values, seeds, (task, quantum), (task, model), metric
                    )
                    comparisons.append(
                        {
                            "claim": claim,
                            "task": task,
                            "commensurability": "PARTIALLY_COMMENSURABLE"
                            if model.startswith("b4")
                            else "COMMENSURABLE",
                            "reason": "B4 is matched on data, not parameters (reference)"
                            if model.startswith("b4")
                            else "same task, data, splits, metric, epochs, tuning rule; matched parameter count or the paper's own baseline",
                            **result,
                        }
                    )
    outcomes = {}
    for claim, tasks in CLAIMS.items():
        per_task = {}
        for task in tasks:
            primary, _ = _metrics_for(task)
            family = _family(task)
            if claim == "MerLin":
                needed = ["correlation_tanh"]
            elif claim == "C6":
                linear = (
                    "quantum_tfim_linear"
                    if family == "fig9_tfim"
                    else "quantum_heis_linear"
                )
                needed = ["b2_pauli_features", linear]
            else:
                needed = ["b2_pauli_features"]
            verdicts = {}
            for control in needed:
                match = [
                    row
                    for row in comparisons
                    if row["task"] == task
                    and row["other_model"] == control
                    and row["metric"] == primary
                ]
                verdicts[control] = (
                    match[0].get("verdict", match[0]["commensurability"])
                    if match
                    else "missing"
                )
            per_task[task] = {
                "verdicts": verdicts,
                "quantum_better": all(
                    value == "supported" for value in verdicts.values()
                ),
            }
        wins = sum(entry["quantum_better"] for entry in per_task.values())
        outcome = (
            "SUPPORTED"
            if wins == len(tasks)
            else ("PARTIALLY_SUPPORTED" if wins > 0 else "UNSUPPORTED")
        )
        outcomes[claim] = {
            "outcome": outcome,
            "tasks_won": wins,
            "tasks": len(tasks),
            "per_task": per_task,
        }
    return {
        "inputs": {
            "final": str(final_dir),
            "photonic_final": str(photonic_dir),
            "reference": str(reference_dir),
        },
        "systems": systems,
        "comparisons": comparisons,
        "b4_selection": b4_selection,
        "claim_outcomes": outcomes,
    }


def main() -> None:
    """Write the comparison JSON and CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("final_dir", type=Path)
    parser.add_argument("photonic_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    args = parser.parse_args()
    result = build(args.final_dir, args.photonic_dir, args.reference_dir)
    output = PROJECT_DIR / "results" / "p5_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = sorted(
        {key for row in result["systems"] for key in row},
        key=lambda key: (key not in ("claim", "task", "system", "role"), key),
    )
    with (PROJECT_DIR / "results" / "p5_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
            for row in result["systems"]
        )
    print(output)
    for claim, entry in result["claim_outcomes"].items():
        print(claim, entry["outcome"], f"{entry['tasks_won']}/{entry['tasks']}")


if __name__ == "__main__":
    main()
