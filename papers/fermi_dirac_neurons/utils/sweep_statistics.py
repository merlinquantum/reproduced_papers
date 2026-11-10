"""Paired statistics and paper placement for a completed paper sweep.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/sweep_statistics.py \
        papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1 \
        --out papers/fermi_dirac_neurons/results/paper_seeds_v1

Implements the comparison rule declared in the sweep spec before launch
(``configs/sweeps/paper_seeds_v1.json``, ``plan``):

* Pairs are formed by seed. Models in one config share the target draw and
  the states; a ``_classical`` config copies the data block of its
  ``_original`` config, so its models are paired with the original's models
  of the same seed.
* Regression metrics are compared as log10 of the loss; accuracies in
  percentage points. Difference = claimed-better model minus the other, signed
  so that a positive value favours the claimed-better model.
* Two-sided Wilcoxon signed-rank test (scipy, exact for n = 10). Verdict:
  ``tie`` when |median difference| is below the tolerance (0.1 decade, 1
  point), ``supported`` when the median favours the claimed-better model and
  p < 0.05, otherwise ``not_supported``.

Paper placement: each single paper value (released CSV last epoch, Table II,
Fig. 9 pixel reading) is located in our 10-seed distribution of the same
config and model as its rank (number of our seeds strictly below it).

Only ``metrics.json`` files listed as COMPLETED in ``sweep_status.json`` are
read; a missing seed or a non-finite value raises an error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.lib.data import load_reference_curve  # noqa: E402
from fermi_dirac_neurons.lib.paper_values import (  # noqa: E402
    FIG7_REFERENCE_CSV,
    FIG8_REFERENCE_CSV,
    FIG9_FINAL_TRAIN_LOSS_READING,
    TABLE_II_VALIDATION_ACCURACY_PCT,
)
from fermi_dirac_neurons.utils.compare_reference import default_data_root  # noqa: E402

LOSS_TOLERANCE_DECADES = 0.1
ACCURACY_TOLERANCE_POINTS = 1.0
ALPHA = 0.05
CONFIG_PATTERN = re.compile(r"(fig[789])_([a-z]+)_(\d)q_([a-z0-9_]+)")


def load_values(sweep_dir: Path) -> tuple[dict, list[int]]:
    """Return ``{(config, model, metric): {seed: value}}`` and the planned seeds."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"Sweep status is {status['status']}, not COMPLETED")
    seeds = sorted(set(status["spec"]["seeds"]))
    values: dict = {}
    for run in status["runs"]:
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        for model, record in metrics["models"].items():
            entries = {"train_loss": record["last_logged_train_loss"]}
            validation = record["final"]["validation"]
            for name in ("mse", "accuracy_linear", "accuracy_activation"):
                if name in validation:
                    entries[f"validation_{name}"] = validation[name]
            for metric, value in entries.items():
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {metric} in {run['run_id']}/{model}")
                bucket = values.setdefault((run["config"], model, metric), {})
                if run["seed"] in bucket:
                    raise ValueError(
                        f"Duplicate seed {run['seed']} for {run['config']}/{model}"
                    )
                bucket[run["seed"]] = value
    for key, bucket in values.items():
        if sorted(bucket) != seeds:
            raise ValueError(f"Incomplete seeds for {key}: {sorted(bucket)}")
    return values, seeds


def _transform(metric: str, value: float) -> float:
    if metric.startswith("validation_accuracy"):
        return 100.0 * value
    return math.log10(value) if value > 0 else -math.inf


def paired_comparison(
    values: dict, seeds: list[int], better: tuple, other: tuple, metric: str
) -> dict:
    """Compare two (config, model) pairs on one metric, paired by seed."""
    higher_is_better = metric.startswith("validation_accuracy")
    better_values = np.array(
        [_transform(metric, values[(*better, metric)][seed]) for seed in seeds]
    )
    other_values = np.array(
        [_transform(metric, values[(*other, metric)][seed]) for seed in seeds]
    )
    difference = (
        better_values - other_values
        if higher_is_better
        else other_values - better_values
    )
    finite = difference[np.isfinite(difference)]
    if len(finite) != len(difference):
        raise ValueError(
            f"Non-finite paired difference for {better} vs {other} on {metric}"
        )
    if higher_is_better:
        # Accuracy differences are exact multiples of 100 / n_states points;
        # rounding removes float noise so zero and tied differences are exact
        # for the Wilcoxon test (found by utils/recheck_paired_summaries.py).
        difference = np.round(difference, 9)
    median = float(np.median(difference))
    tolerance = (
        ACCURACY_TOLERANCE_POINTS if higher_is_better else LOSS_TOLERANCE_DECADES
    )
    p_value = (
        1.0
        if np.all(difference == 0)
        else float(wilcoxon(difference, alternative="two-sided").pvalue)
    )
    if abs(median) < tolerance:
        verdict = "tie"
    elif median > 0 and p_value < ALPHA:
        verdict = "supported"
    else:
        verdict = "not_supported"
    return {
        "metric": metric,
        "unit": "accuracy points" if higher_is_better else "decades of loss",
        "better_config": better[0],
        "better_model": better[1],
        "other_config": other[0],
        "other_model": other[1],
        "n_pairs": len(seeds),
        "median_advantage": median,
        "mean_advantage": float(np.mean(difference)),
        "min_advantage": float(np.min(difference)),
        "max_advantage": float(np.max(difference)),
        "seeds_favouring": int(np.sum(difference > 0)),
        "wilcoxon_p": p_value,
        "verdict": verdict,
    }


def planned_comparisons(values: dict) -> list[tuple]:
    """Return (better, other, metric) triples for every declared comparison."""
    models_by_config: dict = {}
    for config, model, _ in values:
        models_by_config.setdefault(config, set()).add(model)
    triples = []
    for config, models in sorted(models_by_config.items()):
        figure, family, n_text, variant = CONFIG_PATTERN.fullmatch(config).groups()
        if variant == "classical":
            continue
        metrics = (
            [
                "validation_accuracy_linear",
                "validation_accuracy_activation",
                "train_loss",
            ]
            if figure == "fig8"
            else ["train_loss", "validation_mse"]
        )
        quantum = sorted(model for model in models if model.startswith("quantum"))
        classical = sorted(model for model in models if model.startswith("classical"))
        pairs = [((config, q), (config, c)) for q in quantum for c in classical]
        if figure == "fig9":
            tanh = next(model for model in quantum if model.endswith("_tanh"))
            linear = next(model for model in quantum if model.endswith("_linear"))
            pairs.append(((config, tanh), (config, linear)))
        baseline_config = f"{figure}_{family}_{n_text}q_classical"
        if variant == "original" and baseline_config in models_by_config:
            for baseline in sorted(models_by_config[baseline_config]):
                for q in quantum:
                    if baseline == "b3_quantum_linear" and figure != "fig7":
                        continue
                    pairs.append(((config, q), (baseline_config, baseline)))
        triples += [
            (better, other, metric) for better, other in pairs for metric in metrics
        ]
    return triples


def paper_placement(values: dict, seeds: list[int]) -> list[dict]:
    """Locate each single paper value within our 10-seed distribution."""
    data_root = default_data_root()
    config_root = {"data_root": str(data_root)}
    rows = []

    def place(config, model, metric, paper_value, source):
        ours = np.array([values[(config, model, metric)][seed] for seed in seeds])
        rows.append(
            {
                "config": config,
                "model": model,
                "metric": metric,
                "paper_value": paper_value,
                "paper_source": source,
                "our_median": float(np.median(ours)),
                "our_min": float(ours.min()),
                "our_max": float(ours.max()),
                "our_seeds_below_paper": int(np.sum(ours < paper_value)),
                "inside_our_range": bool(ours.min() <= paper_value <= ours.max()),
            }
        )

    for (activation, n_qubits), filename in FIG7_REFERENCE_CSV.items():
        curve = load_reference_curve(config_root, filename)
        config = f"fig7_{activation}_{n_qubits}q_original"
        place(
            config,
            "quantum_tfim",
            "train_loss",
            float(curve["Quantum_Loss"][-1]),
            "released CSV",
        )
        place(
            config,
            "classical_im",
            "train_loss",
            float(curve["Classical_Loss"][-1]),
            "released CSV",
        )
    for n_qubits, filename in FIG8_REFERENCE_CSV.items():
        curve = load_reference_curve(config_root, filename)
        config = f"fig8_logistic_{n_qubits}q_original"
        for model, prefix in (
            ("quantum_heisenberg", "Quantum"),
            ("classical_fcim", "Classical"),
        ):
            place(
                config,
                model,
                "train_loss",
                float(curve[f"{prefix}_Loss"][-1]),
                "released CSV",
            )
            accuracies = curve[f"{prefix}_Accuracy_Pct"]
            last_logged = float(accuracies[np.isfinite(accuracies)][-1]) / 100
            place(
                config,
                model,
                "validation_accuracy_linear",
                last_logged,
                "released CSV, last logged",
            )
            place(
                config,
                model,
                "validation_accuracy_linear",
                TABLE_II_VALIDATION_ACCURACY_PCT[model][n_qubits] / 100,
                "Table II",
            )
    for (family, n_qubits), readings in FIG9_FINAL_TRAIN_LOSS_READING.items():
        for model, reading in readings.items():
            place(
                f"fig9_{family}_{n_qubits}q_original",
                model,
                "train_loss",
                reading,
                "Fig. 9 pixel reading",
            )
    return rows


def _write(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Compute and write the comparison and placement tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output path stem")
    args = parser.parse_args()
    values, seeds = load_values(args.sweep_dir)
    comparisons = [
        paired_comparison(values, seeds, *triple)
        for triple in planned_comparisons(values)
    ]
    placement = paper_placement(values, seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "sweep_dir": str(args.sweep_dir),
        "seeds": seeds,
        "tolerances": {
            "loss_decades": LOSS_TOLERANCE_DECADES,
            "accuracy_points": ACCURACY_TOLERANCE_POINTS,
        },
        "alpha": ALPHA,
        "comparisons": comparisons,
        "paper_placement": placement,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write(comparisons, args.out.with_name(args.out.name + "_comparisons.csv"))
    _write(placement, args.out.with_name(args.out.name + "_paper_placement.csv"))
    print(args.out.with_suffix(".json"))


if __name__ == "__main__":
    main()
