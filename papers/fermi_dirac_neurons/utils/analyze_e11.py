"""E11 analysis: training success of the Fig. 7a protocol vs qubit count.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_e11.py \
        papers/fermi_dirac_neurons/outdir/sweeps/e11_training_vs_n_v4

Applies the rule declared before launch: per (n, training-state set, model),
the median and min-max over seeds of the training squared loss logged at the
last epoch and of the validation MSE; paired by seed, the quantum-minus-IM
difference in log10 training loss as a median and the number of seeds
favouring each model. No significance test: with 5 pairs a two-sided
Wilcoxon cannot reach p < 0.05. Writes ``results/e11_training_vs_n_summary.json``
and the figure ``e11_training_vs_n.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fermi_dirac_neurons.lib.plotting import (  # noqa: E402
    COLOURS,
    apply_style,
    finish,
    model_role,
    save,
)

CONFIG_PATTERN = re.compile(r"e11_tanh_(\d)q_(code|haar)")
MODEL_LABELS = {
    "quantum_tfim": "TFIM neuron (quantum)",
    "classical_im": "IM neuron (commuting)",
}


def load(sweep_dir: Path) -> dict:
    """Return ``{(n, state_set): {model: {seed: (train_loss, validation_mse)}}}``."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"Sweep is {status['status']}, not COMPLETED")
    table: dict = defaultdict(lambda: defaultdict(dict))
    for run in status["runs"]:
        n_text, state_set = CONFIG_PATTERN.fullmatch(run["config"]).groups()
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        for model, record in metrics["models"].items():
            values = (
                record["last_logged_train_loss"],
                record["final"]["validation"]["mse"],
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite metric in {run['run_id']}/{model}")
            table[(int(n_text), state_set)][model][run["seed"]] = values
    return table


def summarize(table: dict) -> list[dict]:
    """Medians, ranges and paired quantum-minus-IM differences per (n, state set)."""
    rows = []
    for (n_qubits, state_set), models in sorted(table.items()):
        seeds = sorted(models["quantum_tfim"])
        if seeds != sorted(models["classical_im"]):
            raise ValueError(f"Unpaired seeds at n = {n_qubits}, {state_set}")
        row = {"n_qubits": n_qubits, "state_set": state_set, "seeds": seeds}
        for model, by_seed in models.items():
            train = np.array([by_seed[seed][0] for seed in seeds])
            validation = np.array([by_seed[seed][1] for seed in seeds])
            row[model] = {
                "train_loss_median": float(np.median(train)),
                "train_loss_min": float(train.min()),
                "train_loss_max": float(train.max()),
                "validation_mse_median": float(np.median(validation)),
            }
        advantage = np.array(
            [
                math.log10(models["classical_im"][seed][0])
                - math.log10(models["quantum_tfim"][seed][0])
                for seed in seeds
            ]
        )
        row["quantum_advantage_decades_median"] = float(np.median(advantage))
        row["seeds_favouring_quantum"] = int(np.sum(advantage > 0))
        rows.append(row)
    return rows


def plot(rows: list[dict], sweep_dir: Path) -> None:
    """Median training loss vs n with min-max band, one panel per state set."""
    apply_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4.4), constrained_layout=True, sharey=True
    )
    for axis, state_set in zip(axes, ("code", "haar")):
        cells = [row for row in rows if row["state_set"] == state_set]
        qubits = [row["n_qubits"] for row in cells]
        for model, label in MODEL_LABELS.items():
            colour = COLOURS[model_role(model)]
            axis.fill_between(
                qubits,
                [row[model]["train_loss_min"] for row in cells],
                [row[model]["train_loss_max"] for row in cells],
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
            axis.plot(
                qubits,
                [row[model]["train_loss_median"] for row in cells],
                color=colour,
                marker="o",
                markersize=4,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_xticks(qubits)
        axis.set_xlabel("qubits n")
        axis.set_title(
            "authors' training states (2^(n+1)+5)"
            if state_set == "code"
            else "Haar-random training states (same count)"
        )
    axes[0].set_ylabel("training squared loss after 2000 epochs")
    handles, labels = axes[0].get_legend_handles_labels()
    finish(
        figure,
        handles,
        labels,
        "E11: Fig. 7a protocol (tanh, η = 0.1, 2000 epochs) vs qubit count — median over 5 seeds (band: min–max)",
        "extension (not in the paper); seeds 0–4; n = 2…8 (n = 9 dropped for run time, LOG.md)",
        columns=2,
    )
    for path in save(figure, "e11_training_vs_n", sweep_dir):
        print(path)


def main() -> None:
    """Write the summary and the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    rows = summarize(load(args.sweep_dir))
    output = PROJECT_DIR / "results" / "e11_training_vs_n_summary.json"
    output.write_text(
        json.dumps({"sweep_dir": str(args.sweep_dir), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    plot(rows, args.sweep_dir)


if __name__ == "__main__":
    main()
