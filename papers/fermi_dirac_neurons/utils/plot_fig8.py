"""Fig. 8 / Table II reproduction: logistic-loss curves and validation accuracy vs n.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/plot_fig8.py \
        papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1

Writes two figures:

* ``fig8_reproduction``: training logistic loss (Eq. (56)) for n = 2…7,
  Heisenberg vs FCIM neurons, 10-seed median and min-max band, with the
  authors' released curves dashed. B2 is omitted here because its training
  loss is the classical logistic loss Eq. (52), a different quantity.
* ``table2_accuracy``: validation accuracy (linear prediction rule, as in the
  released code, D4) vs n: mean ± standard deviation over seeds for the
  Heisenberg neuron, the FCIM neuron and the fair baseline B2, with Table II's
  single-run values as hollow markers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fermi_dirac_neurons.lib.data import load_reference_curve  # noqa: E402
from fermi_dirac_neurons.lib.paper_values import (  # noqa: E402
    FIG8_REFERENCE_CSV,
    TABLE_II_VALIDATION_ACCURACY_PCT,
)
from fermi_dirac_neurons.lib.plotting import (  # noqa: E402
    COLOURS,
    ROLE_LABELS,
    SweepReader,
    apply_style,
    band,
    finish,
    model_role,
    save,
)
from fermi_dirac_neurons.utils.compare_reference import default_data_root  # noqa: E402

QUBITS = range(2, 8)


def plot_losses(reader: SweepReader, sweep_dir: Path) -> None:
    """Draw the six logistic-loss panels."""
    data_cfg = {"data_root": str(default_data_root())}
    figure, axes = plt.subplots(
        2, 3, figsize=(11, 6.5), constrained_layout=True, sharex=True
    )
    for axis, n_qubits in zip(axes.flat, QUBITS):
        config = f"fig8_logistic_{n_qubits}q_original"
        for model in ("quantum_heisenberg", "classical_fcim"):
            epochs, curves = reader.histories(config, model)
            role = model_role(model)
            label = (
                "Heisenberg neuron" if role == "quantum" else "FCIM (commuting) neuron"
            )
            band(axis, epochs, curves, role, label)
        reference = load_reference_curve(data_cfg, FIG8_REFERENCE_CSV[n_qubits])
        axis.plot(
            reference["Epoch"],
            reference["Quantum_Loss"],
            color=COLOURS["quantum"],
            linestyle="--",
            linewidth=1.2,
            label="paper CSV: Heisenberg",
        )
        axis.plot(
            reference["Epoch"],
            reference["Classical_Loss"],
            color=COLOURS["commuting"],
            linestyle="--",
            linewidth=1.2,
            label="paper CSV: FCIM",
        )
        axis.set_xlim(0, int(epochs[-1]))
        axis.set_title(f"{n_qubits} qubits")
        axis.set_ylabel("training logistic loss")
        if n_qubits >= 5:
            axis.set_xlabel("epoch")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    finish(
        figure,
        handles,
        labels,
        "Fig. 8 reproduction: logistic loss, median over 10 seeds (band: min–max), authors' released curves dashed",
        f"paper-accurate (released-code protocol: 1000 Haar training states, labels sign Tr[H*ρ], η_eff = 0.05; seeds {reader.seeds[0]}–{reader.seeds[-1]}, sweep {reader.sweep_id}); the n = 6 CSV runs 750 epochs, ours 500",
        columns=4,
    )
    for path in save(figure, "fig8_reproduction", sweep_dir):
        print(path)


def _final_accuracies(reader: SweepReader, config: str, model: str) -> np.ndarray:
    return 100 * np.array(
        [
            run["models"][model]["final"]["validation"]["accuracy_linear"]
            for run in reader.metrics(config)
        ]
    )


def plot_accuracy(reader: SweepReader, sweep_dir: Path) -> None:
    """Draw validation accuracy vs n against Table II."""
    figure, axis = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    qubits = np.array(list(QUBITS))
    series = (
        ("original", "quantum_heisenberg", "Heisenberg neuron (ours)"),
        ("original", "classical_fcim", "FCIM (commuting) neuron (ours)"),
        ("classical", "b2_pauli_features", ROLE_LABELS["b2"] + " (ours)"),
    )
    for offset, (variant, model, label) in zip((-0.08, 0.0, 0.08), series):
        values = np.array(
            [
                _final_accuracies(reader, f"fig8_logistic_{n}q_{variant}", model)
                for n in qubits
            ]
        )
        axis.errorbar(
            qubits + offset,
            values.mean(1),
            yerr=values.std(1, ddof=1),
            color=COLOURS[model_role(model)],
            marker="o",
            markersize=5,
            linewidth=1.6,
            capsize=3,
            label=label,
        )
    for model, label in (
        ("quantum_heisenberg", "Table II: quantum"),
        ("classical_fcim", "Table II: classical"),
    ):
        table = [TABLE_II_VALIDATION_ACCURACY_PCT[model][n] for n in qubits]
        axis.plot(
            qubits,
            table,
            linestyle="none",
            marker="D",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.4,
            color=COLOURS[model_role(model)],
            label=label,
        )
    axis.set_xticks(qubits)
    axis.set_xlabel("qubits n")
    axis.set_ylabel("validation accuracy (%)")
    axis.set_ylim(50, 100)
    handles, labels = axis.get_legend_handles_labels()
    finish(
        figure,
        handles,
        labels,
        "Table II reproduction: validation accuracy on 500 Haar states (mean ± std over 10 seeds)",
        f"paper-accurate (released-code protocol, linear prediction rule sign Tr[H(θ)ρ] (D4); seeds {reader.seeds[0]}–{reader.seeds[-1]}, sweep {reader.sweep_id})",
        columns=2,
    )
    for path in save(figure, "table2_accuracy", sweep_dir):
        print(path)


def main() -> None:
    """Draw both Fig. 8 figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    reader = SweepReader(args.sweep_dir)
    apply_style()
    plot_losses(reader, args.sweep_dir)
    plot_accuracy(reader, args.sweep_dir)


if __name__ == "__main__":
    main()
